################### LIBRARIES ###################
### Basic Libraries
from audioop import avg
from tkinter.font import names
from traceback import print_tb
import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import parameters    as par
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.attack_parameters(parser)
parser = par.s2sd_parameters(parser)
opt = parser.parse_args()

### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import metrics       as metrics
import evaluation    as eval
from utilities import logger
from architectures import net
from architectures.attacker import PGDAttacker
from functools import partial

opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])
opt.kernels = 4*len(opt.gpu[0].replace(",",""))
"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)
"""==================================================================================================="""

##################### NETWORK SETUP ##################
class MixBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn1 = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.aux_bn2 = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv1':
            input = self.aux_bn1(input)
        elif self.batch_type == 'adv2':
            input = self.aux_bn2(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn1(input[batch_size // 2:3*(batch_size // 4)])
            input2 = self.aux_bn2(input[3*(batch_size // 4):])
            input = torch.cat((input0, input1, input2), 0)
        return input

attacker1 = PGDAttacker(num_iter=opt.attack_itr,epsilon=opt.epsilon, masterface_targets=opt.attacker1_targets)
attacker2 = PGDAttacker(num_iter=opt.attack_itr,epsilon=opt.epsilon, masterface_targets=opt.attacker2_targets)

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status
to_clean_status = partial(to_status, status='clean')
to_adv1_status = partial(to_status, status='adv1')
to_adv2_status = partial(to_status, status='adv2')
to_mix_status = partial(to_status, status='mix')

opt.device = torch.device('cuda')

# Loading ResNet50 model
model      = net.__dict__['resnet50'](num_classes=opt.embed_dim, norm_layer=MixBatchNorm2d, opt=opt)
model.set_attacker(attacker1,attacker2)
model = nn.DataParallel(model)
_  = model.to("cuda")
######################################################


"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets    = datasets.select(opt.dataset, opt, opt.source_path)
dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['testing']    = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
if opt.use_tv_split:
    dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels, batch_size=opt.bs,shuffle=False)
train_data_sampler      = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict, datasets['training'].image_list)
if train_data_sampler.requires_storage:
    train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)
opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)
"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)
"""============================================================================"""

if opt.eval_params=='cub200_multisimilarity':
    param_dir = './author_params/CUB200_MULTISIMILARITY'

if opt.eval_params=='cub200_s2sd':
    param_dir = './author_params/CUB200_S2SD'

if opt.eval_params=='cars196_multisimilarity':
    param_dir = './author_params/CARS196_MULTISIMILARITY'

if opt.eval_params=='sop_multisimilarity':
    param_dir = './author_params/ONLINE_PRODUCTS_MULTISIMILARITY'

checkpoints = torch.load(param_dir+'/checkpoint_Test_discriminative_e_recall@1.pth.tar')
pretrained_params = checkpoints['state_dict']
model.load_state_dict(pretrained_params)
_ = model.eval()

if 'multisimilarity' in opt.eval_params:
    eval_loss = 'Multisimilarity'
elif 's2sd' in opt.eval_params:
    eval_loss = 's2sd'

print('\n\nMDProp Evaluation Data: '+str(opt.dataset))
print('\nTraining Loss Function: '+eval_loss+'\n')


eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing']],    model, opt, opt.evaltypes, opt.device, log_key='Test')
if opt.use_tv_split:
    print('\nComputing Validation Metrics...')
    eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['validation']], model, opt, opt.evaltypes, opt.device, log_key='Val')
if not opt.no_train_metrics:
    print('\nComputing Training Metrics...')
    eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['evaluation']], model, opt, opt.evaltypes, opt.device, log_key='Train')

