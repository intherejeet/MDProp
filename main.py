"""==================================================================================================="""
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import parameters    as par
"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.attack_parameters(parser)
### Include S2SD Parameters
parser = par.s2sd_parameters(parser)
##### Read in parameters
opt = parser.parse_args()
"""==================================================================================================="""
# setting automated naming of savefolder depending on the parameter setting
opt.savename = str(opt.loss)+'_'+str(opt.batch_mining)+'_attacktargets_'+str(opt.attacker1_targets)+'_'+str(opt.attacker2_targets)+'_epsilon_'+str(opt.epsilon)+'_attackitrs_'+str(opt.attack_itr)+'_seed_'+str(opt.seed)

"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import metrics       as metrics
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger
from architectures import net
import torch.backends.cudnn as cudnn
# from architectures import attacker
from architectures.attacker import PGDAttacker
from functools import partial
from torchsummary import summary

"""==================================================================================================="""
full_training_start_time = time.time()
"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'
opt.pretrained = not opt.not_pretrained
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


######################################################
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

print('Loading pretrained ResNet50 model')
model      = net.__dict__['resnet50'](num_classes=opt.embed_dim, norm_layer=MixBatchNorm2d, opt=opt)
# Check if ResNet50 parameters file exists in the params folder
params_file = os.path.join(os.getcwd(), 'params', 'resnet50-19c8e357.pth')
if not os.path.exists(params_file):
    raise FileNotFoundError("ResNet50 official PyTorch parameters file not found in the 'params' folder. "
                            "Please download and place them in the 'params' folder.")

# Load ResNet50 parameters
pretrained_params = torch.load(params_file)

model.set_attacker(attacker1,attacker2)

print("Applying to the Mix status")
model.set_mixbn(True)
model.apply(to_mix_status)


##################### Loading partial pretrained parameters #######################

model_dict = model.state_dict()
for name, param in pretrained_params.items():
    if name in model_dict.keys():
        if name not in ['fc.weight', 'fc.bias']:
            model_dict[name].copy_(param)   
        else:
            pass
    else:
        continue

model.load_state_dict(model_dict)
### Initializing aux_bn parameters
model_dict = model.state_dict()
for name, param in model.named_parameters():
    if 'aux_bn1' in name:
        target = name.replace('.aux_bn1', '')
        model_dict[name]=model_dict[target]
    if 'aux_bn2' in name:
        target = name.replace('.aux_bn2', '')
        model_dict[name]=model_dict[target]
model.load_state_dict(model_dict)

##################################################################################
model.train()

if opt.fc_lr<0 and opt.separate_bn_lr==False:
    print("Assigning same learning rates for all the parameters...")
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]

elif opt.separate_bn_lr:
    print("Assigning different learning rates for BN layers...")
    aux_bn_params = [x[-1] for x in list(filter(lambda x: 'aux_bn' in x[0], model.named_parameters()))]
    main_bn_params = [x[-1] for x in list(filter(lambda x: 'aux_bn' not in x[0] and 'bn' in x[0], model.named_parameters()))]
    other_params = [x[-1] for x in list(filter(lambda x: 'aux_bn' not in x[0] and 'bn' not in x[0], model.named_parameters()))]
    to_optim          = [{'params':other_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':main_bn_params,'lr':opt.main_bn_lr,'weight_decay':opt.decay},
                         {'params':aux_bn_params,'lr':opt.aux_bn_lr,'weight_decay':opt.decay}]
else:
    print("Using different learning rates for FC layer parameters...")
    all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'fc' not in x[0], model.named_parameters()))]
    fc_params         = model.fc.parameters()
    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.fc_lr,'weight_decay':opt.decay}]


if 'frozen' in opt.arch:
    print("Freezing the BatchNorm layer parameters...")
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
            module.train = lambda _: None

if 'freeze_main' in opt.arch:
    print("Freezing the Main BatchNorm layer parameters only...")
    for name, module in model.named_modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d) and 'aux_bn' not in name:
            ### Freezing batch norms from the original implementation
            module.eval()
            module.train = lambda _: None



model = nn.DataParallel(model)
_  = model.to("cuda")

######################################################
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
"""============================================================================"""
#################### LOSS SETUP ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
_ = criterion.to(opt.device)
if 'criterion' in train_data_sampler.name:
    train_data_sampler.internal_criterion = criterion
"""============================================================================"""
#################### OPTIM SETUP ####################
if opt.optim == 'adam':
    optimizer    = torch.optim.Adam(to_optim)
elif opt.optim == 'sgd':
    optimizer    = torch.optim.SGD(to_optim, momentum=0.9)
else:
    raise Exception('Optimizer <{}> not available!'.format(opt.optim))
scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)
"""============================================================================"""
################### Summary #########################3
data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary    = data_text+'\n'+setup_text+'\n'+miner_text+'\n'+arch_text
print(summary)


"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}


for epoch in range(opt.n_epochs):
    epoch_start_time = time.time()
    

    if epoch>0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
        train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

    opt.epoch = epoch
    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    """======================================="""
    if train_data_sampler.requires_storage:
        train_data_sampler.precompute_indices()

    """======================================="""
    ### Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = []
    data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))


    for i,out in enumerate(data_iterator):
        class_labels, input, input_indices = out
        ### Compute Embedding
        input      = input.to(opt.device)

        model_args = {'x':input.to(opt.device)}
        model_args['labels'] = class_labels

        out_dict, target_labels  = model(**model_args)
        embeds, avg_features, features, extra_embeds = [out_dict[key] for key in ['embeds', 'avg_features', 'features', 'extra_embeds']]
        ### Compute Loss
        loss_args['input_batch']    = input
        loss_args['batch']          = embeds
        loss_args['labels']         = target_labels
        loss_args['f_embed']        = model.module.fc
        loss_args['batch_features'] = features.reshape((-1,2048,7,7))
        loss_args['avg_batch_features'] = avg_features
        loss_total      = criterion(**loss_args)

        ###
        optimizer.zero_grad()
        loss_total.backward()
        ### Compute Model Gradients and log them!
        grads              = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

        ### Update network weights!
        optimizer.step()
        optimizer.zero_grad()
        ###
        loss_collect.append(loss_total.item())

        ###
        iter_count += 1

        if i==len(dataloaders['training'])-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

        """======================================="""
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)

    result_metrics = {'loss': np.mean(loss_collect)}

    ####
    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))



    """======================================="""
    ### Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()
    print('\nComputing Testing Metrics...')
    eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing']],    model, opt, opt.evaltypes, opt.device, log_key='Test')
    if opt.use_tv_split:
        print('\nComputing Validation Metrics...')
        eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['validation']], model, opt, opt.evaltypes, opt.device, log_key='Val')
    if not opt.no_train_metrics:
        print('\nComputing Training Metrics...')
        eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['evaluation']], model, opt, opt.evaltypes, opt.device, log_key='Train')

    LOG.update(all=True)


    """======================================="""
    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('Total Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
    print('\n-----\n')

"""======================================================="""
### CREATE A SUMMARY TEXT FILE
summary_text = ''
full_training_time = time.time()-full_training_start_time
summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))

summary_text += '---------------\n'
for sub_logger in LOG.sub_loggers:
    metrics       = LOG.graph_writer[sub_logger].ov_title
    summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), metrics)

with open(opt.save_path+'/training_summary.txt','w') as summary_file:
    summary_file.write(summary_text)
