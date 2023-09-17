import numpy as np, copy
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer as bmine
import criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()

        self.opt = opt

        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        self.name           = 'S2SD'
        self.d_mode         = 'cosine'
        self.iter_count     = 0
        self.embed_dim      = opt.embed_dim

        self.optim_dict_list = []

        self.T      = opt.loss_s2sd_T
        self.w      = opt.loss_s2sd_w
        self.feat_w = opt.loss_s2sd_feat_w
        self.pool_aggr     = opt.loss_s2sd_pool_aggr
        self.match_feats   = opt.loss_s2sd_feat_distill
        self.max_feat_iter = opt.loss_s2sd_feat_distill_delay

        f_dim = 1024 if 'bninception' in opt.arch else 2048
        self.target_nets  = torch.nn.ModuleList([nn.Sequential(nn.Linear(f_dim, t_dim), nn.ReLU(), nn.Linear(t_dim, t_dim)) for t_dim in opt.loss_s2sd_target_dims])
        self.optim_dict_list.append({'params':self.target_nets.parameters(), 'lr':opt.lr})

        old_embed_dim = copy.deepcopy(opt.embed_dim)
        self.target_criteria = nn.ModuleList()
        for t_dim in opt.loss_s2sd_target_dims:
            opt.embed_dim = t_dim

            batchminer       = bmine.select(opt.batch_mining, opt)
            target_criterion = criteria.select(opt.loss_s2sd_target, opt, batchminer=batchminer)
            self.target_criteria.append(target_criterion)

            if hasattr(target_criterion, 'optim_dict_list'):
                self.optim_dict_list.extend(target_criterion.optim_dict_list)
            else:
                self.optim_dict_list.append({'params':target_criterion.parameters(), 'lr':opt.lr})

        opt.embed_dim = old_embed_dim
        batchminer   = bmine.select(opt.batch_mining, opt)
        self.source_criterion = criteria.select(opt.loss_s2sd_source, opt, batchminer=batchminer)

        if hasattr(self.source_criterion, 'optim_dict_list'):
            self.optim_dict_list.extend(self.source_criterion.optim_dict_list)
        else:
            self.optim_dict_list.append({'params':self.source_criterion.parameters(), 'lr':opt.lr})



    def prep(self, thing):
        return 1.*torch.nn.functional.normalize(thing, dim=1)


    def forward(self, batch, labels, batch_features, avg_batch_features, f_embed, **kwargs):
        ###
        bs          = len(batch)
        batch       = self.prep(batch)
        self.labels = labels.unsqueeze(1)

        source_loss = self.source_criterion(batch, labels, batch_features=batch_features, f_embed=f_embed, **kwargs)
        source_smat = self.smat(batch, batch, mode=self.d_mode)
        loss        = source_loss

        if self.pool_aggr:
            avg_batch_features = nn.AdaptiveAvgPool2d(1)(batch_features).view(bs,-1)+nn.AdaptiveMaxPool2d(1)(batch_features).view(bs,-1)
        else:
            avg_batch_features = avg_batch_features.view(bs,-1)

        kl_divs, target_losses  = [], []
        for i,out_net in enumerate(self.target_nets):
            target_batch   = F.normalize(out_net(avg_batch_features.view(bs, -1)), dim=-1)
            target_loss    = self.target_criteria[i](target_batch, labels, batch_features=batch_features, f_embed=f_embed, **kwargs)
            target_smat    = self.smat(target_batch, target_batch, mode=self.d_mode)

            kl_divs.append(self.kl_div(source_smat, target_smat.detach()))
            target_losses.append(target_loss)

        loss = (torch.mean(torch.stack(target_losses)) + loss)/2. + self.w*torch.mean(torch.stack(kl_divs))

        if self.match_feats and self.iter_count>=self.max_feat_iter:
            n_avg_batch_features = F.normalize(avg_batch_features, dim=-1).detach()
            avg_feat_smat        = self.smat(n_avg_batch_features, n_avg_batch_features, mode=self.d_mode)
            avg_batch_kl_div     = self.kl_div(source_smat, avg_feat_smat.detach())
            loss += self.feat_w*avg_batch_kl_div

        self.iter_count+=1

        return loss



    def kl_div(self, A, B):
        log_p_A = F.log_softmax(A/self.T, dim=-1)
        p_B     = F.softmax(B/self.T, dim=-1)
        kl_div  = F.kl_div(log_p_A, p_B, reduction='sum') * (self.T**2) / A.shape[0]
        return kl_div


    def smat(self, A, B, mode='cosine'):
        if mode=='cosine':
            return A.mm(B.T)
        elif mode=='euclidean':
            As, Bs = A.shape, B.shape
            return (A.mm(A.T).diag().unsqueeze(-1)+B.mm(B.T).diag().unsqueeze(0)-2*A.mm(B.T)).clamp(min=1e-20).sqrt()
