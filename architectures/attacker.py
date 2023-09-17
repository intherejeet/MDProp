import torch 
import torch.nn as nn 
import torch.nn.functional as F

class PGDAttacker():
    def __init__(self, num_iter, epsilon, masterface_targets=3 ,kernel_size=15, device='cuda'):
        step_size = epsilon / num_iter
        self.num_iter = num_iter
        self.epsilon = epsilon 
        self.step_size = step_size
        self.masterface_targets = masterface_targets
        self.device=device
    
    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=199)
        return (label + label_offset) % 199

    def attack_feature(self, image_clean, label, model, original=False):
        image_clean_saved = image_clean.clone()
        out_dict  = model(image_clean_saved.to(self.device))
        embeds, avg_features, features, extra_embeds = [out_dict[key] for key in ['embeds', 'avg_features', 'features', 'extra_embeds']]
        embeddings_tar_list = []
        for q in range(self.masterface_targets):
            embeddings_tar_list.append(embeds.detach().clone()[torch.randperm(embeds.size()[0])])

        lower_bound = torch.clamp(image_clean - self.epsilon, min=image_clean.min().item(), max=image_clean.max().item())
        upper_bound = torch.clamp(image_clean + self.epsilon, min=image_clean.min().item(), max=image_clean.max().item())

        init_start = torch.empty_like(image_clean).uniform_(-0.0005, 0.0005)
        start_adv = image_clean + init_start

        adv = start_adv
        for i in range(self.num_iter):
            adv.requires_grad = True
            out_dict  = model(adv.to(self.device))
            adv_feat, avg_features, features, extra_embeds = [out_dict[key] for key in ['embeds', 'avg_features', 'features', 'extra_embeds']]
            
            loss_attack = torch.Tensor([0]).to(self.device)
            for embeds_tar in embeddings_tar_list:
                diff = (embeds_tar - adv_feat)
                loss_attack += torch.mean(torch.norm(diff, p=2, dim=1))
            loss_attack = (loss_attack/self.masterface_targets).pow(2) 
            ####################################################
            
            g = torch.autograd.grad(loss_attack, adv, 
                                    retain_graph=False, create_graph=False)[0]
            if original:
                adv = adv + torch.sign(g)*self.step_size
            else:
                adv = adv - torch.sign(g) * self.step_size
            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
        return adv, label

