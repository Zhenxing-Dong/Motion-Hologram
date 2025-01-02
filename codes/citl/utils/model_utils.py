import torch
import torch.nn as nn
import os
from collections import OrderedDict
# import sys 
# sys.path.append("..") 
# from unet import norm_layer

def norm_layer(norm_str):
    if norm_str.lower() == 'instance':
        return nn.InstanceNorm2d
    elif norm_str.lower() == 'group':
        return nn.GroupNorm
    elif norm_str.lower() == 'batch':
        return nn.BatchNorm2d


def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

# def load_newcheckpoint(model, pretrained_weights):
#     pretrained_dict = torch.load(pretrained_weights)
#     model_dict = model.state_dict()

#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

#     model_dict.update(pretrained_dict)

#     model.load_state_dict(model_dict)


def load_partial_and_freeze(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        if name in pretrained_dict:
            param.requires_grad = False







def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import CNNpropCNN, Multi_CNNpropCNN

    arch = opt.arch

    ## propagation parameter
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    prop_dist = [-3 * mm, -2 * mm, -1 * mm, 0 * mm, 1 * mm, 2 * mm, 3 * mm]
    # prop_dist = [100 * mm]
    # prop_dist = [-1 * mm, 1 * mm]
    mid_prop = 0 * cm

    prop_dist = [x + mid_prop for x in prop_dist]

    wavelength= 680 * nm
    feature_size = (3.74* um, 3.74 * um)

    print('You choose '+ arch +'...')
    if arch == 'CNNpropCNN':
        model_citl = CNNpropCNN(prop_dist, wavelength, feature_size,
                                prop_type='ASM',
                                F_aperture=opt.F_aperture,
                                num_downs_slm=opt.num_downs_slm,
                                num_feats_slm_min=opt.num_feats_slm_min,
                                num_feats_slm_max=opt.num_feats_slm_max,
                                num_downs_target=opt.num_downs_target,
                                num_feats_target_min=opt.num_feats_target_min,
                                num_feats_target_max=opt.num_feats_target_max,
                                slm_latent_amp=opt.slm_latent_amp,
                                slm_latent_phase=opt.slm_latent_phase,
                                norm=norm_layer(opt.norm)
                                )
        
    elif arch == 'Multi_CNNpropCNN':
        model_citl = Multi_CNNpropCNN(prop_dist, wavelength, feature_size,
                                      prop_type='ASM',
                                      F_aperture=opt.F_aperture,
                                      num_downs_slm=opt.num_downs_slm,
                                      num_feats_slm_min=opt.num_feats_slm_min,
                                      num_feats_slm_max=opt.num_feats_slm_max,
                                      num_downs_target=opt.num_downs_target,
                                      num_feats_target_min=opt.num_feats_target_min,
                                      num_feats_target_max=opt.num_feats_target_max,
                                      slm_latent_amp=opt.slm_latent_amp,
                                      slm_latent_phase=opt.slm_latent_phase,
                                      norm=norm_layer(opt.norm)
                                      )    
    else:
        raise Exception("Arch error!")

    return model_citl