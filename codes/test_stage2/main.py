import os
from dataset_load import *
from prop_ideal import *
import torch
import numpy as np
import utils
import optics
from optimization import *
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

## load dataset
root_dir = ''

dataset = FocalstackDataset(root_dir)

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

## propagation parameter
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = [-3 * mm, -2 * mm, -1 * mm, 0 * mm, 1 * mm, 2 * mm, 3 * mm]
mid_prop = 0 * mm

prop_dist = [x + mid_prop for x in prop_dist]

channel = 1  ## blue:0 ; green:1; red: 2

wavelength= 520 * nm  ## blue:450 ; green:520; red: 680

feature_size = (3.74 * um, 3.74 * um)
sim_prop = Propagation(prop_dist, wavelength, feature_size, 'ASM', F_aperture=1, dim=1)

## save files
save_files = ''

if not os.path.exists(save_files):
    os.makedirs(save_files)

for i, images_mask in enumerate(dataloader, 0): 
    print(i)
    focal_stack = torch.cat(images_mask, dim=0)

    target_amp = focal_stack[:, channel, :, :].unsqueeze(1)

    H,W = target_amp.shape[2], target_amp.shape[3]

    init_phase = (-0.5  + 1.0 * torch.rand(1, 1, 1152, 2048)).cuda()
    filter = optics.np_filter(1, 1, 1152, 2048, 1152//2, 2048//2)
    filter = torch.tensor(filter, dtype=torch.float32).cuda()
    
    ## optimizarion
    final_phase, _ = FocalStackShift(init_phase, target_amp, sim_prop, filter, num_iters = 2000, 
                                  lr = 0.05)
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)
    cv2.imwrite(os.path.join(save_files, f'{i:04d}.png'), phase_out_8bit)
