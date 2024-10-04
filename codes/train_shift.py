import os
from dataset_load import *
from prop_ideal import *
import torch
import numpy as np
import utils
import optics
from optimization import *
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

## load dataset
# root_dir = '/mnt/data/zhenxing/multi_plane'
# depth_dir = '/mnt/data/zhenxing/1024_physical_depth'
# dataset = FocalstackDDataset(root_dir, depth_dir)

## load dataset
root_dir = '/mnt/data/zhenxing/data/ar/images'

dataset = FocalstackDataset(root_dir)


batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


## propagation parameter
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = [-3 * mm, -2 * mm, -1 * mm, 0 * mm, 1 * mm, 2 * mm, 3 * mm]
mid_prop = 0 * cm

prop_dist = [x + mid_prop for x in prop_dist]

channel = 1  ## blue:0 ; green:1; red: 2

wavelength= 520 * nm  ## blue:450 ; green:520; red: 680/633

feature_size = (3.74 * um, 3.74 * um)
sim_prop = Propagation(prop_dist, wavelength, feature_size, 'ASM', F_aperture=1, dim=1)


## save files
save_files = '/mnt/data/zhenxing/data/ar/hologram/proposed'
if not os.path.exists(save_files):
    os.makedirs(save_files)

for i, images in enumerate(dataloader, 0): 
    print(i)

    focal_stack = torch.cat(images, dim=0)

    mask = 0
    target_amp = focal_stack[:, channel, :, :].unsqueeze(1)
    H,W = target_amp.shape[2], target_amp.shape[3]

    init_num_iters, init_phase_range, init_learning_rate = utils.random_gen()

    print(init_num_iters, init_phase_range, init_learning_rate)

    init_phase = (init_phase_range * (-0.5  + 1.0 * torch.rand(1, 1, 574, 1022))).cuda()
    circ_filter = optics.np_filter(1, 1, 576, 1024, 576//2, 576//2)
    circ_filter = torch.tensor(circ_filter, dtype=torch.float32).cuda()

    ## optimizarion
    final_phase = FocalStackShift(init_phase, target_amp, mask, sim_prop, circ_filter, num_iters = init_num_iters, 
                                  lr = init_learning_rate)
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)
    cv2.imwrite(os.path.join(save_files, str(i+1349).zfill(4) + '.png'), phase_out_8bit)
