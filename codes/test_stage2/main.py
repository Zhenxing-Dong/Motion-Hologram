import os
from dataset_load import *
from prop_ideal import *
import torch
import numpy as np
import utils
import optics
from optimization import *
import time
import argparse

parser = argparse.ArgumentParser(description='CGH Optimization using Motion Hologram')

parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', type=str, default ='./data/example',  
                    help='dir of test data')
parser.add_argument('--hologram_dir', type=str, default ='./hologram',  
                    help='dir of hologram data')
parser.add_argument('--recon_dir', type=str, default ='./recon',  
                    help='dir of hologram data')
parser.add_argument('--channel', type=int, default = 1,  help='color')
parser.add_argument('--pixel_size', type=float, default = 3.74,  help='SLM pixel pitch')
parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
parser.add_argument('--num_iters', type=int, default = 2000,  help='iterations')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

## load dataset
dataset = FocalstackDataset(args.data_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

## propagation parameter
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = [-3 * mm, -2 * mm, -1 * mm, 0 * mm, 1 * mm, 2 * mm, 3 * mm]
mid_prop = 0 * mm

prop_dist = [x + mid_prop for x in prop_dist]

wavelength= [450* nm, 520 * nm, 680*nm][args.channel]  ## blue:450 ; green:520; red: 680

feature_size = (args.pixel_size * um, args.pixel_size * um)
sim_prop = Propagation(prop_dist, wavelength, feature_size, 'ASM', F_aperture=1, dim=1)

## save files
if not os.path.exists(args.hologram_dir):
    os.makedirs(args.hologram_dir)
    os.makedirs(args.recon_dir)

for i, images in enumerate(dataloader, 0): 
    print(i)
    focal_stack = torch.cat(images, dim=0)

    target_amp = focal_stack[:, args.channel, :, :].unsqueeze(1)

    H,W = target_amp.shape[2], target_amp.shape[3]

    init_phase = (-0.5  + 1.0 * torch.rand(1, 1, 1152, 2048)).cuda()
    filter = optics.np_filter(1, 1, 1152, 2048, 1152//2, 2048//2)
    filter = torch.tensor(filter, dtype=torch.float32).cuda()
    
    ## optimizarion
    final_phase, recon_amp = FocalStackShift(init_phase, target_amp, sim_prop, filter, 
                                             num_iters = args.num_iters, lr = args.lr)
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)
    cv2.imwrite(os.path.join(args.hologram_dir, f'{i:04d}.png'), phase_out_8bit)

    recon_amp = recon_amp.squeeze().cpu().detach().numpy()
    recon_srgb = optics.srgb_lin2gamma(np.clip(recon_amp**2, 0.0, 1.0))

    for j in range(len(prop_dist)):

        files = os.path.join(args.recon_dir, f'{i:04d}')
        if not os.path.exists(files):
            os.makedirs(files)

        recon = recon_srgb[j,:,:]
        cv2.imwrite(os.path.join(files, f'{j:04d}.png'), 
                    (np.clip(recon, 0.0, 1.0) * np.iinfo(np.uint8).max).round().astype(np.uint8))
