import os
import sys
from scipy.io import loadmat
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Camera in the loop')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np
import datetime
from pdb import set_trace as stx

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data
from utils.optics import *
######### Logs dir ###########
log_dir = os.path.join(dir_name,'log', opt.arch, opt.channel, opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_citl = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_citl)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_citl.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_citl.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########
model_citl = torch.nn.DataParallel (model_citl)
model_citl.cuda()

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_citl,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 5e1000
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()


######### Loss ###########
criterion = nn.L1Loss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
train_dataset = get_training_data(opt.train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Size of training set: ", len_trainset,", size of validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//2
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(train_loader, 0): 
        # zero_grad
        optimizer.zero_grad()

        phase = data[0].cuda()

        capture_amp = data[1].cuda()
        capture_amp = capture_amp.permute(1, 0, 2, 3)
        out_field = model_citl(phase)

        out_amp = crop_image(out_field.abs(), target_shape=(540, 960), pytorch=True, stacked_complex=False)

        loss = criterion(out_amp, capture_amp)

        loss_scaler(
                loss, optimizer,parameters=model_citl.parameters())

        epoch_loss += loss.item()

        #### Evaluation ####
        if (i+1) % eval_now==0 and i>0:
            with torch.no_grad():
                model_citl.eval()
                psnr_val = []

                for ii, data_val in enumerate((val_loader), 0):

                    phase = data_val[0].cuda()
                    capture_amp = data_val[1].cuda()
                    capture_amp = capture_amp.permute(1, 0, 2, 3)
                    out_field = model_citl(phase)

                    out_amp = crop_image(out_field.abs(), target_shape=(540, 960), pytorch=True, stacked_complex=False)

                    # with torch.no_grad():
                    #     s = (out_amp * capture_amp).mean() / \
                    #         (out_amp ** 2).mean() 

                    psnr_val.append(utils.batch_PSNR(out_amp, capture_amp, True).item())

                psnr_val = sum(psnr_val) / len_valset

                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_citl.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR image: %.4f\t] ----  [best_Ep_image %d best_it_image %d Best_PSNR_image %.4f] " % (epoch, i, psnr_val, best_epoch, best_iter, best_psnr))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR image: %.4f\t] ----  [best_Ep_image %d best_it_image %d Best_PSNR_image %.4f] " \
                        % (epoch, i, psnr_val, best_epoch, best_iter, best_psnr)+'\n')
                model_citl.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.6f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.6f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_citl.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_citl.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
