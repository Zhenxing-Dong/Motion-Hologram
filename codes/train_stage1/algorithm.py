import torch.optim as optim
import torch
import torch.nn as nn
import torch.fft as tfft
from dataset_load import *
import utils
import optics
from prop_ideal import *

def FocalStackShift(init_phase, target_amp, mask, model_citl, path, circ_filter, num_iters, 
                    roi_res=(1080, 1920), loss=nn.MSELoss(), lr=0.06):

    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)
    # optimization variables and adam optimizer
    optvars = [{'params': slm_phase}]
    optimizer = optim.Adam(optvars, lr=lr)
    # crop target roi
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)

    coords = torch.nonzero(path == 1)
    coords = coords - 32
    for k in range(num_iters):

        optimizer.zero_grad()

        slm_phase1 = utils.pad_image(slm_phase, (1152, 2048), stacked_complex=False)

        real1, imag1 = utils.polar_to_rect(torch.ones_like(slm_phase1), slm_phase1)
        slm_field1 = torch.complex(real1, imag1)
        slm_field1 = tfft.fftshift(tfft.fftn(slm_field1, dim=(-2, -1), norm='ortho'), (-2, -1))
        slm_field1 = slm_field1 * circ_filter
        slm_field1 = tfft.ifftn(tfft.ifftshift(slm_field1, (-2, -1)), dim=(-2, -1), norm='ortho')

        centre_amp = torch.abs(model_citl(slm_field1))
        centre_amp = centre_amp.permute(1, 0, 2, 3)

        shifts_amp = []
        for coord in coords:
            shift_amp = torch.roll(centre_amp, shifts=(coord[0], coord[1]), dims=(2, 3))
            shifts_amp.append(shift_amp)
        shifts_amp = torch.cat(shifts_amp, dim=1)
        recon_amp = torch.mean(shifts_amp, dim=1, keepdim=True)
        
        recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)

        with torch.no_grad():
            s = (recon_amp * target_amp).mean() / \
                (recon_amp ** 2).mean() 

        lossValue_MSE = loss(s*recon_amp, target_amp)
        loss_infocus = loss(s*recon_amp*mask, target_amp*mask)
        lossvalue = lossValue_MSE + loss_infocus

        lossvalue.backward()
        optimizer.step()

    return slm_phase, s*recon_amp

def PSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

class Hologram():
    def __init__(self, model):
        self.model = model

    def forward(self, target_amp, mask, path):
        init_phase = (-0.5  + 1.0 * torch.rand(1, 1, 1152, 2048)).cuda()
        circ_filter = optics.np_filter(1, 1, 1152, 2048, 1152//2, 2048//2)
        circ_filter = torch.tensor(circ_filter, dtype=torch.float32).cuda()
        _, recon_amp = FocalStackShift(init_phase, target_amp, mask, self.model, path, circ_filter, num_iters=300, lr=0.4)
        psnr_value = PSNR(recon_amp, target_amp)
        psnr_infocus_value = PSNR(recon_amp*mask, target_amp*mask)
        psnr_all = psnr_value + psnr_infocus_value
        recon_amp = recon_amp.permute(1, 0, 2, 3)
        return recon_amp, psnr_all