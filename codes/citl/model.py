import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import math
import numpy as np
import time
from unet import UnetGenerator, init_weights
from prop_ideal import *
from utils.optics import *

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

class CNNpropCNN(nn.Module):
    """
    A parameterized model with CNNs

    """

    def __init__(self, prop_dist, wavelength, feature_size, prop_type, F_aperture=1.0,
                 num_downs_slm=0, num_feats_slm_min=0, num_feats_slm_max=0,
                 num_downs_target=0, num_feats_target_min=0, num_feats_target_max=0,
                 slm_latent_amp=False, slm_latent_phase=False,
                 norm=nn.InstanceNorm2d):
        super(CNNpropCNN, self).__init__()

        ##################
        # Model pipeline #
        ##################
        slm_res = (704, 1200)

        # Learned SLM amp/phase
        if slm_latent_amp:
            self.slm_latent_amp = nn.Parameter(torch.ones(1, 1, *slm_res, requires_grad=True))
            
        if slm_latent_phase:
            self.slm_latent_phase = nn.Parameter(torch.zeros(1, 1, *slm_res, requires_grad=True))

        ## SLM CNN
        self.slm_cnn = UnetGenerator(input_nc=2, output_nc=2,
                                     num_downs=num_downs_slm, nf0=num_feats_slm_min,
                                     max_channels=num_feats_slm_max, norm_layer=norm, outer_skip=True)
        init_weights(self.slm_cnn, init_type='normal')

        ## ASM prop
        self.prop = Propagation(prop_dist, wavelength, feature_size, prop_type, F_aperture, dim=1)


        ## Target CNN
        self.target_cnn = UnetGenerator(input_nc=2, output_nc=2,
                                        num_downs=num_downs_target, nf0=num_feats_target_min,
                                        max_channels=num_feats_target_max, norm_layer=norm, outer_skip=True)
        init_weights(self.target_cnn, init_type='normal')


    def forward(self, hologram):

        input_field = torch.exp(1j * hologram)

        # 1) Learned phase offset
        if self.slm_latent_phase is not None:
            input_field = input_field * torch.exp(1j * self.slm_latent_phase)

        # 2) Learned amplitude
        if self.slm_latent_amp is not None:
            input_field = self.slm_latent_amp * input_field

 
        field = torch.cat((input_field.real, input_field.imag), dim=1)
        slm_field = self.slm_cnn(field)
        slm_field = torch.complex(slm_field[:, 0, :, : ].unsqueeze(1), slm_field[:, 1, :, : ].unsqueeze(1))

        target_field = self.prop(slm_field)

        target_field = torch.cat((target_field.real, target_field.imag), dim=1)

        output_field = self.target_cnn(target_field)
        output_field = torch.complex(output_field[:, 0, :, : ].unsqueeze(1), output_field[:, 1, :, : ].unsqueeze(1))

        return output_field

class Multi_CNNpropCNN(nn.Module):
    """
    A parameterized model with CNNs

    """

    def __init__(self, prop_dist, wavelength, feature_size, prop_type, F_aperture=1.0,
                 num_downs_slm=0, num_feats_slm_min=0, num_feats_slm_max=0,
                 num_downs_target=0, num_feats_target_min=0, num_feats_target_max=0,
                 slm_latent_amp=False, slm_latent_phase=False,
                 norm=nn.InstanceNorm2d):
        super(Multi_CNNpropCNN, self).__init__()

        ##################
        # Model pipeline #
        ##################

        self.slm_latent_amp = None
        self.slm_latent_phase = None

        slm_res = (576, 1024)

        # Learned SLM amp/phase
        if slm_latent_amp:
            self.slm_latent_amp = nn.Parameter(torch.ones(1, 1, *slm_res, requires_grad=True))
            
        if slm_latent_phase:
            self.slm_latent_phase = nn.Parameter(torch.zeros(1, 1, *slm_res, requires_grad=True))

        ## SLM CNN
        self.slm_cnn = UnetGenerator(input_nc=2, output_nc=2,
                                     num_downs=num_downs_slm, nf0=num_feats_slm_min,
                                     max_channels=num_feats_slm_max, norm_layer=norm, outer_skip=True)
        init_weights(self.slm_cnn, init_type='normal')

        ## ASM prop
        self.prop = Propagation(prop_dist, wavelength, feature_size, prop_type, F_aperture, dim=1)


        ## Target CNN
        self.target_cnn = UnetGenerator(input_nc=2, output_nc=2,
                                        num_downs=num_downs_target, nf0=num_feats_target_min,
                                        max_channels=num_feats_target_max, norm_layer=norm, outer_skip=True)
        init_weights(self.target_cnn, init_type='normal')

    def forward(self, hologram):

        input_field = torch.exp(1j * hologram)

        # 1) Learned phase offset
        if self.slm_latent_phase is not None:
            input_field = input_field * torch.exp(1j * self.slm_latent_phase)

        # 2) Learned amplitude
        if self.slm_latent_amp is not None:
            input_field = self.slm_latent_amp * input_field
 
        field = torch.cat((input_field.real, input_field.imag), dim=1)
        field = utils.pad_image(field, (576, 1024), pytorch=True, stacked_complex=False, padval=0)
        
        slm_field = self.slm_cnn(field)

        slm_amp, slm_phs = utils.rect_to_polar(slm_field[:, 0, :, : ].unsqueeze(1), slm_field[:, 1, :, : ].unsqueeze(1))

        slm_field = torch.cat((slm_amp, slm_phs), dim=1)

        slm_real, slm_imag = polar_to_rect(slm_field[:, 0, :, : ].unsqueeze(1), slm_field[:, 1, :, : ].unsqueeze(1))
        slm_field = torch.complex(slm_real, slm_imag)
        

        target_field = self.prop(slm_field)

        target_field =  target_field.permute(1, 0, 2, 3)

        target_field = torch.cat((target_field.real, target_field.imag), dim=1)

        output_field = self.target_cnn(target_field)

        output_field = torch.complex(output_field[:, 0, :, : ].unsqueeze(1), output_field[:, 1, :, : ].unsqueeze(1))

        return output_field
    



