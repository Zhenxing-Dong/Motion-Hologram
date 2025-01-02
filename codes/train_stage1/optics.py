import abc
import numpy as np
import torch
import math
import torch.fft as tfft
import torch.nn as nn

def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out

def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def torch_compl_exp(phase):
    real = torch.cos(phase)
    imag = torch.sin(phase)
    return torch.complex(real, imag)


def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    """Helper for pad_image() that pads a real padval in a complex-aware manner"""
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)

def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field

def np_circ_filter(batch,
                   num_channels,
                   res_h,
                   res_w,
                   filter_radius,
                   ):
    """create a circular low pass filter
    """
    y,x = np.meshgrid(np.linspace(-(res_w-1)/2, (res_w-1)/2, res_w), np.linspace(-(res_h-1)/2, (res_h-1)/2, res_h))
    mask = x**2+y**2 <= filter_radius**2
    np_filter = np.zeros((res_h, res_w))
    np_filter[mask] = 1.0
    np_filter = np.tile(np.reshape(np_filter, [1,1,res_h,res_w]), [batch, num_channels, 1, 1])
    return np_filter


def np_filter(batch,
                   num_channels,
                   res_h,
                   res_w,
                   filter_radius_x,
                   filter_radius_y,
                   ):
    """create a low pass filter
    """
    y,x = np.meshgrid(np.linspace(-(res_w-1)/2, (res_w-1)/2, res_w), np.linspace(-(res_h-1)/2, (res_h-1)/2, res_h))
    mask = (x**2 / filter_radius_x**2) + (y**2 / filter_radius_y**2) <= 1.0
    np_filter = np.zeros((res_h, res_w))
    np_filter[mask] = 1.0
    np_filter = np.tile(np.reshape(np_filter, [1,1,res_h,res_w]), [batch, num_channels, 1, 1])
    return np_filter














def generate_2d_gaussian(kernel_length=[21, 21], nsigma=[3, 3], mu=[0, 0], normalize=False):

    x = torch.linspace(-kernel_length[0]/2., kernel_length[0]/2., kernel_length[0])
    y = torch.linspace(-kernel_length[1]/2., kernel_length[1]/2., kernel_length[1])
    X, Y = torch.meshgrid(x, y)
    if nsigma[0] == 0:
        nsigma[0] = 1e-5
    if nsigma[1] == 0:
        nsigma[1] = 1e-5
    kernel_2d = 1. / (2. * np.pi * nsigma[0] * nsigma[1]) * torch.exp(-((X - mu[0])**2. / (2. * nsigma[0]**2.) + (Y - mu[1])**2. / (2. * nsigma[1]**2.)))
    if normalize:
        kernel_2d = kernel_2d / kernel_2d.max()
    return kernel_2d


def np_gaussian(shape=(3,3), sigma=0.5, reshape_4d=True):
    """
    modified from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    2D Gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    
    if reshape_4d:
        h = h[np.newaxis, np.newaxis, :, :]
        # h = h.repeat(3,2)
       
    return h








def torch_wrap_phs(phs_only, 
                   phs_max=[2*np.pi]*3, 
                   adaptive_phs_shift=False):
    def wrap_less_equal_than_phs_max(phs_only, phs_max, phs_per_channel_max, phs_per_channel_min):
        return phs_only + (phs_max-phs_per_channel_min-phs_per_channel_max) / 2.0
    
    def wrap_greater_than_phs_max(phs_only, phs_max):
        phs_only = phs_only + phs_max/2.0
        phs_only = torch.where(phs_only < 0, phs_only + 2.0*math.pi, phs_only)
        phs_only = torch.where(phs_only > phs_max, phs_only - 2.0*math.pi, phs_only)
        return phs_only

    if phs_max is not None:
        # wrap out-of-range phase
        if adaptive_phs_shift:
            phs_per_channel_list = []
            for i in range(3):
                phase_per_channel = phs_only[:,i,:,:]
                phs_max_channel = phs_max[i]
                phase_per_channel_max = torch.max(phase_per_channel)
                phase_per_channel_min = torch.min(phase_per_channel)
                phase_per_channel = torch.where((phase_per_channel_max - phase_per_channel_min) <= phs_max_channel, 
                    wrap_less_equal_than_phs_max(phase_per_channel, phs_max_channel, phase_per_channel_max, phase_per_channel_min), 
                    wrap_greater_than_phs_max(phase_per_channel, phs_max_channel))
                phs_per_channel_list.append(phase_per_channel)
            phs_only = torch.stack(phs_per_channel_list, dim=1)
        else:
            phs_max_4d = phs_max.reshape(1, 3, 1, 1)
            phs_only = wrap_greater_than_phs_max(phs_only, phs_max_4d) 
    
    return phs_only




class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x



class Propagation(nn.Module):
    def __init__(self,
                 input_shape,
                 pitch,
                 wavelengths,
                 double_pad):
        super(Propagation, self).__init__()
        self.input_shape  = input_shape
        if double_pad:
            self.m_pad    = input_shape[0] // 2
            self.n_pad    = input_shape[1] // 2
        else:
            self.m_pad    = 0
            self.n_pad    = 0
        self.wavelengths  = wavelengths[None, :, None, None]
        self.wave_nos     = 2. * math.pi / wavelengths
        self.pitch        = pitch
        self.fx, self.fy  = self._torch_xy_grid()
        self.unit_phase_shift = None

    # torch grid
    def _torch_xy_grid(self):

        M, N = self.input_shape[0] + 2 * self.m_pad, self.input_shape[1] + 2 * self.n_pad

        fy = torch.linspace(-1 / (2 * self.pitch), 1 / (2 * self.pitch), M)
        fx = torch.linspace(-1 / (2 * self.pitch), 1 / (2 * self.pitch), N)

        fx, fy = torch.meshgrid(fx, fy)
        fx = torch.transpose(fx, 0, 1)
        fy = torch.transpose(fy, 0, 1)
        fx = fx[None, None, :, :]
        fy = fy[None, None, :, :]    
  
        return fx, fy        

    def _unit_phase_shift(self):
        """Compute unit distance phase shift
        """
    def _propagate(self, input_field, z_dist):
        padded_input_field = pad_image(input_field, [self.input_shape[0] + 2 * self.m_pad, self.input_shape[1] + 2 * self.n_pad], 
                                       stacked_complex=False, padval=0)
        
        real, imag = polar_to_rect(torch.ones_like(z_dist * self.unit_phase_shift), z_dist * self.unit_phase_shift)

        H = torch.complex(real, imag)
        H = H.to(input_field.device)

        obj_fft = tfft.fftshift(tfft.fftn(padded_input_field, dim=(-2, -1), norm='ortho'), (-2, -1))

        out_field = tfft.ifftn(tfft.ifftshift(obj_fft*H, (-2, -1)), dim=(-2, -1), norm='ortho')

        return crop_image(out_field, [self.input_shape[0], self.input_shape[1]], pytorch=True, stacked_complex=False)

    def __call__(self, input_field, z_dist):
        return self._propagate(input_field, z_dist)

# Fresnel approximation
class FresnelPropagation(Propagation):
    def __init__(self, 
                 input_shape,
                 pitch,
                 wavelengths,
                 double_pad):
        super(FresnelPropagation, self).__init__(input_shape, pitch, wavelengths, double_pad)
        self.unit_phase_shift = self._unit_phase_shift()

    def _unit_phase_shift(self):
        squared_sum = torch.square(self.fx) + torch.square(self.fy)
        phase_shift = -1. * self.wavelengths * math.pi * squared_sum
        return phase_shift

# angular spectrum propagation
class ASPropagation(Propagation):
    def __init__(self,
                 input_shape,
                 pitch,
                 wavelengths,
                 double_pad):
        super(ASPropagation, self).__init__(input_shape, pitch, wavelengths, double_pad)
        self.unit_phase_shift = self._unit_phase_shift()

    def _unit_phase_shift(self):
        phase_shift = 2 * math.pi * (1 / self.wavelengths**2 - (self.fx ** 2 + self.fy ** 2)).sqrt()
        return phase_shift.cuda()

def torch_propagator(input_shape,
                     pitch,
                     wavelengths,
                     method = "as",
                     double_pad = False):
    switcher = {
        "as": ASPropagation,
        "fresnel": FresnelPropagation
    }
    propagator = switcher.get(method, "invalid method")
    if propagator == "invalid method":
        raise ValueError("invalid propgation method")
    return propagator(input_shape=input_shape,
                      pitch=pitch,
                      wavelengths=wavelengths,
                      double_pad=double_pad)



# add propagation, check normalize
def torch_dpm_maimone(cpx, 
                      propagator=None,
                      depth_shift=0,
                      adaptive_phs_shift=False,
                      batch=1, 
                      num_channels=3, 
                      res_h=384, 
                      res_w=384,
                      dim=2,
                      phs_max=[2*np.pi]*3, 
                      amp_max=None, 
                      clamp=False,
                      normalize=True,
                      wavelength=[0.000450, 0.000520, 0.000638]):
    """
    Double phase method of [Maimone et al. 2017]
    """
    # shift the hologram to hologram plane
    assert (depth_shift == 0 or propagator != None)
    if depth_shift != 0:
        torch_wavelength = torch.tensor(wavelength, dtype=torch.float32).reshape(1,3,1,1)

        cpx = propagator(cpx, depth_shift) * torch_compl_exp(-2*math.pi*depth_shift/torch_wavelength)

    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    # normalize amplitude
    if amp_max is None:
        # avoid acos producing nan
        amp_max = torch.max(amp) + 1e-6
    amp = amp / amp_max

    # clamp maximum value to 1.0
    if clamp:
        amp = torch.min(amp, 1.0-1e-6)

    # center phase for each color channel
    phs_zero_mean = phs - torch.mean(phs, dim = [2,3], keepdim=True)

    # discard every other pixel
    if dim == 3:    # reduce columns
        amp = amp[:,:,:,0::2]
        phs_zero_mean = phs_zero_mean[:,:,:,0::2]
    elif dim == 2:  # reduce rows
        amp = amp[:,:,0::2,:]
        phs_zero_mean = phs_zero_mean[:,:,0::2,:]

    # compute two phase maps
    phs_offset = torch.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset 

    # arrange in checkerboard pattern
    if dim == 3:
        phs_1_1 = phs_low[:,:,0::2,:]
        phs_1_2 = phs_high[:,:,0::2,:]
        phs_2_1 = phs_high[:,:,1::2,:]
        phs_2_2 = phs_low[:,:,1::2,:]
        phs_only = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    elif dim == 2:
        phs_1_1 = phs_low[:,:,:,0::2]
        phs_1_2 = phs_high[:,:,:,0::2]
        phs_2_1 = phs_high[:,:,:,1::2]
        phs_2_2 = phs_low[:,:,:,1::2]
        phs_only = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    else:
        raise ValueError("axis has to be 2 or 3")
    phs_only = DepthToSpace(2)(phs_only)

    if phs_max != None:
        phs_only = torch_wrap_phs(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize:
        phs_max_4d = phs_max.reshape(1,3,1,1)
        phs_only = phs_only / phs_max_4d

    return phs_only, amp_max



def torch_aadpm(cpx, 
                propagator=None,
                depth_shift=0,
                adaptive_phs_shift=False,
                batch=1, 
                num_channels=3, 
                res_h=384, 
                res_w=384,
                sigma=0.5, 
                kernel_width=5, 
                phs_max=[2*np.pi]*3, 
                amp_max=None, 
                clamp=False,
                normalize=True,
                wavelength=[0.000450, 0.000520, 0.000638]):
    """
    Anti-aliasing double phase method
    """
    # shift the hologram to hologram plane
    assert (depth_shift == 0 or propagator != None)
    if depth_shift != 0:
        torch_wavelength = torch.tensor(wavelength, dtype=torch.float32).reshape(1,3,1,1)
        cpx = propagator(cpx, depth_shift) * torch_compl_exp(-2*math.pi*depth_shift/torch_wavelength)

    # apply pre-blur
    if sigma > 0.0:
 
        blur_kernel = torch.tensor(np_gaussian([kernel_width, kernel_width], sigma, reshape_4d=True), dtype=torch.float32).cuda()
        blur_kernel = blur_kernel.repeat(3,1,1,1)
        cpx_imag = torch.imag(cpx)
        cpx_real = torch.real(cpx)
        cpx_imag = torch.nn.functional.conv2d(cpx_imag, blur_kernel, padding=1, groups=3)
        cpx_real = torch.nn.functional.conv2d(cpx_real, blur_kernel, padding=1, groups=3)

        cpx = torch.complex(cpx_real, cpx_imag)

    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    # normalize amplitude
    if amp_max is None:
        # avoid acos producing nan
        amp_max = torch.max(amp) + 1e-6
    amp = amp / amp_max

    # clamp maximum value to 1.0
    if clamp:
        amp = torch.minimum(amp, torch.tensor(1.0-1e-6).cuda())
        # amp = torch.where(amp<1-1e-6, amp, 1-1e-6)

    # center phase for each color channel
    phs_zero_mean = phs - torch.mean(phs, dim = [2,3], keepdim=True)

    # compute two phase maps
    phs_offset = torch.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset 

    # arrange in checkerboard pattern
    phs_1_1 = phs_low[:,:,0::2,0::2]
    phs_1_2 = phs_high[:,:,0::2,1::2]
    phs_2_1 = phs_high[:,:,1::2,0::2]
    phs_2_2 = phs_low[:,:,1::2,1::2]
    phs_only = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    phs_only = DepthToSpace(2)(phs_only)

    if phs_max != None:
        phs_only = torch_wrap_phs(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize:
        phs_max_4d = phs_max.reshape(1,3,1,1)

        phs_only = phs_only / phs_max_4d

    # apply your own lookup table if necessary

    return phs_only, amp_max