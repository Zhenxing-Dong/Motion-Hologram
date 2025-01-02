import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file
import torchvision.transforms.functional as F
import cv2
from utils.optics import *

def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out

def gamma_correction(image):
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected_image = cv2.LUT(image, lookup_table)
    return corrected_image

def inverse_gamma_correction(image):
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    corrected_image = np.power(image, inv_gamma)
    return corrected_image


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')
    
##################################################################################################
def load_planes(file_paths, dtype=torch.float32):
    captured_amp = []
    for file_path in file_paths:
        img = cv2.imread(file_path, 0)
        img = np.sqrt(inverse_gamma_correction(im2float(img, dtype=np.float64)))
        captured_amp.append(torch.tensor(img, dtype=dtype))
    return torch.stack(captured_amp, dim=0)  # 按维度 0 堆叠


class HologramDataset(Dataset):
    def __init__(self, data_dir, mode="train", slm_res=(576, 1024)):
        """
        :param data_dir
        :param mode: ('train', 'val', 'test')
        :param slm_res
        """
        super(HologramDataset, self).__init__()
        self.slm_res = slm_res

        hologram_dir = 'hologram'
        captured_dirs = [f'captured/plane_{i}' for i in range(7)]

        self.hologram_filenames = self._get_files(os.path.join(data_dir, hologram_dir))
        self.captured_plane_filenames = [
            self._get_files(os.path.join(data_dir, captured_dir)) for captured_dir in captured_dirs
        ]

        self.tar_size = len(self.hologram_filenames)

    @staticmethod
    def _get_files(directory):
        return [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if is_png_file(f)]

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        phase_path = self.hologram_filenames[tar_index]
        phase = cv2.imread(phase_path, 0) / 255.0
        phase = torch.tensor((1 - phase) * 2 * np.pi - np.pi, dtype=torch.float32).reshape(1, *self.slm_res)

        captured_files = [files[tar_index] for files in self.captured_plane_filenames]
        captured_amp = load_planes(captured_files)

        hologram_filename = os.path.basename(phase_path)
        captured_filenames = [os.path.basename(f) for f in captured_files]

        return phase, captured_amp, hologram_filename, captured_filenames