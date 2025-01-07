import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils
import cv2
import torch
from skimage.transform import resize

def intoamp(im):

    im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

    # linearize intensity and convert to amplitude
    low_val = im <= 0.04045
    im[low_val] = 25 / 323 * im[low_val]
    im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                    / 211) ** (12 / 5)
    im = np.sqrt(im)  # to amplitude

    # move channel dim to torch convention
    im = np.transpose(im, axes=(2, 0, 1))

    return im


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return np.transpose(image, axes=(2, 0, 1))

class FocalstackDDataset(Dataset):
    def __init__(self, root_dir, depth_dir, depth_bins=7):
        self.root_dir = root_dir
        self.depth_dir = depth_dir
        self.folder_names = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        self.depth_names = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.depth_bins = depth_bins

    def __len__(self):
        return len(self.folder_names)

    def discretize_depth(self, depth):
        bin_edges = np.linspace(0, 6, self.depth_bins + 1)
        depth_discrete = np.digitize(depth, bin_edges, right=True) - 1
        depth_discrete = depth_discrete / (self.depth_bins - 1)
        return depth_discrete.astype('float32')

    def __getitem__(self, idx):
        # Load images
        folder_name = self.folder_names[idx]
        folder_path = os.path.join(self.root_dir, folder_name)
        image_names = sorted(os.listdir(folder_path))
        images = []

        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = intoamp(image) 
            image = resize_keep_aspect(image, (1080, 1920))
            image = torch.from_numpy(image).float().cuda()  # HWC -> CHW
            images.append(image)

        # Load depth
        depth_name = self.depth_names[idx]
        depth_path = os.path.join(self.depth_dir, depth_name)
        depth = cv2.imread(depth_path, 0) / 255.0
        depth = resize_keep_aspect(depth, (1080, 1920))
        depth = self.discretize_depth(depth)

        # Create depth mask
        depth = torch.from_numpy(depth).float().cuda()
        mask = torch.zeros((self.depth_bins, depth.shape[0], depth.shape[1]), dtype=torch.float32).cuda()
        for i in range(self.depth_bins):
            mask[i] = (depth == i / (self.depth_bins - 1)).float()

        return images, mask
