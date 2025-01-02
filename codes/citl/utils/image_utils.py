import torch
import numpy as np
import pickle
import cv2
from scipy.io import loadmat

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.imread(filepath, 0)
    img = img[:, :992]
    img = img[:,:,None]
    img = img.astype(np.float32)
    img = img/255.
    return img


def load_data_mat(filepath):
    img = loadmat(filepath)
    img = img['slice_to_save']
    # img = img*10000
    # # img = img['data']
    # # img = img['Spectraldata']
    # pad_width = ((0, 0), (0, 192))
    # img = np.pad(img, pad_width, mode='constant')
    img = img[:, :992]
    img = img[:,:,None]
    img = img.astype(np.float32)
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath, img)

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def myMSE(tar_img, prd_img):
    imdff = prd_img - tar_img
    mse = (imdff**2).mean()
    return mse

def batch_MSE(img1, img2, average=True):
    MSE = []
    for im1, im2 in zip(img1, img2):
        mse = myMSE(im1, im2)
        MSE.append(mse)
    return sum(MSE)/len(MSE) if average else sum(MSE)
