import os

from dataset import HologramDataset

def get_training_data(train_dir):
    assert os.path.exists(train_dir)
    return HologramDataset(train_dir, mode="train")

def get_validation_data(val_dir):
    assert os.path.exists(val_dir)
    return HologramDataset(val_dir, mode="val")


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return HologramDataset(rgb_dir, mode="test")