#File for easily editing settings

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
WM38Data = "datasets/MixedWM38.npz"
WM811KData = "datasets/WM811K.pkl"

# CNN
CNN_EPOCHS = 50
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT = 0.25

# GAN
GAN_LATENT_DIM = 100
GAN_EPOCHS = 200
GAN_LR_GENERATOR = 0.0002
GAN_LR_DISCRIMINATOR = 0.00002
GAN_LAMBDA_GP = 10
GAN_N_CRITIC = 5
GAN_BETA1 = 0.7
GAN_BETA2 = 0.99
GAN_TARGET_CLASSES = [5, 7, 8]

# WM-811K label map
WM811K_LABEL_MAP = {
    'Center'    : 0,
    'Donut'     : 1,
    'Edge-Loc'  : 2,
    'Edge-Ring' : 3,
    'Loc'       : 4,
    'Scratch'   : 5,
    'Random'    : 6,
    'Near-full' : 7,
    'none'      : 8
}

# Class names
CLASS_NAMES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
