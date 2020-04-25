import multiprocessing
multiprocessing.set_start_method('spawn', True)

import argparse
import os
import numpy as np
import math
import sys

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from train import train
from test import test, test_moving
from models import UNet, Discriminator, weights_init
from functions import *
from data_util import MyDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def main(opt):
    writer = SummaryWriter()
    log_dir = writer.get_logdir()
    os.makedirs(os.path.join(log_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "test"), exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize generator and discriminator
    generator = UNet(opt.sample_num, opt.channels, opt.batch_size, opt.alpha)
    discriminator = Discriminator(opt.batch_size, opt.alpha)

    generator.to(device=device)
    discriminator.to(device=device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    if opt.mode == 'train':
        generator = train(writer, log_dir, device, generator, discriminator, optimizer_G, optimizer_D, opt)
        test(opt, log_dir, generator=generator)
    if opt.mode == 'test':
        test(opt, log_dir)
        test_moving(opt, log_dir)
        
    
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--n_epochs_g", type=int, default=3, help="number of epochs of training only g")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr_d", type=float, default=0.000001, help="adam: learning rate d")
    parser.add_argument("--lr_g", type=float, default=0.00004, help="adam: learning rate g")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--alpha", type=float, default=0.2, help="Randomized Leaky ReLU activation layer")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Loss weight for gradient penalty")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=30, help="interval betwen image samples")
    parser.add_argument("--load_model", type=str, default='', help="path to model to continue training")
    parser.add_argument("--windows", type=bool, default=False, help="run on windows")
    parser.add_argument("--mode", type=str, default='test', help="train, test")


    parser.add_argument("--dataset_dir", type=str, default='/Users/Maayan/Documents/databases/mit_100_frames', help="path to dataset directory")
    parser.add_argument("--reference_dataset_path", type=str, default='/Users/Maayan/Documents/databases/mit', help="path to ground thruth dataset")
    parser.add_argument("--test_dataset_path", type=str, default='/Users/Maayan/Documents/databases/test/frames_256', help="path to test_dataset")


    parser.add_argument("--num_workers_dataloader", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--sample_num", type=int, default=20, help="number of images to random sample from each video")

    opt = parser.parse_args()
    print(opt)
    main(opt)
