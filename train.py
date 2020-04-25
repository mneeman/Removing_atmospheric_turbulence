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
#import matplotlib.pyplot as plt
from models import UNet, Discriminator, weights_init
from functions import *
from data_util import MyDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train(writer, log_dir, device, generator, discriminator, optimizer_G, optimizer_D, opt):
    torch.cuda.empty_cache()
    opt.n_epochs_g = opt.n_epochs_g -1 

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(MyDataset(opt), opt.batch_size, shuffle=True, num_workers=opt.num_workers_dataloader)

    if opt.load_model == '':
        # Initialize weights
        generator.apply(weights_init)
        discriminator.apply(weights_init)

        # load fixed batch for log
        data_iter = iter(dataloader)
        batch_fixed, GT_image_fixed = next(data_iter)
        del data_iter
        torch.cuda.empty_cache()
        start_epoch = 0
        batches_done = 0
    
    else:
        checkpoint = torch.load(opt.load_model)
        generator.load_state_dict(checkpoint['g_state_dict'])
        discriminator.load_state_dict(checkpoint['d_state_dict'])
        optimizer_G.load_state_dict(checkpoint['g_optimizer_state_dict'])
        optimizer_D.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        batch_fixed = checkpoint['batch_fixed']
        GT_image_fixed = checkpoint['GT_image_fixed']
        batches_done = checkpoint['batches_done']
        del checkpoint
        torch.cuda.empty_cache()
        generator.train()
        discriminator.train()


    # ----------
    #  Training
    # ----------
    
    for epoch in range(start_epoch, opt.n_epochs):
        for i, (imgs, GT_image) in enumerate(dataloader):
            torch.cuda.empty_cache()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            fake_img = generator(imgs.to(device))

            if epoch > opt.n_epochs_g: #train only generator for first opt.n_epochs_g
                    retain_g_flag = True
                    fake_validity = discriminator(fake_img)
                    g_loss = -torch.mean(fake_validity) + compute_L1_loss(imgs, fake_img.to('cpu'), opt, device)
            else:
                    retain_g_flag = False
                    g_loss = compute_L1_loss(imgs, fake_img.to('cpu'), opt, device)

            g_loss.backward(retain_graph=retain_g_flag)
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            #train only generator for first opt.n_epochs_g
            if epoch > opt.n_epochs_g: 

                optimizer_D.zero_grad()

                # Real images
                real_validity = discriminator(GT_image.to(device))
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, GT_image.to(device), fake_img)
                # Adversarial loss
                d_loss = -torch.mean(real_validity.detach()) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty

                
                d_loss.backward(retain_graph=False)
                optimizer_D.step()

                writer.add_scalar('D_Loss/train', d_loss.item(), batches_done)
            
            writer.add_scalar('G_Loss/train', g_loss.item(), batches_done)
                
                
            if i % opt.sample_interval == 0:
                if epoch > opt.n_epochs_g:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )
                else:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: none - first epoch] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item())
                    )
                
                with torch.no_grad():
                    fake_img_fixed = generator(batch_fixed.to(device))

                    # save GT image only at first iteration
                    if batches_done == 0:
                        GT_image_fixed = denormalize(GT_image_fixed)
                        GT_image_fixed = convert_im(GT_image_fixed, os.path.join(log_dir, "images/GT_image.png"), nrow=5, normalize=False, save_im=True)
                        writer.add_image('images/real', GT_image_fixed, batches_done)
                    
                    fake_img_fixed = denormalize(fake_img_fixed)
                    fake_img_fixed = convert_im(fake_img_fixed, os.path.join(log_dir, "images/%d.png" % batches_done), nrow=5, normalize=False, save_im=True)
                    writer.add_image('images/fake', fake_img_fixed, batches_done)
            
            del g_loss, fake_img, GT_image, imgs
            torch.cuda.empty_cache()
            batches_done += 1
            if batches_done % 1000 == 0:
                opt.sample_interval = int(opt.sample_interval * 1.2)
    
        # save model after each epoch
        path = os.path.join(log_dir, 'model_%d' % epoch)
        torch.save({
            'epoch': epoch,
            'batch_fixed': batch_fixed,
            'GT_image_fixed': GT_image_fixed,
            'g_state_dict': generator.state_dict(),
            'g_optimizer_state_dict': optimizer_G.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            'd_optimizer_state_dict': optimizer_D.state_dict(),
            'batches_done': batches_done
            }, path)

    return generator