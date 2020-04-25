import torch
from torch.autograd import Variable
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
from PIL import Image
from torchvision.utils import make_grid


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 3, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], d_interpolates.shape[1]).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_L1_loss(input_imgs, output_im, opt, device):
    loss_L1 = nn.L1Loss()
    loss = torch.zeros(1)
    for i in range(opt.sample_num):
        loss += loss_L1(input_imgs[:,i,:,:], output_im)
    loss = (1000//opt.sample_num)*loss
    #loss = loss/opt.sample_num

    return loss.to(device=device)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def convert_image(inp):
    inp = denorm(inp)
    inp = inp.to('cpu')
    inp = np.clip(inp,0,1)
    return inp

def convert_im(tensor, filename='', nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, save_im = False):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if save_im:
        im.save(filename)
    return np.rollaxis(ndarr, 2, 0)

def denormalize(image_tensor):
    # denormalize the normalized image tensor(N,C,H,W) with mean=0.5 and std=0.5 for each channel
    return (image_tensor + 1) / 2.0
