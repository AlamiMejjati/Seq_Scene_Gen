# -*- coding: utf-8 -*-
"""
mask generation

@author: motur
"""

import os, time
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_loader_bgfg_2 import Inferer
from utils import  show_result
from models_res import Generator_Baseline_2
from networks import Generator_FG
import argparse
import glob
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--fgmodel', type=str, required=True,   help='batch size')
parser.add_argument('--maskmodel', type=str, required=True,   help='batch size')
parser.add_argument('--batch_size', type=int, default =4,   help='batch size')
parser.add_argument('--num_test_img', type=int, default =4,   help='num test images')
parser.add_argument('--bg_ims', type=str,  help='path to annotations')
parser.add_argument('--mask_ims', type=str, help='dataset path')
parser.add_argument('--category_name', type=str, default='giraffe',help='List of categories in MS-COCO dataset')
opt = parser.parse_args()

log_numb = 0
epoch_bias = 0
save_models = True
load_params = False
category_names = [opt.category_name]
#Hyperparameters
noise_size = 128
batch_size = opt.batch_size
num_test_img = opt.num_test_img
lr = 0.0002
train_epoch = 400
img_size = 256
use_LSGAN_loss = False
use_history_batch = False
one_sided_label_smoothing = 1 #default = 1, put 0.9 for smoothing
optim_step_size = 80
optim_gamma = 0.5
CRITIC_ITERS = 5
lambda_FM = 10
num_res_blocks = 2

#----------------------------Load Dataset--------------------------------------
transform = transforms.Compose([transforms.Scale((img_size,img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
transform2 = transforms.Compose([transforms.Scale((img_size, img_size)),transforms.ToTensor()])

allmasks = sorted(glob.glob(os.path.join(opt.mask_ims, '*.png')))
if len(allmasks)==0:
    allmasks = sorted(glob.glob(os.path.join(opt.mask_imgs, '*.jpg')))
allimages = sorted(glob.glob(os.path.join(opt.bg_ims, '*.png')))
if len(allimages)==0:
    allimages = sorted(glob.glob(os.path.join(opt.bg_ims, '*.jpg')))

nbmasks = len(allmasks)
print('Number of masks: %d'%nbmasks)
dataset = Inferer(imfile= allimages, mfiles = allmasks,  category_names = category_names,
                  transform=transform, final_img_size=img_size)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)


#--------------------------Define Networks------------------------------------
G_local = Generator_Baseline_2(z_dim=noise_size, label_channel=len(category_names),num_res_blocks=num_res_blocks)
G_local.cuda()
G_local.load_state_dict(torch.load(opt.maskmodel))
print('Parameters for mask model are loaded!')

G_fg = Generator_FG(z_dim=noise_size, label_channel=len(category_names),num_res_blocks=5)
G_fg.load_state_dict(torch.load(opt.fgmodel))
print('Parameters for FG model are loaded!')
G_fg.cuda()


#---------------------- results save folder-----------------------------------
root = './results_KID'
mask_folder = os.path.join(root, opt.category_name, 'ms')
im_folder = os.path.join(root, opt.category_name, 'ims')

os.makedirs(mask_folder, exist_ok=True)
os.makedirs(im_folder, exist_ok=True)

#Save the file for inspection later

#-------------------------------------------------------------------------
#----------------------------MAIN-----------------------------------------
#-------------------------------------------------------------------------
print('nb images: %d' %len(train_loader))
print('inference starts!')

start_time = time.time()
data_iter = iter(train_loader)
num_iter = 0
while num_iter < len(train_loader):
    bg_nb = num_iter//nbmasks
    box_id = np.mod(num_iter, nbmasks)
    sample_batched = data_iter.next()
    num_iter += 1
    x_ = sample_batched['seg_mask']
    x_ = torch.squeeze(x_, dim =1)
    x_ = torch.sum(x_,dim=1)
    x_ = x_.view(x_.size()[0],1,x_.size()[1],x_.size()[2])
    x_ = Variable(x_.cuda())
    y_ = sample_batched['bboxes']
    y_ = torch.squeeze(y_, dim=1)
    y_ = Variable(y_.cuda())

    # train local discriminator D ------------------------------------------
    mini_batch = x_.size()[0]

    # Fake ----------------------------------------------------------------
    z_ = torch.randn((mini_batch, noise_size))
    z_ = Variable(z_.cuda())

    maski = G_local(z_, y_)
    maski = maski.cpu().data.numpy().transpose(0,2,3,1)
    Image.fromarray((maski*255).squeeze().astype(np.uint8)).save(os.path.join(mask_folder, 'm_%d_bg_%d.png'%(box_id, bg_nb)))
    seg_masks = torch.zeros([1, img_size, img_size])
    mask = transform2(Image.fromarray(maski.squeeze()))
    if torch.max(mask) != 0:
        mask = mask / torch.max(mask)

    seg_masks[0, :, :] += mask.squeeze()
    x_fixed = sample_batched['image']
    x_fixed = Variable(x_fixed.cuda())
    z_fixed = torch.randn((mini_batch,noise_size))
    z_fixed = Variable(z_fixed.cuda())
    y_fixed = seg_masks[None, :, :, :]
    y_fixed = Variable(y_fixed.cuda())
    img = G_fg(z_fixed, y_fixed, torch.mul(x_fixed, (1 - torch.sum(y_fixed, 1).view(y_fixed.size(0), 1, img_size, img_size))))
    img = (img.cpu().data.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
    imgbg = (x_fixed.cpu().data.numpy().squeeze().transpose(1, 2, 0) + 1) / 2

    img_to_save = img*maski[0] + imgbg*(1 - maski[0])
    Image.fromarray((img_to_save*255).astype(np.uint8)).save(os.path.join(im_folder, 'im_%d_bg_%d.png'%(box_id, bg_nb)))