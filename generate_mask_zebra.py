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
from data_loader_bgfg_2 import zebra_silvia
from utils import  show_result
from models_res import Discriminator, Generator_Baseline_2
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default =4,   help='batch size')
parser.add_argument('--num_test_img', type=int, default =4,   help='num test images')
parser.add_argument('--train_imgs', type=str, help='dataset path')
parser.add_argument('--category_names', type=str, default='zebra',help='List of categories in MS-COCO dataset')
opt = parser.parse_args()


log_numb = 0
epoch_bias = 0
save_models = True
load_params = False
category_names = [opt.category_names]
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

allmasks = sorted(glob.glob(os.path.join(opt.train_imgs, '*.png')))

dataset = zebra_silvia(root = allmasks,  category_names = category_names, transform=transform, final_img_size=img_size)


#Discarding images contain small instances  
# dataset.discard_small(min_area=0.01, max_area= 0.5)
# dataset.discard_bad_examples('bad_examples_list.txt')
# dataset.discard_num_objects()

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)


# For evaluation, fixed masks and noize
data_iter = iter(train_loader)
sample_batched = data_iter.next() 
y_fixed = sample_batched['bboxes'][0:num_test_img]
y_fixed = torch.squeeze(y_fixed, dim=1)
y_fixed = Variable(y_fixed.cuda())
print(y_fixed)
x_fixed = sample_batched['seg_mask'][0:num_test_img]
x_fixed = torch.squeeze(x_fixed, dim=1)
x_fixed = torch.sum(x_fixed,dim=1)
x_fixed = x_fixed.view(x_fixed.size()[0],1,x_fixed.size()[1],x_fixed.size()[2])
x_fixed = Variable(x_fixed.cuda())


z_fixed = torch.randn((num_test_img, noise_size))
z_fixed= Variable(z_fixed.cuda())
    
#--------------------------Define Networks------------------------------------
G_local = Generator_Baseline_2(z_dim=noise_size, label_channel=len(category_names),num_res_blocks=num_res_blocks)
D_local = Discriminator(channels=1+len(category_names))
G_local.cuda()
D_local.cuda()

#Load parameters from pre-trained model
if load_params:
    G_local.load_state_dict(torch.load('C:/Users/motur/coco/cocoapi/PythonAPI/result_local/models_local_'+str(log_numb)+'/coco_model_G_glob_epoch_'+str(epoch_bias)+'.pth'))
    D_local.load_state_dict(torch.load('C:/Users/motur/coco/cocoapi/PythonAPI/result_local/models_local_'+str(log_numb)+'/coco_model_D_glob_epoch_'+str(epoch_bias)+'.pth'))
    print('Parameters are loaded from logFile: models_local_' +str(log_numb) +' ---- Epoch: '+str(epoch_bias))


# Binary Cross Entropy loss
if use_LSGAN_loss:
    BCE_loss= nn.MSELoss()
else:
    BCE_loss = nn.BCELoss()


if use_history_batch:
    img_hist_1 = torch.zeros([batch_size//2,4,img_size,img_size])
    img_hist_2 = torch.zeros([batch_size//2,4,img_size,img_size])
    
#Feature matching criterion
#criterionVGG = VGGLoss()
#criterionVGG = criterionVGG.cuda()
criterionCE = nn.BCELoss()

# Adam optimizer
G_local_optimizer = optim.Adam(G_local.parameters(), lr=lr, betas=(0.0, 0.9))
D_local_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D_local.parameters()), lr=lr, betas=(0.0,0.9))

#Learning rate scheduler
scheduler_G = lr_scheduler.StepLR(G_local_optimizer, step_size=optim_step_size, gamma=optim_gamma)
scheduler_D = lr_scheduler.StepLR(D_local_optimizer, step_size=optim_step_size, gamma=optim_gamma)

#---------------------- results save folder-----------------------------------
root = 'result_local/'  +  opt.category_names + '/'
model = 'coco_model_'
result_folder_name = 'local_result_' + str(log_numb)
model_folder_name = 'models_local_' + str(log_numb)
if not os.path.isdir(root):
    os.makedirs(root)
if not os.path.isdir(root + result_folder_name):
    os.makedirs(root + result_folder_name)
if not os.path.isdir(root + model_folder_name):
    os.makedirs(root + model_folder_name)

#Save the file for inspection later
copyfile(os.path.basename(__file__), root + result_folder_name + '/' + os.path.basename(__file__))


#-------------------------------------------------------------------------
#----------------------------MAIN-----------------------------------------
#-------------------------------------------------------------------------
print('training start!')
start_time = time.time()

for epoch in range(train_epoch):
    scheduler_G.step()
    scheduler_D.step()
     
    D_local_losses = []
    G_local_losses = []

    y_real_ = torch.ones(batch_size)*one_sided_label_smoothing
    y_fake_ = torch.zeros(batch_size)
    y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    epoch_start_time = time.time()

    data_iter = iter(train_loader)
    num_iter = 0
    while num_iter < len(train_loader):
        
        j=0
        while j < CRITIC_ITERS and num_iter < len(train_loader):
            j += 1
            sample_batched = data_iter.next()  
            num_iter += 1            
            #x_ = torch.sum(sample_batched['single_fg_mask'],dim=1)
            #x_ = x_.view(x_.size()[0],1,x_.size()[1],x_.size()[2])
            #y_ = sample_batched['bbox_mask']
            
            x_ = sample_batched['seg_mask']
            x_ = torch.squeeze(x_, dim =1)
            x_ = torch.sum(x_,dim=1)
            x_ = x_.view(x_.size()[0],1,x_.size()[1],x_.size()[2])
            y_ = sample_batched['bboxes']
            y_ = torch.squeeze(y_, dim=1)
            
            # train local discriminator D ------------------------------------------
            D_local.zero_grad()
            mini_batch = x_.size()[0]
    
            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch)*one_sided_label_smoothing
                y_fake_ = torch.zeros(mini_batch)
                y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    
            x_, y_ = Variable(x_.cuda()) , Variable(y_.cuda()) 
            x_d = torch.cat([x_,y_],1)
            
            D_result = D_local(x_d.detach()).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)
            D_real_loss.backward()
            
            # Fake ----------------------------------------------------------------
            z_ = torch.randn((mini_batch, noise_size))
            z_ = Variable(z_.cuda())
    
            G_result = G_local(z_, y_)
            G_result_d = torch.cat([G_result,y_],1) 
            D_result = D_local(G_result_d.detach()).squeeze()
            
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_loss.backward()
            D_local_optimizer.step()
            D_fake_score = D_result.data.mean()
            D_train_loss = D_real_loss + D_fake_loss
            D_local_losses.append(D_train_loss.data)


        # train generator G  --------------------------------------------------
        G_local.zero_grad()   
        D_result = D_local(G_result_d).squeeze() 

        G_train_loss = BCE_loss(D_result, y_real_) 
        
        #Feature matching loss between generated image and corresponding ground truth
        #FM_loss = criterionVGG(G_result,x_)
        #FM_loss = criterionVGG(G_result.expand(batch_size,3,128,128),x_.expand(batch_size,3,128,128))
        Recon_loss = criterionCE(G_result, x_)
           
        total_loss = G_train_loss + lambda_FM*Recon_loss 
        total_loss.backward()
        G_local_optimizer.step()
        G_local_losses.append(G_train_loss.data)

        print('loss_d: %.3f, loss_g: %.3f' % (D_train_loss.data,G_train_loss.data))
        if (num_iter % 100) == 0:
            print('%d - %d complete!' % ((epoch+1), num_iter))
            print(result_folder_name)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    epoch_biassed = epoch + epoch_bias
    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_local_losses)),
                                                              torch.mean(torch.FloatTensor(G_local_losses))))

    fixed_p1 = root + result_folder_name+ '/' + model + str(epoch_biassed + 1 ) + '.png'
    fixed_p3 = root + result_folder_name+ '/' + model + str(epoch_biassed + 1 ) + '_gt.png'
    
    #Evaluate
    if epoch == 0:
        show_result((epoch_biassed+1),x_fixed ,save=True, path=fixed_p3)
        for t in range(y_fixed.size()[1]):
            fixed_p2 = root + result_folder_name+ '/' + model + str(epoch_biassed + 1 + t ) + '_gt_y.png'
            show_result((epoch_biassed+1), y_fixed[:,t:t+1,:,:] ,save=True, path=fixed_p2)
        
    show_result((epoch_biassed+1),G_local(z_fixed, y_fixed) ,save=True, path=fixed_p1)
    
    #Save model params
    if save_models and (epoch_biassed>21 and epoch_biassed % 10 == 0 ):
        torch.save(G_local.state_dict(), root +model_folder_name+ '/' + model + 'G_glob_epoch_'+str(epoch_biassed)+'.pth')
        torch.save(D_local.state_dict(), root + model_folder_name +'/'+ model + 'D_glob_epoch_'+str(epoch_biassed)+'.pth')
          
        
end_time = time.time()
total_ptime = end_time - start_time
print("Training finish!... save training results")
print('Training time: ' + str(total_ptime))