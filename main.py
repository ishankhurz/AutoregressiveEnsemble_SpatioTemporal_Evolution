#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:08:39 2024

@author: ishan
"""

##Import modules
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
#from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import model_forward, model_forward_conv
#from torch.utils.data import TensorDataset, DataLoader
from utils import (imgdata_loader, save_checkpoint, conv_forward_pass_unet, 
                   recursive_pred_unet, recursive_pred_moe, addition_moe_forward_pass)
from models import Unet2D


##Set device
torch.cuda.empty_cache()    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gpu = False


##Set file-paths for all datasets
data_dir = '/Users/ishan/JHU/Data/HighDim_datasets/datasets/planetswe/data_proc/'
#path_ip_test = data_dir + 'inputs_test_planetswe_96traj_50steps_udim1_timelb3.npz'
#path_op_test = data_dir + 'outputs_test_planetswe_96traj_50steps_udim1_timelb3.npz'
#path_load_test = data_dir + 'params_test_planetswe_96traj_50steps_udim1_timelb3.npz'
stats = np.load(data_dir+'stats_commonnorm_planetswe_96traj_50steps_udim1.npz', allow_pickle=True)


data_dir = '/Users/ishan/JHU/Data/HighDim_datasets/datasets/gray_scott_reaction_diffusion/data_proc/'
#path_ip_test = data_dir + 'inputs_test_grayscott_960traj_1to50stepsinc1_speciesA_timelb5.npz'
#path_op_test = data_dir + 'outputs_test_grayscott_960traj_1to50stepsinc1_speciesA_timelb5.npz'
#path_load_test = data_dir + 'params_test_grayscott_960traj_1to50stepsinc1_speciesA_timelb5.npz'
stats = np.load(data_dir+'stats_commonnorm_grayscott_960traj_50steps_speciesA.npz', allow_pickle=True)


data_dir = '/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/'
path_ip_test = data_dir + 'inputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
path_op_test = data_dir + 'outputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
path_load_test = data_dir + 'loading_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
stats = np.load(data_dir+'stats_commonnorm_20fiber_elastoplastic_3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load.npz', allow_pickle=True)


save_path = '/Users/ishan/JHU/Data/Outputs/'  
expt_path = '20fiber_loadpathvar_branchfnn_trunkunet_loadimgtrunkip'
full_path = save_path + expt_path

##Define params
timesteps = 30
bs_train = 10
bs_test = 12
iterations = 10
lr_init = 0.001
lr_decay = 0.1
model = 'unet'
n_basis = 200
grid_dim = 128
learners = 4
model_list = []
pretrained = True
batch_size_viz = timesteps
data_samp = 1

##Load stats 
mean_ip = stats['arr_0'][0]['mean_ip']
std_ip = stats['arr_0'][0]['std_ip']


##Create train and test data loaders
ds_train = imgdata_loader(path_ip_train, path_op_train, path_load_train,
                          bs_train, 1, stats, shuffle = False)
ds_test = imgdata_loader(path_ip_test, path_op_test, path_load_test, bs_test, 
                        1, stats, shuffle = False)



if pretrained ==False:
    ##Training
    if model == 'gnn':
        deep_net = model_forward.GraphDeepONet(node_feat_size = 6, edge_feat_size = 4, output_size=n_basis, latent_size_mp=64, message_passing_steps=4, window = 1, trunk_feat_size = 6, grid_dim = 128)
    elif model == 'cnn':
        deep_net = model_forward_conv.ConvDeepONet(branch_feat_size = 1, output_size=n_basis, trunk_feat_size = 1, grid_dim = 128)
    elif model == 'unet':
        for i in range(learners):
            deep_net = Unet2D(dim=16, out_dim = 1, dim_mults=(1, 2, 4, 8), channels = 5, ip_time_dim = 8, attn = True)
            model_list.append(deep_net)
else:
    ##Testing pre trained modesls
  for i in range(learners):
     # model = torch.load("/Users/ishan/JHU/Data/Results/planetswe_96traj50stepsudim1_3lb1lf+timeid_unet_timemlpmembed_attn_commonnorm_onlymodel"+str(i),
     #                    map_location = device)
     # model = torch.load("/Users/ishan/JHU/Data/Results/grayscott_960traj1to50stepsinc1_speciesA5lb1lf+timeid_unet_timemlpmembed_attn_commonnorm_80iter_onlymodel"+str(i),
     #                    map_location = device)
      model = torch.load("/Users/ishan/JHU/Data/Outputs/20fiber_loadonlypath_micro70loadpath10var_3lb1lf+timeid_sig11_unet_timemlpmembed_attn_commonnorm_onlymodel"+str(i),
                         map_location = device)
      model_list.append(model)
  
    

params = []
for i in range(learners):
    params = params + list(model_list[i].parameters())

##Define optimizer and lr scheduler
optimizer = torch.optim.Adam(params, lr=lr_init)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

#scheduler = CosineAnnealingLR(optimizer, iterations)
#scheduler = ReduceLROnPlateau(optimizer, 'min')


start_time = time.time()
train_loss, test_loss = [], []
#available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
#branch_net = torch.nn.DataParallel(branch_net, device_ids=[available_gpus])

for model in model_list:
    model.to(device)
    model.train()



####Training + testing loop #########
for i in range(iterations):
        print('\nEpoch:',i)
        t_curr = time.time()
        loss_test_copy, loss_train_copy = 0, 0
        for model in model_list:
            model.train(True)
        for j, data in enumerate(ds_train): 
            print('Data iteration: ', j)
               # targets, preds = conv_forward_pass_unet(data[0], data[1], 
               #                             data[2], deep_net2, device)
            targets, preds = addition_moe_forward_pass(data[0], data[1], data[2],
                                                  model_list, device)
            targets = torch.squeeze(targets, dim=1)
            preds = torch.squeeze(preds, dim=1)
            loss = F.mse_loss(preds, targets)
            # loss = torch.mean(torch.square(preds-targets)/(torch.square(targets)+ 1e-4)).to(device)
            # loss = torch.mean(torch.sum(torch.abs(targets)*torch.square(preds-targets), dim=-1)/torch.sum(targets,dim=-1))
            loss = loss.to(device)
           
            loss_train_copy += loss.item()
            
            # Compute the gradient of loss and updated weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

               ##Loss printing
            if True:
                print("\nImage processed: ", (j+1)*bs_train)
                t_curr1 = time.time()
                print('Time elapsed:', t_curr1 - t_curr)
                
        train_loss.append(loss_train_copy/(j+1))
        
        
        ##Test mode
        for model in model_list:
            model.eval()

        with torch.no_grad():
            for k, data_test in enumerate(ds_test):
               # targets_test, preds_test = conv_forward_pass_unet(data_test[0], data_test[1], 
               #                             data_test[2], deep_net2, device)
                targets_test, preds_test = addition_moe_forward_pass(data_test[0], data_test[1], 
                                    data_test[2],model_list, device)
                targets_test = torch.squeeze(targets_test, dim=1)
                preds_test = torch.squeeze(preds_test, dim=1)
                loss_test =  F.mse_loss(preds_test, targets_test)
                loss_test = loss_test.to(device)
                loss_test_copy += loss_test.item()  
  
        test_loss.append(loss_test_copy/(k+1))
        print(f"\nTraining step: {i+1}/{iterations}. Train loss: {train_loss[-1]}. Test loss: {test_loss[-1]}\n") 
       # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
       #scheduler.step(loss_test)

##Save models
for i in range(len(model_list)):
    torch.save(model_list[i], save_path + expt_path + 'model' + str(i))       


###### Metric viz ##########
plt.figure()
plt.semilogy(np.asarray(train_loss), label = 'Train')
plt.semilogy(np.asarray(test_loss), label = 'Test')
plt.xlabel('# Training epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(save_path + expt_path)  



##Test predictions
stress_preds, stress_targets, stress_br, stress_tr, stress_loads = [], [], [], [], []    
ds_viz = imgdata_loader(path_ip_test, path_op_test, path_load_test, batch_size_viz, 
                        1, stats, shuffle = False)

for model in model_list:
    model.eval()
start_time = time.time()    
with torch.no_grad():
    for j, data_viz in enumerate(ds_viz):
            print('Processing test sample: ', j)
           # targets_viz, preds_viz, = recursive_pred_unet(data_viz[0], data_viz[1], 
           #                                         data_viz[2], deep_net1, device, timesteps)
            targets_viz, preds_viz = recursive_pred_moe(data_viz[0], data_viz[1], 
                                                        data_viz[2], model_list[3:], device, timesteps)
            stress_preds.append(preds_viz.cpu().detach().numpy())
            stress_targets.append(targets_viz.cpu().numpy())

end_time = time.time()
print('Time elapsed:', end_time-start_time)
        
##Save predictions + ground truth
expt_data = []
outputs = {}
outputs['gt'] = stress_targets
outputs['preds'] = stress_preds
expt_data.append(outputs)
np.savez(save_path + '20fiber_loadonlypathvar_3lb1lf_conv2dpenultimatemoe_recursive', expt_data)








