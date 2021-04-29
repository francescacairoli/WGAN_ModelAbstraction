import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from scipy.stats import wasserstein_distance
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from Dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=480, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--traj_len", type=int, default=32, help="number of steps")
parser.add_argument("--x_dim", type=int, default=2, help="number of channels of x")
parser.add_argument("--model_name", type=str, default="eSIRS", help="name of the model")
parser.add_argument("--species_labels", type=str, default=["S", "I"], help="list of species names")
parser.add_argument("--training_flag", type=bool, default=False, help="do training or not")
parser.add_argument("--loading_id", type=str, default="", help="id of the model to load")
parser.add_argument("--po_flag", type=bool, default=False, help="id of the model to load")
opt = parser.parse_args()
print(opt)

if opt.model_name == "Oscillator":
    opt.species_labels = ["A", "B", "C"]
if opt.model_name == "SIR":
    opt.species_labels = ["S", "I", "R"]
if opt.model_name == "ToggleSwitch":
    opt.species_labels = ["P1", "P2"]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.traj_len // int(opt.traj_len/2)

        self.padd = 1
        self.n_filters = 2*self.padd+1
        self.Q = 2
        self.Nch = 512

        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.Nch * self.Q))

        if opt.traj_len == 32:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(self.Nch+opt.x_dim, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv1d(128, opt.x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )
        else:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(self.Nch+opt.x_dim, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv1d(128, opt.x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )


    def forward(self, noise, conditions):
        conds_flat = conditions.view(conditions.shape[0],-1)
        conds_rep = conds_flat.repeat(1,self.Q).view(conditions.shape[0], opt.x_dim, self.Q)
        noise_out = self.l1(noise)
        noise_out = noise_out.view(noise_out.shape[0], self.Nch, self.Q)
        gen_input = torch.cat((conds_rep, noise_out), 1)
        traj = self.conv_blocks(gen_input)
        
        return traj


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

opt.y_dim = opt.x_dim

model_name = opt.model_name
PO_FLAG = opt.po_flag
if PO_FLAG:
    trainset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_dataset_fixed_param_1000x50.pickle"
    test_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_grid_dataset_fixed_param_100x5000.pickle"
else:
    trainset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_training_set.pickle"
    test_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_grid_validation_set.pickle"

ds = Dataset(trainset_fn, "", opt.x_dim, opt.y_dim, opt.traj_len)
ds.load_train_data()
ds.add_grid_data(test_fn)
ds.load_grid_test_data()

ID = opt.loading_id
plots_path = model_name+"/MA_w_Transp/ID_"+ID
MODEL_PATH = plots_path+"/generator_{}epochs.pt".format(opt.n_epochs)
print("MODEL_PATH: ", MODEL_PATH)
generator = torch.load(MODEL_PATH)
generator.eval()
if cuda:
    generator.cuda()


n_gen_trajs = ds.n_traj_per_point_grid #5000

DO_COMPUTATIONS = False

if DO_COMPUTATIONS:
    print("Computing test trajectories...")

    gen_trajectories = np.empty(shape=(ds.n_points_grid, n_gen_trajs, opt.x_dim, opt.traj_len))
    for iii in range(ds.n_points_grid):
        print("Test point nb ", iii+1, " / ", ds.n_points_grid)
        for jjj in range(n_gen_trajs):
            z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
            temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_test_grid_transp[iii]])))
            gen_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]

    MEAN_DIFF = np.empty((ds.n_points_grid, opt.x_dim, opt.traj_len))
    VAR_DIFF = np.empty((ds.n_points_grid, opt.x_dim, opt.traj_len))
    WASS_DIST = np.empty((ds.n_points_grid, opt.x_dim, opt.traj_len))
    for k in range(ds.n_points_grid):
        for s in range(opt.x_dim):
            for t in range(opt.traj_len):
                R = ds.X_test_grid_transp[k,:,s, t]
                G = gen_trajectories[k,:,s, t]

                MEAN_DIFF[k, s, t] = np.abs(np.mean(R)-np.mean(G))
                VAR_DIFF[k, s, t] = np.abs(np.var(R)-np.var(G))
                WASS_DIST[k, s, t] = wasserstein_distance(R, G)

    distances_dict = {"MEAN_DIFF": MEAN_DIFF, "VAR_DIFF": VAR_DIFF, "WASS_DIST": WASS_DIST, "gen_trajectories": gen_trajectories}
    file = open(plots_path+'/grid_distances.pickle', 'wb')
    # dump information to that file
    pickle.dump(distances_dict, file)
    # close the file
    file.close()
else:
    file = open(plots_path+'/grid_distances.pickle', 'rb')
    # dump information to that file
    distances_dict = pickle.load(file)
    # close the file
    file.close()

    ABS_ERR = False

    gen_trajectories = distances_dict["gen_trajectories"]
    if ABS_ERR:
        MEAN_DIFF = distances_dict["MEAN_DIFF"]
        VAR_DIFF = distances_dict["VAR_DIFF"]
        WASS_DIST = distances_dict["WASS_DIST"]
    else:
        MEAN_DIFF = np.empty((ds.n_points_grid, opt.x_dim, opt.traj_len))
        VAR_DIFF = np.empty((ds.n_points_grid, opt.x_dim, opt.traj_len))
        for k in range(ds.n_points_grid):
            G = np.array([np.round(ds.HMIN+(gen_trajectories[k, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_traj_per_point_grid)])
            R = np.array([np.round(ds.HMIN+(ds.X_test_grid_transp[k, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_traj_per_point_grid)])

            for s in range(opt.x_dim):
                for t in range(opt.traj_len):
                    
                    MEAN_DIFF[k, s, t] = np.abs(np.mean(R[:,s,t])-np.mean(G[:,s,t]))/np.mean(R[:,s,t])
                    VAR_DIFF[k, s, t] = np.abs(np.var(R[:,s,t])-np.var(G[:,s,t]))/np.var(R[:,s,t])

grid_len = int(np.sqrt(ds.n_points_grid))
if model_name == "ToggleSwitch":
    xlb, xub = 50, 200
    ylb, yub = 200, 50
elif model_name == "eSIRS":
    xlb, xub = 5, 45
    ylb, yub = 45, 5
else: #MAPK
    xlb, xub = 0.1, 2.5
    ylb, yub = 290, 10

if True:
    mean_diff_grid = np.reshape(MEAN_DIFF, (grid_len, grid_len, opt.x_dim, opt.traj_len))
    var_diff_grid = np.reshape(VAR_DIFF, (grid_len, grid_len, opt.x_dim, opt.traj_len))
    #wass_dist_grid = np.reshape(WASS_DIST, (grid_len, grid_len, opt.x_dim, opt.traj_len))
    
    time_instants = [0,15,31]
    grid_plots_path = plots_path+"/GridPlots/"
    os.makedirs(grid_plots_path, exist_ok=True)

    fig1, ax1 = plt.subplots(3,opt.x_dim, figsize=(4*3, 8*opt.x_dim))
    for s in range(opt.x_dim):
        for it, t in enumerate(time_instants):
            c1 = ax1[it, s].imshow(mean_diff_grid[:,:,s, t], extent=[xlb, xub, ylb, yub])
            fig1.colorbar(c1, ax = ax1[it, s])
            ax1[it, s].set_xlabel(opt.species_labels[0])
            ax1[it, s].set_ylabel(opt.species_labels[1])
            ax1[it, s].set_title(opt.species_labels[s]+" at step {}".format(t+1))
        
    plt.tight_layout()
    fig1.savefig(grid_plots_path+model_name+'_mean_relative_distance_at_times_{}.png'.format(time_instants))
    plt.close()

    fig2, ax2 = plt.subplots(3,opt.x_dim, figsize=(4*3, 8*opt.x_dim))
    for s in range(opt.x_dim):
        for it, t in enumerate(time_instants):
            c2 = ax2[it, s].imshow(var_diff_grid[:,:,s, t], extent=[xlb, xub, ylb, yub])
            fig2.colorbar(c2, ax = ax2[it, s])
            ax2[it, s].set_xlabel(opt.species_labels[0])
            ax2[it, s].set_ylabel(opt.species_labels[1])
            ax2[it, s].set_title(opt.species_labels[s]+" at step {}".format(t+1))
        
    plt.tight_layout()
    fig2.savefig(grid_plots_path+model_name+'_var_relative_distance_at_times_{}.png'.format(time_instants))
    plt.close()
    '''
    fig3, ax3 = plt.subplots(3,opt.x_dim, figsize=(4*3, 8*opt.x_dim))
    for s in range(opt.x_dim):
        for it, t in enumerate(time_instants):
            c3 = ax3[it, s].imshow(wass_dist_grid[:,:,s, t], extent=[xlb, xub, ylb, yub])
            fig3.colorbar(c3, ax = ax3[it, s])
            ax3[it, s].set_xlabel(opt.species_labels[0])
            ax3[it, s].set_ylabel(opt.species_labels[1])
            ax3[it, s].set_title(opt.species_labels[s]+" at step {}".format(t+1))
        
    plt.tight_layout()
    fig3.savefig(grid_plots_path+model_name+'_wass_distance_at_times_{}.png'.format(time_instants))
    plt.close()
    '''

if False: # Plots of the averages
    avg_mean = np.mean(np.mean(MEAN_DIFF, axis = 1), axis = 1)
    avg_var = np.mean(np.mean(VAR_DIFF, axis = 1), axis = 1)
    avg_wass = np.mean(np.mean(WASS_DIST, axis = 1), axis = 1)

    
    
    mean_grid = np.reshape(avg_mean, (grid_len, grid_len))
    var_grid = np.reshape(avg_var, (grid_len, grid_len))
    wass_grid = np.reshape(avg_wass, (grid_len, grid_len))


    print(mean_grid)
    fig1 = plt.figure(figsize = (12,12))
    c1 = plt.imshow(mean_grid, extent=[xlb, xub, ylb, yub])
    plt.xlabel(opt.species_labels[0])
    plt.ylabel(opt.species_labels[1])
    plt.title("average mean distance")
    fig1.colorbar(c1)
    plt.tight_layout()
    fig1.savefig(plots_path+'/average_mean_distance.png')
    plt.close()
    fig2 = plt.figure(figsize = (12,12))
    c2 = plt.imshow(var_grid, extent=[xlb, xub, ylb, yub])
    plt.xlabel(opt.species_labels[0])
    plt.ylabel(opt.species_labels[1])
    plt.title("average variance distance")
    fig2.colorbar(c2)
    plt.tight_layout()
    fig2.savefig(plots_path+'/average_var_distance.png')
    plt.close()
    fig3 = plt.figure(figsize = (12,12))
    c3 = plt.imshow(wass_grid, extent=[xlb, xub, ylb, yub])
    plt.xlabel(opt.species_labels[0])
    plt.ylabel(opt.species_labels[1])
    plt.tight_layout()
    plt.title("average wasserstein distance")
    fig3.colorbar(c3)
    fig3.savefig(plots_path+'/average_wass_distance.png')
    plt.close()