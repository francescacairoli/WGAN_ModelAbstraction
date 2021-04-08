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
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.95, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
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
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.y_dim, 64 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, opt.x_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, conditions):
        conds_flat = conditions.view(conditions.shape[0],-1)
        gen_input = torch.cat((conds_flat, noise), 1)
        out = self.l1(gen_input)     
        out = out.view(out.shape[0], 64, self.init_size)
        traj = self.conv_blocks(out)
        return traj

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

opt.y_dim = opt.x_dim
cuda = True if torch.cuda.is_available() else False

model_name = opt.model_name

trainset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_dataset_fixed_param_1000x50.pickle"
test_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_grid_dataset_fixed_param_100x5000.pickle" # GRID

ds = Dataset(trainset_fn, "", opt.x_dim, opt.y_dim, opt.traj_len)
ds.load_train_data()
ds.add_grid_data(test_fn)
ds.load_grid_test_data()

ID = opt.loading_id
plots_path = model_name+"/MA_Plots/ID_"+ID
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
        #print("k = ", k)
        for s in range(opt.x_dim):
            #print("s = ", s)
            for t in range(opt.traj_len):
                #print("t = ", t)
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

    MEAN_DIFF = distances_dict["MEAN_DIFF"]
    VAR_DIFF = distances_dict["VAR_DIFF"]
    WASS_DIST = distances_dict["WASS_DIST"]

    avg_mean = np.mean(np.mean(MEAN_DIFF, axis = 1), axis = 1)
    avg_var = np.mean(np.mean(VAR_DIFF, axis = 1), axis = 1)
    avg_wass = np.mean(np.mean(WASS_DIST, axis = 1), axis = 1)

    grid_len = int(np.sqrt(ds.n_points_grid))
    
    mean_grid = np.reshape(avg_mean, (grid_len, grid_len))
    var_grid = np.reshape(avg_var, (grid_len, grid_len))
    wass_grid = np.reshape(avg_wass, (grid_len, grid_len))

    xlb, xub = 50, 200
    ylb, yub = 50, 200

    fig1 = plt.figure(figsize = (12,12))
    c1 = plt.imshow(mean_grid, extent=[xlb, xub, ylb, yub])
    plt.xlabel(opt.species_labels[0])
    plt.ylabel(opt.species_labels[1])
    plt.title("average mean distance")
    fig1.colorbar(c1)
    plt.tight_layout()
    fig1.savefig(plots_path+'/average_mean_distance.png')
    fig2 = plt.figure(figsize = (12,12))
    c2 = plt.imshow(var_grid, extent=[xlb, xub, ylb, yub])
    plt.xlabel(opt.species_labels[0])
    plt.ylabel(opt.species_labels[1])
    plt.title("average variance distance")
    fig2.colorbar(c2)
    plt.tight_layout()
    fig2.savefig(plots_path+'/average_var_distance.png')
    fig3 = plt.figure(figsize = (12,12))
    c3 = plt.imshow(wass_grid, extent=[xlb, xub, ylb, yub])
    plt.xlabel(opt.species_labels[0])
    plt.ylabel(opt.species_labels[1])
    plt.tight_layout()
    plt.title("average wasserstein distance")
    fig3.colorbar(c3)
    fig3.savefig(plots_path+'/average_wass_distance.png')