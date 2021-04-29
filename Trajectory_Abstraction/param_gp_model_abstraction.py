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

from ParamDataset import *

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
parser.add_argument("--x_dim", type=int, default=1, help="number of channels of x")
parser.add_argument("--par_dim", type=int, default=1, help="number of channels of par")
parser.add_argument("--model_name", type=str, default="eSIRS", help="name of the model")
parser.add_argument("--species_labels", type=str, default=["S", "I"], help="list of species names")
parser.add_argument("--training_flag", type=bool, default=False, help="do training or not")
parser.add_argument("--loading_id", type=str, default="", help="id of the model to load")
parser.add_argument("--po_flag", type=bool, default=False, help="id of the model to load")
parser.add_argument("--n_filters_critic", type=int, default=64, help="number of filters in critic")

opt = parser.parse_args()


if opt.model_name == "Oscillator":
    opt.species_labels = ["A", "B", "C"]
if opt.model_name == "SIR":
    opt.species_labels = ["S", "I", "R"]
if opt.model_name == "ToggleSwitch":
    opt.species_labels = ["P1", "P2"]
if opt.model_name == "MAPK":
    opt.species_labels = ["V1", "MAPK-PP"]
print(opt)

cuda = True if torch.cuda.is_available() else False

model_name = opt.model_name
PO_FLAG = opt.po_flag
if PO_FLAG:
    trainset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_training_set.pickle"
    testset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_validation_set.pickle"
    gridset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_PO_grid_validation_set.pickle"
else:
    trainset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_training_set.pickle"
    testset_fn = "../Dataset_Generation/data/"+model_name+"/"+model_name+"_validation_set.pickle"




ds = ParamDataset(trainset_fn, testset_fn, opt.x_dim, opt.par_dim, opt.traj_len)
ds.load_train_data()
ds.load_test_data()

opt.y_dim = 1+opt.par_dim

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.padd = 1
        self.n_filters = 2*self.padd+1
        self.Q = opt.traj_len // (2**4)
        self.Nch = 512
        
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.Nch * self.Q))
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(self.Nch+opt.x_dim+opt.par_dim, 128, 4, stride=2, padding=self.padd),
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

    def forward(self, noise, init_state, param):
        init_state_rep = init_state.repeat(1,1,self.Q)
        
        param_rep = param.repeat(1, 1, self.Q)

        noise_out = self.l1(noise)
        noise_out = noise_out.view(noise_out.shape[0], self.Nch, self.Q)
        gen_input = torch.cat((init_state_rep, param_rep, noise_out), 1)
        
        traj = self.conv_blocks(gen_input)
        return traj


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, L):
            padd = 1
            n_filters = 2*padd+2
            block = [nn.Conv1d(in_filters, out_filters, n_filters, stride=2, padding=padd), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.1)]
            block.append(nn.LayerNorm([out_filters, L]))

            return block

        self.n_filters = opt.n_filters_critic
        self.model = nn.Sequential(
            *discriminator_block(opt.x_dim+opt.par_dim, self.n_filters, (opt.traj_len+1)//2),
            *discriminator_block(self.n_filters, self.n_filters, (opt.traj_len+1)//4)
            
        )

        # The height and width of downsampled image
        ds_size = (opt.traj_len + 1) // (2**2)
        self.adv_layer = nn.Sequential(nn.Linear(self.n_filters * ds_size, 1))
        
    def forward(self, traj, init_state, param):
        #print("A------", trajs.shape, conditions.shape)
        full_traj = torch.cat((init_state, traj), 2)
        param_rep = param.repeat(1, 1, opt.traj_len+1)

        d_in = torch.cat((full_traj, param_rep), 1)

        out = self.model(d_in)
        #print("B------", out.shape)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        #print("C------", validity.shape)
        return validity

DO_TRAINING = True

if DO_TRAINING:
    ID = str(np.random.randint(0,100000))
    print("ID = ", ID)
else:
    ID = opt.loading_id

plots_path = model_name+"/MA_w_Repeat/ID_"+ID
os.makedirs(plots_path, exist_ok=True)
f = open(plots_path+"/log.txt", "w")
f.write(str(opt))
f.close()

MODEL_PATH = plots_path+"/generator_{}epochs.pt".format(opt.n_epochs)

# Loss weight for gradient penalty
lambda_gp = 10


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples, init_state, param):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, init_state, param)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.reshape(gradients.shape[0], opt.traj_len*opt.x_dim)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generate_random_conditions():
    rnd_s0 = (np.random.rand(opt.batch_size, opt.x_dim, 1)-0.5)*2
    rnd_p = (np.random.rand(opt.batch_size, opt.par_dim, 1)-0.5)*2
    
    return rnd_s0, rnd_p

# ----------
#  Training
# ----------
# Initialize generator and discriminator


if DO_TRAINING:

    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    
    batches_done = 0
    G_losses = []
    D_losses = []
    real_comp = []
    gen_comp = []
    gp_comp = []

    full_G_loss = []
    full_D_loss = []
    for epoch in range(opt.n_epochs):
        bat_per_epo = int(ds.n_points_dataset / opt.batch_size)
        n_steps = bat_per_epo * opt.n_epochs
        
        tmp_G_loss = []
        tmp_D_loss = []

        
        for i in range(bat_per_epo):
            trajs_np, _, init_states_np, params_np = ds.generate_full_mini_batches(opt.batch_size)
            # Configure input
            real_trajs = Variable(Tensor(trajs_np))
            init_states = Variable(Tensor(init_states_np))
            params = Variable(Tensor(params_np))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

            # Generate a batch of images
            fake_trajs = generator(z, init_states, params)
            # Real images
            real_validity = discriminator(real_trajs, init_states, params)
            # Fake images
            fake_validity = discriminator(fake_trajs, init_states, params)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_trajs.data, fake_trajs.data, init_states.data, params.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            real_comp.append(torch.mean(real_validity).item())
            gen_comp.append(torch.mean(fake_validity).item())
            gp_comp.append(lambda_gp * gradient_penalty.item())
            tmp_D_loss.append(d_loss.item())
            full_D_loss.append(d_loss.item())

            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                gen_init_states_np, gen_params_np = generate_random_conditions()
                gen_init_states = Variable(Tensor(gen_init_states_np))
                gen_params = Variable(Tensor(gen_params_np))

                # Generate a batch of images
                gen_trajs = generator(z, gen_init_states, gen_params)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(gen_trajs, gen_init_states, gen_params)
                g_loss = -torch.mean(fake_validity)
                tmp_G_loss.append(g_loss.item())
                full_G_loss.append(g_loss.item())
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs, i, bat_per_epo, d_loss.item(), g_loss.item())
                )

                batches_done += opt.n_critic

        if (epoch+1) % 250 == 0:
            torch.save(generator, plots_path+"/generator_{}epochs.pt".format(epoch+1))    
        
        D_losses.append(np.mean(tmp_D_loss))
        G_losses.append(np.mean(tmp_G_loss))
    
    fig, axs = plt.subplots(2,1,figsize = (12,6))
    axs[0].plot(np.arange(opt.n_epochs), G_losses)
    axs[1].plot(np.arange(opt.n_epochs), D_losses)
    axs[0].set_title("generator loss")
    axs[1].set_title("critic loss")
    plt.tight_layout()
    fig.savefig(plots_path+"/losses.png")
    plt.close()

    fig1, axs1 = plt.subplots(2,1,figsize = (12,6))
    axs1[0].plot(np.arange(len(full_G_loss)), full_G_loss)
    axs1[1].plot(np.arange(len(full_D_loss)), full_D_loss)
    axs1[0].set_title("generator loss")
    axs1[1].set_title("critic loss")
    plt.tight_layout()
    fig1.savefig(plots_path+"/full_losses.png")
    plt.close()

    fig2, axs2 = plt.subplots(3,1, figsize = (12,9))
    axs2[0].plot(np.arange(n_steps), real_comp)
    axs2[1].plot(np.arange(n_steps), gen_comp)
    axs2[2].plot(np.arange(n_steps), gp_comp)
    axs2[0].set_title("real term")
    axs2[1].set_title("generated term")
    axs2[2].set_title("gradient penalty term")
    plt.tight_layout()
    fig2.savefig(plots_path+"/components.png")
    plt.close()

    # save the ultimate trained generator    
    torch.save(generator, MODEL_PATH)
else:
    # load the ultimate trained generator
    print("MODEL_PATH: ", MODEL_PATH)
    generator = torch.load(MODEL_PATH)
    generator.eval()
    if cuda:
        generator.cuda()

print("Computing test trajectories...")
ds.load_test_data()
n_gen_trajs = ds.n_traj_per_point
gen_trajectories = np.empty(shape=(ds.n_points_test, n_gen_trajs, opt.x_dim, opt.traj_len))
for iii in range(ds.n_points_test):
    print("Test point nb ", iii+1, " / ", ds.n_points_test)
    for jjj in range(n_gen_trajs):
        z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
        temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.I_test_transp[iii]])), Variable(Tensor([ds.P_test_transp[iii]])))
        gen_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]

trajs_dict = {"gen_trajectories": gen_trajectories}
file = open(plots_path+'/generated_validation_trajectories.pickle', 'wb')
# dump information to that file
pickle.dump(trajs_dict, file)
# close the file
file.close()

colors = ['blue', 'orange']
leg = ['real', 'gen']
#PLOT TRAJECTORIES 
n_trajs_to_plot = 10
print("Plotting test trajectories...")      
tspan = range(opt.traj_len)
for kkk in range(ds.n_points_test):
    print("Test point nb ", kkk+1, " / ", ds.n_points_test)
    fig, ax = plt.subplots(opt.x_dim)
    G = np.array([np.round(ds.XMIN+(gen_trajectories[kkk, it].T+1)*(ds.XMAX-ds.XMIN)/2).T for it in range(ds.n_traj_per_point)])
    R = np.array([np.round(ds.XMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.XMAX-ds.XMIN)/2).T for it in range(ds.n_traj_per_point)])

    for d in range(opt.x_dim):
        if opt.x_dim == 1:
            axd = ax
        else:
            axd = ax[d]
        for traj_idx in range(n_trajs_to_plot):
            #axd.plot(tspan, ds.X_test_transp[kkk,traj_idx, d], color=colors[0])
            #axd.plot(tspan, gen_trajectories[kkk,traj_idx,d], color=colors[1])
            axd.plot(tspan, R[traj_idx, d], color=colors[0])
            axd.plot(tspan, G[traj_idx,d], color=colors[1])
    plt.tight_layout()
    #fig.savefig(plots_path+"/"+opt.model_name+"_Trajectories"+str(kkk)+".png")
    fig.savefig(plots_path+"/"+opt.model_name+"_Rescaled_Trajectories"+str(kkk)+".png")
    plt.close()

#PLOT HISTOGRAMS
if True:

    bins = 50
    time_instant = -1
    print("Plotting histograms...")
    for kkk in range(ds.n_points_test):
        fig, ax = plt.subplots(opt.x_dim,1, figsize = (12,opt.x_dim*3))
        for d in range(opt.x_dim):
            #XXX = np.vstack((ds.X_test_transp[kkk,:,d, time_instant], gen_trajectories[kkk,:,d, time_instant])).T
            G = np.array([np.round(ds.XMIN+(gen_trajectories[kkk, it].T+1)*(ds.XMAX-ds.XMIN)/2).T for it in range(ds.n_traj_per_point)])
            R = np.array([np.round(ds.XMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.XMAX-ds.XMIN)/2).T for it in range(ds.n_traj_per_point)])

            XXX = np.vstack((R[:,d, time_instant], G[:,d, time_instant])).T
            
            if opt.x_dim == 1:
                axd = ax
            else:
                axd = ax[d]
            axd.hist(XXX, bins = bins, stacked=False, density=False, color=colors, label=leg)
            axd.legend()
            axd.set_ylabel(opt.species_labels[d])

        #figname = plots_path+"/"+opt.model_name+"_hist_comparison_{}th_timestep_{}.png".format(time_instant, kkk)
        figname = plots_path+"/"+opt.model_name+"_rescaled_hist_comparison_{}th_timestep_{}.png".format(time_instant, kkk)
        fig.savefig(figname)

        plt.close()


#COMPUTE WASSERSTEIN DISTANCES

if True:
    dist = np.zeros(shape=(ds.n_points_test, opt.x_dim, opt.traj_len))
    print("Computing and Plotting Wasserstein distances...") 
    for kkk in range(ds.n_points_test):
        print("\tinit_state n = ", kkk)
        for m in range(opt.x_dim):
            for t in range(opt.traj_len):    
                A = ds.X_test_transp[kkk,:,m,t]
                B = gen_trajectories[kkk,:,m,t]
                dist[kkk, m, t] = wasserstein_distance(A, B)
                

    avg_dist = np.mean(dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(opt.x_dim):
        plt.plot(np.arange(opt.traj_len), avg_dist[spec], markers[spec], label=opt.species_labels[spec])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.tight_layout()

    figname = plots_path+"/"+opt.model_name+"_Traj_avg_wass_distance_{}epochs_{}steps.png".format(opt.n_epochs, opt.traj_len)
    fig.savefig(figname)
    distances_dict = {"gen_hist":B, "ssa_hist":A, "wass_dist":dist}
    file = open(plots_path+'/wgan_gp_distances.pickle', 'wb')
    # dump information to that file
    pickle.dump(distances_dict, file)
    # close the file
    file.close()
