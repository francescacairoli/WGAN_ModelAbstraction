import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle
import time

class AbstractionDataset(object):

    def __init__(self, n_init_states, n_trajs, state_space_dim, param_space_bound, model_name, time_step, T):

        self.n_init_states = n_init_states
        self.n_trajs = n_trajs
        self.n_training_points = n_init_states*n_trajs
        self.state_space_dim = state_space_dim
        self.stoch_mod = stochpy.SSA(IsInteractive=False)
        self.stoch_mod.Model(model_name+'.psc')
        self.directory_name = model_name
        self.time_step = time_step
        self.T = T # end time
        self.param_space_bound = param_space_bound
        self.param_space_dim = param_space_bound.shape[0]

    def time_resampling(self, data):
        time_index = 0
        # Il nuovo array dei tempi
        time_array = np.linspace(0, self.T, num=int(self.T / self.time_step+1))
        # new_data conterrà i dati con la nuova scansione temporale
        # la prima colonna contiene gli istanti di tempo, e quindi corrisponde a time_array
        new_data = np.zeros((time_array.shape[0], data.shape[1]))
        new_data[:, 0] = time_array
        for j in range(len(time_array)):
            while time_index < data.shape[0] - 1 and data[time_index + 1][0] < time_array[j]:
                time_index = time_index + 1
            if time_index == data.shape[0] - 1:
                new_data[j, 1:] = data[time_index, 1:]
            else:
                new_data[j, 1:] = data[time_index, 1:]
        return new_data


    def set_initial_states(self, init_state):
        
        list_of_species = ["M3K", "M3Kp", "M2K", "M2Kp", "M2Kpp", "MAPK", "MAPKp", "MAPKpp"]
        for i in range(self.state_space_dim):
            self.stoch_mod.ChangeInitialSpeciesCopyNumber(list_of_species[i], init_state[i])
        


    def sample_initial_states(self, n_points=None):
        if n_points == None:
            n_points = self.n_init_states

        set_of_init_states = np.empty((n_points, self.state_space_dim))
        for i in range(n_points):
            m3k = np.random.randint(low=0, high=100)
            m2k = np.random.randint(low=0, high=300)
            m2kp = np.random.randint(low=0, high=int(300-m2k))
            mapk = np.random.randint(low=0, high=300)
            mapkp = np.random.randint(low=0, high=int(300-mapk))
            
            set_of_init_states[i] = np.array([m3k, 100-m3k, m2k, m2kp, 300-m2k-m2kp, mapk, mapkp, 300-mapk-mapkp])

        return set_of_init_states

    def set_parameters(self, V1):
        self.stoch_mod.ChangeParameter("V1", V1)

    def sample_parameters_settings(self, n_points = None):
        if n_points == None:
            n_points = self.n_init_states

        set_of_params = (self.param_space_bound[1] - self.param_space_bound[0])*random(size=(n_points,))+self.param_space_bound[0]

        return set_of_params

    def sample_grid_config(self, n_points):
        
        param_grid = np.linspace(self.param_space_bound[0], self.param_space_bound[1], n_points)
        output_grid = np.linspace(0,300,n_points)
        
        return param_grid, output_grid


    def generate_training_set(self):
        Yp = np.zeros((self.n_training_points,self.param_space_dim))
        Ys = np.zeros((self.n_training_points,self.state_space_dim))
        X = np.zeros((self.n_training_points, int(self.T/self.time_step), self.state_space_dim))

        initial_states = self.sample_initial_states()
        set_of_params = self.sample_parameters_settings()

        count, avg_time = 0, 0
        for i in tqdm(range(self.n_init_states)):
            self.set_initial_states(initial_states[i,:])
            self.set_parameters(set_of_params[i])
            for k in range(self.n_trajs):
                begin_time = time.time()
                self.stoch_mod.DoStochSim(method="Direct", trajectories=3, mode="time", end=self.T)
                if False:
	                self.stoch_mod.PlotSpeciesTimeSeries(species2plot =['MAPK', 'MAPKpp'])
	                stochpy.plt.savefig('NEW_V1={}_MAPK_{}{}.png'.format(set_of_params[i], i,k))
                ntraj_time = time.time()-begin_time

                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)
                avg_time += ntraj_time
                datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()

                new_datapoint = self.time_resampling(datapoint)
                #print(new_datapoint)
                #print(new_datapoint.shape)
                X[count,:,:] = new_datapoint[1:,1:self.state_space_dim+1]
                Ys[count,:] = initial_states[i,:self.state_space_dim]
                Yp[count] = set_of_params[i]
                count += 1
        print("average time to gen 1 traj with SSA: ", avg_time/self.n_trajs)
        self.X = X
        self.Y_s0 = Ys
        self.Y_par = Yp
        self.Y = np.hstack((Yp,Ys))

    def generate_validation_set(self, n_val_points, n_val_trajs_per_point):
        Yp = np.zeros((n_val_points,self.param_space_dim))
        Ys = np.zeros((n_val_points,self.state_space_dim))
        X = np.zeros((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.state_space_dim))

        initial_states = self.sample_initial_states(n_val_points)
        set_of_params = self.sample_parameters_settings(n_val_points)
            
        for ind in range(n_val_points):
            self.set_initial_states(initial_states[ind,:])
            self.set_parameters(set_of_params[ind])
            Ys[ind,:] = initial_states[ind,:self.state_space_dim]
            Yp[ind,:] = set_of_params[ind]
                
            for k in range(n_val_trajs_per_point):
                if k%100 == 0:
                    print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
                
                self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
                
                new_datapoint = self.time_resampling(datapoint)
                X[ind,k,:,:] = new_datapoint[1:,1:self.state_space_dim+1]
            
        self.X = X
        self.Y_s0 = Ys
        self.Y_par = Yp
        self.Y = np.hstack((Yp,Ys))

    def generate_grid_validation_set(self, n_val_points, n_val_trajs_per_point):
        Yp = np.zeros((n_val_points,self.param_space_dim))
        Ys = np.zeros((n_val_points,self.state_space_dim))
        X = np.zeros((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.state_space_dim))

        initial_states = self.sample_initial_states(n_val_points)
        set_of_params = self.sample_parameters_settings(n_val_points)
            
        for ind in range(n_val_points):
            self.set_initial_states(initial_states[ind,:])
            self.set_parameters(set_of_params[ind])
            Ys[ind,:] = initial_states[ind,:self.state_space_dim]
            Yp[ind,:] = set_of_params[ind]
                
            for k in range(n_val_trajs_per_point):
                if k%100 == 0:
                    print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
                
                self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
                
                new_datapoint = self.time_resampling(datapoint)
                X[ind,k,:,:] = new_datapoint[1:,1:self.state_space_dim+1]
            
        self.X = X
        self.Y_s0 = Ys
        self.Y_par = Yp
        self.Y = np.hstack((Yp,Ys))

    def save_dataset(self, filename):
        dataset_dict = {"X": self.X, "Y_s0": self.Y_s0, "Y": self.Y, "Y_par": self.Y_par}
        with open(filename, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    




def run_training(filename):
    n_init_states = 5000
    n_trajs = 10

    state_space_dim = 8

    time_step = 60
    n_steps = 32
    T = n_steps*time_step

    param_space_dim = 1
    param_space_bounds = np.array([0.1,2.5])
    
    mapk_dataset = AbstractionDataset(n_init_states, n_trajs, state_space_dim, param_space_bounds, 'MAPK', time_step, T)

    start_time = time.time()
    mapk_dataset.generate_training_set()
    print("Time to generate the training set w fixed param =", time.time()-start_time)

    mapk_dataset.save_dataset(filename)


def run_validation(filename):
    n_init_states = 0
    n_trajs = 0

    state_space_dim = 8

    time_step = 60
    n_steps = 32
    T = n_steps*time_step

    n_val_points = 100
    n_trajs_per_point = 5000

    param_space_dim = 1
    param_space_bounds = np.array([0.1,2.5])

    sir_dataset = AbstractionDataset(n_init_states, n_trajs, state_space_dim, param_space_bounds, 'EGF', time_step, T)
    start_time = time.time()
    sir_dataset.generate_validation_set(n_val_points, n_trajs_per_point)
    print("Time to generate the validation set w fixed param =", time.time()-start_time)

    sir_dataset.save_dataset(filename)

run_training("../../data/MAPK/MAPK_training_set_one_param.pickle")
#run_validation("../../data/MAPK/MAPK_validation_one_param.pickle")
