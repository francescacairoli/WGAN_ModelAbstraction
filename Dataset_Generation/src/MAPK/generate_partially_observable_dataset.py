import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle
import time

class PO_AbstractionDataset(object):

    def __init__(self, n_init_settings, n_trajs, obs_state_bounds, param_space_bounds, model_name, time_step, T, obs_state_dim, non_obs_state_dim,):
    
        self.n_init_settings = n_init_settings
        self.n_trajs = n_trajs
        self.n_training_points = n_init_settings*n_trajs
        self.obs_state_bounds = obs_state_bounds
        self.obs_state_dim = obs_state_dim
        self.non_obs_state_dim = non_obs_state_dim
        self.global_state_space_dim = obs_state_dim+non_obs_state_dim
        self.stoch_mod = stochpy.SSA(IsInteractive=False)
        self.stoch_mod.Model(model_name+'.psc')
        self.directory_name = model_name
        self.time_step = time_step
        self.T = T # end time
        self.param_space_bounds = param_space_bounds
        self.param_space_dim = param_space_bounds.shape[0]
  
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


    def set_initial_states(self, obs_state, non_obs_states):
        list_of_nonobs_species = ["M3K", "M3Kp", "M2K", "M2Kp", "M2Kpp", "MAPK", "MAPKp"]
        for i in range(self.non_obs_state_dim):
            self.stoch_mod.ChangeInitialSpeciesCopyNumber(list_of_nonobs_species[i], int(non_obs_states[i]))
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("MAPKpp", int(obs_state))
        

    def sample_obs_state(self, n_points):
        
        mapkpp = np.random.randint(low=self.obs_state_bounds[0], high=self.obs_state_bounds[1], size = (n_points,))
            
        return mapkpp

    def sample_grid_obs_state(self, n_points):
        
        mapkpp = np.round(np.linspace(self.obs_state_bounds[0], self.obs_state_bounds[1], n_points))
            
        return mapkpp

    def sample_non_obs_states(self, n_points, obs_state):
        
        set_of_non_obs_states = np.empty((n_points, self.non_obs_state_dim))
        for i in range(n_points):
            m3k = np.random.randint(low=0, high=100)
            m2k = np.random.randint(low=0, high=300)
            m2kp = np.random.randint(low=0, high=int(300-m2k))
            mapk = np.random.randint(low=0, high=int(300-obs_state))
            mapkp = np.random.randint(low=0, high=int(300-mapk))
            
            set_of_non_obs_states[i] = np.array([m3k, 100-m3k, m2k, m2kp, 300-m2k-m2kp, mapk, mapkp])

        return set_of_non_obs_states

    def set_parameters(self, V1):
        self.stoch_mod.ChangeParameter("V1", V1)

    def sample_parameters_settings(self, n_points):
        
        set_of_params = (self.param_space_bounds[1] - self.param_space_bounds[0])*random(size=(n_points,))+self.param_space_bounds[0]

        return set_of_params

    def sample_grid_parameters_settings(self, n_points):
        
        set_of_params = np.round(np.linspace(self.param_space_bounds[0], self.param_space_bounds[1], n_points))
        
        return set_of_params

    def generate_training_set(self):
        Yp = np.zeros((self.n_training_points,self.param_space_dim))
        Y_obs = np.zeros((self.n_training_points,self.obs_state_dim))
        Y_non_obs = np.zeros((self.n_training_points,self.non_obs_state_dim))
        X = np.zeros((self.n_training_points, int(self.T/self.time_step), self.global_state_space_dim))
        X_obs = np.zeros((self.n_training_points, int(self.T/self.time_step), self.obs_state_dim))

        initial_obs_states = self.sample_obs_state(self.n_init_settings)
        set_of_params = self.sample_parameters_settings(self.n_init_settings)

        count, avg_time = 0, 0
        for i in tqdm(range(self.n_init_settings)):
            self.set_parameters(set_of_params[i])
            
            non_obs_states = self.sample_non_obs_states(self.n_trajs, initial_obs_states[i])
            
            for k in range(self.n_trajs):

                self.set_initial_states(initial_obs_states[i], non_obs_states[k,:])
            
                begin_time = time.time()
                self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
                if False:
	                self.stoch_mod.PlotSpeciesTimeSeries(species2plot =['MAPK', 'MAPKpp'])
	                stochpy.plt.savefig('NEW_V1={}_MAPK_{}{}.png'.format(set_of_params[i], i,k))
                ntraj_time = time.time()-begin_time

                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)
                avg_time += ntraj_time
                #datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
                datapoint = pd.read_csv(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).values

                new_datapoint = self.time_resampling(datapoint)
                X[count,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
                X_obs[count] = new_datapoint[1:,8:9]
                Y_obs[count,:] = initial_obs_states[i]
                Y_non_obs[count,:] = non_obs_states[k]
                Yp[count] = set_of_params[i]
                count += 1
        print("average time to gen 1 traj with SSA: ", avg_time/self.n_trajs)
        self.X = X
        self.X_obs = X_obs
        self.Y_obs = Y_obs
        self.Y_non_obs = Y_non_obs
        self.Y_par = Yp
        
    def generate_validation_set(self, n_val_points, n_val_trajs_per_point):
        Yp = np.zeros((n_val_points,self.param_space_dim))
        Y_obs = np.zeros((n_val_points, self.obs_state_dim))
        Y_non_obs = np.zeros((n_val_points, n_val_trajs_per_point, self.non_obs_state_dim))

        X = np.zeros((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.global_state_space_dim))
        X_obs = np.zeros((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.obs_state_dim))

        initial_obs_states = self.sample_obs_state(n_val_points)
        set_of_params = self.sample_parameters_settings(n_val_points)
            
        for ind in range(n_val_points):
            self.set_parameters(set_of_params[ind])
            non_obs_states = self.sample_non_obs_states(n_val_trajs_per_point, initial_obs_states[ind])
            
            Y_obs[ind,:] = initial_obs_states[ind]
            Yp[ind,:] = set_of_params[ind]
                
            for k in range(n_val_trajs_per_point):
                if k%100 == 0:
                    print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
                self.set_initial_states(initial_obs_states[ind], non_obs_states[k,:])
                
                self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                #datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
                datapoint = pd.read_csv(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).values

                new_datapoint = self.time_resampling(datapoint)
                X[ind,k,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
                X_obs[ind, k] = new_datapoint[1:,8:9]
                Y_non_obs[ind,k,:] = non_obs_states[k]
                
        self.X = X
        self.X_obs = X_obs
        self.Y_obs = Y_obs
        self.Y_non_obs = Y_non_obs
        self.Y_par = Yp
        

    def generate_grid_validation_set(self, grid_len, n_val_trajs_per_point):
        Yp = np.zeros((grid_len,self.param_space_dim))
        Y_obs = np.zeros((grid_len, self.obs_state_dim))
        Y_non_obs = np.zeros((grid_len**2, n_val_trajs_per_point, self.non_obs_state_dim))

        X = np.zeros((grid_len**2,  n_val_trajs_per_point,int(self.T/self.time_step), self.global_state_space_dim))
        X_obs = np.zeros((grid_len**2,  n_val_trajs_per_point,int(self.T/self.time_step), self.obs_state_dim))

        initial_obs_states = self.sample_grid_obs_state(grid_len)
        print(initial_obs_states)
        set_of_params = self.sample_grid_parameters_settings(grid_len)
        count = 0
        for i in range(grid_len):
            Yp[i,:] = set_of_params[i]
            for j in range(grid_len):
                self.set_parameters(set_of_params[i])
                non_obs_states = self.sample_non_obs_states(n_val_trajs_per_point, initial_obs_states[j])
                
                Y_obs[j,:] = initial_obs_states[j]
                
                    
                for k in range(n_val_trajs_per_point):
                    if k%100 == 0:
                        print(count, "/", grid_len**2, "------------------K iter: ", k, "/", n_val_trajs_per_point)
                    self.set_initial_states(initial_obs_states[j], non_obs_states[k,:])
                    
                    self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
                    self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                    #datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
                    datapoint = pd.read_csv(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).values

                    new_datapoint = self.time_resampling(datapoint)
                    X[count,k,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
                    X_obs[count,k,:,:] = new_datapoint[1:,8:9]
                    Y_non_obs[count,k,:] = non_obs_states[k]
                count += 1
        self.X = X
        self.X_obs = X_obs
        self.Y_obs = Y_obs
        self.Y_non_obs = Y_non_obs
        self.Y_par = Yp
        
    def save_dataset(self, filename):
        dataset_dict = {"X_full": self.X, "X": self.X_obs, "Y_s0": self.Y_obs, "Y_non_obs": self.Y_non_obs, "Y_par": self.Y_par}
        
        with open(filename, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    




def run_training(filename):
    n_init_settings = 2000
    n_trajs = 50

    obs_state_dim = 1
    non_obs_state_dim = 7

    time_step = 60
    n_steps = 32
    T = n_steps*time_step

    obs_state_bounds = np.array([10,290])
    param_space_bounds = np.array([0.1,2.5])
    
    mapk_dataset = PO_AbstractionDataset(n_init_settings, n_trajs, obs_state_bounds, param_space_bounds, 'MAPK', time_step, T, obs_state_dim, non_obs_state_dim)
    
    start_time = time.time()
    mapk_dataset.generate_training_set()
    print("Time to generate the training set w fixed param =", time.time()-start_time)

    mapk_dataset.save_dataset(filename)


def run_validation(filename):
    n_init_settings = 0
    n_trajs = 0

    obs_state_dim = 1
    non_obs_state_dim = 7

    time_step = 60
    n_steps = 32
    T = n_steps*time_step

    n_val_points = 25
    n_trajs_per_point = 2000

    obs_state_bounds = np.array([10,290])
    param_space_bounds = np.array([0.1,2.5])

    mapk_dataset = PO_AbstractionDataset(n_init_settings, n_trajs, obs_state_bounds, param_space_bounds, 'MAPK', time_step, T, obs_state_dim, non_obs_state_dim)
    

    start_time = time.time()
    mapk_dataset.generate_validation_set(n_val_points, n_trajs_per_point)
    print("Time to generate the validation set w fixed param =", time.time()-start_time)

    mapk_dataset.save_dataset(filename)

def run_grid_validation(filename):
    n_init_settings = 0
    n_trajs = 0

    obs_state_dim = 1
    non_obs_state_dim = 7

    time_step = 60
    n_steps = 32
    T = n_steps*time_step

    grid_len = 10
    n_trajs_per_point = 5000

    obs_state_bounds = np.array([10,290])
    param_space_bounds = np.array([0.1,2.5])

    mapk_dataset = PO_AbstractionDataset(n_init_settings, n_trajs, obs_state_bounds, param_space_bounds, 'MAPK', time_step, T, obs_state_dim, non_obs_state_dim)
    

    start_time = time.time()
    mapk_dataset.generate_grid_validation_set(grid_len, n_trajs_per_point)
    print("Time to generate the validation set w fixed param =", time.time()-start_time)

    mapk_dataset.save_dataset(filename)

run_training("../../data/MAPK/MAPK_PO_training_set.pickle")
run_validation("../../data/MAPK/MAPK_PO_validation_one_param.pickle")
run_grid_validation("../../data/MAPK/MAPK_PO_grid_validation_one_param.pickle")
