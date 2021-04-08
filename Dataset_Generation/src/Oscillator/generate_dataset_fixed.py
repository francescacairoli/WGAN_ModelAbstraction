import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle


class AbstractionDataset(object):

    def __init__(self, n_init_states, n_trajs, state_space_bounds, model_name, time_step, T):

        self.n_init_states = n_init_states
        self.n_trajs = n_trajs
        self.n_training_points = n_init_states*n_trajs
        self.state_space_bounds = state_space_bounds
        self.state_space_dim = state_space_bounds.shape[0]      
        self.stoch_mod = stochpy.SSA(IsInteractive=False)
        self.stoch_mod.Model(model_name+'.psc')
        self.directory_name = model_name
        self.time_step = time_step
        self.T = T # end time
        self.conc_flag = False



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
        A = int(init_state[0])
        B = int(init_state[1])
        C = int(init_state[2])
        
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("A", A)
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("B", B)
        self.stoch_mod.ChangeInitialSpeciesCopyNumber("C", C)



    def sample_initial_states(self, n_points=None):
        if n_points == None:
            n_points = self.n_init_states
        set_of_init_states = np.ones((n_points,self.state_space_dim))
        for i in range(n_points):
            
            if self.conc_flag:
                a = random()
                b = random()
                c = random()
            else:
                a = randint(low = self.state_space_bounds[0,0], high = self.state_space_bounds[0,1])
                b = randint(low = self.state_space_bounds[1,0], high = self.state_space_bounds[1,1])
                c = randint(low = self.state_space_bounds[2,0], high = self.state_space_bounds[2,1])
            set_of_init_states[i,:] = np.array([a, b, c])
    
        return set_of_init_states




    def generate_training_set(self):

        Ys = np.zeros((self.n_training_points,self.state_space_dim))

        X = np.zeros((self.n_training_points, int(self.T/self.time_step), self.state_space_dim))

        initial_states = self.sample_initial_states()
            
        count = 0
        for i in tqdm(range(self.n_init_states)): 
            self.set_initial_states(initial_states[i,:])

            for k in range(self.n_trajs):
                self.stoch_mod.DoStochSim(method="Direct", trajectories=self.n_trajs, mode="time", end=self.T)
            
                self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

                datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
                
                new_datapoint = self.time_resampling(datapoint)
                X[count,:,:] = new_datapoint[1:,1:]
                Ys[count,:] = initial_states[i]
                
                count += 1

        self.X = X
        self.Y_s0 = Ys


    def generate_fixed_param_validation_set(self, n_val_points, n_val_trajs_per_point):

        initial_states = self.sample_initial_states(n_val_points)

        Ys = initial_states[:,:self.state_space_dim]

        X = np.empty((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.state_space_dim))

        
        for ind in range(n_val_points):
            self.set_initial_states(initial_states[ind,:])

              
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


    def save_dataset(self, filename):
        dataset_dict = {"X": self.X, "Y_s0": self.Y_s0}
        with open(filename, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



def run_train():
    n_init_states = 2000
    n_trajs = 10

    state_space_dim = 3

    time_step = 1
    n_steps = 32
    T = n_steps*time_step
    
    state_space_bounds = np.array([[20,100], [20,100], [20,100]])

    clock_dataset = AbstractionDataset(n_init_states, n_trajs, state_space_bounds, 'Clock', time_step, T)

    clock_dataset.generate_training_set()
    clock_dataset.save_dataset("../../data/Clock/Clock_training_set_dt=1_32steps.pickle")



def run_validation_fixed_param():
    n_init_states = 0
    n_trajs = 0

    state_space_dim = 3

    time_step = 1
    n_steps = 32
    T = n_steps*time_step

    n_val_points = 100
    n_trajs_per_point = 5000
    
    state_space_bounds = np.array([[20,100], [20,100], [20,100]])

    clock_dataset = AbstractionDataset(n_init_states, n_trajs, state_space_bounds, 'Clock', time_step, T)

    clock_dataset.generate_fixed_param_validation_set(n_val_points, n_trajs_per_point)
    clock_dataset.save_dataset("../../data/Oscillator/Oscillator_validation_set_large.pickle")




#run_train()
import time
start_time = time.time()
run_validation_fixed_param()
print("time CLOCK=", time.time()-start_time)
        