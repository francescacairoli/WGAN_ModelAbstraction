import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle


class PO_AbstractionDataset(object):
	
	def __init__(self, n_init_obs_states, n_trajs, obs_states_bounds, non_obs_states_bounds, model_name, time_step, T, obs_state_dim, non_obs_state_dim,):
		self.n_init_obs_states = n_init_obs_states
		self.n_trajs = n_trajs
		self.n_training_points = n_init_obs_states*n_trajs
		self.obs_states_bounds = obs_states_bounds
		self.non_obs_states_bounds = non_obs_states_bounds
		self.obs_state_dim = obs_state_dim
		self.non_obs_state_dim = non_obs_state_dim
		self.global_state_space_dim = obs_state_dim+non_obs_state_dim
		self.stoch_mod = stochpy.SSA(IsInteractive=False)
		self.stoch_mod.Model(model_name+'.psc')
		self.directory_name = model_name
		self.time_step = time_step
		self.T = T # end time
		

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


	def set_initial_states(self, obs_states, non_obs_states):
		g1_off = int(non_obs_states[0])
		g1_on = int(non_obs_states[1])
		g2_off = int(non_obs_states[2])
		g2_on = int(non_obs_states[3])
		p1 = int(obs_states[0])
		p2 = int(obs_states[1])
		self.stoch_mod.ChangeInitialSpeciesCopyNumber("G1_on", g1_on)
		self.stoch_mod.ChangeInitialSpeciesCopyNumber("G1_off", g1_off)
		self.stoch_mod.ChangeInitialSpeciesCopyNumber("P1", p1)
		self.stoch_mod.ChangeInitialSpeciesCopyNumber("G2_on", g2_on)
		self.stoch_mod.ChangeInitialSpeciesCopyNumber("G2_off", g2_off)
		self.stoch_mod.ChangeInitialSpeciesCopyNumber("P2", p2)

	def set_parameters(self, params):
		self.stoch_mod.ChangeParameter("Kp1", params[0])
		self.stoch_mod.ChangeParameter("Kd1", params[1])
		self.stoch_mod.ChangeParameter("Kb1", params[2])
		self.stoch_mod.ChangeParameter("Ku1", params[3])
		self.stoch_mod.ChangeParameter("Kp2", params[4])
		self.stoch_mod.ChangeParameter("Kd2", params[5])    
		self.stoch_mod.ChangeParameter("Kb2", params[6])
		self.stoch_mod.ChangeParameter("Ku2", params[7])


	def sample_obs_states(self, n_points):
		
		set_of_obs_states = np.ones((n_points,self.obs_state_dim))
		for i in range(n_points):
			p1 = randint(low = self.obs_states_bounds[0,0], high = self.obs_states_bounds[0,1])
			p2 = randint(low = self.obs_states_bounds[1,0], high = self.obs_states_bounds[1,1])            
			set_of_obs_states[i,:] = np.array([p1, p2])
	
		return set_of_obs_states

	def sample_grid_obs_states(self, n_points):
		
		p1 = np.round(np.linspace(self.obs_states_bounds[0,0], self.obs_states_bounds[0,1], n_points))
		p2 = np.round(np.linspace(self.obs_states_bounds[1,0], self.obs_states_bounds[1,1], n_points))           
			
		return p1,p2

	def sample_non_obs_states(self, n_points):

		set_of_non_obs_states = np.ones((n_points,self.non_obs_state_dim))
		for i in range(n_points):
			g1_on = randint(low = self.non_obs_states_bounds[0,0], high = self.non_obs_states_bounds[0,1])
			g2_on = randint(low = self.non_obs_states_bounds[1,0], high = self.non_obs_states_bounds[1,1])
			set_of_non_obs_states[i,:] = np.array([1-g1_on,g1_on,1-g2_on, g2_on])
	
		return set_of_non_obs_states


	def generate_training_set(self, fixed_param):

		Y_obs = np.zeros((self.n_training_points,self.obs_state_dim))
		Y_non_obs = np.zeros((self.n_training_points,self.non_obs_state_dim))

		X = np.zeros((self.n_training_points, int(self.T/self.time_step), self.global_state_space_dim))

		obs_states = self.sample_obs_states(self.n_init_obs_states)

		count = 0

		self.set_parameters(fixed_param)
		
		for i in tqdm(range(self.n_init_obs_states)):
			print("state ", i+1, "/", self.n_init_obs_states)

			non_obs_states = self.sample_non_obs_states(self.n_trajs)
			
			for k in range(self.n_trajs):

				self.set_initial_states(obs_states[i,:], non_obs_states[k,:])

				self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
			
				self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

				datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
				
				new_datapoint = self.time_resampling(datapoint)
				X[count,:,:] = new_datapoint[1:,1:]
				Y_obs[count,:] = obs_states[i]
				Y_non_obs[count,:] = non_obs_states[k]
				
				count += 1

		self.X = X
		self.Y_obs = Y_obs
		self.Y_non_obs = Y_non_obs
		


	def generate_fixed_param_validation_set(self, n_val_points, n_val_trajs_per_point, fixed_param):

		X = np.empty((n_val_points,  n_val_trajs_per_point, int(self.T/self.time_step), self.global_state_space_dim))
		Y_obs = np.zeros((n_val_points, self.obs_state_dim))
		Y_non_obs = np.zeros((n_val_points, n_val_trajs_per_point, self.non_obs_state_dim))

		obs_states = self.sample_obs_states(n_val_points) # initial_obs_states
	
		self.set_parameters(fixed_param)

		for ind in range(n_val_points):
			
			non_obs_states = self.sample_non_obs_states(n_val_trajs_per_point)
			  
			for k in range(n_val_trajs_per_point):
				if k%100 == 0:
					print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
				
				self.set_initial_states(obs_states[ind,:], non_obs_states[k,:])

				self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
				self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

				datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
				
				new_datapoint = self.time_resampling(datapoint)
				X[ind,k,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
				Y_obs[ind,:] = obs_states[ind]
				Y_non_obs[ind,k,:] = non_obs_states[k]
				

		self.X = X
		self.Y_obs = Y_obs
		self.Y_non_obs = Y_non_obs
		
	def generate_grid_validation_set(self, len_grid, n_val_trajs_per_point, fixed_param):

		X = np.empty((len_grid**2,  n_val_trajs_per_point, int(self.T/self.time_step), self.global_state_space_dim))
		Y_obs = np.zeros((len_grid**2, self.obs_state_dim))
		Y_non_obs = np.zeros((len_grid**2, n_val_trajs_per_point, self.non_obs_state_dim))

		p1v, p2v = self.sample_grid_obs_states(len_grid) # initial_obs_states
	
		self.set_parameters(fixed_param)
		count = 0
		for i in range(len_grid):
			for j in range(len_grid):
			
				non_obs_states = self.sample_non_obs_states(n_val_trajs_per_point)
				  
				for k in range(n_val_trajs_per_point):
					if k%100 == 0:
						print(i+1, "/", len_grid, "------------------K iter: ", k, "/", n_val_trajs_per_point)
					obs_states = np.array([p1v[i], p2v[j]])
					self.set_initial_states(obs_states, non_obs_states[k,:])

					self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
					self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

					datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
					
					new_datapoint = self.time_resampling(datapoint)
					X[count,k,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
					Y_obs[count,:] = obs_states
					Y_non_obs[count,k,:] = non_obs_states[k]
				count += 1

		self.X = X
		self.Y_obs = Y_obs
		self.Y_non_obs = Y_non_obs
		

	def save_dataset(self, filename):
		dataset_dict = {"X": self.X, "Y_obs": self.Y_obs, "Y_non_obs": self.Y_non_obs}
		with open(filename, 'wb') as handle:
			pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	


def run_train(fixed_param):
	n_init_obs_states = 1000
	n_trajs = 50

	obs_state_dim = 2
	non_obs_state_dim = 4


	time_step = 0.1
	n_steps = 32
	T = n_steps*time_step

	obs_states_bounds = np.array([[5,20], [5, 20]])
	non_obs_states_bounds = np.run_validation_fixed_param(fixed_param)
#print("TEST SET COMPLETED!")array([[0,2], [0,2]])

	gts_dataset = PO_AbstractionDataset(n_init_obs_states, n_trajs, obs_states_bounds, non_obs_states_bounds, 'ToggleSwitch', time_step, T, obs_state_dim, non_obs_state_dim)
	
	gts_dataset.generate_training_set(fixed_param)
	gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_PO_dataset_fixed_param_{}x{}_dt={}.pickle".format(n_init_obs_states,n_trajs, time_step))


def run_validation_fixed_param(fixed_param):
	n_init_obs_states = 0
	n_trajs = 0

	obs_state_dim = 2
	non_obs_state_dim = 4

	time_step = 0.1
	n_steps = 32
	T = n_steps*time_step

	n_val_points = 25#100
	n_trajs_per_point = 2000#5000

	obs_states_bounds = np.array([[5,20], [5, 20]])
	non_obs_states_bounds = np.array([[0,2], [0,2]])

	gts_dataset = PO_AbstractionDataset(n_init_obs_states, n_trajs, obs_states_bounds, non_obs_states_bounds, 'ToggleSwitch', time_step, T, obs_state_dim, non_obs_state_dim)
	
	gts_dataset.generate_fixed_param_validation_set(n_val_points, n_trajs_per_point, fixed_param)
	gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_PO_dataset_fixed_param_{}x{}_dt={}.pickle".format(n_val_points,n_trajs_per_point, time_step))

def run_grid_validation_fixed_param(fixed_param):
	n_init_obs_states = 0
	n_trajs = 0

	obs_state_dim = 2
	non_obs_state_dim = 4

	time_step = 0.1
	n_steps = 32
	T = n_steps*time_step

	grid_len = 10
	n_trajs_per_point = 5000

	obs_states_bounds = np.array([[5,20], [5, 20]])
	non_obs_states_bounds = np.array([[0,2], [0,2]])

	gts_dataset = PO_AbstractionDataset(n_init_obs_states, n_trajs, obs_states_bounds, non_obs_states_bounds, 'ToggleSwitch', time_step, T, obs_state_dim, non_obs_state_dim)
	
	gts_dataset.generate_grid_validation_set(grid_len, n_trajs_per_point, fixed_param)
	gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_PO_grid_dataset_fixed_param_{}x{}_dt={}.pickle".format(grid_len**2,n_trajs_per_point, time_step))



fixed_param = np.array([1.00873131e+02, 1.82719256e+00, 6.86173329e-04, 1.83068190e-01, 1.11966802e+02, 9.17581745e-01, 6.59146786e-04, 4.26048009e-01])
#run_train(fixed_param=fixed_param)
#print("TRAINING SET COMPLETED!")
#run_validation_fixed_param(fixed_param)
#print("TEST SET COMPLETED!")

run_grid_validation_fixed_param(fixed_param)
print("TEST SET COMPLETED!")
