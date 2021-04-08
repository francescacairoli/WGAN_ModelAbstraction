import numpy as np
from numpy.random import randint, random
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm
import pickle


class AbstractionDataset(object):

	def __init__(self, n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, model_name, time_step, T, global_state_space_dim):
		# state_space_bounds : shape = (state_space_dim,2)
		# param_space_bounds : shape = (param_space_dim,2)
		self.n_init_states = n_init_states
		self.n_params = n_params
		self.n_trajs = n_trajs
		self.n_training_points = n_init_states*n_params*n_trajs
		self.state_space_bounds = state_space_bounds
		self.param_space_bounds = param_space_bounds
		self.global_state_space_dim = global_state_space_dim
		self.state_space_dim = state_space_bounds.shape[0]
		self.param_space_dim = param_space_bounds.shape[0]
		self.stoch_mod = stochpy.SSA(IsInteractive=False)
		self.stoch_mod.Model(model_name+'.psc')
		self.directory_name = model_name
		self.time_step = time_step
		self.T = T # end time
		self.N = None # population size

	def set_popul_size(self, N):
		self.N = N


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
		g1_off = int(init_state[0])
		g1_on = int(init_state[1])
		g2_off = int(init_state[2])
		g2_on = int(init_state[3])
		p1 = int(init_state[4])
		p2 = int(init_state[5])
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


	def sample_initial_states(self, n_points = None):
		if n_points == None:
			n_points = self.n_init_states
		set_of_init_states = np.ones((n_points,self.global_state_space_dim))
		for i in range(n_points):
			g1_on = randint(low = self.state_space_bounds[0,0], high = self.state_space_bounds[0,1])
			g2_on = randint(low = self.state_space_bounds[1,0], high = self.state_space_bounds[1,1])
			p1 = randint(low = self.state_space_bounds[2,0], high = self.state_space_bounds[2,1])
			p2 = randint(low = self.state_space_bounds[3,0], high = self.state_space_bounds[3,1])            
			set_of_init_states[i,:] = np.array([1-g1_on,g1_on,1-g2_on, g2_on, p1, p2])
	
		return set_of_init_states


	def sample_parameters_settings(self, n_points = None):
		if n_points == None:
			n_points = self.n_params
		set_of_params = np.zeros((n_points, self.param_space_dim))
		for i in range(self.param_space_dim):
			set_of_params[:,i] = (self.param_space_bounds[i,1] - self.param_space_bounds[i,0])*random(size=(n_points,))+self.param_space_bounds[i,0]

		return set_of_params


	def generate_training_set(self, fixed_param_flag, fixed_param):

		Yp = np.zeros((self.n_training_points,self.param_space_dim))
		Ys = np.zeros((self.n_training_points,self.global_state_space_dim))

		X = np.zeros((self.n_training_points, int(self.T/self.time_step), self.global_state_space_dim))


		#set_of_params = self.sample_parameters_settings()
		initial_states = self.sample_initial_states()

		count = 0
		for p in range(self.n_params):
			#self.set_parameters(set_of_params[p,:])
			self.set_parameters(fixed_param)
			
			for i in tqdm(range(self.n_init_states)):
				print("state ", i+1, "/", self.n_init_states)
				self.set_initial_states(initial_states[i,:])

				for k in range(self.n_trajs):
					self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
				
					self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

					datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
					
					new_datapoint = self.time_resampling(datapoint)
					X[count,:,:] = new_datapoint[1:,1:]
					#Yp[count,:] = set_of_params[p,:]
					Yp[count,:] = fixed_param
					Ys[count,:] = initial_states[i]
					
					count += 1

		self.X = X
		self.Y_par = Yp
		self.Y_s0 = Ys
		self.Y = np.hstack((Yp,Ys))

		if fixed_param_flag:
			return Yp[0]

	def generate_full_validation_set(self, n_val_points, n_val_trajs_per_point):

		Yp = np.zeros((n_val_points,self.param_space_dim))
		Ys = np.zeros((n_val_points,self.global_state_space_dim))

		X = np.zeros((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.global_state_space_dim))


		set_of_params = self.sample_parameters_settings(n_val_points)
		initial_states = self.sample_initial_states(n_val_points)
			
		for ind in range(n_val_points):
			self.set_parameters(set_of_params[ind,:])
			self.set_initial_states(initial_states[ind,:])

			Yp[ind,:] = set_of_params[ind]
			Ys[ind,:] = initial_states[ind]
				
			for k in range(n_val_trajs_per_point):
				if k%100 == 0:
					print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
				
				self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
				self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

				datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
				
				new_datapoint = self.time_resampling(datapoint)
				X[ind,k,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
			
		self.X = X
		self.Y_par = Yp
		self.Y_s0 = Ys
		self.Y = np.hstack((Yp,Ys))

	def generate_fixed_param_validation_set(self, n_val_points, n_val_trajs_per_point, fixed_param):

		initial_states = self.sample_initial_states(n_val_points)

		Yp = fixed_param*np.ones((n_val_points,self.param_space_dim))
		Ys = initial_states[:,:self.global_state_space_dim]

		X = np.empty((n_val_points,  n_val_trajs_per_point,int(self.T/self.time_step), self.global_state_space_dim))

		#set_of_params = self.sample_parameters_settings(n_val_points)
		initial_states = self.sample_initial_states(n_val_points)
		
		for ind in range(n_val_points):
			self.set_parameters(fixed_param)
			self.set_initial_states(initial_states[ind,:])

			  
			for k in range(n_val_trajs_per_point):
				if k%100 == 0:
					print(ind, "/", n_val_points, "------------------K iter: ", k, "/", n_val_trajs_per_point)
				
				self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.T)
				self.stoch_mod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory=self.directory_name, quiet=False)

				datapoint = pd.read_table(filepath_or_buffer=self.directory_name+'/'+self.directory_name+'.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).as_matrix()
				
				new_datapoint = self.time_resampling(datapoint)
				X[ind,k,:,:] = new_datapoint[1:,1:self.global_state_space_dim+1]
			
		self.X = X
		self.Y_par = Yp
		self.Y_s0 = Ys
		self.Y = np.hstack((Yp,Ys))


	def save_dataset(self, filename):
		dataset_dict = {"X": self.X, "Y": self.Y, "Y_par": self.Y_par, "Y_s0": self.Y_s0}
		with open(filename, 'wb') as handle:
			pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	


def run_train(fixed_param_flag, fixed_param):
	n_init_states = 1000
	n_params = 1
	n_trajs = 50

	state_space_dim = 4
	global_state_space_dim = state_space_dim + 2

	param_space_dim = 8

	time_step = 0.1
	n_steps = 32
	T = n_steps*time_step


	param_space_bounds = np.array([[50,150], [0.1,5], [0.0001,0.001], [0.1,1],[50,150], [0.1,5], [0.0001,0.001], [0.1,1]])
	state_space_bounds = np.array([[0,2], [0,2], [5,20], [5, 20]])

	gts_dataset = AbstractionDataset(n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, 'ToggleSwitch', time_step, T, global_state_space_dim)
	if fixed_param_flag:
		fixed_param = gts_dataset.generate_training_set(fixed_param_flag, fixed_param)
		gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_dataset_fixed_param_{}x{}_dt={}.pickle".format(n_init_states,n_trajs, time_step))
		return fixed_param
	else:
		gts_dataset.generate_training_set(fixed_param_flag)
		gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_training_set_full.pickle")


def run_validation_full():
	n_init_states = 0
	n_params = 0
	n_trajs = 0

	state_space_dim = 4
	global_state_space_dim = state_space_dim + 2

	param_space_dim = 8

	time_step = 0.1
	n_steps = 32#128
	T = n_steps*time_step

	n_val_points = 1#20
	n_trajs_per_point = 20000#2000

	param_space_bounds = np.array([[50,150], [0.1,5], [0.0001,0.001], [0.1,1],[50,150], [0.1,5], [0.0001,0.001], [0.1,1]])
	state_space_bounds = np.array([[0,2], [0,2], [5,20], [5, 20]])

	gts_dataset = AbstractionDataset(n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, 'ToggleSwitch', time_step, T, global_state_space_dim)

	gts_dataset.generate_full_validation_set(n_val_points, n_trajs_per_point)
	#gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_validation_set_full.pickle")

def run_validation_fixed_param(fixed_param):
	n_init_states = 0
	n_params = 0
	n_trajs = 0

	state_space_dim = 4
	global_state_space_dim = state_space_dim + 2

	param_space_dim = 8

	time_step = 0.1
	n_steps = 32
	T = n_steps*time_step

	n_val_points = 100
	n_trajs_per_point = 5000

	param_space_bounds = np.array([[50,150], [0.1,5], [0.0001,0.001], [0.1,1],[50,150], [0.1,5], [0.0001,0.001], [0.1,1]])
	state_space_bounds = np.array([[0,2], [0,2], [5,20], [5, 20]])

	gts_dataset = AbstractionDataset(n_init_states, n_params, n_trajs, state_space_bounds, param_space_bounds, 'ToggleSwitch', time_step, T, global_state_space_dim)

	gts_dataset.generate_fixed_param_validation_set(n_val_points, n_trajs_per_point, fixed_param)
	gts_dataset.save_dataset("../../data/ToggleSwitch/ToggleSwitch_dataset_fixed_param_{}x{}_dt={}.pickle".format(n_val_points,n_trajs_per_point, time_step))



FIX_FLAG = True
if FIX_FLAG:
	fixed_param = np.array([1.00873131e+02, 1.82719256e+00, 6.86173329e-04, 1.83068190e-01, 1.11966802e+02, 9.17581745e-01, 6.59146786e-04, 4.26048009e-01])
	#fixed_param = np.array([100., 0.1, 10., 10., 100., 0.1, 10., 10.])
	#fixed_param = np.array([3., 5*10**(-2), 10**(-6), 3*10**(-4), 3., 5*10**(-2), 10**(-6), 3*10**(-4)]) #SLOW BINDING
	#fixed_param = 10*np.array([3., 5*10**(-2), 6*10**(-3), 3*10**(-3), 3., 5*10**(-2), 6*10**(-3), 3*10**(-3)]) #SLOW BINDING BIS
	fixed_param = run_train(fixed_param_flag = FIX_FLAG, fixed_param=fixed_param)
	print("fixed_param = ", fixed_param)
	print("TRAINING SET COMPLETED!")
	run_validation_fixed_param(fixed_param)
	print("TEST SET COMPLETED!")
else:
	run_train(fixed_param_flag = FIX_FLAG)
	run_validation_full()
'''
import time
start_time = time.time()
run_validation_full()
print("time TS=", time.time()-start_time)
'''