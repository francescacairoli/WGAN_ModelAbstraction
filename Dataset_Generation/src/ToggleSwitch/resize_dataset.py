import numpy as np
from numpy.random import randint, random
import os
import shutil
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

TRAINING = False
if TRAINING:
	filename = "../../data/ToggleSwitch/ToggleSwitch_PO_dataset_fixed_param_1000x50_dt=0.1.pickle"
	with open(filename, 'rb') as handle:
	    dataset_dict = pickle.load(handle)

	id_x = [1,3]
	id_s0 = [4, 5]
	print("x-----", dataset_dict["X"][0])
	print("s0----", dataset_dict["Y_obs"][0])
	
	T = dataset_dict["Y_obs"]
	X = dataset_dict["X"][:, :, id_x]

	indip_filename = "../../data/ToggleSwitch/ToggleSwitch_PO_dataset_fixed_param_1000x50_dt=0.1_red.pickle"

	ord_dataset_dict = {"X": X, "Y_s0": T}
	with open(indip_filename, 'wb') as handle:
		pickle.dump(ord_dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
	filename = "../../data/ToggleSwitch/ToggleSwitch_PO_grid_dataset_fixed_param_100x5000_dt=0.1.pickle"
	with open(filename, 'rb') as handle:
		dataset_dict = pickle.load(handle)

	id_x = [1,3]
	id_s0 = [4, 5]
	
	print("x-----", dataset_dict["X"][0])
	#print("s0----", dataset_dict["Y_s0"][0])
	
	T = dataset_dict["Y_obs"]
	X = dataset_dict["X"][:, :, :, id_x]
	
	#print("par-----", dataset_dict["Y_par"][0])

	indip_filename = "../../data/ToggleSwitch/ToggleSwitch_PO_grid_dataset_fixed_param_100x5000.pickle"

	ord_dataset_dict = {"X": X, "Y_s0": T}
	with open(indip_filename, 'wb') as handle:
		pickle.dump(ord_dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
_, p = Y_par.shape

timeline = np.linspace(0,n_steps,n_steps)

fig1, ax1 = plt.subplots(1, 2, figsize=(12, 8))
for ggg in range(0,20):
	ax1[0].plot(timeline, X_ord[ggg,:,2], "r")
	ax1[1].plot(timeline, X_ord[ggg,:,3], "b")
	
ax1[0].set_xlabel("time")	
ax1[0].set_title("P1")
ax1[1].set_xlabel("time")
ax1[1].set_title("P2")
plt.tight_layout()
string_name_1 = "Plots/proteins_bistability.png"
fig1.savefig(string_name_1)

for ttt in range(3):
	fig, ax = plt.subplots(1, 2, figsize=(12, 8))
	ax[0].plot(timeline, X_ord[ttt,:,0], "r", label = "G1")
	ax[0].plot(timeline, X_ord[ttt,:,1], "b", label = "G2")
	ax[0].legend()
	ax[0].set_xlabel("time")
	ax[0].set_ylabel("genes")
	ax[0].set_title("Genes")
	ax[1].plot(timeline, X_ord[ttt,:,2], "r", label = "P1")
	ax[1].plot(timeline, X_ord[ttt,:,3], "b", label = "P2")
	ax[1].set_xlabel("time")
	ax[1].set_ylabel("proteins")
	ax[0].set_title("Proteins")
	ax[1].legend()
	plt.tight_layout()
	string_name = "Plots/genes_and_proteins_{}.png".format(ttt)
	fig.savefig(string_name)
'''