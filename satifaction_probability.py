import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries

loc = "WGAN_for_Trajectories/"
ID = "17398"
compute_flag = True

MODEL_NAME = "SIR_fixed_s0"
RESULTS_PATH = MODEL_NAME + "/Results/ID_" + ID
traj_filename = loc+RESULTS_PATH+'/validation_trajectories_GAN_vs_SSA.pickle'

file = open(traj_filename, 'rb')
data = pickle.load(file)
file.close()

gen_traj_scaled = data["gen"]
ssa_traj_scaled = data["ssa"]
HMAX = 50.0
gen_traj = np.round((gen_traj_scaled+1)*HMAX)
ssa_traj = np.round((ssa_traj_scaled+1)*HMAX)


valset_file = "_validation_set_dt=5_large.pickle"
valset_filename = open('Dataset_Generation/data/'+MODEL_NAME+'/'+MODEL_NAME+valset_file, 'rb')
val_data = pickle.load(valset_filename)
valset_filename.close()

par_val = val_data["Y_par"]

n_val_points, n_trajs, traj_len, n_species = gen_traj.shape

formula = '(I > 0) U_[100,150] (I <= 0)'
#formula = '( F_[0.0,7.5] (I > 0) )'
variables = ['I']


if compute_flag:
	gen_phi = np.empty((n_val_points, n_trajs))
	ssa_phi = np.empty((n_val_points, n_trajs))
	timeline = np.arange(0,150,5)
	print("timeline = ",timeline)
	for i in range(n_val_points):
		print(i+1,"/",n_val_points)
		for j in range(n_trajs):
			serie_gen = TimeSeries(variables, timeline, gen_traj[i,j,:,1].reshape((1,traj_len)))

			serie_ssa = TimeSeries(variables, timeline, ssa_traj[i,j,:,1].reshape((1,traj_len)))    
			
			G = stlBooleanSemantics(serie_gen, 0, formula)
			S = stlBooleanSemantics(serie_ssa, 0, formula)
			if G or S:
				print(G,S)
				print(gen_traj[i,j,:,1])
				print(ssa_traj[i,j,:,1])
			gen_phi[i,j] = G
			ssa_phi[i,j] = S

	gen_phi_bool = copy.deepcopy(gen_phi)
	ssa_phi_bool = copy.deepcopy(ssa_phi)

	gen_phi[(gen_phi==True)] = 1
	gen_phi[(gen_phi==False)] = 0

	ssa_phi[(ssa_phi==True)] = 1
	ssa_phi[(ssa_phi==False)] = 0

	gen_sat = np.mean(gen_phi, axis=1)
	ssa_sat = np.mean(ssa_phi, axis=1)

	sat_dict = {"gen_phi": gen_phi, "ssa_phi": ssa_phi, "gen_sat": gen_sat, "ssa_sat": ssa_sat}
	sat_file = open('Satisfaction_Probs/GEN_vs_SSA_satisfaction_prob_ID_'+ID+'.pickle', 'wb')
	pickle.dump(sat_dict, sat_file)
	sat_file.close()
else:
	sat_file = open('Satisfaction_Probs/GEN_vs_SSA_satisfaction_prob_ID_'+ID+'.pickle', 'rb')
	sat_dict = pickle.load(sat_file)
	sat_file.close()
	gen_phi = sat_dict["gen_phi"]
	ssa_phi = sat_dict["ssa_phi"] 
	gen_sat = sat_dict["gen_sat"]
	ssa_sat = sat_dict["ssa_sat"]

print("gen_sat: ", np.mean(gen_sat))
print("ssa_sat: ", np.mean(ssa_sat))

'''
gen_sat[(gen_sat>0.1)] = 1
gen_sat[(gen_sat<=0.1)] = 0

ssa_sat[(ssa_sat>0.1)] = 1
ssa_sat[(ssa_sat<=0.1)] = 0
'''

fig,ax = plt.subplots(1,2, figsize = (18,9))
img0 = ax[0].scatter(par_val[:,0], par_val[:,1], c = ssa_sat)
ax[0].set_title("ssa")
img0.set_clim(0,1)
cbar0 = fig.colorbar(img0, ax=ax[0])
ax[0].set_xlim(0.00005,0.003)
ax[0].set_ylim(0.005,0.2)

img1 = ax[1].scatter(par_val[:,0], par_val[:,1], c = gen_sat)
ax[1].set_title("gen")
img1.set_clim(0,1)
cbar1 = fig.colorbar(img1, ax=ax[1])
ax[1].set_xlim(0.00005,0.003)
ax[1].set_ylim(0.005,0.2)

#fig.tight_layout()
fig.savefig("Satisfaction_Probs/"+str(n_val_points)+"_valid_points_ID_"+ID+".png")