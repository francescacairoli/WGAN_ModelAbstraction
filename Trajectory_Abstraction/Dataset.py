import numpy as np
import math
import sys

import pickle

class Dataset(object):
    def __init__(self, trainset_fn, test_fn, x_dim, y_dim, traj_len):
        self.trainset_fn = trainset_fn
        self.test_fn = test_fn
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.traj_len = traj_len

    def add_grid_data(self, grid_fn):
        self.grid_test_fn = grid_fn

    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["X"][:,:self.traj_len,:]
        Y = data["Y_s0"]
        print("DATASET SHAPES: ", X.shape, Y.shape)

        self.HMAX = np.max(np.max(X, axis = 0),axis=0)

        self.HMIN = np.min(np.min(X, axis = 0),axis=0)

        self.X_train = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        Y = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        self.Y_train = np.expand_dims(Y,axis=2)
        self.n_points_dataset = self.X_train.shape[0]

        Xt = np.empty((self.n_points_dataset, self.x_dim, self.traj_len))
        Yt = np.empty((self.n_points_dataset, self.y_dim, 1))
        for j in range(self.n_points_dataset):
            Xt[j] = self.X_train[j].T
            Yt[j] = self.Y_train[j]
        self.X_train_transp = Xt
        self.Y_train_transp = Yt        
        
        
    def load_test_data(self):

        file = open(self.test_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["X"][:,:,:self.traj_len,:]
        Y = data["Y_s0"]        
        
        self.X_test = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        
        Y = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        
        self.n_points_test = self.X_test.shape[0]
        self.n_traj_per_point = self.X_test.shape[1]
        
        Xt = np.empty((self.n_points_test, self.n_traj_per_point, self.x_dim, self.traj_len))
        
        for j in range(self.n_points_test):
            for k in range(self.n_traj_per_point):
                Xt[j, k] = self.X_test[j, k].T

        self.X_test_transp = Xt
        self.Y_test_transp = np.expand_dims(Y,axis=2) 
        print("xxxxx", self.Y_test_transp.shape)
        

    def load_grid_test_data(self):

        file = open(self.grid_test_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["X"][:,:,:self.traj_len,:]
        Y = data["Y_s0"]        
        
        self.X_test_grid = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        
        Y = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        
        self.n_points_grid = self.X_test_grid.shape[0]
        self.n_traj_per_point_grid = self.X_test_grid.shape[1]
        
        Xt = np.empty((self.n_points_grid, self.n_traj_per_point_grid, self.x_dim, self.traj_len))
        
        for j in range(self.n_points_grid):
            for k in range(self.n_traj_per_point_grid):
                Xt[j, k] = self.X_test_grid[j, k].T

        self.X_test_grid_transp = Xt
        self.Y_test_grid_transp = np.expand_dims(Y,axis=2) 


    def generate_mini_batches(self, n_samples):
        
        ix = np.random.randint(0, self.X_train_transp.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]

        return Xb, Yb
