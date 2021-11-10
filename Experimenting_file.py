# -*- coding: utf-8 -*-
""" This file used to experiment and test behaviour of intitial and time dependent momentum 
matching, on different datasets/parameters.
"""


import numpy as np
from scipy.linalg import norm as norm
import matplotlib.pyplot as plt
import scipy.integrate
import matching
import toydatagen


#%% storage for optimised momenta
store_in = []
store_td = []

#%% generate data and plot templates using data functions imported from toydatagen file
data = toydatagen.squish_data(.3)

gen_pts = data[0] # initial template which we will deform
match_pts = data[1] # target template 

# visualise points and join with lines. 

plt.scatter(match_pts[:,0],match_pts[:,1],marker="d",s=100)
plt.scatter(gen_pts[:,0],gen_pts[:,1],marker="x",s=100)


gen_plot = np.vstack((gen_pts,gen_pts[0]))
match_plot = np.vstack((match_pts,match_pts[0]))
plt.plot(gen_plot[:,0],gen_plot[:,1],marker="x")
plt.plot(match_plot[:,0],match_plot[:,1],marker="d")

#%% set up and discretize initial momentum matching

x = matching.initial_momentum(gen_pts, match_pts) # intialize

x.sigma =.5 #smoothness param
N = 20 #number of int points for initial momentum solve
match_param = .1 #matching tolerance


x.discretize(N, match_param) #discretize
#%% optimization params
n_iters = 5000 #number of gd iters
step_size = .001 #gd step size
#%% optimize
x.optimize(n_iters, step_size) 

#%% plot deformed templates for initial momentum formulation
x.show()

#%%  store initial momenta

store_in.append(x.in_vel)

#%% set up and discretize td_momentum matching

y = matching.td_momentum(gen_pts, match_pts) #initialize

y.sigma = .1 #smoothness param
mesh_points= 100 # number of integration points
y.factor = 100 # matching tolerance


y.discretize(mesh_points) # discretize 
#%% optimization params
n_iters_1 = 5000 #number of gd iterates
step_size_1 = .1#step_size

y.optimize(n_iters_1, step_size_1)
#%% show deformed templates for time dependent momentum
y.show()

#%% store time dependent momenta

store_td.append(y.z_init)
















