# -*- coding: utf-8 -*-
"""
functions for creating toy data: circles, ellipses, banana and random points.

"""

import numpy as np

#%% 

"""currently implement
matching with known correpsondences, so must be careful with noise variance parameter"""

#function for generating circular points
def circle_points(r, n):
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

# generating circular data radii r1,r2,number of points n and noise variance
def circular_data(r1,r2,n,noise):
    
    gen_pts, match_pts = circle_points([r1,r2], [n,n])
    #some noise to target
    match_pts = match_pts + np.random.normal(loc=0,scale=noise,size=(n,2)  )
    return [gen_pts,match_pts]

# data to generate elliptical deformation. n number of points, mutiplier is stretch factor, noise is noise variance
def elliptical_data(n,multiplier,noise):
    
    gen_pts, match_pts = circle_points([1,1], [n,n])
    
    match_pts[:,0] = match_pts[:,0]/multiplier
    match_pts[:,1] = match_pts[:,1]*multiplier
    
    #add some noise to target
    match_pts = match_pts + np.random.normal(loc=0,scale=noise,size=(n,2)  )
    
    return [gen_pts,match_pts]

# function to generate some landmarks randomly
def random_landmark_data(n,m1,v1,v2):
    gen_pts = np.random.multivariate_normal(mean=m1,cov=v1*np.array([1,0,0,1]).reshape((2,2)),size=(n,))
    match_pts = gen_pts + np.random.multivariate_normal(mean=[0,0],cov=v2*np.array([1,0,0,1]).reshape((2,2)),size=(n,))

    return [gen_pts,match_pts]

# squish templates (banana interpolation), takes a rate parameter between 0 and 0.4 for squish rate
def squish_data(rate):
    gen_pts = circle_points([1,1], [10,10])[0]
    
    gen_pts[:,0] = gen_pts[:,0]/2.0
    gen_pts[:,1] = gen_pts[:,1]*2.0
    
    match_pts = gen_pts.copy()
    
    match_pts[1] = match_pts[1] - (rate +.05)
    match_pts[0] = match_pts[0] - (rate + .2)
    match_pts[-1] = match_pts[-1] - (rate+.05)
    
    return [gen_pts,match_pts]