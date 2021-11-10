# -*- coding: utf-8 -*-
""" This file contains the classes for certain variations of initial and time dependent momentum matching"""
 
from __future__ import division
import numpy as np
from scipy.linalg import norm as norm
import matplotlib.pyplot as plt
import scipy.integrate



""" In future perhaps can add a more general class with discretize, optimize and solve_fwd methods which the individual classes
implement for specific algorithms"""
#%%
class initial_momentum:
    
    #initialise
    def __init__(self,template,target):
        self.template = template
        self.target = target
        self.sigma = None
        self.match_param = None
        self.n = len(template)
        
        
    #kernel function
    def kernel(self,x,y):
        return  np.exp(-(norm(x-y)**2)/(2*self.sigma**2))*np.eye(2)
    
    
    #gradient of kernel function wrt first arg
    def kernel_gradient(self,x,y,k):
        return (-(x-y)[k]/self.sigma**2)*self.kernel(x,y)
   
    
   # double gradients
    def double_gradient(self,x,y,k,p):
        if k!=p:
            return (-(x-y)[k]/(self.sigma**2))*self.kernel_gradient(x,y,p)
        else: 
            return (-1/(self.sigma**2))*self.kernel(x,y) +  (-(x-y)[k]/(self.sigma**2))*self.kernel_gradient(x,y,p)
        
    # generate kernel matrix (to be vectorised)
    def S_func(self,q):
        S = np.zeros((self.n,2,self.n,2))
        for i in range(len(q)):
            for j in range(len(q)):
                for l in range(2):
                    for k in range(2):
                        S[i,k,j,l] = self.kernel(q[i],q[j])[l,k]    
        return S
    #generate derivative kernel matrix (to be vectorised)
    def S_der_func(self,q):
        S_der = np.zeros((2,len(q),2,len(q),2))
        for k in range(2):
            for j in range(len(q)):
                for l in range(2):
                    for m in range(len(q)):
                        for o in range(2):
                            S_der[k,j,l,m,o] = self.kernel_gradient(q[j], q[m], k)[l,o]
        return S_der
    # generate double derivative matrix (to be vectorised)
    def S_dd_func(self,q):
        
        S_dd = np.zeros((2,2,len(q),2,len(q),2))
        
        for k in range(2):
            for e in range(2):
                for j in range(len(q)):
                    for l in range(2):
                        for m in range(len(q)):
                            for o in range(2):
                                S_dd[k,e,j,l,m,o] = self.double_gradient(q[j], q[m], k, e)[l,o]
        return S_dd
    
    # landmark evolution dynamic
    def q_dot(self,q,p,S):
   
        qdot = S@(p.flatten())

        return qdot.reshape((len(q),2))
    
    # momentum evolution dynamic
    def p_dot(self,q,p,S_der):
   
        pdot = np.zeros((len(q),2))
        
        for k in range(2):
            pdot[:,k] = np.sum(p*((S_der[k]@p.flatten()).reshape((len(q),2))),axis=1)
        
        return -1*pdot

   
 # time evolution potentials of derivatives
    def dot_der(self,q,p,q_der,p_der,S,S_der,S_der_mat,S_dd):
    
        qdot_der = np.zeros((len(q),2,len(q),2))
        qdot_der_1 = np.zeros((len(q),2,len(q),2))
        pdot_der1 = np.zeros((len(q),2,len(q),2))
        qdot_der_2 = np.zeros((len(q),2,len(q),2))
        
        #no derviatives terms
        for r in range(len(q)):
            for s in range(2):
                qdot_der[:,:,r,s] = (S@(p_der[:,:,r,s].flatten())).reshape((len(q),2))
        
        #single derivative term 1
        for r in range(len(q)):
            for s in range(2):
                q_der1 = np.tile(q_der[:,:,r,s],[2,1,1])
                qdot_der_1[:,:,r,s] = np.sum(np.transpose(((S_der_mat)@p.flatten()).reshape((2,len(q),2)),[2,1,0])*q_der1,axis=2).T.reshape((len(q),2)) 
        
        qdot_der = qdot_der + qdot_der_1
        
        #single derivative term 2
        for r in range(len(q)):
            for s in range(2):
                p_1 = np.transpose(np.tile(p,[2,len(q),2,1,1]),[0,3,4,1,2])
                q_der1 = np.transpose(np.tile(q_der[:,:,r,s],[2,len(q),2,1,1]),[4,3,0,1,2])
                qdot_der_2[:,:,r,s] = np.sum(np.sum(np.sum(S_der*q_der1*p_1,axis=2),1),axis=0)
                    
        qdot_der = qdot_der + qdot_der_2
                   
            
        pdot_der = np.zeros((len(q),2,len(q),2))
        
        #single derivative terms
        for r in range(len(q)):
            for s in range(2):
                p_der1 = np.tile(p_der[:,:,r,s],[2,1,1])
                p_1 = np.tile(p,[2,1,1])
                pdot_der[:,:,r,s]  = np.sum((p_1*(((S_der_mat)@p_der[:,:,r,s].flatten()).reshape((2,len(q),2)))),axis=2).T.reshape((len(q),2))  + np.sum(p_der1*(((S_der_mat)@p.flatten()).reshape((2,len(q),2))),axis=2).T.reshape((len(q),2)) 
                                   
                                       
        #double derivative terms
        for r in range(len(q)):
            for s in range(2):
                q_der1 = np.transpose(np.tile(q_der[:,:,r,s],[2,2,1,1]),[0,3,2,1])
                q_der2 = np.transpose(np.tile(q_der[:,:,r,s],[2,2,self.n,2,1,1]),[0,5,4,1,2,3])
                p_1  = np.tile(p,[2,2,1,1])
                p_2 = np.tile(p,[2,2,self.n,2,1,1])
                p_3 = np.transpose(np.tile(p,[2,2,self.n,2,1,1]),[0,1,4,5,2,3])
        
                pdot_der1[:,:,r,s] =  np.sum(np.sum(np.sum(np.sum(q_der2*p_2*(S_dd)*p_3,axis=5),3),axis=2),axis=1).T.reshape((len(q),2)) + np.sum(np.sum(p_1*q_der1*((S_dd.reshape((2,2,2*self.n,2*self.n))@p.flatten()).reshape((2,2,self.n,2))),axis=3),axis=1).T.reshape((len(q),2))
        pdot_der = pdot_der + pdot_der1
                        
        return [qdot_der,-1*pdot_der]
    
    # dynamics function for all four ode to be solved with solve_ivp in initial momentum matching
    def syspot(self,t,s):
        n = self.n
        q = s[:2*n].reshape((n,2))
        p = s[2*n:4*n].reshape((n,2))
        
        q_der = s[4*n:((4*(n**2)) + (4*n))].reshape((n,2,n,2))
        p_der = s[((4*(n**2)) + (4*n)):((8*(n**2)) + (4*n) )].reshape((n,2,n,2))
        
        S = self.S_func(q).reshape((2*n,2*n))
        S_der =  self.S_der_func(q)
        S_der_mat = S_der.reshape((2,2*n,2*n))
        S_dd = self.S_dd_func(q)
        
        qdot = self.q_dot(q, p, S)
        pdot = self.p_dot(q, p, S_der_mat)
        res = self.dot_der(q, p, q_der, p_der, S, S_der, S_der_mat, S_dd)
        
        qdot_der = res[0]
        pdot_der = res[1]
        
        sysvec = np.concatenate((qdot.reshape((-1,1)),-1*pdot.reshape((-1,1)),qdot_der.reshape((-1,1)),-1*pdot_der.reshape((-1,1))))
        
        return sysvec.flatten()
        
        
    # discretizes the ODEs over the interval and also sets initial vaalues of key quantities
    def discretize(self,N,match_param):
        
        # set all appropriate parameters and t=0 values
        
        self.match_param = match_param
        
        T=1
        self.N = N
        self.h = T/(N-1)
        
        gen_pts = self.template
        match_pts = self.target
        
        n = len(self.template)
        
        self.S_or = np.zeros((n,2,n,2))

        for i in range(n):
            for j in range(n):
                for l in range(2):
                    for k in range(2):
                        self.S_or[i,l,j,k] = self.kernel(gen_pts[i],gen_pts[j])[l,k]    
        
        self.kern_mat = self.S_or.reshape((2*n,2*n))
        
        self.S_inv = np.linalg.inv(self.kern_mat).reshape((n,2,n,2))
        
        self.init_velocity = self.h*(match_pts-gen_pts)                 #initialise the velocity
        
        self.init_momentum = np.linalg.inv(self.kern_mat)@(self.init_velocity.flatten()) #initialise the momentum
        
        #initial derivatives
        self.ini_mom_der = np.eye((2*n)).reshape((n,2,n,2))
        self.init_velocity_der = np.zeros((n,2,n,2))
       
        self.q_mat = np.zeros((N,n,2))
        self.p_mat = np.zeros((N,n,2))
        self.p_der_mat = np.zeros((N,n,2,n,2))
        self.q_der_mat = np.zeros((N,n,2,n,2))
        
        self.q_mat[0] = gen_pts
        self.p_mat[0] = self.init_momentum.reshape((n,2))
        
        self.p_der_mat[0] = self.ini_mom_der
        self.q_der_mat[0] = self.init_velocity_der
        
        self.in_vel = self.init_velocity
        
        return None
        
    #objective function 
    def J(self,v):
    
        v_1 = v.reshape((-1,1))
        
        # energy term
        first  = .5*(v_1.T@self.kern_mat@v_1)
       
        # term involving result of solving ode
        second = .5*(1/self.match_param**2)*(norm((self.solve_fwd(self.template)[-1]-self.target).flatten())**2)
        
        return first + second
    
    #optimize for initial momentum with gradient descent using analytic gradients
    # at each step solve forward from current initial momentum using fw Euler
    def optimize(self,num_iters,step_size):
        
        self.step_size = step_size
        N = self.N
        M = num_iters
        gen_pts = self.template
        match_pts = self.target
        n = len(self.template)
        
        # gradient steps
        for j in range(M):
            
            self.q_mat[0] = gen_pts
            
            # solve landmarks and momenta forward with Euler 
            for i in range(N-1):
                S = self.S_func(self.q_mat[i]).reshape((2*n,2*n))
                S_der = self.S_der_func(self.q_mat[i])
                S_der_mat = S_der.reshape((2,2*n,2*n))
                S_dd = self.S_dd_func(self.q_mat[i])
            
                self.q_mat[i+1] = self.q_mat[i] + self.h*self.q_dot(self.q_mat[i],self.p_mat[i],S)
                self.p_mat[i+1] = self.p_mat[i] + self.h*self.p_dot(self.q_mat[i],self.p_mat[i],S_der_mat)
                
                der = self.dot_der(self.q_mat[i],self.p_mat[i],self.q_der_mat[i],self.p_der_mat[i],S,S_der,S_der_mat,S_dd) 
            
                
                self.q_der_mat[i+1] = self.q_der_mat[i] + self.h*der[0]
                self.p_der_mat[i+1] = self.p_der_mat[i] + self.h*der[1]
            
            # set up gradients of objective J using output of ode solver
            self.gradients = np.zeros((n,2))
            
            self.gradients = self.gradients + self.q_dot(gen_pts, self.p_mat[0],self.kern_mat)
                
            for r in range(n):
                for s in range(2):
                    
                    self.gradients[r,s] = self.gradients[r,s] + (1/self.match_param**2)*np.sum(((self.q_mat[-1] - match_pts)*self.q_der_mat[-1][:,:,r,s])) #(1/match_param**2)*((q_mat[-1][i,k] - match_pts[i,k]))*q_der_mat[-1][i,k,r,s]
            
           
            # gradient step update initial velocity
            self.in_vel = self.in_vel - self.step_size*self.gradients   
            
            # update initial momentum
            self.in_mom = np.linalg.inv(self.kern_mat)@(self.in_vel.flatten())
            self.p_mat[0] = self.in_mom.reshape((n,2))
            print("gradient norm: " + str(norm(self.gradients)) + " and distance " + str(norm(self.q_mat[-1].flatten()-match_pts.flatten()))+ " funcval "+str(self.J(self.in_vel)))
     
            
    # optimises the objective but solving state and momenta forward with solve_ivp
    def ode_solver_opt(self,num_iters,step_size):
        n = self.n
        gen_pts = self.template
        match_pts = self.target
        self.in_mom = np.linalg.inv(self.kern_mat)@(self.in_vel.flatten())
        
        for j in range(num_iters):
            
            in_sys = np.concatenate((gen_pts.reshape((-1,1)),self.in_mom.reshape((-1,1)),self.init_velocity_der.reshape((-1,1)),self.ini_mom_der.reshape((-1,1))))
            ode_solve = scipy.integrate.solve_ivp(fun=self.syspot, t_span=(0,1), y0=in_sys.flatten(),method='RK45')
            res_fin = ode_solve['y'][:,-1]
            q_fin = res_fin[:2*n].reshape((n,2))
            q_der_fin = res_fin[4*n:((4*(n**2)) + (4*n) )].reshape((n,2,n,2))
            v_1 = self.in_vel.flatten()
            first  = .5*(v_1.T@self.kern_mat@v_1)
            second = .5*(1/self.match_param**2)*(norm((q_fin-match_pts).flatten())**2)
            gradients = np.zeros((n,2))
            
            gradients = gradients + self.q_dot(gen_pts, self.in_mom,self.kern_mat)
            
             
            for r in range(n):
                for s in range(2):
                    
                    gradients[r,s] = gradients[r,s] + (1/self.match_param**2)*np.sum(((q_fin - match_pts)*q_der_fin[:,:,r,s])) #(1/match_param**2)*((q_mat[-1][i,k] - match_pts[i,k]))*q_der_mat[-1][i,k,r,s]
            
            self.in_vel = self.in_vel - step_size*gradients   
            self.in_mom = (np.linalg.inv(self.kern_mat)@(self.in_vel.flatten())).reshape((n,2))
            #p_mat[0] = in_mom.reshape((n,2))
            
           
            J = first + second
            print("gradient norm: " + str(norm(gradients)) + " and distance " + str(norm(q_fin-match_pts)) + " functionval "+str(J))
               
            # solve forward on landmark points (for objective)- to be fixed  
            self.q_fin = q_fin
    
    
    #function to solve forward on landmarks (to be generalised to any points) using current estimate of initial velocity
    def solve_fwd(self,x):
        M = self.n
        
        
        #initial derivatives
        ini_mom_der = np.eye((2*M)).reshape((M,2,M,2))
        init_velocity_der = np.zeros((M,2,M,2))
       
        q_mat = np.zeros((self.N,M,2))
        p_mat = np.zeros((self.N,M,2))
        p_der_mat = np.zeros((self.N,M,2,M,2))
        q_der_mat = np.zeros((self.N,M,2,M,2))
        
        
        init_momentum = np.linalg.inv(self.kern_mat)@(self.in_vel.flatten()) #initialise the momentum
        
        q_mat[0] = x
        p_mat[0] = init_momentum.reshape((M,2))
        
        
        
        p_der_mat[0] = ini_mom_der
        q_der_mat[0] = init_velocity_der
        
        
        for i in range(self.N-1):
            S = self.S_func(q_mat[i]).reshape((2*self.n,2*self.n))
            S_der = self.S_der_func(q_mat[i])
            S_der_mat = S_der.reshape((2,2*self.n,2*self.n))
            S_dd = self.S_dd_func(q_mat[i])
            
            q_mat[i+1] = q_mat[i] + self.h*self.q_dot(q_mat[i],p_mat[i],S)
            p_mat[i+1] = p_mat[i] + self.h*self.p_dot(q_mat[i],p_mat[i],S_der_mat)
                
            der = self.dot_der(q_mat[i],p_mat[i],q_der_mat[i],p_der_mat[i],S,S_der,S_der_mat,S_dd) 
            
                
            q_der_mat[i+1] = q_der_mat[i] + self.h*der[0]
            p_der_mat[i+1] = p_der_mat[i] + self.h*der[1]
        
        gradients = np.zeros((self.n,2))
        
        gradients = self.q_dot(self.template, p_mat[0],self.kern_mat)
        
        for r in range(self.n):
               for s in range(2):
                   
                   gradients[r,s] = gradients[r,s] + (1/self.match_param**2)*np.sum(((q_mat[-1] - self.target)*q_der_mat[-1][:,:,r,s]))
        
        return q_mat
     
        
    # utility function for gradient checking
    def grad_J(self):
        M = self.n
        
        #initial derivatives
        ini_mom_der = np.eye((2*M)).reshape((M,2,M,2))
        init_velocity_der = np.zeros((M,2,M,2))
       
        q_mat = np.zeros((self.N,M,2))
        p_mat = np.zeros((self.N,M,2))
        p_der_mat = np.zeros((self.N,M,2,M,2))
        q_der_mat = np.zeros((self.N,M,2,M,2))
        
        
        init_momentum = np.linalg.inv(self.kern_mat)@(self.in_vel.flatten()) #initialise the momentum
        
        q_mat[0] = self.template
        p_mat[0] = init_momentum.reshape((M,2))
        
        
        
        p_der_mat[0] = ini_mom_der
        q_der_mat[0] = init_velocity_der
        
        
        for i in range(self.N-1):
            S = self.S_func(q_mat[i]).reshape((2*self.n,2*self.n))
            S_der = self.S_der_func(q_mat[i])
            S_der_mat = S_der.reshape((2,2*self.n,2*self.n))
            S_dd = self.S_dd_func(q_mat[i])
            
            q_mat[i+1] = q_mat[i] + self.h*self.q_dot(q_mat[i],p_mat[i],S)
            p_mat[i+1] = p_mat[i] + self.h*self.p_dot(q_mat[i],p_mat[i],S_der_mat)
                
            der = self.dot_der(q_mat[i],p_mat[i],q_der_mat[i],p_der_mat[i],S,S_der,S_der_mat,S_dd) 
            
                
            q_der_mat[i+1] = q_der_mat[i] + self.h*der[0]
            p_der_mat[i+1] = p_der_mat[i] + self.h*der[1]
        
        gradients = np.zeros((self.n,2))
        
        gradients = self.q_dot(self.template, p_mat[0],self.kern_mat)
        
        for r in range(self.n):
               for s in range(2):
                   
                   gradients[r,s] = gradients[r,s] + (1/self.match_param**2)*np.sum(((q_mat[-1] - self.target)*q_der_mat[-1][:,:,r,s]))
        
        return gradients
        
    
    # show deformed template superimposed on initial and final    
    def show(self):
        q = self.q_mat[-1]
        
        gen_pts = self.template
        match_pts = self.target
        
        gen_plot = np.vstack((gen_pts,gen_pts[0]))
        match_plot = np.vstack((match_pts,match_pts[0]))
        res_plot = np.vstack((q,q[0]))
        
        plt.scatter(gen_pts[:,0],gen_pts[:,1],marker="x",s=100)
        plt.scatter(match_pts[:,0],match_pts[:,1],marker="d",s=100)
        plt.scatter(q[:,0],q[:,1],marker="x")
        
        plt.plot(gen_plot[:,0],gen_plot[:,1],marker='x')
        plt.plot(match_plot[:,0],match_plot[:,1],marker="d")
        plt.plot(res_plot[:,0],res_plot[:,1])
        

       

       
#%%
class td_momentum:
    
    #initialise
    def __init__(self,template,target):
        self.template = template
        self.target = target
        self.sigma = None
        self.lambd = self.template - self.target
        self.factor = None
        self.n = len(template)
    
    #kernel   
    def kernel(self,x,y):
        return np.exp(-(norm(x-y)**2)/(2*self.sigma**2) )
    
    #gradient
    def kernel_gradient(self,x,y):
        return (-(x-y)/(self.sigma**2))*self.kernel(x,y)
    
    # negative dicrepancy
    def final_disc(self,y):
        return -2*self.factor*(y-self.target)
    
    # landmark evolution potentials for matching
    def p_disc(self,i,y,*args):
        
        init = args[0][i].reshape((-1,2))
        y = y.reshape((-1,2))
        var = np.zeros((len(y),2))
        for i in range(self.n):
            for l in range(self.n):
                var[i] = var[i] + self.kernel(y[i],y[l])*init[l] #replace loops
        return var

    def p_disc_final(self,i,y,*args):
        yj = args[0][i].reshape((-1,2))
        z = args[1][i].reshape((-1,2))
        y = y.reshape((-1,2))
        var = np.zeros((len(y),2))
        for i in range(int(y.shape[0])):
            for l in range(self.n):
                var[i] = var[i] + self.kernel(y[i],yj[l])*z[l] #Creplace loops
        return var
    
    # potnetial for co-state evolution
    def etapot_disc(self,i,eta,*args):
        
        init = args[0][i].reshape((-1,2))
        eta = eta.reshape((-1,2))
        var = np.zeros((self.n,2))
        y = args[1][i].reshape((self.n,2))
        for k in range(self.n):
            for l in range(self.n):
                for i in range(2):
                    var[k] = var[k] - eta[l][i]*init[k][i]*self.kernel_gradient(y[k],y[l]) - init[l][i]*eta[k][i]*self.kernel_gradient(y[k],y[l]) + 2*init[l][i]*init[k][i]*self.kernel_gradient(y[k],y[l])    
                
        return var
    
    # discretize problem over time interval
    def discretize(self,N):
        
        self.N = N
        self.T = 1
        self.h = 1/N
        
        self.mesh = np.linspace(0,self.T,N)
        
        self.z_init = np.random.randn(N,self.n,2)
        self.z_init[N-1] = 0
        
        self.y_init = self.template
        
        self.y_store = np.zeros((N+1,self.n,2)) #store landmark trajectories
        self.eta_store = np.zeros((N,self.n,2)) 

        self.y_store[0] = self.y_init
        
        return None
    
    # calculate gradients by solving state and costate equations and take gradient steps
    def optimize(self,num_iters,step_size):
        
        self.step_size = step_size
        N = self.N
        M = num_iters
        
        # gradient descent steps
        for l in range(M):
            
            #solve forward for landmark states
            for i in range(self.N):
                self.y_store[i+1] = self.y_store[i] + self.h*self.p_disc(i,self.y_store[i],self.z_init) #unchanged for trap rule
            
            
            
            eta_1 = self.final_disc(self.y_store[N]) 
            # store eta trajectories
            self.eta_store[N-1] = eta_1.copy()
        
            #solve backwards for eta costates
            for i in range(N-1):
             
                self.eta_store[N-1-(i+1)] = self.eta_store[N-1-i] + self.h*self.etapot_disc(N-1-i,self.eta_store[N-1-i],self.z_init,self.y_store)
            
            # calculate gradients
            self.grads = self.h*self.eta_store - self.h*self.z_init
            
            # gradient step on initial momentum vector
            self.z_init = self.z_init + self.step_size*self.grads
            
            # print progress
            print("distance "+str(norm(self.final_disc(self.y_store[N])/(2*self.factor))**2),"gradient norm "+str(np.average(self.h*norm(norm(self.grads,axis=2),axis=0)**2)),"iteration number ",l)
            
           
            
    # solve forward for general sets of landmarks       
    def solve_fwd(self,x):
        M = x.shape[0] #x is M by 2
        xstore = np.zeros((self.N+1,M,2));
        
        xstore[0,:,0] = x[:,0]
        xstore[0,:,1] = x[:,1]
        for i in range(self.N):
            xstore[i+1,:,:] = xstore[i,:,:] + self.h*self.p_disc_final(i,xstore[i,:,:],self.y_store,self.z_init ) 
        return xstore
    
    # plotter for showing experiment results
    def show(self):
        gen_pts = self.template
        match_pts = self.target
        
        gen_plot = np.vstack((gen_pts,gen_pts[0]))
        match_plot = np.vstack((match_pts,match_pts[0]))
        
        result_td = self.solve_fwd(gen_pts)[-1]
        result_plot = np.vstack((result_td,result_td[0]))
        
        fig = plt.figure(1)
        plt.scatter(gen_plot[:,0],gen_plot[:,1],marker="x")
        plt.scatter(match_plot[:,0],match_plot[:,1],marker="d")
        plt.scatter(result_plot[:,0],result_plot[:,1],marker="o")
        
        plt.plot(gen_plot[:,0],gen_plot[:,1],marker="x")
        plt.plot(match_plot[:,0],match_plot[:,1],marker="d")
    
        plt.plot(result_plot[:,0],result_plot[:,1],marker="o")
        
        return fig

    
    