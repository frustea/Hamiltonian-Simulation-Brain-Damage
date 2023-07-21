import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

class normal_modes:
    # init method or constructor   
    def __init__(self, num_ions, omega_a,omega_r,mass,l_fac,prefac):  
        self.num_ions = num_ions
        self.omega_a = omega_a  #axial freq
        self.omega_r = omega_r #radial freq
        self.l_fac = l_fac # np.pow(z^2 e^2/(4 pi epsilon_0 M omega_a^2),1/3)
        self.prefac=prefac  
        self.mass=mass
        
    def potential(self,x):
        harmonic_term = np.sum(x**2)
        coulomb_term = np.sum(1/np.abs(x[:,None]-x[None,:])[~np.eye(len(x),dtype=bool)])
        return harmonic_term+coulomb_term
    
    def find_eq_pos(self):
        temp_sol = minimize(self.potential,np.random.rand(self.num_ions))
        return np.sort(temp_sol.x)
    
    def find_axial(self):
        A_mat = np.zeros((self.num_ions,self.num_ions))
        x_vec = np.sort(self.find_eq_pos())
        u_vec = x_vec/np.power(2*self.prefac/(self.mass*self.omega_a**2),1/3)
        for i in range(self.num_ions):
            for j in range(self.num_ions):
                if i!=j:
                    A_mat[i,j]= -2/np.power(np.abs(u_vec[i]-u_vec[j]),3)
        for i in range(self.num_ions):
            A_mat[i,i]=1-np.sum(A_mat[i])

        freqs_squared, b_vecs = np.linalg.eigh(self.omega_a**2*A_mat)
        #freqs = np.sqrt(freqs_squared) 
        freqs = freqs_squared
        
        return freqs,b_vecs,A_mat  
    
    def find_transverse(self):
        A_mat = np.zeros((self.num_ions,self.num_ions))
        x_vec = np.sort(self.find_eq_pos())
        u_vec = x_vec/np.power(2*self.prefac/(self.mass*self.omega_a**2),1/3)
        for i in range(self.num_ions):
            for j in range(self.num_ions):
                if i!=j:
                    A_mat[i,j]= 1/np.power(np.abs(u_vec[i]-u_vec[j]),3)
        for i in range(self.num_ions):
            A_mat[i,i]= (self.omega_r/self.omega_a)**2-np.sum(A_mat[i])

        freqs_squared, b_vecs = np.linalg.eigh(self.omega_a**2*A_mat)
        freqs = np.sqrt(freqs_squared)
        return freqs,b_vecs, A_mat   
        
