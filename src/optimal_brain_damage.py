"""
This script uses TensorFlow and Numba to simulate a Quantum Many-Body system,
applying tensor computations to perform the simulation. 
The script covers various operations including creating a Hamiltonian,
optimizing and pruning the system, and calculating the cost function.
"""

# Importing required libraries
"""
The code starts by importing the necessary libraries. Numpy is used for numerical operations, 
TensorFlow for tensor operations, Matplotlib for plotting, and tqdm for progress bars.
The trapped_ions is a module  in a /utility module to perform quantum mechanical simulations of trapped ions.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import pickle
from trapped_ions import normal_modes
from scipy.optimize import minimize
import numba as nb
from numba import njit,orange

# Configuring Tensorflow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
# Setting environmental variables for multiprocessing and Tensorflow logging level
os.environ["OMP_NUM_THREADS"] = ncores
os.environ["OPENBLAS_NUM_THREADS"] = ncores
os.environ["MKL_NUM_THREADS"] = ncores
os.environ["VECLIB_MAXIMUM_THREADS"] = ncores
os.environ["NUMEXPR_NUM_THREADS"] = ncores
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Printing the current Conda environment and the number of GPUs available
print (os.environ['CONDA_DEFAULT_ENV'])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Functions to manipulate matrices and vectors, get indices for 2D and 3D grids, and create a hopping matrix
def matrix_to_vector(A):
    vec=np.empty(A.size,dtype=np.complex128)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            vec[i*A.shape[0]+j]=A[i,j]
    return vec
def vec_to_array(vec):
    L=int(np.sqrt(vec.size))
    arr=np.zeros((L,L),dtype=np.complex128)
    for i in range(L):
        for j in range(L):
            arr[i,j]=vec[Index(i, j, L)]
    return arr



def Index(L1,dim1,ix, iy,iz=0):
    if dim1==2:
        return iy*L1+ix
    elif dim1==3:
        return iz*L1*L1+iy*L1+ix

def invIndex(L1,dim1,I):
    if dim1==2:
        return I % L1, I // L1
    elif dim1==3:
        return (I%L1**2)%L1,(I%L1**2)//L1,I//L1**2



def create_nn_hopping(L1, dim=1, bc='obc',ty=1,tx=1,phase=1):
    if dim == 1:
        hopping_matrix = 1. * np.diag(np.ones(L1 - 1), k=1)
        if bc == 'pbc':
            hopping_matrix[0, -1] = 1.
        hopping_matrix += hopping_matrix.T
        return hopping_matrix.astype(np.complex128)
    elif dim == 2:
        hopping_matrix = np.zeros((L1 ** 2, L1 ** 2), dtype=np.complex128)
        for I in range(L1 ** 2):
            ix, iy = invIndex(L1, dim, I)
            # print(ix,iy)
            if ix < L1 - 1:
                hopping_matrix[I, Index(L1, dim, ix + 1, iy)] = tx* np.exp(1j*iy*phase)
            if iy < L1 - 1:
                hopping_matrix[I, Index(L1, dim, ix, iy + 1)] = ty
            if bc == 'pbc':
                if ix == (L1 - 1):
                    hopping_matrix[I, Index(L1, dim, 0, iy)] = tx* np.exp(1j*iy*phase)
                if iy == (L1 - 1):
                    hopping_matrix[I, Index(L1, dim, ix, 0)] = ty
        hopping_matrix += hopping_matrix.T
        return hopping_matrix
    elif dim == 3:
        hopping_matrix = np.zeros((L1 ** 3, L1 ** 3), dtype=np.complex128)
        for I in range(1, L1 ** 3):
            ix, iy, iz = invIndex(L1, dim, I)
            # print(ix,iy)
            if ix < L1 - 1:
                hopping_matrix[I, Index(L1, dim, ix + 1, iy, iz)] = 1.
            if iy < L1 - 1:
                hopping_matrix[I, Index(L1, dim, ix, iy + 1, iz)] = 1.
            if iz < L1 - 1:
                hopping_matrix[I, Index(L1, dim, ix, iy, iz + 1)] = 1.
            if bc == 'pbc':
                if ix == (L1 - 1):
                    hopping_matrix[I, Index(L1, dim, 0, iy, iz)] = 1.
                if iy == (L1 - 1):
                    hopping_matrix[I, Index(L1, dim, ix, 0, iz)] = 1.
                if iz == (L1 - 1):
                    hopping_matrix[I, Index(L1, dim, ix, iy, 0)] = 1.
    #return csc_matrix(hopping_matrix)
        hopping_matrix += hopping_matrix.T
        return hopping_matrix

# Creating a Hamiltonian matrix for the quantum system
j_target=create_nn_hopping(L1=7, dim=2, bc='pbc',ty=1.,tx=1,phase=0.3)
j_up=np.triu(j_target, k=0)
j_full=(np.transpose(np.conjugate(j_up))+j_up)

#creates a Hamiltonian matrix
# Initialization for trapped ions simulation
N_row=7
N_col=7
hopp_amph=-1
hopp_ampv=-1
N_sites=N_row*N_col
N_ion=N_sites
omega_a=1
omega_r=20
mass=1
l_fac=1
prefac=0.1
ions_instance = normal_modes(N_ion,omega_a,omega_r,mass,l_fac,prefac)
freqs,vecs,A_mat = ions_instance.find_transverse()
etas = (vecs/np.sqrt(freqs))
mus = freqs+0.1*np.append([np.mean(np.diff(freqs))],np.diff(freqs))
Fijn = np.array([[[np.sum(vecs[i_val]*vecs[j_val]/(mus[n_val]**2-freqs**2))*(i_val!=j_val) for n_val in range(N_ion)] for j_val in range(N_ion)] for i_val in range(N_ion)])
f_tensor = tf.cast(tf.convert_to_tensor(Fijn),'complex128')
j_tensor = tf.cast(tf.convert_to_tensor(j_full),'complex128')

norm = np.linalg.norm(j_full)

def cost_j_tf(var):
    #jmat_error = tf.reduce_sum(tf.square(j_target-tf.einsum('in,jn,ijn->ij',var,var,f_tensor)))
    jmat_error = tf.reduce_sum(1*tf.math.real(j_tensor-tf.einsum('in,jn,ijn->ij',var,tf.math.conj(var),f_tensor))**2
                               +1*tf.math.imag(j_tensor-tf.einsum('in,jn,ijn->ij',var,tf.math.conj(var),f_tensor))**2)
    return jmat_error

def j_from_omega(omega):
    return np.einsum('in,jn,ijn->ij',omega,tf.math.conj(omega),Fijn)


delta_mn = (mus[:,None]-freqs[None,:])
Delta_mn = (mus[:,None]+freqs[None,:])

t_list = np.linspace(0,2,10)
I_tmn = np.array([-1j*((-1+np.exp(2j*np.pi*Delta_mn*t))/Delta_mn
                    -(-1+np.exp(-2j*np.pi*delta_mn*t))/delta_mn) for t in t_list])

#def cost_j_tf(var):
#    #jmat_error = tf.reduce_sum(tf.square(j_target-tf.einsum('in,jn,ijn->ij',var,var,f_tensor)))
#    jmat_error = tf.reduce_sum((j_target-tf.einsum('in,jn,ijn->ij',var,tf.math.conj(var),f_tensor))*tf.math.conj(j_target-tf.einsum('in,jn,ijn->ij',var,tf.math.conj(var),f_tensor)))
#    return jmat_error

def first_order_errs(var):
    K_itm = tf.square(tf.abs(tf.einsum('in,tmn->itm',tf.cast(var,'complex128'),I_tmn)))
    cost_imt = tf.einsum('im,itm->imt',np.square(np.abs(etas)),K_itm)
    return cost_imt.numpy()

def cost_reg_tf(var):
    K_itm = tf.square(tf.abs(tf.einsum('in,tmn->itm',tf.cast(var,'complex128'),I_tmn)))
    cost_it = tf.einsum('im,itm->it',np.square(np.abs(etas)),K_itm)
    cost_reduced = tf.reduce_sum(cost_it)
    return tf.cast(cost_reduced/np.prod(I_tmn.shape),'float64') #normalize by t*n_variables
def hess_reg(j,l):
    return 2*np.einsum('tm,m->...',np.abs(I_tmn[:,:,l])**2,np.abs(etas[j,:])**2)/np.prod(I_tmn.shape)

def hess_j(omega,k,m):
    hess_temp = sum((omega[j,m]*Fijn[k,j,m])*tf.math.conj((omega[j,m]*Fijn[k,j,m])) for j in range(N_ion))
    hess_temp+= sum((tf.math.conj(omega[i,m])*Fijn[i,k,m])*tf.math.conj((tf.math.conj(omega[i,m])*Fijn[i,k,m])) for i in range(N_ion))
    return 2*hess_temp 

@tf.function
def opt_step_fine(omega_in,mask,alpha):
    with tf.GradientTape() as t:
        t.watch(omega_in)
        #current_loss = cost_j_tf(omega_in*mask)+alpha*cost_reg_tf(omega_in*mask)
        current_loss = cost_j_tf(omega_in*mask)#+alpha*cost_reg_tf(omega_in*mask)
    domega = t.gradient(current_loss, [omega_in])
    opt.apply_gradients(zip(domega,[omega_in]))
    return current_loss

def optimize_and_prune(var,alpha,thresh=2000,init_epochs=20000,fine_epochs=10000,custom_mask=None):
    results = {}
    n_dims = var.shape[0]
    
    loss_hist = []
    
    mask = np.ones_like(np.real(var.numpy())).astype('complex128')
    for _ in tqdm(range(init_epochs)):
        current_loss = opt_step_fine(var,tf.constant(mask),alpha)
        loss_hist.append(current_loss.numpy())
    
    init_cost = cost_j_tf(var).numpy()
    init_omega = var.numpy()

    results["f_opt"] = init_cost
    results["x_opt"] = init_omega

    if custom_mask is None:
        hess_mat = np.array([[hess_j(init_omega,i,j) for j in range(n_dims)] for i in range(n_dims)])
        #hess_mat += alpha*np.array([[hess_reg(i,j) for j in range(n_dims)] for i in range(n_dims)])

        #make hessian mask
        num_nonzero = np.sum(np.abs(init_omega)!=0)
        i_ctr = 0 
        #mindices = np.argsort((hess_mat*init_omega**2).flatten())
        #mindices = np.argsort((hess_mat*init_omega*tf.math.conj(init_omega))).flatten()
        mindices=np.argsort(np.abs(hess_mat*init_omega*np.conj(init_omega)).flatten())
#         mindices = np.argsort((init_omega**2).flatten())


        while num_nonzero>thresh:
            mask[mindices[i_ctr]//N_ion,mindices[i_ctr]%N_ion] = 0
            i_ctr+=1 
            
            num_nonzero = np.sum(np.abs(init_omega*mask)!=0) 
            
            

    else: 
        mask = custom_mask 

    #train with mask
    
    results["mask"] = mask 
    
    for _ in tqdm(range(fine_epochs)):
        current_loss = opt_step_fine(var,tf.constant(mask),alpha)
        loss_hist.append(current_loss.numpy())
        
    omega_pruned_opt = mask*var.numpy()
    cost_after_opt = cost_j_tf(tf.constant(omega_pruned_opt)).numpy()
    
    results["f_pruned"] = cost_after_opt
    results["x_pruned"] = omega_pruned_opt
    results["history"] = loss_hist
    
    return results

def optimize_lbfgs(var_numpy,mask):
    def f_func(y): #need to define a single input function np.array -> np.array 
        y_temp = y.reshape(var_numpy.shape)
        return cost_j_tf(y_temp*mask).numpy()
        
    def ham_jac(y):
        with tf.GradientTape() as g:
            y_tf  = tf.convert_to_tensor(y.reshape(var_numpy.shape))
            g.watch(y_tf)
            f = cost_j_tf(y_tf*mask)
        jacobian = g.gradient(f, y_tf)
        return jacobian.numpy().flatten()
            
    res_opt_temp = scipy.optimize.minimize(f_func, var_numpy.flatten(), method="L-BFGS-B",jac=ham_jac,)
    
    return res_opt_temp

## Run and plotting the result:
th_range = np.array([500,750,1000,1250,1500,1750,2000,2250,2401])
prune_cost = []
for j in range(2):
    prune_cost_temp=[]
    for th_val in th_range:
        res_prune = pickle.load(open(f"/***Your Address***/res_prune_{th_val}_{j}.pkl", "rb" ))
        
        prune_cost_temp.append(res_prune["f_pruned"])
    prune_cost.append(prune_cost_temp)
prune_cost = np.array(prune_cost)

for i in range(len(th_range)):
    plt.scatter(th_range[i]*np.ones(prune_cost.shape[0]),prune_cost[:,i],c='C1')

plt.plot(th_range,np.mean(prune_cost,axis=0), label='Prune')
