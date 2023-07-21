import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import pickle
from trapped_ions import normal_modes
from scipy.optimize import minimize

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
ions_instance = normal_modes(N_ion,omega_a,omega_r,mass,l_fac,prefac) #num_ions, omega_a,omega_r,mass,l_fac,prefac
freqs,vecs,A_mat = ions_instance.find_transverse() 
etas = (vecs/np.sqrt(freqs))
mus = freqs+0.1*np.append([np.mean(freqs)],np.diff(freqs))
Fijn = np.array([[[np.sum(vecs[i_val]*vecs[j_val]/(mus[n_val]**2-freqs**2))*(i_val!=j_val) for n_val in range(N_ion)] for j_val in range(N_ion)] for i_val in range(N_ion)])



## defining the interaction matrix, J, for the one dimensional Schwinger Model 

j_target = np.zeros((N_sites,N_sites),'float64')
for m in range(N_sites-2):
    for n in range(m+1,N_sites-1):
        j_target[m,n] = (N_sites-(n+1))/N_sites
j_target = j_target + j_target.T



f_tensor = tf.cast(tf.convert_to_tensor(Fijn),'float64')
j_tensor = tf.cast(tf.convert_to_tensor(j_target),'float64')
norm = np.linalg.norm(j_target)

delta_mn = (mus[:,None]-freqs[None,:])
Delta_mn = (mus[:,None]+freqs[None,:])


t_list = np.linspace(0,2,10)
I_tmn = np.array([-1j*((-1+np.exp(2j*np.pi*Delta_mn*t))/Delta_mn
                    -(-1+np.exp(-2j*np.pi*delta_mn*t))/delta_mn) for t in t_list])

def cost_j_tf(var):
    jmat_error = tf.reduce_sum(tf.square(j_target-tf.einsum('in,jn,ijn->ij',var,var,f_tensor)))
    return jmat_error
def j_from_omega(omega):
    return np.einsum('in,jn,ijn->ij',omega,omega,Fijn)

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
    hess_temp = sum((omega[j,m]*Fijn[k,j,m])**2 for j in range(N_sites))
    hess_temp+= sum((omega[i,m]*Fijn[i,k,m])**2 for i in range(N_sites))
    return 2*hess_temp
@tf.function
def opt_step_fine(omega_in,mask,alpha):
    with tf.GradientTape() as t:
        t.watch(omega_in)
        current_loss = cost_j_tf(omega_in*mask)+alpha*cost_reg_tf(omega_in*mask)
    domega = t.gradient(current_loss, [omega_in])
    opt.apply_gradients(zip(domega,[omega_in]))
    return current_loss
def optimize_and_prune(var,alpha,thresh=2000,init_epochs=20000,fine_epochs=10000,custom_mask=None):
    results = {}
    n_dims = var.shape[0]
    
    loss_hist = []
    
    mask = np.ones_like(var.numpy())
    for _ in tqdm(range(init_epochs)):
        current_loss = opt_step_fine(var,tf.constant(mask),alpha)
        loss_hist.append(current_loss.numpy())
    
    init_cost = cost_j_tf(var).numpy()
    init_omega = var.numpy()

    results["f_opt"] = init_cost
    results["x_opt"] = init_omega

    if custom_mask is None:
        hess_mat = np.array([[hess_j(init_omega,i,j) for j in range(n_dims)] for i in range(n_dims)])
        hess_mat += alpha*np.array([[hess_reg(i,j) for j in range(n_dims)] for i in range(n_dims)])

        #make hessian mask
        num_nonzero = np.sum(init_omega!=0)
        i_ctr = 0 
        mindices = np.argsort((hess_mat*init_omega**2).flatten())
#         mindices = np.argsort((init_omega**2).flatten())


        while num_nonzero>thresh:
            mask[mindices[i_ctr]//N_sites,mindices[i_ctr]%N_sites] = 0
            i_ctr+=1 
            num_nonzero = np.sum((init_omega*mask)!=0) 

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
opt = tf.keras.optimizers.Adam(0.003)
omega_tensor_0 = np.random.rand(N_sites,N_sites).astype('float64')
omega_tensor = tf.Variable(omega_tensor_0)#,constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

th_range = np.array([500,750,1000,1250,1500,1750,2000,2250,2401])

for j in [0,1,2,3,4,5,6,7,8,9]:
    for th_val in th_range:
        try:
            foo = pickle.load(open(f"**Your-Address**/res_prune_{th_val}_{j}.pkl","rb") )
        except (EOFError,OSError, IOError) as e:
            omega_init=np.random.rand(N_ion,N_ion).astype('float64')
            omega_tensor = tf.Variable(omega_init)
            omega_tensor.assign(omega_tensor)
            opt.set_weights(0*np.array(opt.get_weights(),dtype='object'))
            res_temp_prune = optimize_and_prune(omega_tensor,0,thresh=th_val,init_epochs=10000,
                                                fine_epochs = 20000)
            f = open(f"/**Your-Address**/res_prune_{th_val}_{j}.pkl","wb")
            pickle.dump(res_temp_prune,f)
            f.close()
for j in range(10):
    for th_val in th_range:
        try:
            foo = pickle.load(open(f"/**Your-Address**/res_rand_{th_val}_{j}.pkl","rb") )
        except (EOFError,OSError, IOError) as e:
            
            omega_init=np.random.rand(N_ion,N_ion).astype('float64')
            #Omega_init_tf=tf.random.uniform(shape=[N_ion,N_p],dtype='complex128')
            omega_tensor = tf.Variable(omega_init)

            opt.set_weights(0*np.array(opt.get_weights(),dtype='object'))
            rand_mask = np.zeros(np.prod(omega_tensor.shape),dtype='float64') # complex or flot 64??
            rand_mask[:th_val] = 1 
            np.random.shuffle(rand_mask)
            rand_mask = rand_mask.reshape(omega_tensor.shape)

            res_temp_rand = optimize_and_prune(omega_tensor,alpha=1,thresh=th_val,init_epochs=0,
                                                fine_epochs = 3000,custom_mask=rand_mask)    
            f = open(f"/**Your-Address**/res_rand_{th_val}_{j}.pkl","wb")
            pickle.dump(res_temp_rand,f)
            f.close()



