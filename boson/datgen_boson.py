# necessary imports

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import pickle
import time
import sys

from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardChain

from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
from tenpy.algorithms import tdvp


# observables to evaluate during time evolution

def observables(eng, data):
    keys = ['t', 'entropy', 'N', 'B', 'g1', 'g2', 'trunc_err']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    data['entropy'].append(eng.psi.entanglement_entropy())
    data['N'].append(eng.psi.expectation_value('N'))
    data['g1'].append(eng.psi.correlation_function('Bd', 'B'))
    data['g2'].append(eng.psi.correlation_function('N', 'N'))
    data['trunc_err'].append(eng.trunc_err.eps)
    return data

# to save after each time step on drive
def save_obj(obj, name ):
  with open(name + '.pkl', 'wb') as f:
      pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
      
      
# main execution     
if __name__ == "__main__":
    
    # SYSTEM PARAMETERS
    
    # command line input
    N_site = int(sys.argv[1]) # number of sites
    U = float(sys.argv[2]) # interaction strength
    
    # filling of particles, otherwise set to 0.1
    try:
        V0 = -float(sys.argv[3])
    except:
        V0 = -0.3
    
    # boundary conditions, otherwise set to "finite"
    try:
        bc_MPS = str(sys.argv[4])
    except:    
        bc_MPS = 'finite'
    
    # folder to store, otherwise set to "store"    
    try:
        folder = str(sys.argv[5])
    except:
        folder = "store"
    
    # fixed parameters
    J = 0.5 # tunneling
    V = 0. # nearest-neighbor interaction
    mu = 0. # chemical potential
    filling = 0.1
    
    # starting potential V0*exp( -(x-x0)**2/sigma**2 )
    x0 = N_site/2.
    sigma = 0.1*N_site
    
    # for time evolution
    t_ev = N_site / 2
    save = True
    prefix = "/data_"
    
    n_max = 7 # maximal occupation
    
    print("Running for N={}, U={:.2f}, V0={:.2f}...".format(N_site, U, V0))
    
    # numerical parameters dmrg
    dmrg_params = {
        'trunc_params': {
            'chi_max': 300,
            'svd_min': 1.e-8,
        },
        'update_env': 5,
        'start_env': 5,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'max_sweeps': 5, 
        'mixer': False,
    
    }
    
    
    # numerical parameters TEBD
    tebd_params = { # everything for TEBD
        'order': 4,
        'dt': 0.1,
        'N_steps': 5,
        'max_error_E': 1.e-6,
        'trunc_params': {
            'chi_max': 300,
            'svd_min': 1.e-7
        },
        'verbose': 10,
    }
    
    # DMRG WITH POTENTIAL

    # number of particles in system
    N_particle = int(filling*N_site)
    
    # the potential
    x = np.linspace(0,N_site-1,N_site)
    potential = V0*np.exp( -(x-x0)**2/sigma**2 )
    
    # Hamiltonian with J_start = 0
    model_params = dict(L=N_site, t=J, U=U, V=V, mu=mu, bc_MPS=bc_MPS, conserve="N", n_max=n_max)
    BH0 = BoseHubbardChain(model_params)
    #BH0.n_max = n_max
    BH0.add_onsite(potential, 0, "N", category=None)
    BH0.init_H_from_terms()
    
    # initial state
    occ = np.zeros(N_site, dtype="int")
    occ[:N_particle] = 1
    occ = np.roll(occ,(N_site-N_particle)//2 )
    psi = MPS.from_product_state(BH0.lat.mps_sites(), occ, bc=BH0.lat.bc_MPS)
    
    # TEBD engine
    #eng = dmrg.DMRGEngine(psi, BH0, dmrg_params)
    #info = dmrg.run(psi, BH0, dmrg_params)
    print("Start DMRG...")
    start = time.time()
    eng = dmrg.TwoSiteDMRGEngine(psi, BH0, dmrg_params)
    E, psi = eng.run()
    
    print("time elapsed for DMRG: {:.2f}s".format(time.time()-start))
    
    # TEBD TO OBTAIN TIME EVOLUTION
    
    # make model with params above
    model_params = dict(L=N_site, t=J, U=U, V=V, mu=mu, bc_MPS=bc_MPS, conserve="N", n_max=n_max)
    BH = BoseHubbardChain(model_params)
    
    # TEBD engine with DMRG obtained psi
    eng = tebd.TEBDEngine(psi, BH, tebd_params)
    
    # intialize data
    data = observables(eng, None)
    
    # the count and number of time steps
    i = 0;
    i_tot = int(t_ev / tebd_params["dt"] / tebd_params["N_steps"])
    
    start = time.time()
    print("Start TEBD...")
    while eng.evolved_time < t_ev:
        
        t_start = time.time()
        eng.run()
        observables(eng, data)
    
        if save:
          # file to save all collected data
          suffix = "N_" + str(N_site) + "_V0_" + "{:.2f}".format(V0) + "_U_" + "{:.2f}".format(U) + "_J_" + "{:.2f}".format(J) + "_T_" + "{:.2f}".format(t_ev)
          filename = folder + prefix + suffix
          save_obj(data,filename)
    
    
        i += 1
        print("run " + str(i) + " of " + str(i_tot) + " finished in {:.2f}s".format(time.time()-t_start))
        
    print("Full run TEBD finished in {:.2f}s".format(time.time()-start))
    