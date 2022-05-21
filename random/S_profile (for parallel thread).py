import numpy as np
import random
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from scipy.linalg import eig, eigh
import itertools as it
import sys
import os
from pathlib import Path
import time

from functions_S import *
from functions_U import *

from joblib import Parallel, delayed


## MAIN EXECUTION
if __name__ == "__main__":
    
    # obtain input from command line
    
    # input parameters
    N = int(sys.argv[1]) # number of modes
    r = float(sys.argv[2]) # squeezing
    
    # LON parameters
    LON_type = str(sys.argv[3]) # type of LON
    D = int(sys.argv[4]) # depth
    w = float(sys.argv[5]) # control parameter (strength disorder or probability BS)
    
    print("Running for N="+str(N) + ", r="+str(r) + ", type=" + LON_type + ", D=" + str(D) + " and w=" + str(w))
    
    # where to save data
    folder = str(sys.argv[6]) 
    
    print("Data will be saved in folder " + folder)
    
    # traverse whole chain for entropies
    max_l = N
    
    # local function to loop over in parallel threads
    def get_instance(dummy):
        if LON_type=="localization":
            U = np.matrix( network_U_eff(N,D,w).todense() )
        elif LON_type=="percolation":
            U = np.matrix( network_U_percolation(N,D,w) )
            
        C = get_covariance_matrix(U*np.cosh(r), U*np.sinh(r))
    
        S = [0.]
        for i in range(1,max_l):
            S.append( get_entropy(C, type="ada", cut=i) )
        S.append(0.)
    
        return S
    
    # parallel function execution, evaluate run time
    t1 = time.time()
    cpuCount = os.cpu_count()
    S_dat = Parallel(n_jobs=-1)(delayed(get_instance)(i) for i in range(cpuCount))
    t2 = time.time()
    
    print("time: " + str(t2-t1))
    print("CPU: " + str(cpuCount))
    
    # save data
    filename = "/dat_N_" + str(N) + "_r_" + "{:.2f}".format(r) + "_D_" + str(D) + "_w_" + "{:.2f}".format(w)
    fullname = folder + filename + ".npy"
    
    # append
    if Path(fullname).is_file():
        S_prev = np.load(fullname)
        np.save(fullname, np.concatenate((S_prev,S_dat), axis=0))
    # create new file
    else:
        np.save( fullname, np.array(S_dat) )
        
        
        
        
    