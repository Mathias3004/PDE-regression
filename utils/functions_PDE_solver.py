import numpy as np
import time
import functools
import progressbar
import operator

from importlib import reload

import functions_regression
reload( functions_regression )
from functions_regression import *

# get all additions/subtractions first
def get_u_term_from_descr(descr_term, u, dx, bc="periodic"):
  # check if plus or minus
  tp = descr_term.count('+') > 0
  tm = descr_term.count('-') > 0

  if tp:
    terms = descr_term.split('+')
    return get_u_term_from_descr_el(terms[0], u, dx, bc) + get_u_term_from_descr(terms[1], u, dx, bc)
  
  elif tm:
    terms = descr_term.split('-')
    return get_u_term_from_descr_el(terms[0], u, dx, bc) - get_u_term_from_descr(terms[1], u, dx, bc)

  else:
    return get_u_term_from_descr_el(descr_term, u, dx, bc)


# get all separate terms in sum
def get_u_term_from_descr_el(descr_term, u, dx, bc="periodic"):
    """Parse a string representing a given term in PDE.

    Notation examples 1D: u_x = \partial_x u
                       u_xx = \partial_xx u
                       u*u_x = u*(\partial_x u)
                       u_x^2 = (\partial_x u)^2
    Notation examples 2D: u_sg = (\partial_x + \partial_y) u
                      u_ag = (\partial_x - \partial_y) u
                      u_sgg = (\partial_xx + \partial_yy) u
                      etc
    Supports only * and ^ binary operations
    Returns: np.array(complex) term
    """

    # Split multipliers according to * sign
    multis = descr_term.split('*')
    collection = []
    obs_dict = {'u': u, 'uc' : np.conj(u), '': np.ones(u.shape)}

    if u.ndim==2:
      obs_dict['nx'] = np.tile(np.diag(u), (u.shape[0],1) ).T
      obs_dict['ny'] = np.tile(np.diag(u), (u.shape[0],1) )
      obs_dict['Q'] = np.diag(u.shape)
    
    for m in multis:
        expr = m.split('^')  # check if have a power sign
        if len(expr) > 1:
            m, pow = expr[0], int(expr[1])
        elif len(expr) == 1:
            m, pow = expr[0], 1
        else:
            raise ValueError("Found multiple symbols for u, should be one")
        f = m.split('_')
        obs = f[0]
        # assert obs == 'u'
        if len(f) > 1:
            if obs_dict[obs].ndim == 1:
                der = f[1].count('x')
                dat_der = FiniteDiff(obs_dict[obs], dx, der, bc=bc)

            if obs_dict[obs].ndim == 2:
              if f[1].count('g') > 0:
                sym = f[1][0]
                der = f[1].count('g')

                # x derivatives
                dat_x = MultiFiniteDiff(obs_dict[obs], dx, der, axis=0, bc=bc)
                # y derivatives
                dat_y = MultiFiniteDiff(obs_dict[obs], dx, der, axis=1, bc=bc)

                if sym == 's':
                  dat_der = dat_x + dat_y
                elif sym == 'a':
                  dat_der = dat_x - dat_y

              if f[1].count('x') + f[1].count('y') > 0:
                der_x = f[1].count('x')
                der_y = f[1].count('y')

                # x derivatives
                dat_x = MultiFiniteDiff(obs_dict[obs], dx, der_x, axis=0, bc=bc)
                # y derivatives
                dat_der = MultiFiniteDiff(dat_x, dx, der_y, axis=1, bc=bc)

        # for "u" or ""
        elif len(f) == 1:
            dat_der = obs_dict[obs]

        collection.append(dat_der**pow)
    term = functools.reduce(operator.mul, collection, 1)
    return term

def scalar_pde_solver(descr, coefs, u0, t, dx, Q=None, v0=None, bc="periodic", nskip=100, dirichlet_bc=None):
    """Solver for the first order PDE in form u_t=F(u, u_x, u_xx, ...).
       Assumes scalar complex-valued function u(t,x).
       Builds equation from symbolic description.

       This PDE solver is very basic (explicit Euler method).
       Pros - Provides flexibility of implementation for many types of PDEs.
       Cons - Need extra time steps

       Parameters:
            (complex) coefs -- vector of coefficients multiplying each term in F(.)
            (int) nskip -- number of additional time steps for PDE solver (per dt step)
                           (increases number of steps in vector 't' by a factor 'nskip')
            (complex) u0 -- initial condition for function u(t, x) at t=0

    """
    # remove all zero terms from description for evaluation  
    
    threshold = np.max(np.abs(u0))*1e-7
    ind = np.squeeze(np.abs(coefs) > threshold)
    coefs = coefs[ind]
    descr = list(np.array(descr)[ind])

    t = np.linspace(t[0], t[-1], len(t)*nskip)  # Create a refined t-grid
    nt = len(t)

    if u0.ndim==1:
      nx = u0.size
      u_ev = np.zeros((nt,nx), dtype=complex)
    else:
      nx, ny = u0.shape[0], u0.shape[1]
      u_ev = np.zeros((nt,nx,ny), dtype=complex)
    
    u = u0

    if not v0 is None:
        u_t = v0

    dt = t[1] - t[0]

    for it, t in enumerate(t):
        #print(it)
        der = np.sum([coefs[i]*get_u_term_from_descr(descr_i, u, dx) \
                    for i, descr_i in enumerate(descr)], axis=0)

        if not Q is None:
            der += Q

        # second-order time derivative?
        if not v0 is None:
            u_t = u_t + dt*der
            u_next = u + dt*u_t
        else:   
            u_next = u + dt*der

        if not dirichlet_bc is None:
          for i, val in enumerate(dirichlet_bc):
            u_next[val[:-1]] = val[-1]

        u_ev[it] = u.copy()
        u = u_next

        if np.isnan(np.sum(u)):
            # Solution exploded, interrupt
            return np.array([np.nan])

    return u_ev[::nskip]