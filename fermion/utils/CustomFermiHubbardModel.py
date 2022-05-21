import numpy as np
import copy

import tenpy.linalg.np_conserved as npc
import tenpy.networks.mps as mps
import tenpy.networks.site as site
from tenpy.algorithms import tdvp, tebd
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.site import FermionSite
from tenpy.tools.params import asConfig

import matplotlib.pyplot as plt

from six.moves import cPickle as pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


class CustomFermiHubbardModel(CouplingMPOModel):
    r"""Spinless fermions with particle number conservation.
        The Hamiltonian reads:
        .. math ::
            H = \sum_{\langle i,j\rangle, i<j}
                  - \mathtt{J}~(c^{\dagger}_i c_j + c^{\dagger}_j c_i) + \mathtt{V}~n_i n_j \\
                - \sum_i
                  \mathtt{mu}~n_{i} + U2 term(!)

        Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
        All parameters are collected in a single dictionary `model_params`, which
        is turned into a :class:`~tenpy.tools.params.Config` object.
        .. warning ::
            Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
            See :doc:`/intro/JordanWigner` for details.
        Parameters
        ----------
        model_params : :class:`~tenpy.tools.params.Config`
            Parameters for the model. See :cfg:config:`FermionModel` below.
        Options
        -------
        .. cfg:config :: FermionModel
            :include: CouplingMPOModel
            conserve : 'best' | 'N' | 'parity' | None
                What should be conserved. See :class:`~tenpy.networks.Site.FermionSite`.
                For ``'best'``, we check the parameters what can be preserved.
            J, V, mu : float | array
                Hopping, interaction and chemical potential as defined for the Hamiltonian above.
        """
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        if conserve == 'best':
            conserve = 'N'
            if self.verbose:
                print(self.name + ": set conserve to", conserve)
        site = FermionSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = model_params.get('t', 1.)
        U = model_params.get('U', 0)
        U2 = model_params.get('U2', 0)
        mu = model_params.get('mu', 0.)

        print('U2', U2)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'N')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cd', u2, 'C', dx, plus_hc=True)
            self.add_coupling(U, u1, 'N', u2, 'N', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(U2, u1, 'N', u2, 'N', dx)

       # potential = model_params['potential']
        #assert self.lat.N_sites == len(potential), f"L {self.lat.N_sites}, pot {len(potential)}"
        #plt.plot(potential)
        #plt.title('potential')
        #self.add_onsite(potential, 0, 'N')



class CustomFermionChain(CustomFermiHubbardModel, NearestNeighborModel):
    """The :class:`FermionModel` on a Chain, suitable for TEBD.
    See the :class:`FermionModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
