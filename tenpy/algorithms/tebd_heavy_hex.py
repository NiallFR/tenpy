import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError, truncate
from ..linalg import random_matrix
from .tebd import TEBDEngine
from ..tools.params import asConfig

class TEBDEngineHeavyHex(TEBDEngine):
    def __init__(self, psi, model, options, **kwargs):
        TEBDEngine.__init__(self, psi, model, options, **kwargs)
    
    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        U_param = dict(order=order, delta_t=delta_t, type_evo=type_evo, E_offset=E_offset)
        if type_evo == 'real':
            U_param['tau'] = delta_t
        elif type_evo == 'imag':
            U_param['tau'] = -1.j * delta_t
        else:
            raise ValueError("Invalid value for `type_evo`: " + repr(type_evo))
        if self._U_param == U_param and not self.force_prepare_evolve:
            logger.debug("Skip recalculation of U with same parameters as before")
            return  # nothing to do: U is cached
        self._U_param = U_param
        logger.info("Calculate U for %s", U_param)

        self._U = []
        
        if self.psi.L == 3 or self.psi.L == 2:
            num_layers = 1
            
        elif self.psi.L == 6:
            num_layers = 2
            
        else:
            num_layers = 3
            
            
        if order == 1:
            for k in range(num_layers):
                U_bond_layer = [
                    self._calc_U_bond(i_bond, delta_t) for i_bond in self.model.layers[k]
                    ]
                self._U.append(U_bond_layer)
                
        elif order == 2:
            U_bond_layer = [
                self._calc_U_bond(i_bond, 0.5*delta_t) for i_bond in self.model.layers[0]
                ]
            self._U.append(U_bond_layer)
            U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5*delta_t) for i_bond in self.model.layers[1]
                    ]
            self._U.append(U_bond_layer)
            U_bond_layer = [
                    self._calc_U_bond(i_bond, delta_t) for i_bond in self.model.layers[2]
                    ]
            self._U.append(U_bond_layer)
            U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5*delta_t) for i_bond in self.model.layers[1]
                    ]
            self._U.append(U_bond_layer)
            U_bond_layer = [
                self._calc_U_bond(i_bond, 0.5*delta_t) for i_bond in self.model.layers[0]
                ]
            self._U.append(U_bond_layer)
            
        else:
            raise NotImplementedError()
        
        

    def _calc_U_bond(self, i_bond, dt):
        i, j = self.model._connections[i_bond]
        h = self.model.H_bond[(i, j)]
        if h is None:
            return None  # don't calculate exp(i H t), if `H` is None
        H2 = h.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
        #H2 = H2 - npc.diag(E_offset[i_bond], H2.legs[0])
        H2 = (-1.j * dt) * H2
        U = npc.expm(H2)
        assert (tuple(U.get_leg_labels()) == ('(p0.p1)', '(p0*.p1*)'))
        return U.split_legs()
        
    
    
    def evolve(self, N_steps, dt):
        if dt is not None:
            assert dt == self._U_param['delta_t']
        trunc_err = TruncationError()
        order = self._U_param['order']
        #for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
        # for U_idx_dt, odd in self._decomp:
            
            
        for layer_no in range(len(self._U)):
            trunc_err += self.evolve_step(layer_no)
               
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `evolv`)
        return trunc_err
        
    def evolve_step(self, layer_no):
        #trunc_par = asConfig({'chi_max':  max(max(self.psi.chi), 100)}, 'trunc_params')
        trunc_par = self.trunc_params
        #print('chi_max = ', self.trunc_params['chi_max'])
        order = self._U_param['order']
        Us = self._U[layer_no]

        #print('len(Us) = ', len(Us))
        #print('Us = ', [Us[j] for j in range(len(Us))])
        #print('Us._data = ', [Us[j]._data for j in range(len(Us))])
        
        if order == 1:
            connection_numbers = self.model.layers[layer_no]
        elif order == 2:
            if layer_no == 3:
                layer_no = 1
            elif layer_no == 4:
                layer_no = 0
            connection_numbers = self.model.layers[layer_no]
        
        trunc_err = TruncationError()
        for k in range(len(Us)):
            U = Us[k]
            bond_no = connection_numbers[k]
            i, j = self.model._connections[bond_no]
            assert i < j
            if i+1 == j:
                trunc_err += self.update_bond(j, U)
            else:
                perm2, a, b = self.calc_perm(i,j)
                perm1, a, b = self.calc_reverse_perm(i,j)
                #print('final_perm1 = ', final_perm1)              
                self.psi.permute_sites(perm1, trunc_par=trunc_par)
                #self.psi.permute_sites(perm1)
                trunc_err += self.update_bond(j-b, U)
                self.psi.permute_sites(perm2, trunc_par=trunc_par)
                #self.psi.permute_sites(perm2)

        return trunc_err
        
    def calc_perm(self, i,j):
        assert i < j
        if (j-i-1) % 2 == 0:
            a = int((j-i-1) / 2)
            b = int((j-i-1) / 2)
        else:
            a = int((j-i) / 2)
            b = int((j-i) / 2 - 1)
        init_perm = np.array([k for k in range(self.psi.L)])
        final_perm = init_perm.copy()
        
        for k in range(a):
            final_perm[i+k] = init_perm[i+k+1]
        for k in range(b):
            final_perm[j-k] = init_perm[j-1-k]
        
        final_perm[i+a] = i
        final_perm[j-b] = j
        
        return final_perm, a, b
    
    def calc_reverse_perm(self, i,j):
        assert i < j
        if (j-i-1) % 2 == 0:
            a = int((j-i-1) / 2)
            b = int((j-i-1) / 2)
        else:
            a = int((j-i) / 2)
            b = int((j-i) / 2 - 1)
        init_perm = np.array([k for k in range(self.psi.L)])
        final_perm = init_perm.copy()
        
        for k in range(a):
            final_perm[i+k+1] = init_perm[i+k]
        for k in range(b):
            final_perm[j-1-k] = init_perm[j-k]
        
        final_perm[i] = i+a
        final_perm[j] = j-b
        
        return final_perm, a, b
        
        
        