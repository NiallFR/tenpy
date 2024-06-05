import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..linalg import random_matrix
from .tebd_qc import TEBDEngine_qc

__all__ = ['TEBDEngine', 'Engine', 'RandomUnitaryEvolution', 'TimeDependentTEBD']

class TEBDEngine_qc_hx(TEBDEngine_qc):
    def __init__(self, psi, model, model2, model3, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, psi, model, options, **kwargs)
        self.model2 = model2
        self.model3 = model3
        self._trunc_err_bonds = [TruncationError() for i in range(psi.L + 1)]
        self._U = None
        self._U_param = {}
        self._update_index = None
        # self._decomp = [(0, 1), (2, 1), (1, 0), (2, 1), (0, 1)]
       
    @staticmethod
    def suzuki_trotter_decomposition(order, N_steps):
        even, odd = 0, 1
        if N_steps == 0:
            return []
        if order == 1:
            a = (0, odd)
            b = (0, even)
            return [a, b] * N_steps
        elif order == 2:
            a = (0, 1)  # dt/2
            hz = (2, 1)  # dt/2
            hx = (3, 1) # dt/2
            b = (1, 0)  # dt
            c = (1, 1)  # dt
            return [a, hz, hx, b, hx, hz] + [c, hz, hx, b, hx, hz] * (N_steps - 1) + [a]
        elif order == 4:
            a = (0, odd)  # t1/2
            a2 = (1, odd)  # t1
            b = (1, even)  # t1
            c = (2, odd)  # (t1 + t3) / 2 == (1 - 3 * t1)/2
            d = (3, even)  # t3 = 1 - 4 * t1
            # From Schollwoeck 2011 (:arxiv:`1008.3477`):
            # U = U(t1) U(t2) U(t3) U(t2) U(t1)
            # with U(dt) = U(dt/2, odd) U(dt, even) U(dt/2, odd) and t1 == t2
            # Using above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            steps = [a, b, a2, b, c, d, c, b, a2, b]
            steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            steps = steps + [a]
            return steps
    
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

        L = self.psi.L
        self._U = []
        if order == 2:
            for bond_type in ['half_time', 'full_time', 'magz', 'magx']:
                if bond_type == 'magx' or bond_type == 'magz' or bond_type == 'half_time':
                    dt = 0.5
                elif bond_type == 'full_time':
                    dt = 1.0
                else:
                    NotImplementedError("bond type must be half_time, full_time or mag")
                U_bond = [
                    self._calc_U_bond_order2(i_bond, bond_type, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
                ]
                self._U.append(U_bond)

        elif order == 4:
            for dt in self.suzuki_trotter_time_steps(order):
                U_bond = [
                    self._calc_U_bond(i_bond, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
                ]
                self._U.append(U_bond)

        # for dt in self.suzuki_trotter_time_steps(order):
        #     U_bond = [
        #         self._calc_U_bond(i_bond, bond_type, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
        #     ]
        #     self._U.append(U_bond)
        self.force_prepare_evolve = False
        
        
        
    def _calc_U_bond_order2(self, i_bond, bond_type, dt, type_evo, E_offset):
        """Calculate exponential of a bond Hamitonian.

        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
        """

        if bond_type == 'half_time' or bond_type == 'full_time':
            h = self.model.H_bond[i_bond]
        elif bond_type == 'magz':
            h = self.model2.H_bond[i_bond]
        elif bond_type == 'magx':
            h = self.model3.H_bond[i_bond]
        else:
            NotImplementedError("bond type must be half_time, full_time or mag")

        if h is None:
            return None  # don't calculate exp(i H t), if `H` is None
        H2 = h.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
        if type_evo == 'imag':
            H2 = (-dt) * H2
        elif type_evo == 'real':
            if E_offset is not None:
                H2 = H2 - npc.diag(E_offset[i_bond], H2.legs[0])
            H2 = (-1.j * dt) * H2
        else:
            raise ValueError("Expect either 'real' or 'imag'inary time, got " + repr(type_evo))
        U = npc.expm(H2)
        assert (tuple(U.get_leg_labels()) == ('(p0.p1)', '(p0*.p1*)'))
        return U.split_legs()
    
    
    
    
    
    