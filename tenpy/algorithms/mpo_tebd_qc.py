import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..linalg import random_matrix
from .tebd import TEBDEngine
from .tebd_qc import TEBDEngine_qc

__all__ = ['TEBDEngine', 'Engine', 'RandomUnitaryEvolution', 'TimeDependentTEBD']

class MPO_TEBDEngine_qc(TEBDEngine_qc):
    def __init__(self, F, model, model2, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, F, model, options, **kwargs)
        self.model2 = model2
        self._trunc_err_bonds = [TruncationError() for i in range(F.L + 1)]
        self._U = None
        self._U_param = {}
        self._update_index = None
        self.F = F
        self.conj = False
        # self._decomp = [(0, 1), (2, 1), (1, 0), (2, 1), (0, 1)]

    def update_bond(self, i, U_bond):
        """Updates the B matrices on a given bond.

        Function that updates the B matrices, the bond matrix s between and the
        bond dimension chi for bond i. The correponding tensor networks look like this::

        |           --S--B1--B2--           --B1--B2--
        |                |   |                |   |
        |     theta:     U_bond        C:     U_bond
        |                |   |                |   |

        Parameters
        ----------
        i : int
            Bond index; we update the matrices at sites ``i-1, i``.
        U_bond : :class:`~tenpy.linalg.np_conserved.Array`
            The bond operator which we apply to the wave function.
            We expect labels ``'p0', 'p1', 'p0*', 'p1*'``.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced by the truncation
            during this update step.
        """
        # print('i = ', i)
        i0, i1 = i - 1, i
        logger.debug("Update sites (%d, %d)", i0, i1)
        # Construct the theta matrix
        # C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C0 = self.F.get_W(i0)
        C1 = self.F.get_W(i1)

        C = npc.tensordot(C0, C1, axes=(['wR'],['wL']))
        new_labels = ['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR']
        C.iset_leg_labels(new_labels)
        # print('C0 = ', C0)
        # print('C1 = ', C1)
        # print('C = ', C)

        # print('U_bond = ', U_bond)
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # print('U_bond_C = ', C)
        # C.itranspose(['vL', 'p0', 'p1', 'vR'])
        # print('indices = ', C.get_leg_indices(['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR']))
        C.itranspose(['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR'])
        # print('U_bond_C_transpose = ', C)
        # theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        # print('S = ', self.F.Ss[i0])
        theta = C.scale_axis(self.F.Ss[i0], 'wL')
        # print('indices2 = ', theta.get_leg_indices(['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR']))
        # now theta is the same as if we had done
        #   theta = self.psi.get_theta(i0, n=2)
        #   theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # but also have C which is the same except the missing "S" on the left
        # so we don't have to apply inverses of S (see below)

        # theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])
        # print('theta_before_reshape = ', theta)
        # theta = theta.combine_legs([('wL', 'p0', 'p0*'), ('p1', 'p1*', 'wR')], qconj=[+1, -1])
        theta = theta.combine_legs([[0,1,2], [3,4,5]], qconj=[+1, -1])
        # print('theta_after_reshape = ', theta)
        # theta = theta.split_legs([0,1])
        # print('theta_after_split = ', theta)
        # Perform the SVD and truncate the wavefunction
        # U, S, V, trunc_err, renormalize = svd_theta(theta,
        #                                             self.trunc_params,
        #                                             [self.psi.get_B(i0, None).qtotal, None],
        #                                             inner_labels=['vR', 'vL'])
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    [None, None],
                                                    # [self.psi.get_W(i0, None).qtotal, None],
                                                    inner_labels=['wR', 'wL'])

        # print('renormalize = ', renormalize)
        # Split tensor and update matrices
        # print('S = ', S)
        # print('V = ', V)
        B_R = V.split_legs(1).ireplace_labels(['p1', 'p1*'], ['p', 'p*'])
        # print('B_R = ', B_R)
        # print(B_R)

        # In general, we want to do the following:
        #     U = U.iscale_axis(S, 'vR')
        #     B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)**-1, 'vL')
        #     B_L = B_L.ireplace_label('p0', 'p')
        # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**-1 U S``
        #
        # However, the inverse of SL is problematic, as it might contain very small singular
        # values.  Instead, we use ``C == SL**-1 theta == SL**-1 U S V``,
        # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
        # here, C is the same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)
        # B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
        #                     V.conj(),
        #                     axes=['(p1.vR)', '(p1*.vR*)'])
        # print(V.conj())
        B_L = npc.tensordot(C.combine_legs(('p1', 'p1*', 'wR'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(p1.p1*.wR)', '(p1*.p1.wR*)'])
        B_L.ireplace_labels(['wL*', 'p0', 'p0*'], ['wR', 'p', 'p*'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        # self.psi.set_SR(i0, S)
        self.F.Ss[i1] = S
        # self.psi.set_B(i0, B_L, form='B')
        self.F.set_W(i0, B_L)
        # self.psi.set_B(i1, B_R, form='B')
        self.F.set_W(i1, B_R)
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def evolve_step(self, U_idx_dt, odd):
        Us = self._U[U_idx_dt]
        # print('U_idx_dt = ', U_idx_dt)
        trunc_err = TruncationError()
        for i_bond in np.arange(int(odd) % 2, self.psi.L, 2):
            # print('i_bond = ', i_bond)
            # print('Us[i_bond] = ', Us[i_bond])
            if Us[i_bond] is None:
                continue  # handles finite vs. infinite boundary conditions
            self._update_index = (U_idx_dt, i_bond)
            if self.conj:
                self.update_bond_conj(i_bond, Us[i_bond])
            else:
                trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err

    def update_bond_conj(self, i, U_bond):
        # print('i = ', i)
        i0, i1 = i - 1, i
        logger.debug("Update sites (%d, %d)", i0, i1)
        # Construct the theta matrix
        # C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C0 = self.F.get_W(i0)
        C1 = self.F.get_W(i1)

        C = npc.tensordot(C0, C1, axes=(['wR'], ['wL']))
        new_labels = ['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR']
        C.iset_leg_labels(new_labels)
        # print('C0 = ', C0)
        # print('C1 = ', C1)
        # print('C = ', C)

        # print('U_bond = ', U_bond)
        C = npc.tensordot(U_bond, C, axes=(['p0', 'p1'], ['p0*', 'p1*']))  # apply U
        # print('U_bond_C = ', C)
        # C.itranspose(['vL', 'p0', 'p1', 'vR'])
        # print('indices = ', C.get_leg_indices(['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR']))
        C.itranspose(['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR'])
        # print('U_bond_C_transpose = ', C)
        # theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        # print('S = ', self.F.Ss[i0])
        theta = C.scale_axis(self.F.Ss[i0], 'wL')
        # print('indices2 = ', theta.get_leg_indices(['wL', 'p0', 'p0*', 'p1', 'p1*', 'wR']))
        # now theta is the same as if we had done
        #   theta = self.psi.get_theta(i0, n=2)
        #   theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # but also have C which is the same except the missing "S" on the left
        # so we don't have to apply inverses of S (see below)

        # theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])
        # print('theta_before_reshape = ', theta)
        # theta = theta.combine_legs([('wL', 'p0', 'p0*'), ('p1', 'p1*', 'wR')], qconj=[+1, -1])
        theta = theta.combine_legs([[0, 1, 2], [3, 4, 5]], qconj=[+1, -1])
        # print('theta_after_reshape = ', theta)
        # theta = theta.split_legs([0,1])
        # print('theta_after_split = ', theta)
        # Perform the SVD and truncate the wavefunction
        # U, S, V, trunc_err, renormalize = svd_theta(theta,
        #                                             self.trunc_params,
        #                                             [self.psi.get_B(i0, None).qtotal, None],
        #                                             inner_labels=['vR', 'vL'])
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    [None, None],
                                                    inner_labels=['wR', 'wL'])

        # print('renormalize = ', renormalize)
        # Split tensor and update matrices
        # print('S = ', S)
        # print('V = ', V)
        B_R = V.split_legs(1).ireplace_labels(['p1', 'p1*'], ['p', 'p*'])
        # print('B_R = ', B_R)
        # print(B_R)

        # In general, we want to do the following:
        #     U = U.iscale_axis(S, 'vR')
        #     B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)**-1, 'vL')
        #     B_L = B_L.ireplace_label('p0', 'p')
        # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**-1 U S``
        #
        # However, the inverse of SL is problematic, as it might contain very small singular
        # values.  Instead, we use ``C == SL**-1 theta == SL**-1 U S V``,
        # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
        # here, C is the same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)
        # B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
        #                     V.conj(),
        #                     axes=['(p1.vR)', '(p1*.vR*)'])
        # print(V.conj())
        B_L = npc.tensordot(C.combine_legs(('p1', 'p1*', 'wR'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(p1.p1*.wR)', '(p1*.p1.wR*)'])
        B_L.ireplace_labels(['wL*', 'p0', 'p0*'], ['wR', 'p', 'p*'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        # self.psi.set_SR(i0, S)
        self.F.Ss[i1] = S
        # self.psi.set_B(i0, B_L, form='B')
        self.F.set_W(i0, B_L)
        # self.psi.set_B(i1, B_R, form='B')
        self.F.set_W(i1, B_R)
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err