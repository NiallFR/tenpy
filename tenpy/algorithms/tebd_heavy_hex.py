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

    def calc_U(self, order, delta_t, type_evo="real", E_offset=None):
        U_param = dict(
            order=order, delta_t=delta_t, type_evo=type_evo, E_offset=E_offset
        )
        if type_evo == "real":
            U_param["tau"] = delta_t
        elif type_evo == "imag":
            U_param["tau"] = -1.0j * delta_t
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
            num_layers = len(self.model.layers)

        if order == 1:
            for k in range(num_layers):
                U_bond_layer = [
                    self._calc_U_bond(i_bond, delta_t)
                    for i_bond in self.model.layers[k]
                ]
                self._U.append(U_bond_layer)

        elif order == 2:
            if len(self.model.layers) == 2:
                U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5 * delta_t)
                    for i_bond in self.model.layers[0]
                ]
                self._U.append(U_bond_layer)
                U_bond_layer = [
                    self._calc_U_bond(i_bond, delta_t)
                    for i_bond in self.model.layers[1]
                ]
                self._U.append(U_bond_layer)
                U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5 * delta_t)
                    for i_bond in self.model.layers[0]
                ]
                self._U.append(U_bond_layer)
            elif len(self.model.layers) == 3:
                U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5 * delta_t)
                    for i_bond in self.model.layers[0]
                ]
                self._U.append(U_bond_layer)
                U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5 * delta_t)
                    for i_bond in self.model.layers[1]
                ]
                self._U.append(U_bond_layer)
                U_bond_layer = [
                    self._calc_U_bond(i_bond, delta_t)
                    for i_bond in self.model.layers[2]
                ]
                self._U.append(U_bond_layer)
                U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5 * delta_t)
                    for i_bond in self.model.layers[1]
                ]
                self._U.append(U_bond_layer)
                U_bond_layer = [
                    self._calc_U_bond(i_bond, 0.5 * delta_t)
                    for i_bond in self.model.layers[0]
                ]
                self._U.append(U_bond_layer)
            else:
                raise NotImplementedError()
        elif order == 4:
            """
            (2k)th-order Suzuki formula S2k, defined recursively in eq 4, 5 of C.1, arXiv:1711.10980v1
            defined for the first time in https://doi.org/10.1063/1.529425
            H = H_1 + H_2 + ... + H_n
            S_2(λ) := \Prod_{j=1}^n exp(Hjλ/2) \Prod_{j=n}^1 exp(Hjλ/2)
            S_{2k}(λ) := [S_{2k-2}(p_k λ)]^2 S_{2k-2}((1 - 4p_k)λ)[S_{2k-2}(p_k λ)]^2
            note that the square means to apply 2 times the operator.
            p_k = 1/(4-4^{1/(2k-1)})
            """
            if len(self.model.layers) == 2:
                # Eq (30a) of arXiv:1901.04974
                a1 = 0.095848502741203681182
                b1 = 0.42652466131587616168
                a2 = -0.078111158921637922695
                b2 = -0.12039526945509726545
                a3 = 0.5 - a1 - a2
                b3 = 1.0 - 2 * (b1 + b2)
                # symmetric: a1 b1 a2 b2 a3 b3 a3 b2 a2 b1 a1
                tks = [a1, b1, a2, b2, a3, b3, a3, b2, a2, b1, a1]
                # steps = [(0, odd), (1, even), (2, odd), (3, even), (4, odd),  (5, even),
                #     (4, odd), (3, even), (2, odd), (1, even), (0, odd)]
                for i, tk in enumerate(tks):
                    U_bond_layer = [
                        self._calc_U_bond(i_bond, tk * delta_t)
                        for i_bond in self.model.layers[i % 2]
                    ]
                    self._U.append(U_bond_layer)
            elif len(self.model.layers) == 3:

                p2 = 1.0 / (4.0 - 4.0 ** (1 / 3.0))
                p = 1.0 - 4.0 * p2
                coefs = [p2 / 2]
                for _ in range(3):
                    coefs.extend([p2 / 2, p2])
                coefs.extend(
                    [p2 / 2, (p2 + p) / 2, p / 2, p, p / 2, (p + p2) / 2, p2 / 2]
                )
                for _ in range(3):
                    coefs.extend([p2, p2 / 2])
                coefs.append(p2 / 2)

                pattern = [0, 1, 2, 1]

                for i, tk in enumerate(coefs):
                    ly = pattern[i % 4]
                    U_bond_layer = [
                        self._calc_U_bond(i_bond, tk * delta_t)
                        for i_bond in self.model.layers[ly]
                    ]
                    self._U.append(U_bond_layer)

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def _calc_U_bond(self, i_bond, dt):
        i, j = self.model._connections[i_bond]
        h = self.model.H_bond[(i, j)]
        if h is None:
            return None  # don't calculate exp(i H t), if `H` is None
        H2 = h.combine_legs([("p0", "p1"), ("p0*", "p1*")], qconj=[+1, -1])
        # H2 = H2 - npc.diag(E_offset[i_bond], H2.legs[0])
        H2 = (-1.0j * dt) * H2
        U = npc.expm(H2)
        assert tuple(U.get_leg_labels()) == ("(p0.p1)", "(p0*.p1*)")
        return U.split_legs()

    def evolve(self, N_steps, dt):
        if dt is not None:
            assert dt == self._U_param["delta_t"]
        trunc_err = TruncationError()
        order = self._U_param["order"]
        # for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
        # for U_idx_dt, odd in self._decomp:

        for layer_no in range(len(self._U)):
            trunc_err += self.evolve_step(layer_no)

        self.evolved_time = self.evolved_time + N_steps * self._U_param["tau"]
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `evolv`)
        return trunc_err

    def evolve_step(self, layer_no):
        # trunc_par = asConfig({'chi_max':  max(max(self.psi.chi), 100)}, 'trunc_params')
        trunc_par = self.trunc_params
        # print('chi_max = ', self.trunc_params['chi_max'])
        order = self._U_param["order"]
        Us = self._U[layer_no]

        # print('len(Us) = ', len(Us))
        # print('Us = ', [Us[j] for j in range(len(Us))])
        # print('Us._data = ', [Us[j]._data for j in range(len(Us))])

        if order == 1:
            connection_numbers = self.model.layers[layer_no]
        elif order == 2:
            if layer_no == 3:
                layer_no = 1
            elif layer_no == 4:
                layer_no = 0
            connection_numbers = self.model.layers[layer_no]
        elif order == 4:
            pattern = [0, 1, 2, 1]
            connection_numbers = self.model.layers[pattern[layer_no % 4]]
        else:
            raise NotImplementedError()

        trunc_err = TruncationError()
        for k in range(len(Us)):
            U = Us[k]
            bond_no = connection_numbers[k]
            i, j = self.model._connections[bond_no]
            assert i < j
            if i + 1 == j:
                trunc_err += self.update_bond(j, U)
            else:
                perm2, a, b = self.calc_perm(i, j)
                perm1, a, b = self.calc_reverse_perm(i, j)
                # print('final_perm1 = ', final_perm1)
                self.psi.permute_sites(perm1, trunc_par=trunc_par)
                # self.psi.permute_sites(perm1)
                trunc_err += self.update_bond(j - b, U)
                self.psi.permute_sites(perm2, trunc_par=trunc_par)
                # self.psi.permute_sites(perm2)

        return trunc_err

    def calc_perm(self, i, j):
        assert i < j
        if (j - i - 1) % 2 == 0:
            a = int((j - i - 1) / 2)
            b = int((j - i - 1) / 2)
        else:
            a = int((j - i) / 2)
            b = int((j - i) / 2 - 1)
        init_perm = np.array([k for k in range(self.psi.L)])
        final_perm = init_perm.copy()

        for k in range(a):
            final_perm[i + k] = init_perm[i + k + 1]
        for k in range(b):
            final_perm[j - k] = init_perm[j - 1 - k]

        final_perm[i + a] = i
        final_perm[j - b] = j

        return final_perm, a, b

    def calc_reverse_perm(self, i, j):
        assert i < j
        if (j - i - 1) % 2 == 0:
            a = int((j - i - 1) / 2)
            b = int((j - i - 1) / 2)
        else:
            a = int((j - i) / 2)
            b = int((j - i) / 2 - 1)
        init_perm = np.array([k for k in range(self.psi.L)])
        final_perm = init_perm.copy()

        for k in range(a):
            final_perm[i + k + 1] = init_perm[i + k]
        for k in range(b):
            final_perm[j - 1 - k] = init_perm[j - k]

        final_perm[i] = i + a
        final_perm[j] = j - b

        return final_perm, a, b


class TEBDEngineHeavyHex_qc(TEBDEngineHeavyHex):
    def __init__(self, psi, model, model2, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, psi, model, options, **kwargs)
        self.model2 = model2
        self._trunc_err_bonds = [TruncationError() for i in range(psi.L + 1)]
        self._U = None
        self._U_param = {}
        self._update_index = None
