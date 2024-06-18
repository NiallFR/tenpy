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
from .mpo_tebd_qc import MPO_TEBDEngine_qc
from .tebd_heavy_hex import TEBDEngineHeavyHex
from tenpy.networks.mpo import MPO
from tenpy.linalg.charges import ChargeInfo


__all__ = ['TEBDEngine', 'Engine', 'RandomUnitaryEvolution', 'TimeDependentTEBD']

class MPO_TEBDEngine_HeavyHex(TEBDEngineHeavyHex):
    def __init__(self, F, model, layer_no, conserve, conj, options, **kwargs):
        self._layer_no = layer_no
        TimeEvolutionAlgorithm.__init__(self, F, model, options, **kwargs)
        self._U_param = {}
        self.F = F
        self.conserve = conserve
        self.conj = conj
        
        
    def set_identity_MPO(self, L, M):
        B = np.zeros([4, 2, 2, 4], dtype=float)        
        B_left = np.zeros([1, 2, 2, 4], dtype=float)
        B_right = np.zeros([4, 2, 2, 1], dtype=float)
        for i1 in range(4):
            for i2 in range(2):
                B[i1,i2,i2,i1] = 1
        for i2 in range(2):
            B_left[0,i2,i2,0] = 1
        for i2 in range(2):
            B_right[0,i2,i2,0] = 1
        labels = ['wL', 'p', 'p*', 'wR']
        
        if self.conserve:
            leg_charge1 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [-2,0,0,2], qconj=1)
            leg_charge2 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1], qconj=1)
            leg_charge3 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1], qconj=-1)
            leg_charge4 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [-2,0,0,2], qconj=-1)
            leg_charge = [leg_charge1, leg_charge2, leg_charge3, leg_charge4]
            
            leg_charge1_left = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1], qconj=1)
            leg_charge4_right = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1], qconj=-1)
            
            left_leg_charge = [leg_charge1_left, leg_charge2, leg_charge3, leg_charge4]
            right_leg_charge = [leg_charge1, leg_charge2, leg_charge3, leg_charge4_right]
            
            B_array = npc.Array.from_ndarray(B, legcharges=leg_charge, labels=labels)
            B_left_array = npc.Array.from_ndarray(B_left, legcharges=left_leg_charge, labels=labels)
            B_right_array = npc.Array.from_ndarray(B_right, legcharges=right_leg_charge, labels=labels)
        
        else:
            B_array = npc.Array.from_ndarray_trivial(B, labels=labels)
            B_left_array = npc.Array.from_ndarray_trivial(B_left, labels=labels)
            B_right_array = npc.Array.from_ndarray_trivial(B_right, labels=labels)
        
        F = MPO.from_wavepacket(M.lat.mps_sites()[0:L], [1.] * L, 'Id')
        for k in range(1,L-1):
            F.set_W(k, B_array)
        F.set_W(0, B_left_array)
        F.set_W(L-1, B_right_array)
        F.Ss = [[1.]] * L
        
        return F
        
    def svd_u_bond(self, U_bond):        
        #U_bond_matrix = U_bond.combine_legs([[0,1,2], [3,4,5]], qconj=[+1, -1])
        #print('U_bond = ', U_bond)
        U_bond_matrix = U_bond.combine_legs([[0,2], [1,3]], qconj=[+1, -1])
        U, S, V, trunc_err, renormalize = svd_theta(U_bond_matrix,
                                                    self.trunc_params,
                                                    [None, None],
                                                    inner_labels=['wR', 'wL'])
        ulabels = ['p', 'p*', 'wR']
        vlabels = ['wL','p', 'p*']
        
        U_new = U.split_legs(0).iset_leg_labels(ulabels)
        V_new = V.split_legs(1).iset_leg_labels(vlabels)
                
        S = renormalize * S
        return U_new, S, V_new, trunc_err, renormalize
    
        
    def set_USV(self, U, S, V, i, j):
        US = U.iscale_axis(S, 'wR')
        # Define trott_unit as e^{-ih_{ij}t}: first set identity as in
        # initial condition. Then set US as left most unit and V as right most

        trott_unit = self.set_identity_MPO(j-i+1, self.model)
        trott_unit.set_W(0, US)
        trott_unit.set_W(j-i, V)
        

        # Get MPO as in mpo_tebd_qc 
        for k in range(j+1-i):
            F_k = self.F.get_W(k+i)
            trott_k = trott_unit.get_W(k)
            # if F_k.chinfo!=ChargeInfo([],[]):
            #     F_k=(F_k.drop_charge(0))
            # if trott_k.chinfo!=ChargeInfo([],[]):
            #     trott_k=(trott_k.drop_charge(0))
            #print(f"start tensordot")
            if self.conj:
                trott_F_k = npc.tensordot(trott_k, F_k, axes=(['p'],['p*']))
            else:
                trott_F_k = npc.tensordot(trott_k, F_k, axes=(['p*'],['p']))
            print(f"end tensordot")
            
            
            if k == 0:
                trott_F_k.iset_leg_labels(['p','wR1','wL1','p*','wR2'])
                trott_F_k = trott_F_k.combine_legs(['wR1','wR2'], qconj=-1)
                trott_F_k.ireplace_labels(['wL1', '(wR1.wR2)'], ['wL', 'wR'])
                trott_F_k.itranspose(['wL', 'p', 'p*', 'wR'])
            elif k == j-i:
                trott_F_k.iset_leg_labels(['wL1','p','wL2','p*','wR1'])
                trott_F_k = trott_F_k.combine_legs(['wL1','wL2'], qconj=1)
                trott_F_k.ireplace_labels(['wR1', '(wL1.wL2)'], ['wR', 'wL'])
                trott_F_k.itranspose(['wL', 'p', 'p*', 'wR'])
            else:
                trott_F_k.iset_leg_labels(['wL1','p','wR1','wL2','p*','wR2'])
                trott_F_k = trott_F_k.combine_legs(['wL1','wL2'], qconj=1)
                trott_F_k = trott_F_k.combine_legs(['wR1','wR2'], qconj=-1)
                trott_F_k.ireplace_labels(['(wL1.wL2)', '(wR1.wR2)'], ['wL', 'wR'])
                trott_F_k.itranspose(['wL', 'p', 'p*', 'wR'])
            print(f"end combining step")
            self.F.set_W(k+i, trott_F_k)
            print(f"{self.F.chi}")
                    
        
    def evolve(self, N_steps, dt):
        if dt is not None:
            assert dt == self._U_param['delta_t']
        order = self._U_param['order']
        # if type(self._layer_no)!=list:
        #     layers=[self.model.layers]
        for ly in range(len(self.model.layers)):
            connection_numbers = self.model.layers[ly]
            Us = self._U[ly]
            for k in range(len(Us)):
                U_bond = Us[k]
                U, S, V, trunc_err, renormalize = self.svd_u_bond(U_bond)
                bond_no = connection_numbers[k]
                i, j = self.model._connections[bond_no]
                print(f"i: {i}, j: {j}")
                self.set_USV(U, S, V, i, j)
                #print('i = ', i)
                #print('j = ', j)
                #for k in range(self.F.L):
                    #print('k = ', k)
                    #print('F_k = ', self.F.get_W(k).shape)
                
            self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
            self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']


        return trunc_err
    
class MPO_TEBDEngine_HeavyHex_Test(MPO_TEBDEngine_HeavyHex):
    def __init__(self, F, model, layer_no, conserve, conj, options, **kwargs):
        MPO_TEBDEngine_HeavyHex.__init__(self, F, model, layer_no, conserve, conj, options, **kwargs)
        
    
    def set_identity_MPO(self, L, M):
        B = np.zeros([1, 2, 2, 1], dtype=float)       
        B[0, 0, 0, 0] = 1
        B[0, 1, 1, 0] = 1
        labels = ['wL', 'p', 'p*', 'wR']

        if self.conserve:
            leg_charge1 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1], qconj=1)
            leg_charge2 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1], qconj=1)
            leg_charge3 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1], qconj=-1)
            leg_charge4 = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1], qconj=-1)
            leg_charge = [leg_charge1, leg_charge2, leg_charge3, leg_charge4]
            B_array = npc.Array.from_ndarray(B, legcharges=leg_charge, labels=labels)
            
        else:
            B_array = npc.Array.from_ndarray_trivial(B, labels=labels)

        F = MPO.from_wavepacket(M.lat.mps_sites()[0:L], [1.] * L, 'Id')
        for k in range(0,L):
            F.set_W(k, B_array)
        F.Ss = [[1.]] * L
        
        return F
    
    def set_B_left(self):
        B = np.zeros([4, 2, 2, 1], dtype=float)       
        B[0, 0, 0, 0] = 1
        B[0, 1, 1, 0] = 1
        labels = ['wL', 'p', 'p*', 'wR']
    
    
class MPO_TEBDEngine_HeavyHex_qc(MPO_TEBDEngine_HeavyHex):
    def __init__(self, F, model, model2, conj, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, F, model, options, **kwargs)
        self.model2 = model2
        self._trunc_err_bonds = [TruncationError() for i in range(F.L + 1)]
        self._U = None
        self._U_param = {}
        self._update_index = None
        self.F = F
        self.conj = conj
        layer_no=model.layers
        conserve=False
        MPO_TEBDEngine_HeavyHex.__init__(self, F, model, layer_no, conserve, self.conj, options, **kwargs)
    
    