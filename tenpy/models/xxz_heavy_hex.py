import numpy as np
from .lattice import Site, Chain
from ..tools.params import asConfig
from ..linalg import np_conserved as npc
from ..networks.site import SpinHalfSite 
from .model import CouplingModel, MPOModel, HeavyHexModel


class XXZHeavyHex(CouplingModel, HeavyHexModel, MPOModel):
    def __init__(self, model_params):
        model_params = asConfig(model_params, "XXZChain")
        L = model_params.get('L', 2)
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        
        bc_MPS = model_params.get('bc_MPS', 'finite')
        sort_charge = model_params.get('sort_charge', None)
        # 1-3):
        USE_PREDEFINED_SITE = False
        if not USE_PREDEFINED_SITE:
            # 1) charges of the physical leg. The only time that we actually define charges!
            leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1])
            # 2) onsite operators
            Sp = [[0., 1.], [0., 0.]]
            Sm = [[0., 0.], [1., 0.]]
            Sz = [[0.5, 0.], [0., -0.5]]
            # (Can't define Sx and Sy as onsite operators: they are incompatible with Sz charges.)
            # 3) local physical site
            site = Site(leg, ['up', 'down'], sort_charge=sort_charge, Sp=Sp, Sm=Sm, Sz=Sz)
        else:
            # there is a site for spin-1/2 defined in TeNPy, so just we can just use it
            # replacing steps 1-3)
            site = SpinHalfSite(conserve='Sz', sort_charge=sort_charge)
            
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)
        
        connections = self.get_connections(L)
        self._connections = connections
        # Edit line below and loop over all heavy hex connectivities
        connection_no = 0
        for i,j in connections:
            self.add_coupling_term(Jxx[connection_no] * 0.5, i, j, 'Sp', 'Sm', plus_hc=True)
            self.add_coupling_term(Jz[connection_no], i, j, 'Sz', 'Sz')
            connection_no += 1

        self.layers = self.get_layers(L)

        HeavyHexModel.__init__(self, lat, self.calc_H_bond(connections))
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        
    def calc_H_bond(self, connections, tol_zero=1.e-15):

        # Edit below and implement to_heavy_hex_bond_Arrays function in
        # CouplingTerms class in terms.py
    
        sites = self.lat.mps_sites()
        finite = (self.lat.bc_MPS == 'finite')
    
        ct = self.all_coupling_terms()
        ct.remove_zeros(tol_zero)
        H_bond = ct.to_heavy_hex_bond_Arrays(sites, connections)
    
        if self.explicit_plus_hc:
            # self representes the terms of `ct` and `ot` + their hermitian conjugates
            # so we need to explicitly add the hermitian conjugate terms
            for (i,j), Hb in H_bond.items():
                if Hb is not None:
                    H_bond[(i,j)] = Hb + Hb.conj().itranspose(Hb.get_leg_labels())
        return H_bond
    
    
    def get_connections(self, L):
        if L == 3:
            connections = [(0,2)]
            
        elif L == 4:
            connections = [(0,1),(1,2),(2,3),(1,3)]
        
        
        elif L == 6:
            connections = [(0,1),(1,2),(2,3),(3,4),(4,5)]
            
            
        elif L == 8:
            connections = [(0,1),(1,3),(3,4),(4,5),(2,3),(2,7),(6,7)]
         
        elif L == 10:
            connections = [(0,1),(1,3),(3,4),(4,5),(2,3),
                           (2,7),(6,7),(7,8),(8,9)]  
        
            
        elif L == 24:
            connections = [(0,1),(1,3),(2,3),(2,14),(3,4),
                                (4,5),(5,6),(6,7),(7,8),(8,9),
                                (9,10),(10,11),(11,12),(11,13),(12,23),
                                (13,14),(14,15),(15,16),(16,17),(17,18),
                                (18,19),(19,20),(20,21),(20,22),(22,23)]
        else:
            raise NotImplementedError()
            
        return connections
            
    def get_layers(self, L):
        if L == 3:
            layers = [[0]]
            
        elif L == 4:
            layers = [[0,2],[1],[3]]
            
        elif L == 6:
            layers = [[0,2,4],[1,3]]
            
        elif L == 8:
            layers = [[0,2,5],[3,4],[1,6]]
            
        elif L == 10:
            layers = [[0,2,5],[3,4,7],[1,6,8]]
            
        elif L == 24:
             layer1 = [0,3,7,9,11,14,19,21]
             layer2 = [1,5,8,10,13,16,18,20,23]
             layer3 = [2,4,6,12,15,17,22,24]
             layers = [layer1, layer2, layer3]
             
        return layers
            
        