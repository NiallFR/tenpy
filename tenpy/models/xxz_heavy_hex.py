import numpy as np
from .lattice import Site, Chain
from ..tools.params import asConfig
from ..linalg import np_conserved as npc
from ..networks.site import SpinHalfSite, SpinHalfSite2
from .model import CouplingModel, MPOModel, HeavyHexModel


class XXZHeavyHex(CouplingModel, HeavyHexModel, MPOModel):
    def __init__(self, model_params, ordering=None, layers=None):
        model_params = asConfig(model_params, "XXZChain")
        L = model_params.get('L', 2)
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        phi = model_params.get('phi', 1.)
        
        bc_MPS = model_params.get('bc_MPS', 'finite')
        sort_charge = model_params.get('sort_charge', None)
        conserve = model_params.get('conserve', None)
        # 1-3):
        if conserve is not None:
            USE_PREDEFINED_SITE = False
        else: 
            USE_PREDEFINED_SITE = True
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
            #site = SpinHalfSite(conserve='Sz', sort_charge=sort_charge)
            site = SpinHalfSite2(phi, conserve='None', sort_charge=sort_charge)

            
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        PEPS_MOD_2D=False
        if not PEPS_MOD_2D:
            lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        else: #implement lattice model for 2d peps like tensor network
            lat = 'to implement'
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)
        
        connections = self.get_connections(L,ordering,layers)
        self._connections = connections
        # Edit line below and loop over all heavy hex connectivities
        connection_no = 0
        for i,j in connections:
            self.add_coupling_term(Jxx[connection_no] * 0.5, i, j, 'Sp', 'Sm', plus_hc=True)
            self.add_coupling_term(Jz[connection_no], i, j, 'Sz', 'Sz')
            connection_no += 1

        self.layers = self.get_layers(L, layers)

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
    
    
    def get_connections(self, L, ordering, layers):

        if layers != None:
            connections=layers[0]+layers[1]+layers[2]
            return connections
        if L == 2:
            connections = [(0,1)]
        elif L == 3:
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
            
        elif L == 125:
            connections_layer1 = [(4, 5),(7, 8), (9, 10),(13, 14),(15, 16), (19, 21),(11, 22),
             (23, 24),(26, 27),(29, 30),(1, 32),(33, 34),(35, 36),(37, 38),
             (40, 41), (42, 43),(45, 46),(48, 49),(20, 51),(53, 54),(55, 56),
             (57, 58),(59, 60), (61, 62),(64, 65),(67, 69),(39, 70),(72, 73),
             (89, 90),(86, 88),(63, 84),(80, 81),(68, 79),(75, 76),(93, 94),
             (95, 97),(98, 99),(82, 103),(100, 101),(104, 105),(107, 108),
             (109, 110),(96, 124),(122, 123),(119, 120),(117, 118),(115, 116)]
            
            
            connections_layer2 = [(0, 2),(3, 4),(5, 6),(8, 9),(10, 11),(12, 13),
             (16, 17),(18, 19),(21, 22),(24, 25),(27, 28),(29, 31),(32, 33),
             (38, 40),(30, 41),(43, 45),(44, 65),(46, 47),(48, 50),(49, 60),
             (51, 52),(56, 57),(58, 89),(62, 64),(66, 67),
             (69, 70),(71, 72),(73, 74),(76, 78),(77, 108),(79, 80),
             (81, 82),(83, 84),(85, 86),(90, 91),(92, 93),(95, 96),
             (87, 98),(99, 100),(102, 103),(105, 106),(110, 111),(112, 113),
             (114, 115), (116, 117),(101, 120),(121, 122)]
            
            connections_layer3 = [(0, 1),(2, 3),(5, 7),(10, 12),(14, 15),(17, 18),(19, 20),(22, 23),
             (24, 26),(6, 27),(28, 29),(31, 32),(34, 35),(36, 37),(38, 39),
             (41, 42),(43, 44),(25, 46),(47, 48),(50, 51),(52, 53),(54, 55),
             (57, 59),(60, 61),(62, 63),(65, 66),(67, 68),(70, 71),(74, 75),
             (76, 77),(78, 79),(81, 83),(84, 85),(86, 87),(88, 89),(91, 92),
             (94, 95),(97, 98),(100, 102),(103, 104),(105, 107),(108, 109),
             (111, 112),(113, 114),(106, 116),(118, 119),(120, 121),(123, 124)]
            
            connections = connections_layer1 + connections_layer2 + connections_layer3

            if ordering != None:
                print('Applying new ordering:')
                print(ordering)
                #can be optimized
                new_connections = []
                for c in connections:
                    i=ordering.index(c[0])
                    j=ordering.index(c[1])
                    if i<j:
                        new_connections.append((i,j))
                    else:
                        new_connections.append((j,i))
                
                connections = new_connections
                print('New connections:')
                print(connections)

        else:
            raise NotImplementedError()
            
        return connections
            
    def get_layers(self, L, layers=None):
        if layers!=None:
            # return a list of list of indices. The indices refers to the position
            # of the connection in the connections list.
            layers_list=[]
            connect=0
            for i in range(len(layers)):
                layers_list.append([])
                for j in range(len(layers[i])):
                    layers_list[i].append(connect)
                    connect+=1

            return layers_list
        if L == 2:
            layers = [[0]]
        elif L == 3:
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
             
        elif L == 125:
             layer1 = [k for k in range(47)]
             layer2 = [k for k in range(47,94)]
             layer3 = [k for k in range(94,142)]

             layers = [layer1, layer2, layer3]
             
        return layers
            
        