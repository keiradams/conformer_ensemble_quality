import torch
from torch import Tensor
from torch.nn import Linear

import torch_scatter
from torch_scatter import scatter

#from torch_geometric.nn.resolver import activation_resolver
from .models_3D.resolver import activation_resolver

from .global_attention_pool import GlobalAttentionPool, MessagePassingAttentionPool, SelfAttentionPool

from typing import Callable, Optional, Tuple, Union

class GNN3D(torch.nn.Module):    
    
    def __init__(
        self, 
        model_type = 'DimeNetPlusPlus',
        property_type = 'bond', # atom, bond, mol 
        out_emb_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 1,
        act: Union[str, Callable] = 'swish',
        atom_feature_dim: int = 53,
        use_atom_features: bool = False,
        ablate_3D: bool = False,
        ensemble_pooling: bool = False, 
        ensemble_pool_layer_type: str = 'global_attention', # 'global_attention', 'self_attention', 'message_passing'
        device = 'cpu',
    ):
        super().__init__()
        
        self.model_type = model_type
        self.property_type = property_type
        self.out_emb_channels = out_emb_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act
        self.atom_feature_dim = atom_feature_dim
        self.use_atom_features = use_atom_features
        self.ablate_3D = ablate_3D
        self.ensemble_pooling = ensemble_pooling
        self.ensemble_pool_layer_type = ensemble_pool_layer_type if ensemble_pooling else None
        
        self.device = device
                
        act = activation_resolver(act)
        
        if self.property_type in ['bond', 'atom']:
            self.property_predictor = MLP(out_emb_channels*2, hidden_channels, out_channels, act, N_hidden_layers = 2) 
        if self.property_type == 'mol':
            self.property_predictor = MLP(out_emb_channels, hidden_channels, out_channels, act, N_hidden_layers = 2)
        if self.property_type == 'bond':
            self.pairwise_permutation_invariant_MLP = MLP(2*out_emb_channels, hidden_channels, out_emb_channels, act, N_hidden_layers = 1) 
        
        if self.ensemble_pooling:
            if self.ensemble_pool_layer_type == 'global_attention':
                self.ensemble_pool_layer = GlobalAttentionPool(out_emb_channels * (1 + int(self.property_type in ['bond', 'atom'])), hidden_channels)
            if self.ensemble_pool_layer_type == 'self_attention':
                self.ensemble_pool_layer = SelfAttentionPool(out_emb_channels * (1 + int(self.property_type in ['bond', 'atom'])), hidden_channels)
            if self.ensemble_pool_layer_type == 'message_passing':
                self.ensemble_pool_layer = MessagePassingAttentionPool(out_emb_channels * (1 + int(self.property_type in ['bond', 'atom'])), hidden_channels, N_heads = 2, N_layers = 2)
        
        assert model_type in ['DimeNetPlusPlus', 'SchNet', 'EGNN', 'PAINN', 'EquiformerV2']
        
        
        if model_type == 'EquiformerV2':
            from .models_3D.equiformerV2 import EquiformerV2
            assert ablate_3D == False
            assert use_atom_features == False
            self.model = EquiformerV2(
                None, None, None,      # not used
                
                out_emb_channels = self.out_emb_channels,
                
                use_pbc=False,
                regress_forces=False,
                
                otf_graph=True,
                max_neighbors=32,
                max_radius=5.0,
                max_num_elements=90,
        
                num_layers=4,
                sphere_channels=128,
                attn_hidden_channels=64,
                num_heads=4,
                attn_alpha_channels=64, # 64
                attn_value_channels=16,
                ffn_hidden_channels=128,
                
                norm_type='layer_norm_sh',
                
                lmax_list=[4], # [6]
                mmax_list=[2],
                grid_resolution=18, 
        
                num_sphere_samples=128,
        
                edge_channels=128,
                use_atom_edge_embedding=True, 
                share_atom_edge_embedding=False,
                use_m_share_rad=False,
                distance_function="gaussian",
                num_distance_basis=128,
        
                attn_activation='silu',
                use_s2_act_attn=False, 
                use_attn_renorm=True,
                ffn_activation='silu',
                use_gate_act=False,
                use_grid_mlp=True, 
                use_sep_s2_act=True,
        
                alpha_drop=0.0,
                drop_path_rate=0.0, 
                proj_drop=0.0, 
        
                weight_init='uniform',
                
                #use_atom_features = self.use_atom_features, 
                #atom_feature_dim = self.atom_feature_dim if self.use_atom_features else 1, 
                #ablate_3D = self.ablate_3D,
            )
        
        
        if model_type == 'DimeNetPlusPlus':
            from .models_3D.dimenetpp import DimeNetPlusPlus
            self.model = DimeNetPlusPlus(
                # using default hyperparameters, with modifications for readout
                hidden_channels = 128,
                num_blocks = 4,
                int_emb_size = 128,
                basis_emb_size = 8,
                out_emb_channels = self.out_emb_channels,
                num_spherical = 7,
                num_radial = 6,
                cutoff = 5.0,
                max_num_neighbors = 32,
                envelope_exponent = 5,
                num_before_skip = 1,
                num_after_skip = 2,
                num_output_layers = 3,
                act = 'swish',
                
                use_atom_features = self.use_atom_features, 
                atom_feature_dim = self.atom_feature_dim if self.use_atom_features else 1, 
                ablate_3D = self.ablate_3D,
            )
            
        if model_type == 'SchNet':
            from .models_3D.schnet import SchNet
            self.model = SchNet(
                out_emb_channels = self.out_emb_channels,
                
                hidden_channels = 128,
                num_filters = 128,
                num_interactions = 4,
                num_gaussians = 128,
                cutoff = 5.0,
                interaction_graph = None,
                max_num_neighbors = 32,
                
                use_atom_features = self.use_atom_features, 
                atom_feature_dim = self.atom_feature_dim if self.use_atom_features else 1, 
                ablate_3D = self.ablate_3D,
            )
        
        if model_type == 'EGNN':
            from .models_3D.egnn import EGNN
            self.model = EGNN(
                
                in_node_nf = self.atom_feature_dim if self.use_atom_features else 1, # atom_feature_dim
                
                hidden_nf = 128, 
                out_node_nf = self.out_emb_channels, 
                in_edge_nf = 0, 
                cutoff = 5.0, 
                max_num_neighbors = 32, 
                device = self.device, 
                act_fn = torch.nn.SiLU(), 
                n_layers = 4, 
                residual = True, 
                attention = False, 
                normalize = True, 
                tanh = False,
        
                use_atom_features = self.use_atom_features,
                ablate_3D = self.ablate_3D,
            )
        
        if model_type == 'PAINN':
            from .models_3D.painn import PaiNN
            assert ablate_3D == False
            self.model = PaiNN(
                out_embed_dim = self.out_emb_channels,
                hidden_dim = 128,
                num_interactions = 4,
                num_rbf = 128,
                cutoff = 5.0,
                readout= 'sum',
                activation = torch.nn.SiLU(),
                max_atomic_num = 100,
                shared_interactions = False,
                shared_filters = False,
                epsilon = 1e-8,
                
                atom_feature_dim = self.atom_feature_dim if self.use_atom_features else 1, 
                use_atom_features = self.use_atom_features,
            )
        
        self.to(torch.device(self.device))
        
        
    def forward(self, data, z, pos, batch, atom_features = None, select_atom_index = None, select_bond_start_atom_index = None, select_bond_end_atom_index = None, ensemble_batch = None):
        
        # need to debug SchNet, EGNN, and especially PAINN
        if self.model_type in ['DimeNetPlusPlus', 'SchNet', 'EGNN', 'PAINN']:
            h = self.model(
                z, 
                pos, 
                batch,
                atom_features = atom_features,
            )
        if self.model_type == 'EquiformerV2':
            h = self.model(
                data,
            )
            h = h.squeeze(1)            
        
        if self.property_type == 'atom':
            out, att_scores = self.predict_atom_properties(h, batch, select_atom_index, ensemble_batch = ensemble_batch)
        elif self.property_type == 'bond':
            out, att_scores = self.predict_bond_properties(h, batch, select_bond_start_atom_index, select_bond_end_atom_index, ensemble_batch = ensemble_batch)
        elif self.property_type == 'mol':
            out, att_scores = self.predict_mol_properties(h, batch, ensemble_batch = ensemble_batch)
        
        return out, h, att_scores
    
    
    def predict_atom_properties(self, h, batch, select_atom_index, ensemble_batch = None):
        h_select = h[select_atom_index] # subselecting only certain atoms
        h_agg = torch_scatter.scatter_add(h, batch, dim = 0) # sum pooling over all atom embeddings to get molecule-level embedding
        
        h_out = torch.cat([h_select, h_agg], dim = 1)
        
        if self.ensemble_pooling:
            h_out, att_scores = self.ensemble_pool_layer(h_out, ensemble_batch)
        else:
            att_scores = None
        
        out = self.property_predictor(h_out)
        
        return out, att_scores
    
    
    def predict_bond_properties(self, h, batch, select_bond_start_atom_index, select_bond_end_atom_index, ensemble_batch = None):        
        h_start = h[select_bond_start_atom_index] # selecting atoms that start each bond
        h_end = h[select_bond_end_atom_index] # selecting atoms that end each bond
        h_agg = torch_scatter.scatter_add(h, batch, dim = 0) # sum pooling over all atom embeddings to get molecule-level embedding
        h_bond = self.pairwise_permutation_invariant_MLP(torch.cat([h_start, h_end], dim = 1)) + self.pairwise_permutation_invariant_MLP(torch.cat([h_end, h_start], dim = 1))
        
        h_out = torch.cat([h_bond, h_agg], dim = 1)
        
        if self.ensemble_pooling:
            h_out, att_scores = self.ensemble_pool_layer(h_out, ensemble_batch)
        else:
            att_scores = None
        
        out = self.property_predictor(h_out)
        
        return out, att_scores
    
    
    def predict_mol_properties(self, h, batch, ensemble_batch = None):
        h_out = torch_scatter.scatter_add(h, batch, dim = 0) # sum pooling over all atom embeddings to get molecule-level embedding
        
        if self.ensemble_pooling:
            h_out, att_scores = self.ensemble_pool_layer(h_out, ensemble_batch)
        else:
            att_scores = None
        
        out = self.property_predictor(h_out)
        
        return out, att_scores


class MLP(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, act: Callable, N_hidden_layers = 2):
        super().__init__()
        self.act = act
        self.N_hidden_layers = N_hidden_layers
        
        if N_hidden_layers == 0:
            self.output_layer = Linear(input_channels, output_channels) # no activation
        else:
            self.input_layer = Linear(input_channels, hidden_channels) # this counts as 1 hidden layer
            self.lin_layers = torch.nn.ModuleList([
                Linear(hidden_channels, hidden_channels) for _ in range(N_hidden_layers - 1)
            ])
            self.output_layer = Linear(hidden_channels, output_channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.N_hidden_layers == 0:
            return self.output_layer(x)
        
        x = self.act(self.input_layer(x))
        for layer in self.lin_layers:
            x = self.act(layer(x))
        out = self.output_layer(x)
        
        return out