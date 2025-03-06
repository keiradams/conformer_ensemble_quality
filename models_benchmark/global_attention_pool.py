import torch
import torch_scatter
import torch_geometric

# this is a self-gating mechanism, and does not consider the relationships between the conformers (apart from the softmax).
# we could add-in these relationships by using a transformer layer to mix information between the conformers.
# could we add an L1 penalty (regularization) to encourage the network to sparsify scores ? -> this would only work if we added-in relationships between the conformers, as the network would have to pick amongst the options presented to it. 
# I do wonder if this would be a strategy as to "learn" conformational flexibility when using multiple explicit conformers

class GlobalAttentionPool(torch.nn.Module):
    def __init__(self, in_out_dimension, hidden_dimension):
        super(GlobalAttentionPool, self).__init__()
        
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_out_dimension, hidden_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dimension, 1),
        )
        
    def forward(self, h, batch):
        # batch has shape (B)
        # h has shape (B x F)
        
        h_gate = self.gate(h) # (B x F) -> (B X 1)
        scores = torch_scatter.scatter_softmax(h_gate, batch, dim = 0) # (B X 1) -> (B X 1)
        
        h_pool = torch_scatter.scatter_add(h * scores, batch, dim = 0) # (B x  max(batch) + 1)
        
        return h_pool, scores


class MessagePassingAttentionPool(torch.nn.Module):
    def __init__(self, in_out_dimension, hidden_dimension, N_heads = 2, N_layers = 2):
        super(MessagePassingAttentionPool, self).__init__()
        
        self.ensemble_GNN = torch.nn.ModuleList([])
        for n in range(N_layers):
            self.ensemble_GNN.append(
                torch_geometric.nn.GATv2Conv(
                    in_channels = in_out_dimension,
                    out_channels = in_out_dimension,
                    heads = N_heads,
                    concat = False,
                    negative_slope = 0.2,
                    dropout = 0.0,
                    add_self_loops = True,
                    bias = True,
                )
            )

        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_out_dimension, hidden_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dimension, 1),
        )

    def forward(self, H, batch):
        # batch has shape (B)
        # H has shape (B x F)

        # this module is intended to pass messages between conformers of the same molecule in a permutation invariant way.
        # Hence, we first form a conformer ensemble fully-connected graph, do any kind of message passing (e.g., GAT2 for attention-based message passing), and obtain mixed-conformer level embeddings
        # Then, we should perform a learned weighted average over the conformer "node" features, as in GlobalAttentionPool, and output ensemble-level / molecule features.

        # create ensemble_edge_index (fully connected graphs for each molecule in the batch) from batch
        ensemble_adj = (batch.unsqueeze(0) == batch.unsqueeze(1)).long()
        ensemble_adj.fill_diagonal_(0)
        ensemble_edge_index = torch_geometric.utils.dense_to_sparse(ensemble_adj)[0] # get edge index only, ignore weights
        
        # pass H and ensemble_edge_index into any PyG message passing framework to mix information between conformers.
        H_mixed = H
        for GNN in self.ensemble_GNN:
            H_mixed = GNN(H_mixed, ensemble_edge_index)

        # GlobalAttentionPooling on conformers (instead of just sum-pooling node embeddings, etc.)
        h_gate = self.gate(H_mixed) # (B x F) -> (B X 1)
        scores = torch_scatter.scatter_softmax(h_gate, batch, dim = 0) # (B X 1) -> (B X 1)
        h_pool = torch_scatter.scatter_add(H_mixed * scores, batch, dim = 0) # (B x  max(batch) + 1)
        
        return h_pool, scores



class SelfAttentionPool(torch.nn.Module):
    def __init__(self, in_out_dimension, hidden_dimension):
        super(SelfAttentionPool, self).__init__()
        
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(in_out_dimension, in_out_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_out_dimension, in_out_dimension),
            torch.nn.LeakyReLU(),
        )

        self.attention = torch.nn.Linear(in_out_dimension, in_out_dimension)

        self.rho = torch.nn.Sequential(
            torch.nn.Linear(in_out_dimension, in_out_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_out_dimension, in_out_dimension),
            torch.nn.LeakyReLU(),
        )

        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_out_dimension, hidden_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dimension, 1),
        )

        
    def forward(self, H, batch):
        # batch has shape (B)
        # H has shape (B x F)

        # essentially apply dot-product attention across conformers to mix their representations, and then apply GlobalAttentionPool to get attention scores and final molecule embeddings.

        H = self.phi(H)
        attention_vecs = self.attention(H)
        
        dot_product = torch.matmul(attention_vecs, attention_vecs.transpose(1,0))
        
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
        max_values = (dot_product * mask).max(dim = 1, keepdim = True).values

        masked_dot_product = (dot_product - max_values) * mask

        attention_weights = masked_dot_product.exp() / (masked_dot_product.exp() * mask).sum(dim = 1, keepdim = True)
        attention_weights = attention_weights * mask
        
        H_mixed = torch.matmul(attention_weights, H)

        # sum pooling followed by nonlinearity
        #H_pool = torch.scatter(H_mixed, batch, dim = 0, reduce = 'sum') 
        #H_pool = self.rho(H_pool)

        # GlobalAttentionPooling on conformers
        H_mixed = self.rho(H_mixed)
        h_gate = self.gate(H_mixed)
        scores = torch_scatter.scatter_softmax(h_gate, batch, dim = 0) # (B X 1) -> (B X 1)
        h_pool = torch_scatter.scatter_add(H_mixed * scores, batch, dim = 0) # (B x  max(batch) + 1)


        return h_pool, scores