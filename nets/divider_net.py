import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class NeuralDivider(nn.Module):
    def __init__(self, opts, input_dim=3, hidden_dim=128, n_sinkhorn_iters=20, tau=1.0):
        """
        Neural Divider based on Sinkhorn to force balanced clusters.
        
        Args:
            input_dim: 3 (x, y, demand)
            n_splits: Number of partitions (e.g., 4)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_splits = opts.dnc_n_splits
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.tau = tau 
        self.opts = opts

        # Embedding
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Decoder
        self.cluster_queries = nn.Parameter(torch.randn(self.n_splits, hidden_dim))
        self.project_context = nn.Linear(hidden_dim * 3, hidden_dim)
        self.project_k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch, greedy=False, pomo_starts=None):
        coords = batch['coordinates']
        B, Total, _ = coords.size()
        
        # 1. Calcul standardisé des Dummies
        N_DUMMIES = int(self.opts.dummy_rate * self.opts.graph_size)
        
        clients_coords = coords[:, N_DUMMIES:, :] 
        clients_demand = batch['demand'][:, N_DUMMIES:]
        d = clients_demand.unsqueeze(-1)
        
        x = torch.cat([clients_coords, d], dim=-1)
        
        h = self.encoder(self.embed(x)) 
        graph_context = h.mean(dim=1) 
        keys = self.project_k(h) 

        B, N, _ = h.size()
        K = self.n_splits
        target_size = N // K

        assignments = torch.zeros(B, N, dtype=torch.long, device=h.device)
        visited_mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
        
        log_probs_total = torch.zeros(B, device=h.device)

        if pomo_starts is not None:
            last_nodes = []
            for k in range(K):
                start_node_idx = pomo_starts[:, k]

                last_nodes.append(h[torch.arange(B), start_node_idx])
                assignments[torch.arange(B), start_node_idx] = k
                visited_mask[torch.arange(B), start_node_idx] = True

            steps_to_do = target_size - 1
        else:
            last_nodes = [self.cluster_queries[k].unsqueeze(0).expand(B, -1) for k in range(K)]
            steps_to_do = target_size

        for step in range(steps_to_do):
            turn_order = torch.randperm(K, device=h.device)
            
            for i in range(K):
                k = turn_order[i].item()
                
                cluster_q = self.cluster_queries[k].unsqueeze(0).expand(B, -1)
                last_node_embed = last_nodes[k]

                context = torch.cat([graph_context, cluster_q, last_node_embed], dim=-1)
                query = self.project_context(context).unsqueeze(1)

                scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)

                T = 1.0 if greedy else self.opts.temperature
                scores = scores / T

                # --- FIX AUTOGRAD (Contre le crash in-place) ---
                scores = scores.masked_fill(visited_mask, float('-inf'))

                probs = F.softmax(scores, dim=1)
                dist = Categorical(probs)

                if greedy:
                    selected = probs.argmax(dim=1)
                else:
                    selected = dist.sample()
                    log_probs_total += dist.log_prob(selected)

                assignments.scatter_(1, selected.unsqueeze(1), k)
                
                visited_mask = visited_mask.clone()
                visited_mask.scatter_(1, selected.unsqueeze(1), True)

                last_nodes[k] = h[torch.arange(B), selected]

        return assignments, log_probs_total

    def normalize(self, coords):
        """
        Normalize coordinates to [0, 1] preserving aspect ratio.
        coords: [B_total, N_sub, 2]
        """
        min_vals, _ = coords.min(dim=1, keepdim=True)
        max_vals, _ = coords.max(dim=1, keepdim=True)
        ranges = max_vals - min_vals
        scale, _ = ranges.max(dim=2, keepdim=True)
        scale = torch.clamp(scale, min=1e-8)
        return (coords - min_vals) / scale, scale.squeeze(-1)

    def make_sub_batch(self, batch, assignments):
        """
        Construit les sous-instances de manière vectorisée pour le Rollout.
        Compatible avec NeuOpt/POMO (Tenseurs denses).
        
        Args:
            batch: {'coordinates': [B, 600, 2], 'demand': [B, 600]} (200 Dummies + 400 Clients)
            assignments: [B, 400] (Indices de split 0..3 pour les clients)
        
        Returns:
            Dict compatible avec rollout()
        """
        device = batch['coordinates'].device
        B = batch['coordinates'].size(0)
        
        N_DUMMIES = int(self.opts.dummy_rate * self.opts.graph_size)
        N_CLIENTS = self.opts.graph_size
        K = self.n_splits
        
        DUMMIES_PER_SPLIT = N_DUMMIES // K # 50
        CLIENTS_PER_SPLIT = N_CLIENTS // K # 100
        SUB_SIZE = DUMMIES_PER_SPLIT + CLIENTS_PER_SPLIT # 150
        
        all_dummies_coords = batch['coordinates'][:, :N_DUMMIES, :]
        all_dummies_demand = batch['demand'][:, :N_DUMMIES]
        all_dummies_idx = torch.arange(N_DUMMIES, device=device).view(1, N_DUMMIES).expand(B, -1)

        # [B, 400, 2]
        all_clients_coords = batch['coordinates'][:, N_DUMMIES:, :]
        all_clients_demand = batch['demand'][:, N_DUMMIES:]
        all_clients_idx = torch.arange(N_DUMMIES, N_DUMMIES + N_CLIENTS, device=device).view(1, N_CLIENTS).expand(B, -1)

        dummies_coords_split = all_dummies_coords.view(B, K, DUMMIES_PER_SPLIT, 2)
        dummies_demand_split = all_dummies_demand.view(B, K, DUMMIES_PER_SPLIT)
        dummies_idx_split = all_dummies_idx.view(B, K, DUMMIES_PER_SPLIT)

        sort_idx = torch.argsort(assignments, dim=1)
        
        gather_idx_coords = sort_idx.unsqueeze(-1).expand(-1, -1, 2)
        sorted_clients_coords = torch.gather(all_clients_coords, 1, gather_idx_coords)
        
        sorted_clients_demand = torch.gather(all_clients_demand, 1, sort_idx)
        sorted_clients_idx = torch.gather(all_clients_idx, 1, sort_idx)

        clients_coords_split = sorted_clients_coords.view(B, K, CLIENTS_PER_SPLIT, 2)
        clients_demand_split = sorted_clients_demand.view(B, K, CLIENTS_PER_SPLIT)
        clients_idx_split = sorted_clients_idx.view(B, K, CLIENTS_PER_SPLIT)
        
        final_coords = torch.cat([dummies_coords_split, clients_coords_split], dim=2)
        final_demands = torch.cat([dummies_demand_split, clients_demand_split], dim=2)
        final_indices = torch.cat([dummies_idx_split, clients_idx_split], dim=2)

        batch_coords = final_coords.reshape(B * K, SUB_SIZE, 2)
        batch_demands = final_demands.reshape(B * K, SUB_SIZE)
        batch_indices = final_indices.reshape(B * K, SUB_SIZE)
        
        batch_coords_norm, norm_factor = self.normalize(batch_coords)
        
        return {
            'coordinates': batch_coords_norm,   # [B*K, 150, 2] (normalized)
            'demand': batch_demands,            # [B*K, 150]
            'original_indices': batch_indices,  # [B*K, 150] 
            'norm_factor': norm_factor          # [B*K, 1, 1]
        }
