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

    def log_sinkhorn(self, log_alpha, n_iters):
        """
        Applies Sinkhorn algorithm in log-space to produce a doubly stochastic matrix.
        Crucial for encouraging equal-sized clusters.
        """
        batch_size, n, k = log_alpha.size()
        log_P = log_alpha

        
        target_col_sum = torch.log(torch.tensor(n / k, device=log_alpha.device))
        
        for _ in range(n_iters):
            log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)

            current_col_sum = torch.logsumexp(log_P, dim=1, keepdim=True)
            log_P = log_P + (target_col_sum - current_col_sum)
            
        return log_P

    def get_balanced_assignments(self, probs, greedy=False):
        """
        Assignation globale basée sur la probabilité maximale absolue.
        Version optimisée pour garantir le tracking du gradient (Autograd safe).
        """
        B, N, K = probs.size()
        device = probs.device
        target_size = N // K
        
        assignments = torch.zeros(B, N, dtype=torch.long, device=device)
        node_assigned = torch.zeros(B, N, dtype=torch.bool, device=device)
        cluster_counts = torch.zeros(B, K, dtype=torch.long, device=device)

        flat_probs = probs.view(B, -1)
        sorted_probs, sorted_indices = torch.sort(flat_probs, dim=1, descending=True)
        
        nodes = sorted_indices // K 
        clusters = sorted_indices % K 
        
        batch_idx = torch.arange(B, device=device)
        
        for i in range(N * K):
            n = nodes[:, i]
            k = clusters[:, i]
            
            valid_node = ~node_assigned[batch_idx, n]
            valid_cluster = cluster_counts[batch_idx, k] < target_size
            valid = valid_node & valid_cluster
            
            if valid.any():
                valid_b = batch_idx[valid]
                valid_n = n[valid]
                valid_k = k[valid]
                
                assignments[valid_b, valid_n] = valid_k
                node_assigned[valid_b, valid_n] = True
                cluster_counts[valid_b, valid_k] += 1
            
            if node_assigned.all():
                break

        if not greedy:
            chosen_probs = torch.gather(probs, 2, assignments.unsqueeze(-1)).squeeze(-1)

            log_probs_total = torch.log(chosen_probs + 1e-10).sum(dim=1) # [B]
        else:
            log_probs_total = torch.zeros(B, device=device)
        
        return assignments, log_probs_total

    def forward(self, batch, greedy=False, pomo_starts=None):
        coords = batch['coordinates']
        B, Total, _ = coords.size()
        N_DUMMIES = Total - self.opts.graph_size
        
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
            #sans pomo
            last_nodes = [self.cluster_queries[k].unsqueeze(0).expand(B, -1) for k in range(K)]
            steps_to_do = target_size

        for step in range(steps_to_do):
            
            #Round robin
            turn_order = torch.randperm(K, device=h.device)
            
            for i in range(K):
                k = turn_order[i].item()
                
                cluster_q = self.cluster_queries[k].unsqueeze(0).expand(B, -1)
                last_node_embed = last_nodes[k]

                context = torch.cat([graph_context, cluster_q, last_node_embed], dim=-1)
                query = self.project_context(context).unsqueeze(1)

                scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)

                scores[visited_mask] = -float('inf')

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
        
        N_DUMMIES = self.opts.dummy_rate * self.opts.graph_size
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
