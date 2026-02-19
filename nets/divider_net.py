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

        # 1. Feature Embedding (Coordinates + Demand)
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # 2. Encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 3. Logits Head
        self.to_logits = nn.Linear(hidden_dim, self.n_splits) 

    def log_sinkhorn(self, log_alpha, n_iters):
        """
        Applies Sinkhorn algorithm in log-space to produce a doubly stochastic matrix.
        Crucial for encouraging equal-sized clusters.
        """
        # log_alpha: [Batch, N_clients, K_splits]
        batch_size, n, k = log_alpha.size()
        log_P = log_alpha
        
        # columns should sum to N / K.
        
        target_col_sum = torch.log(torch.tensor(n / k, device=log_alpha.device))
        
        for _ in range(n_iters):
            # 1. Row Normalization (Each client chooses exactly 1 cluster)
            # log_P shape: [B, N, K]
            log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)
            
            # 2. Column Normalization (Each cluster receives N/K clients)
            current_col_sum = torch.logsumexp(log_P, dim=1, keepdim=True)
            log_P = log_P + (target_col_sum - current_col_sum)
            
        return log_P

    def get_balanced_assignments(self, probs, greedy=False):
        """
        Assignation globale basée sur la probabilité maximale absolue.
        Aucun cluster n'est favorisé : les assignations se font dans l'ordre 
        strict des probabilités décroissantes, tout en respectant les capacités.
        """
        B, N, K = probs.size()
        device = probs.device
        target_size = N // K
        
        # Initialisations
        assignments = torch.zeros(B, N, dtype=torch.long, device=device)
        log_probs_total = torch.zeros(B, device=device)
        
        # Suivi des contraintes
        node_assigned = torch.zeros(B, N, dtype=torch.bool, device=device)
        cluster_counts = torch.zeros(B, K, dtype=torch.long, device=device)
        
        # --- 1. TRI GLOBAL ---
        # On aplatit les probabilités [B, N*K] pour les trier globalement
        flat_probs = probs.view(B, -1)
        sorted_probs, sorted_indices = torch.sort(flat_probs, dim=1, descending=True)
        
        # On décode les indices plats pour retrouver le Nœud (n) et le Cluster (k)
        # Exemple: index 5 avec K=4 -> Noeud 1 (5//4), Cluster 1 (5%4)
        nodes = sorted_indices // K  # [B, N*K]
        clusters = sorted_indices % K # [B, N*K]
        
        batch_idx = torch.arange(B, device=device)
        
        # --- 2. ASSIGNATION ITÉRATIVE PAR ORDRE DE CONFIANCE ---
        # On parcourt les choix du plus probable au moins probable
        for i in range(N * K):
            # Pour chaque élément du batch, on regarde son i-ème choix préféré
            n = nodes[:, i]
            k = clusters[:, i]
            
            # Vérification des contraintes :
            # 1. Le noeud 'n' ne doit pas être déjà assigné
            valid_node = ~node_assigned[batch_idx, n]
            # 2. Le cluster 'k' ne doit pas être plein
            valid_cluster = cluster_counts[batch_idx, k] < target_size
            
            # Le mouvement est valide si les deux contraintes sont respectées
            valid = valid_node & valid_cluster
            
            if valid.any():
                # On applique l'assignation uniquement pour les éléments valides du batch
                valid_b = batch_idx[valid]
                valid_n = n[valid]
                valid_k = k[valid]
                
                # Mise à jour de la solution et des masques
                assignments[valid_b, valid_n] = valid_k
                node_assigned[valid_b, valid_n] = True
                cluster_counts[valid_b, valid_k] += 1
                
                # Accumulation de la log-probabilité pour l'apprentissage (REINFORCE)
                if not greedy:
                    valid_probs = probs[valid_b, valid_n, valid_k]
                    log_probs_total[valid_b] += torch.log(valid_probs + 1e-10)
            
            # Condition d'arrêt anticipée (Optimisation de vitesse) :
            # Si tous les noeuds de tout le batch sont assignés, on arrête la boucle
            if node_assigned.all():
                break
        
        return assignments, log_probs_total

    def forward(self, batch, greedy=False):
        """
        Args:
            coords: [Batch, N_clients, 2] (No dummies here)
            demand: [Batch, N_clients]
        """
        # 1. Input concatenation [B, N, 3]
        N_DUMMIES = int(self.opts.dummy_rate * self.opts.graph_size) # 200 if graph_size=400
        clients_coords = batch['coordinates'][:, N_DUMMIES:, :] # [B, 400, 2]
        clients_demand = batch['demand'][:, N_DUMMIES:]         # [B, 400]
        if clients_demand.dim() == 2:
            d = clients_demand.unsqueeze(-1)
        else:
            d = clients_demand
        x = torch.cat([clients_coords, d], dim=-1)
        
        # 2. Encode
        h = self.embed(x)
        h = self.encoder(h) # [B, N, H]
        
        # 3. Logits & Sinkhorn
        logits = self.to_logits(h) # [B, N, K]
        if not greedy:
            u = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            logits_noisy = (logits + gumbel) / self.tau
        else:
            logits_noisy = logits / self.tau
        
        # Apply Sinkhorn to encourage balanced scores
        log_P = self.log_sinkhorn(logits_noisy, self.n_sinkhorn_iters)
        probs = torch.exp(log_P)
        
        # 4. Assignment with strict size constraints
        # We generally use Greedy=True during inference to get stable splits
        # During training, we might add noise to logits before Sinkhorn if we want exploration,
        # but here we rely on the probabilistic nature of the output for the loss.
        assignments, log_probs_sum, entropy = self.get_balanced_assignments(probs, greedy=greedy)
            
        return assignments, log_probs_sum, entropy

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
        
        # Constantes (à adapter si nécessaire via self.opts)
        N_DUMMIES = 200 # ou int(self.opts.dummy_rate * self.opts.graph_size)
        N_CLIENTS = 400 # ou self.opts.graph_size
        K = self.n_splits
        
        DUMMIES_PER_SPLIT = N_DUMMIES // K # 50
        CLIENTS_PER_SPLIT = N_CLIENTS // K # 100
        SUB_SIZE = DUMMIES_PER_SPLIT + CLIENTS_PER_SPLIT # 150
        
        # --- 1. Séparation Dummies / Clients ---
        # [B, 200, 2]
        all_dummies_coords = batch['coordinates'][:, :N_DUMMIES, :]
        all_dummies_demand = batch['demand'][:, :N_DUMMIES]
        # Indices globaux des dummies : 0 à 199
        all_dummies_idx = torch.arange(N_DUMMIES, device=device).view(1, N_DUMMIES).expand(B, -1)

        # [B, 400, 2]
        all_clients_coords = batch['coordinates'][:, N_DUMMIES:, :]
        all_clients_demand = batch['demand'][:, N_DUMMIES:]
        # Indices globaux des clients : 200 à 599
        all_clients_idx = torch.arange(N_DUMMIES, N_DUMMIES + N_CLIENTS, device=device).view(1, N_CLIENTS).expand(B, -1)

        # --- 2. Traitement des Dummies (Statique) ---
        # On découpe simplement les 200 dummies en K blocs de 50
        # [B, 200, 2] -> [B, K, 50, 2]
        dummies_coords_split = all_dummies_coords.view(B, K, DUMMIES_PER_SPLIT, 2)
        dummies_demand_split = all_dummies_demand.view(B, K, DUMMIES_PER_SPLIT)
        dummies_idx_split = all_dummies_idx.view(B, K, DUMMIES_PER_SPLIT)

        # --- 3. Traitement des Clients (Dynamique via Assignments) ---
        # Astuce : On trie les clients selon leur assignation (Split 0, puis Split 1...)
        # Cela regroupe les clients par cluster sans boucle for.
        # sort_idx: [B, 400]
        sort_idx = torch.argsort(assignments, dim=1)
        
        # On réordonne les données clients selon ce tri
        # Gather sur la dimension 1 (N_nodes)
        
        # Coords : [B, 400, 2]
        gather_idx_coords = sort_idx.unsqueeze(-1).expand(-1, -1, 2)
        sorted_clients_coords = torch.gather(all_clients_coords, 1, gather_idx_coords)
        
        # Demands & Indices : [B, 400]
        sorted_clients_demand = torch.gather(all_clients_demand, 1, sort_idx)
        sorted_clients_idx = torch.gather(all_clients_idx, 1, sort_idx)
        
        # Maintenant on reshape pour avoir la dimension K
        # [B, 400, ...] -> [B, K, 100, ...]
        clients_coords_split = sorted_clients_coords.view(B, K, CLIENTS_PER_SPLIT, 2)
        clients_demand_split = sorted_clients_demand.view(B, K, CLIENTS_PER_SPLIT)
        clients_idx_split = sorted_clients_idx.view(B, K, CLIENTS_PER_SPLIT)

        # --- 4. Fusion (Concatenation) ---
        # On colle les Dummies et les Clients sur la dimension "Sub_Size" (dim 2)
        # [B, K, 50] + [B, K, 100] -> [B, K, 150]
        
        final_coords = torch.cat([dummies_coords_split, clients_coords_split], dim=2)
        final_demands = torch.cat([dummies_demand_split, clients_demand_split], dim=2)
        final_indices = torch.cat([dummies_idx_split, clients_idx_split], dim=2)

        # --- 5. Aplatissement (Flatten) pour le Rollout ---
        # Le rollout attend [B*K, Sub_Size, ...]
        
        batch_coords = final_coords.reshape(B * K, SUB_SIZE, 2)
        batch_demands = final_demands.reshape(B * K, SUB_SIZE)
        batch_indices = final_indices.reshape(B * K, SUB_SIZE)
        
        # --- 6. Normalisation ---
        # Utilise votre méthode normalize (qui doit renvoyer [B*K, 1, 1] pour le facteur)
        batch_coords_norm, norm_factor = self.normalize(batch_coords)
        
        return {
            'coordinates': batch_coords_norm,   # [B*K, 150, 2] (Normalisé)
            'demand': batch_demands,            # [B*K, 150]
            'original_indices': batch_indices,  # [B*K, 150] (Globaux : 0-599)
            'norm_factor': norm_factor          # [B*K, 1, 1]
        }
        
class NeuralDividerImproved(NeuralDivider):
    def __init__(self, opts, input_dim=3, hidden_dim=128, n_sinkhorn_iters=20, tau=1.0):
        # 1. Initialise la classe parente (récupère opts, n_splits, embed, etc.)
        super().__init__(opts, input_dim, hidden_dim, n_sinkhorn_iters, tau)
        
        # --- AMÉLIORATIONS ARCHITECTURALES (Surcharge les attributs parents) ---
        
        # A. Encodeur plus profond et mieux normalisé (NormFirst=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            batch_first=True, 
            norm_first=True # Aide à la convergence
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4) # 4 couches au lieu de 3

        # B. Remplacement de la tête linéaire par des Prototypes + Attention
        # On supprime l'ancienne couche linéaire si elle existe pour économiser la mémoire
        if hasattr(self, 'to_logits'):
            del self.to_logits
            
        # Nouveaux paramètres : Queries (Centres de clusters apprenables)
        self.cluster_queries = nn.Parameter(torch.randn(1, self.n_splits, hidden_dim))
        
        # Cross-Attention : Les Clusters regardent les Noeuds
        self.decoder_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def get_balanced_assignments(self, probs, greedy=False):
        """
        Surcharge de la méthode parente pour inclure le SHUFFLE des clusters.
        Cela évite que le cluster 0 soit toujours servi en premier et soit le 'meilleur'.
        """
        B, N, K = probs.size()
        device = probs.device
        target_size = N // K
        
        assignments = torch.zeros(B, N, dtype=torch.long, device=device)
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        log_probs_total = torch.zeros(B, device=device)
        
        remaining_probs = probs.clone()
        
        # --- AMÉLIORATION : Ordre aléatoire ---
        # On mélange l'ordre de traitement [0, 1, 2, 3] -> [2, 0, 3, 1]
        cluster_order = torch.randperm(K, device=device)
        
        for i in range(K):
            k = cluster_order[i].item()
            
            p_k = remaining_probs[:, :, k]
            p_k[selected_mask] = -1.0 # Masque les noeuds déjà pris
            
            # Sélection des meilleurs candidats pour ce cluster
            _, top_indices = torch.topk(p_k, k=target_size, dim=1)
            
            assignments.scatter_(1, top_indices, k)
            selected_mask.scatter_(1, top_indices, True)
            
            if not greedy:
                # Accumulation des log-probs pour le gradient RL
                selected_probs = torch.gather(probs[:, :, k], 1, top_indices)
                log_probs_total += torch.log(selected_probs + 1e-10).sum(dim=1)

        # Calcul d'entropie pour monitoring
        dist = Categorical(probs)
        entropy = dist.entropy().mean(dim=1)
        
        return assignments, log_probs_total, entropy

    def forward(self, batch, greedy=False):
        """
        Surcharge du forward pour inclure :
        1. L'Attention Decoder
        2. Le Gumbel Noise (Exploration)
        """
        # 1. Préparation Input (Logique identique, on recalcule N_DUMMIES dynamiquement)
        coords = batch['coordinates']
        B, Total, _ = coords.size()
        N_DUMMIES = Total // 3 # Hypothèse standard ou calcul plus fin
        
        clients_coords = coords[:, N_DUMMIES:, :] 
        clients_demand = batch['demand'][:, N_DUMMIES:]
        
        if clients_demand.dim() == 2: d = clients_demand.unsqueeze(-1)
        else: d = clients_demand
            
        x = torch.cat([clients_coords, d], dim=-1)
        
        # 2. Encode
        h = self.encoder(self.embed(x)) # [B, N, H]
        
        # 3. Decode avec Attention (Queries vs Keys)
        # On répète les queries pour le batch
        queries = self.cluster_queries.repeat(B, 1, 1) # [B, K, H]
        
        # Le decoder ne sert pas à générer une séquence ici, mais à enrichir les queries
        # avec le contexte des noeuds (h). 
        # Mais pour calculer la compatibilité, on peut faire un simple produit scalaire
        # entre les noeuds encodés (h) et les queries de clusters.
        
        # Approche simple et efficace : Attention Dot-Product
        # Score[b, n, k] = Dot(h[b, n], query[b, k])
        logits = torch.bmm(h, queries.transpose(1, 2)) # [B, N, K]
        
        # 4. Gumbel Noise (Exploration stochastique avant Sinkhorn)
        if not greedy:
            u = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            logits_noisy = (logits + gumbel) / self.tau
        else:
            logits_noisy = logits / self.tau
            
        # 5. Sinkhorn (Hérité de la classe parente)
        log_P = self.log_sinkhorn(logits_noisy, self.n_sinkhorn_iters)
        probs = torch.exp(log_P)
        
        # 6. Assignment (Notre version améliorée avec shuffle)
        assignments, log_probs_sum, entropy = self.get_balanced_assignments(probs, greedy=greedy)
            
        return assignments, log_probs_sum, entropy