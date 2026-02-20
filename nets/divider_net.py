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

        self.cluster_queries = nn.Parameter(torch.randn(self.n_splits, hidden_dim))
        self.project_context = nn.Linear(hidden_dim * 3, hidden_dim)
        self.project_k = nn.Linear(hidden_dim, hidden_dim)

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
        Version optimisée pour garantir le tracking du gradient (Autograd safe).
        """
        B, N, K = probs.size()
        device = probs.device
        target_size = N // K
        
        # Initialisations (Pas besoin d'initialiser log_probs_total ici)
        assignments = torch.zeros(B, N, dtype=torch.long, device=device)
        node_assigned = torch.zeros(B, N, dtype=torch.bool, device=device)
        cluster_counts = torch.zeros(B, K, dtype=torch.long, device=device)
        
        # --- 1. TRI GLOBAL ---
        flat_probs = probs.view(B, -1)
        sorted_probs, sorted_indices = torch.sort(flat_probs, dim=1, descending=True)
        
        nodes = sorted_indices // K 
        clusters = sorted_indices % K 
        
        batch_idx = torch.arange(B, device=device)
        
        # --- 2. ASSIGNATION ITÉRATIVE (Sans toucher aux gradients) ---
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

        # --- 3. REINFORCE GRADIENT TRACKING (Le FIX) ---
        # On extrait les probabilités des actions finales de manière vectorisée.
        # Cela crée un lien direct et propre avec `probs` pour le backward pass.
        if not greedy:
            # probs: [B, N, K]
            # assignments: [B, N] -> [B, N, 1]
            # chosen_probs: [B, N]
            chosen_probs = torch.gather(probs, 2, assignments.unsqueeze(-1)).squeeze(-1)
            
            # Somme des log-probabilités pour le batch entier
            log_probs_total = torch.log(chosen_probs + 1e-10).sum(dim=1) # [B]
        else:
            log_probs_total = torch.zeros(B, device=device)
        
        return assignments, log_probs_total

    def forward(self, batch, greedy=False):
        # --- 1. PRÉPARATION ET ENCODAGE ---
        coords = batch['coordinates']
        B, Total, _ = coords.size()
        N_DUMMIES = Total // self.opts.dnc_n_splits
        
        clients_coords = coords[:, N_DUMMIES:, :] 
        clients_demand = batch['demand'][:, N_DUMMIES:]
        d = clients_demand.unsqueeze(-1)
        
        x = torch.cat([clients_coords, d], dim=-1)
        
        # Encodage des noeuds: [B, N, H]
        h = self.encoder(self.embed(x)) 
        
        # Contexte global (moyenne du graphe): [B, H]
        graph_context = h.mean(dim=1) 
        
        # Clés pour l'attention: [B, N, H]
        keys = self.project_k(h) 

        # --- 2. INITIALISATIONS DU DÉCODAGE ---
        B, N, _ = h.size()
        K = self.n_splits
        target_size = N // K

        assignments = torch.zeros(B, N, dtype=torch.long, device=h.device)
        visited_mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
        
        log_probs_total = torch.zeros(B, device=h.device)

        # --- 3. DÉCODAGE SÉQUENTIEL (N étapes) ---
        for k in range(K):
            # L'ancre de base pour le cluster k: [B, H]
            cluster_q = self.cluster_queries[k].unsqueeze(0).expand(B, -1) 
            
            # Au début d'un cluster, le "dernier noeud" est simplement l'ancre
            last_node_embed = cluster_q 

            for step in range(target_size):
                # A. Construction de la Query contextuelle
                # On concatène les 3 informations cruciales
                context = torch.cat([graph_context, cluster_q, last_node_embed], dim=-1)
                query = self.project_context(context).unsqueeze(1) # [B, 1, H]

                # B. Calcul de l'Attention (Dot-Product simple)
                # scores: [B, 1, N] -> [B, N]
                scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)

                # C. Masquage strict (Interdit de reprendre un noeud)
                scores[visited_mask] = -float('inf')

                # D. Probabilités et Échantillonnage
                probs = F.softmax(scores, dim=1)
                dist = Categorical(probs)

                if greedy:
                    selected = probs.argmax(dim=1)
                else:
                    selected = dist.sample()
                    log_probs_total += dist.log_prob(selected)


                # E. Mise à jour de l'état
                assignments.scatter_(1, selected.unsqueeze(1), k)
                visited_mask.scatter_(1, selected.unsqueeze(1), True)

                # F. Mise à jour du contexte pour la prochaine étape
                # Le prochain noeud cherchera autour de celui qu'on vient de sélectionner
                last_node_embed = h[torch.arange(B), selected]

        # On moyenne l'entropie sur les N étapes
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