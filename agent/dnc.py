import torch
from nets.actor_network import Actor
from problems.problem_cvrp import CVRP
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Divider():
    """Divides an instance in sub-instances
    """
    def __init__(self,problem: CVRP, opts):
        self.problem = problem
        self.n_splits = opts.dnc_n_splits
        self.real_size = problem.real_size
    
    def __call__(self, batch):
        dummy_size = self.problem.dummy_size
        coords = batch['coordinates'] # [batch_size, graph_size+dummy_size, 2]
        demands = batch['demand']   # [batch_size, graph_size+dummy_size]
        
        bs, total_size, _ = coords.shape
        
        if self.problem.real_size % self.n_splits != 0:
            raise ValueError("Graph size must be divisible by number of splits")
        
        dummies_coords = coords[:, :dummy_size, :]      # [B, dummy_size, 2]
        real_coords = coords[:, dummy_size:, :]         # [B, real_size, 2]
        real_demands = demands[:, dummy_size:]          # [B, real_size]
        
        depot_ref = dummies_coords[:, 0, :].unsqueeze(1)
        
        #Relative coordinates for angular sweeping
        rel_coords = real_coords - depot_ref
        angles = torch.atan2(rel_coords[:, :, 1], rel_coords[:, :, 0]) # [B, real_size]
        
        sorted_indices = torch.argsort(angles, dim=1)
        idx_expanded = sorted_indices.unsqueeze(-1).expand_as(real_coords)
    
        sorted_real_coords = torch.gather(real_coords, 1, idx_expanded)
        sorted_real_demands = torch.gather(real_demands, 1, sorted_indices)
        
        sub_n_real = self.real_size // self.n_splits
        sub_n_dummy = self.real_size // self.n_splits
        
        split_real_coords = sorted_real_coords.view(bs * self.n_splits, sub_n_real, 2)
        split_real_demands = sorted_real_demands.view(bs * self.n_splits, sub_n_real)
        split_dummies_coords = dummies_coords.view(bs * self.n_splits, sub_n_dummy, 2)
        split_dummies_demands = torch.zeros(bs * self.n_splits, sub_n_dummy, device=coords.device)
        
        new_coords = torch.cat([split_dummies_coords, split_real_coords], dim=1)
        new_demands = torch.cat([split_dummies_demands, split_real_demands], dim=1)
        
        # Retrieve original indices
        global_indices_base = torch.arange(dummy_size, total_size, device=coords.device).expand(bs, -1)
        sorted_global_indices = torch.gather(global_indices_base, 1, sorted_indices)
        split_global_indices = sorted_global_indices.view(bs * self.n_splits, sub_n_real)
        
        return {'coordinates': new_coords,
                'demand': new_demands,
                'original_indices': split_global_indices}
        
    def plot_subdivision(self, batch):
        # 1. Récupération des données (CPU) pour le premier élément du batch
        # On suppose que batch['coordinates'] est [B, N, 2]
        coords = batch['coordinates'][0].cpu() 
        dummy_size = self.problem.dummy_size
        
        # 2. Séparation Dépôt / Clients
        # Le premier dummy est considéré comme le dépôt principal pour le calcul d'angle
        depot = coords[0] 
        real_clients = coords[dummy_size:]
        
        # 3. Réplication de la logique "Angular Sweep"
        # On doit recalculer les angles ici car 'batch' contient les données brutes non triées
        rel_coords = real_clients - depot
        angles = torch.atan2(rel_coords[:, 1], rel_coords[:, 0])
        
        # Tri des indices
        sorted_indices = torch.argsort(angles)
        
        # On réordonne les clients selon l'angle pour visualiser les groupes contigus
        sorted_clients = real_clients[sorted_indices]
        
        # 4. Affichage
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Tracer le Dépôt
        ax.scatter(depot[0], depot[1], c='red', marker='s', s=200, label='Depot', zorder=10)
        
        # Préparation des couleurs
        cmap = cm.get_cmap('tab10') if self.n_splits <= 10 else cm.get_cmap('rainbow')
        colors = [cmap(i / (self.n_splits - 1) if self.n_splits > 1 else 0) for i in range(self.n_splits)]
        
        # Calcul de la taille de chaque sous-groupe (divisibilité déjà vérifiée dans __call__)
        chunk_size = len(real_clients) // self.n_splits
        
        print(f"--- Visualisation : {len(real_clients)} clients divisés en {self.n_splits} groupes de {chunk_size} ---")

        for i in range(self.n_splits):
            # Découpage des indices pour le i-ème sous-problème
            start = i * chunk_size
            end = (i + 1) * chunk_size
            
            # Sélection des clients du secteur
            subset = sorted_clients[start:end]
            
            # Plot des points
            ax.scatter(subset[:, 0], subset[:, 1], 
                       color=colors[i], 
                       s=40, 
                       label=f'Split {i+1}')
            
            # (Optionnel) Ligne pointillée du dépôt vers le centre de masse du secteur
            # pour mieux visualiser l'effet "camembert"
            if len(subset) > 0:
                center = subset.mean(dim=0)
                ax.plot([depot[0], center[0]], [depot[1], center[1]], 
                        color=colors[i], linestyle='--', alpha=0.3)
        
        # Mise en forme
        ax.set_title(f"Divide & Conquer: Angular Sweep (K={self.n_splits})")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        ax.axis('equal') # Indispensable pour ne pas déformer les angles visuellement
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
    
class DNC():
    """Divide and conquer agent inspired by NeuOpt
    """
    def __init__(self, problem, opts):
        self.opts = opts
        self.n_splits = opts.dnc_n_splits
        self.subgraph_size = opts.graph_size // self.n_splits
        self.problem = problem
        self.subproblem = CVRP(p_size = self.subgraph_size,
                               init_val_met = opts.init_val_met,
                               with_assert = opts.use_assert,
                               DUMMY_RATE = opts.dummy_rate,
                               k = opts.k,
                               with_bonus = not opts.wo_bonus,
                               with_regular = not opts.wo_regular)
        
        self.actor = Actor(
            problem = self.subproblem,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.actor_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            v_range = opts.v_range,
            seq_length = self.subgraph_size,
            k = opts.k,
            with_RNN = not opts.wo_RNN,
            with_feature1 = not opts.wo_feature1,
            with_feature3 = not opts.wo_feature3,
            with_simpleMDP = opts.wo_MDP
        )
        
    