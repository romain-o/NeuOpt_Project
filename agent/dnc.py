import torch
from nets.actor_network import Actor
from problems.problem_cvrp import CVRP
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from problems.problem_cvrp import total_history


from utils import torch_load_cpu, get_inner_model, move_to
from agent.utils import validate

feasibility_history_base = [True] * (total_history)

class Divider():
    """Divides an instance in sub-instances
    """
    def __init__(self,problem: CVRP, opts):
        self.problem = problem
        self.n_splits = opts.dnc_n_splits
        self.real_size = problem.real_size
    
    def __call__(self, batch):
        dummy_size = self.problem.dummy_size
        coords = batch['coordinates'] if batch['coordinates'].dim() == 3 else batch['coordinates'].unsqueeze(0)  # [batch_size, graph_size+dummy_size, 2]
        demands = batch['demand'] if batch['demand'].dim() == 2 else batch['demand'].unsqueeze(0)   # [batch_size, graph_size+dummy_size]
        
        bs, total_size, _ = coords.shape
        
        if self.problem.real_size % self.n_splits != 0:
            raise ValueError("Graph size must be divisible by number of splits")
        
        depot_coords = coords[:, 0, :]                  # [B, 2]
        real_coords = coords[:, dummy_size:, :]         # [B, real_size, 2]
        real_demands = demands[:, dummy_size:]          # [B, real_size]
        
        
        #Relative coordinates for angular sweeping
        rel_coords = real_coords - depot_coords.unsqueeze(1)
        angles = torch.atan2(rel_coords[:, :, 1], rel_coords[:, :, 0]) # [B, real_size]
        
        sorted_indices = torch.argsort(angles, dim=1)
        idx_expanded = sorted_indices.unsqueeze(-1).expand_as(real_coords)
    
        sorted_real_coords = torch.gather(real_coords, 1, idx_expanded)
        sorted_real_demands = torch.gather(real_demands, 1, sorted_indices)
        
        sub_n_real = self.real_size // self.n_splits
        sub_n_dummy = dummy_size // self.n_splits
        
        split_real_coords = sorted_real_coords.reshape(bs * self.n_splits, sub_n_real, 2)
        split_real_demands = sorted_real_demands.reshape(bs * self.n_splits, sub_n_real)
        depot_repeated = depot_coords.repeat_interleave(self.n_splits, dim=0)
        split_dummies_coords = depot_repeated.unsqueeze(1).expand(bs * self.n_splits, sub_n_dummy, 2).clone()
        split_dummies_demands = torch.zeros(bs * self.n_splits, sub_n_dummy, device=coords.device)
        
        new_coords = torch.cat([split_dummies_coords, split_real_coords], dim=1)
        new_demands = torch.cat([split_dummies_demands, split_real_demands], dim=1)
        
        # 1. Indices des clients réels (Déjà présent)
        # Ils vont de 'dummy_size' à 'total_size'
        global_indices_base = torch.arange(dummy_size, total_size, device=coords.device).expand(bs, -1)
        sorted_global_indices = torch.gather(global_indices_base, 1, sorted_indices)
        split_real_indices = sorted_global_indices.reshape(bs * self.n_splits, sub_n_real)
        
        # 2. Indices des dummies (AJOUT)
        # Les dummies générés correspondent au dépôt original (index 0).
        # On assigne des ranges de dummies disjoints à chaque split pour permettre la reconstruction.
        split_dummy_indices = torch.arange(0, self.n_splits * sub_n_dummy, device=coords.device).view(self.n_splits, sub_n_dummy).unsqueeze(0).expand(bs, -1, -1).reshape(bs * self.n_splits, sub_n_dummy)
        
        # 3. Concaténation pour avoir la structure [Dummies | Real]
        # Shape finale : [bs * n_splits, sub_n_dummy + sub_n_real]
        split_global_indices = torch.cat([split_dummy_indices, split_real_indices], dim=1)
        
        return {'coordinates': new_coords,
                'demand': new_demands,
                'original_indices': split_global_indices}
        
    def plot_subdivision(self, batch):
        """Visualisation de la division angulaire en K sous-problèmes"""
        
        coords = batch['coordinates'][0].cpu() 
        dummy_size = self.problem.dummy_size
        
        depot = coords[0] 
        real_clients = coords[dummy_size:]
        
        rel_coords = real_clients - depot
        angles = torch.atan2(rel_coords[:, 1], rel_coords[:, 0])
        
        sorted_indices = torch.argsort(angles)
        
        sorted_clients = real_clients[sorted_indices]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(depot[0], depot[1], c='red', marker='s', s=200, label='Depot', zorder=10)
        
        cmap = cm.get_cmap('tab10') if self.n_splits <= 10 else cm.get_cmap('rainbow')
        colors = [cmap(i / (self.n_splits - 1) if self.n_splits > 1 else 0) for i in range(self.n_splits)]
        
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
        self.real_sub_size = problem.real_size // self.n_splits
        self.dummy_sub_size = problem.dummy_size // self.n_splits
        self.total_sub_size = self.real_sub_size + self.dummy_sub_size
        self.total_size = problem.real_size + problem.dummy_size
        
        self.problem = problem
        self.subproblem = CVRP(p_size = self.real_sub_size,
                               init_val_met = opts.init_val_met,
                               with_assert = opts.use_assert,
                               DUMMY_RATE = opts.dummy_rate,
                               k = opts.k,
                               with_bonus = not opts.wo_bonus,
                               with_regular = not opts.wo_regular)
        
        self.divider = Divider(problem = self.problem, opts = opts)
        
        self.actor = Actor(
            problem = self.subproblem,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.actor_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            v_range = opts.v_range,
            seq_length = self.total_sub_size,
            k = opts.k,
            with_RNN = not opts.wo_RNN,
            with_feature1 = not opts.wo_feature1,
            with_feature3 = not opts.wo_feature3,
            with_simpleMDP = opts.wo_MDP
        ).to(opts.device)
        
    def rollout(self, problem, T, val_m, stall_limit, batch, record=False, show_bar=False):
        #l'argument problem sert juste pour que validate soit compatible avec DNC et PPO
        sub_batch_data = self.divider(batch)
        active_problem = self.subproblem
        batch = move_to(sub_batch_data, self.opts.device)
        
        bs, gs, _ = batch['coordinates'].size() #bs = batch_size * n_splits
        
        batch_aug_same = active_problem.augment(batch, val_m, only_copy=True)
        batch_aug = active_problem.augment(batch, val_m)
        batch_feature = active_problem.input_feature_encoding(batch_aug)
        
        solutions = move_to(active_problem.get_initial_solutions(batch_aug_same), self.opts.device)
        solution_best = solutions.clone()
        
        obj, context = active_problem.get_costs(batch_aug_same, solutions, get_context=True, check_full_feasibility=True)
        obj = torch.cat((obj[:,None], obj[:,None], obj[:,None]), -1).clone()
        
        context2 = torch.zeros(bs * val_m, 9).to(solutions.device)
        context2[:, -1] = 1 # Initial state
        
        feasibility_history = torch.tensor(feasibility_history_base).view(-1, total_history).expand(bs * val_m, total_history).to(obj.device)
        
        solution_history = [solutions.clone()]
        solution_best_history = [solution_best.clone()]
        obj_history = [obj.clone()]        
        feasible_history_recorded = [feasibility_history[:, 0]]
        action = None
        reward = []
        stall_cnt_ins = torch.zeros(bs * val_m).to(solution_best.device)

        iterator = range(T)
        if show_bar and not self.opts.no_progress_bar:
             from tqdm import tqdm
             iterator = tqdm(iterator, desc='DNC rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for t in iterator:       

            action = self.actor(active_problem,
                                batch_aug_same,
                                batch_feature,
                                solutions,
                                context,
                                context2,
                                action)[0]

            # Step de l'environnement
            solutions, rewards, obj, feasibility_history, context, context2, info = active_problem.step(
                batch_aug_same, 
                solutions,
                action,
                obj,
                feasibility_history,
                t,
                weights=0
            )
            
            # Mise à jour de la meilleure solution trouvée
            index = rewards[:, 0] > 0.0
            solution_best[index] = solutions[index].clone()

            # Enregistrement
            reward.append(rewards[:, 0].clone())
            obj_history.append(obj.clone())
            
            if record: 
                solution_history.append(solutions.clone())
                solution_best_history.append(solution_best.clone())
                feasible_history_recorded.append(feasibility_history[:, 0].clone())
            
            # Gestion du "Stall" (Augmentation dynamique si on bloque)
            if stall_limit > 0:
                batch_aug_temp = active_problem.augment(batch, val_m)
                stall_cnt_ins = stall_cnt_ins * (1 - index.float()) + 1
                index_aug = stall_cnt_ins >= stall_limit
                
                # Attention : augmentation sur les coordonnées des sous-problèmes
                batch_aug['coordinates'][index_aug] = batch_aug_temp['coordinates'][index_aug]
                batch_feature = active_problem.input_feature_encoding(batch_aug)
                stall_cnt_ins[index_aug] *= 0

        best_length = active_problem.get_costs(batch_aug_same, solution_best, get_context=False, check_full_feasibility=True)
        # assert (best_length - obj[:,1] < 1e-5).all()

        
        out = (
            obj[:, 1].reshape(bs, val_m).min(1)[0], # Best cost per sub-instance
            torch.stack(obj_history, 1).view(bs, val_m, T + 1, -1).min(1)[0], # History
            torch.stack(reward, 1).view(bs, val_m, T).max(1)[0], # Max reward
            None if not record else (solution_history, solution_best_history, feasible_history_recorded),
            sub_batch_data
        )
        
        return out
    
    def reconstruct(self, batch, sub_batch, rollout_output):
        # 1. Récupération des bruts (Raw Data)
        records = rollout_output[3]
        if records is None:
            raise ValueError("Rollout must be called with record=True")
            
        # [BS * K * val_m, Sub_Size]
        # Ce sont toutes les solutions de toutes les augmentations
        best_solutions_aug = records[1][-1] 
        
        # Dimensions
        bs_real = batch['coordinates'].size(0) # Batch size original (ex: 10)
        K = self.n_splits                      # Splits (ex: 4)
        n_splits_total = bs_real * K           # Total sub-problems (ex: 40)
        
        # Détection automatique de val_m
        total_rows = best_solutions_aug.size(0)
        val_m = total_rows // n_splits_total
        
        # 2. Préparation pour la sélection du meilleur val_m
        # On doit étendre les coordonnées/indices originaux pour matcher la taille augmentée
        # sub_batch vient du divider, il est de taille [BS*K, ...]
        
        # On étend les coordonnées pour recalculer le coût local exact
        # [BS*K, N, 2] -> [BS*K*val_m, N, 2]
        coords_sub = sub_batch['coordinates'].repeat_interleave(val_m, dim=0).to(self.opts.device)
        
        # On étend les indices originaux pour le mapping plus tard
        # [BS*K, N] -> [BS*K*val_m, N]
        original_indices_aug = sub_batch['original_indices'].repeat_interleave(val_m, dim=0).to(self.opts.device)
        
        # 3. Calcul des coûts pour départager les augmentations
        # On recrée un mini-batch temporaire
        temp_batch = {'coordinates': coords_sub}
        if 'demand' in sub_batch:
            temp_batch['demand'] = sub_batch['demand'].repeat_interleave(val_m, dim=0).to(self.opts.device)
            
        # Calcul du coût de chaque variation
        costs_aug = self.subproblem.get_costs(temp_batch, best_solutions_aug) # [BS*K*val_m]
        
        # 4. Sélection des Vainqueurs (Best Augmentation)
        # On reshape pour isoler la dimension val_m : [BS*K, val_m]
        costs_view = costs_aug.view(n_splits_total, val_m)
        
        # On trouve l'index de la meilleure augmentation pour chaque sous-problème
        min_vals, min_idxs = torch.min(costs_view, dim=1) # [BS*K]
        
        # On sélectionne les routes et indices gagnants
        # Astuce : on utilise gather ou l'indexation avancée
        # On veut extraire les lignes correspondantes dans les tenseurs augmentés
        
        # Index global dans le tenseur "aug" correspondant au meilleur val_m
        # Ex: Si le split 0 a gagné avec l'aug 3, l'index est 0*val_m + 3
        selection_indices = torch.arange(n_splits_total, device=self.opts.device) * val_m + min_idxs
        
        # [BS*K, Sub_Size] -> On a maintenant UNE solution par split (la meilleure)
        best_sols = best_solutions_aug[selection_indices]
        real_indices = original_indices_aug[selection_indices]
        
        # 5. Reconstruction Vectorisée (Mapping Global)
        
        # Reshape pour séparer Batch et Splits : [BS, K, Sub_Size]
        sols_view = best_sols.view(bs_real, K, -1)
        inds_view = real_indices.view(bs_real, K, -1)
        
        # Initialisation Routes Globales
        reconstructed_routes = torch.zeros((bs_real, self.total_size), dtype=torch.long, device=self.opts.device)
        
        # A. Mapping Local -> Global
        # mapped_sols[b, k, i] = Index global pointé par le nœud i du split k du batch b
        mapped_sols = torch.gather(inds_view, 2, sols_view)
        
        # On écrit tout dans la table globale (sauf les dummies, écrasés sans risque car distincts)
        flat_inds = inds_view.flatten(1)       # [BS, K*Sub_Size]
        flat_vals = mapped_sols.flatten(1)     # [BS, K*Sub_Size]
        reconstructed_routes.scatter_(1, flat_inds, flat_vals)
        
        # 6. Couture (Daisy Chain) - C'est ici qu'on relie les splits
        # Rappel : Chaque split k a ses PROPRES dummies.
        # Le Split k va de Start_k ... à End_k.
        # End_k pointe naturellement vers son dummy (Dummy_k).
        # On veut changer ça : End_k doit pointer vers le dummy du split suivant (Dummy_{k+1}).
        
        # Identifier le Dummy du Split k (C'est toujours l'index 0 local)
        # global_dummies[b, k] = Index global du dummy du split k
        global_dummies = inds_view[:, :, 0] 
        
        # Identifier le nœud de FIN du Split k (Celui qui pointe vers 0 localement)
        # Correction : On filtre pour ne prendre que les nœuds réellement visités qui pointent vers 0
        # (Sinon on risque de prendre un nœud inutilisé qui pointe vers 0 par défaut, brisant la boucle)
        visited_times = self.subproblem.get_order(best_sols, return_solution=False) # [BS*K, Sub_Size]
        visited_times_view = visited_times.view(bs_real, K, -1)
        
        valid_exits = (sols_view == 0) & (visited_times_view > 0)
        
        local_end_indices = valid_exits.float().argmax(dim=2) # [BS, K]
        # local_end_indices = (sols_view == 0).float().argmax(dim=2) # [BS, K] (OLD)

        # Convertir en index global (Source du lien à modifier)
        global_ends = torch.gather(inds_view, 2, local_end_indices.unsqueeze(2)).squeeze(2) # [BS, K]
        
        # Connexions :
        # Split 0 -> Split 1 -> ... -> Split K-1 -> Split 0 (ou Dépôt Global)
        
        # Sources : Les fins des splits 0 à K-2
        sources = global_ends[:, :-1] # [BS, K-1]
        # Cibles : Les dummies des splits 1 à K-1
        targets = global_dummies[:, 1:] # [BS, K-1]
        
        # Appliquer la redirection
        reconstructed_routes.scatter_(1, sources, targets)
        
        # Cas spécial : Le dernier split doit pointer vers le tout premier dummy (le vrai dépôt global)
        # Si Dummy_0 est le dépôt global, on pointe vers global_dummies[:, 0]
        last_source = global_ends[:, -1].unsqueeze(1)
        first_dummy = global_dummies[:, 0].unsqueeze(1)
        reconstructed_routes.scatter_(1, last_source, first_dummy)

        # 7. Finalisation
        # Coût total = Somme des coûts MINIMAUX trouvés à l'étape 4
        # min_vals est [BS*K], on reshape en [BS, K] et on somme
        total_cost = min_vals.view(bs_real, K).sum(dim=1)
        
        # Ordonner pour l'affichage (optionnel, mais propre)
        rec_ordered = self.problem.get_order(reconstructed_routes, True)
        
        return {
            'total_cost': total_cost,
            'routes': rec_ordered,
            'sub_costs': min_vals.view(bs_real, K),
            'pre_manip' : reconstructed_routes
        }
        
    def load(self, load_path):
        assert load_path is not None
        print(f' [*] Loading data from {load_path}')
        load_data = torch_load_cpu(load_path)
        
        # Chargement des poids du modèle (Actor / Critic)
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict(load_data['actor'])
        
    def save(self, save_path):
        torch.save(self.actor.state_dict(), save_path)
        
    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        
    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        
    def solve(self, batch, T, val_m=1, stall_limit=10, show_bar=False):
        self.eval()
        with torch.no_grad():
            rollout_output = self.rollout(
                problem=self.problem,
                T = T,
                val_m = val_m,
                stall_limit = stall_limit,
                batch = batch,
                record = True,
                show_bar = show_bar
            )
            sub_batch = self.divider(batch)
            result = self.reconstruct(batch, sub_batch, rollout_output)
        return result, rollout_output
    
    def start_inference(self, problem, tb_logger, val_dataset=None, input_batch=None, conquer=False):
        validate(0, problem, self, tb_logger , val_dataset=val_dataset, distributed = False, input_batch=input_batch, conquer=conquer)