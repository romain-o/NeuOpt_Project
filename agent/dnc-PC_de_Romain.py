import torch
from nets.actor_network import Actor
from problems.problem_cvrp import CVRP
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from problems.problem_cvrp import total_history
from agent.utils import validate
import torch.multiprocessing as mp

from utils import torch_load_cpu, get_inner_model, move_to

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
        sub_n_dummy = dummy_size // self.n_splits
        
        split_real_coords = sorted_real_coords.reshape(bs * self.n_splits, sub_n_real, 2)
        split_real_demands = sorted_real_demands.reshape(bs * self.n_splits, sub_n_real)
        split_dummies_coords = dummies_coords.reshape(bs * self.n_splits, sub_n_dummy, 2)
        split_dummies_demands = torch.zeros(bs * self.n_splits, sub_n_dummy, device=coords.device)
        
        new_coords = torch.cat([split_dummies_coords, split_real_coords], dim=1)
        new_demands = torch.cat([split_dummies_demands, split_real_demands], dim=1)
        
        
        
        # Retrieve original indices
        global_indices_base = torch.arange(dummy_size, total_size, device=coords.device).expand(bs, -1)
        sorted_global_indices = torch.gather(global_indices_base, 1, sorted_indices)
        split_global_indices = sorted_global_indices.reshape(bs * self.n_splits, sub_n_real)
        
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

        # --- 3. PPO ITERATIONS LOOP ---
        # Cette boucle est identique à l'originale, mais elle optimise les sous-tournées
        iterator = range(T)
        if show_bar and not self.opts.no_progress_bar:
             from tqdm import tqdm
             iterator = tqdm(iterator, desc='DNC rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for t in iterator:       
            
            # Appel à l'acteur (sur le sous-problème)
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

        # --- 4. OUTPUT ---
        # Assertions (optionnelles mais recommandées en debug)
        best_length = active_problem.get_costs(batch_aug_same, solution_best, get_context=False, check_full_feasibility=True)
        # assert (best_length - obj[:,1] < 1e-5).all()

        # Construction de la sortie standard
        # Note : Ces métriques concernent les SOUS-PROBLÈMES.
        # Pour l'entraînement, c'est ce qu'on veut (minimiser la somme des distances locales).
        
        out = (
            obj[:, 1].reshape(bs, val_m).min(1)[0], # Best cost per sub-instance
            torch.stack(obj_history, 1).view(bs, val_m, T + 1, -1).min(1)[0], # History
            torch.stack(reward, 1).view(bs, val_m, T).max(1)[0], # Max reward
            None if not record else (solution_history, solution_best_history, feasible_history_recorded),
            sub_batch_data
        )
        
        return out
    
    def reconstruct(self, batch, rollout_output):

        # 1. Récupération des solutions locales (Indices locaux 0..N_sub)
        records = rollout_output[3]
        if records is None:
            raise ValueError("Rollout must be called with record=True")
        
        sub_batch = rollout_output[4]
            
        # [BS * K * val_m, Sub_Len]
        best_solutions_local = records[1][-1] 
        
        # 2. Gestion des dimensions et de val_m
        # n_splits_total correspond à BS * K (ex: 4 * 4 = 16)
        n_splits_total = sub_batch['original_indices'].size(0) 
        n_augmented = best_solutions_local.size(0)             # ex: 16 * val_m
        
        val_m = self.opts.val_m

        original_indices_expanded = sub_batch['original_indices'].repeat_interleave(val_m, dim=0)
        
        local_size = self.subproblem.size 
        
        mapping_table = torch.zeros(
            n_augmented, 
            local_size, 
            dtype=torch.long, 
            device=self.opts.device
        )
   
        mapping_table[:, self.subproblem.dummy_size:] = original_indices_expanded
 
        all_reconstructed_routes = torch.gather(mapping_table, 1, best_solutions_local.long())

        sub_coords_expanded = sub_batch['coordinates'].repeat_interleave(val_m, dim=0).to(self.opts.device)
        sub_batch_temp = {'coordinates': sub_coords_expanded}
        
        costs = self.subproblem.get_costs(sub_batch_temp, best_solutions_local)
        
        costs_view = costs.view(n_splits_total, val_m)
        min_vals, min_indices = torch.min(costs_view, dim=1)
        
        all_routes_view = all_reconstructed_routes.view(n_splits_total, val_m, -1)
        best_routes_flat = all_routes_view[torch.arange(n_splits_total), min_indices]

        bs = batch['coordinates'].size(0)

        reconstructed_routes = best_routes_flat.view(bs, -1)
 
        costs_per_instance = min_vals.view(bs, self.n_splits)
        total_real_cost = costs_per_instance.sum(dim=1) # [BS]

        batch_temp = {'coordinates': batch['coordinates'].to(self.opts.device),
                      'demand': batch['demand']}
        reconstructed_cost = self.problem.get_costs(batch_temp, reconstructed_routes)
        
        return {
            'total_cost': total_real_cost,
            'routes': reconstructed_routes
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
            result = self.reconstruct(batch, rollout_output)
        return result
    
    def start_inference(self, problem, tb_logger, val_dataset=None, input_batch = None, conquer=False):
        if self.opts.distributed:            
            mp.spawn(validate, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger, True, None, input_batch ))
        else:
            validate(0, problem, self, tb_logger=tb_logger, val_dataset=val_dataset , distributed = False, input_batch=input_batch, conquer=conquer)