from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import pickle
import os
import numpy as np
from utils import augmentation
import vrplib

def get_capacity(n):
    """
    Calcule la capacité dynamiquement en fonction de la taille du graphe n.
    Interpolation linéaire entre les standards connus et extrapolation au-delà.
    Standards: 20->30, 50->40, 100->50, 200->70
    """
    xp = [20, 50, 100, 200]
    yp = [30., 40., 50., 70.]
    
    if n > xp[-1]:
        slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        return yp[-1] + slope * (n - xp[-1])
    elif n < xp[0]:
        slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
        return yp[0] + slope * (n - xp[0])
    else:
        return float(np.interp(n, xp, yp))

def get_epsilon(n):
    """
    Calcule epsilon dynamiquement pour n'importe quelle taille n.
    Standards: 20->0.33, 50->0.625, 100->1.0, 200->1.429
    """
    xp = [20, 50, 100, 200]
    yp = [0.33, 0.625, 1.0, 1.429]
    
    if n > xp[-1]:
        slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        return yp[-1] + slope * (n - xp[-1])
    elif n < xp[0]:
        slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
        return yp[0] + slope * (n - xp[0])
    else:
        return float(np.interp(n, xp, yp))

total_history = 25 # (T_ES in the paper)

class CVRP(object):

    NAME = 'cvrp'  # Capacitiated Vehicle Routing Problem
    
    def __init__(self, p_size, init_val_met = 'random', with_assert = False, DUMMY_RATE = 0.5, k = 5, with_bonus = True, with_regular = True):
        
        self.size = int(np.ceil(p_size * (1 + DUMMY_RATE)))   # the number of real nodes plus dummy nodes in cvrp
        self.real_size = p_size
        self.dummy_size = self.size - self.real_size
        self.init_val_met = init_val_met
        self.k_max = k
        self.state = 'eval'
        self.epsilon = get_epsilon(p_size)
        self.with_bonus = with_bonus
        self.with_regular = with_regular
        self.with_assert = with_assert
        assert self.real_size + self.dummy_size == self.size
        print(f'CVRP with {p_size} nodes and {self.dummy_size} dummy depots (total {self.size}).\n', 
              f'Regulation: {self.with_regular} Bonus: {self.with_bonus} Do assert: {self.with_assert}.\n',
              f'MAX {self.k_max}-opt.\n')
    
    def train(self):
        self.state = 'train'
        
    def eval(self):
        self.state = 'eval'
    
    def augment(self, batch, val_m, only_copy=False):
        bs, gs, dim = batch['coordinates'].size()
        if only_copy:
            coordinates = batch['coordinates'].unsqueeze(1).expand(bs,val_m,gs,dim).clone().reshape(-1,gs,dim)
        else:
            coordinates = batch['coordinates'].unsqueeze(1).expand(bs,val_m,gs,dim).clone()
            coordinates = augmentation(coordinates, val_m).reshape(-1,gs,dim)
        demand = batch['demand'].unsqueeze(1).expand(bs,val_m,gs).clone().reshape(-1,gs)
        return {'coordinates': coordinates,
                'demand':demand}
    
    def input_feature_encoding(self, batch):
        return batch['coordinates'].clone()
    
    def get_initial_solutions(self, batch):
      
      batch_size = batch['coordinates'].size(0)
  
      def get_solution(methods):
          p_size = self.size
          
          if methods == 'random':
              
              candidates = torch.ones(batch_size,self.size).bool()
              candidates[:,:self.dummy_size] = False
              
              rec = torch.zeros(batch_size, self.size).long()
              selected_node = torch.zeros(batch_size, 1).long()
              cum_demand = torch.zeros(batch_size, 2)
              
              demand = batch['demand'].cpu()
              
              for i in range(self.size - 1):
                  
                  dists = torch.arange(p_size).view(-1, p_size).expand(batch_size, p_size).clone()
                  dists.scatter_(1, selected_node, 1e5)
                  dists[~candidates] = 1e5
                  dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                  dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                  
                  next_selected_node = dists.min(-1)[1].view(-1,1)
                  selected_demand = demand.gather(1,next_selected_node)
                  cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                  cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
    
                  rec.scatter_(1,selected_node, next_selected_node)
                  candidates.scatter_(1, next_selected_node, 0)
                  selected_node = next_selected_node  
                  
              return rec
          
          
          elif methods == 'greedy':

              candidates = torch.ones(batch_size,self.size).bool()
              candidates[:,:self.dummy_size] = False
              
              rec = torch.zeros(batch_size, self.size).long()
              selected_node = torch.zeros(batch_size, 1).long()
              cum_demand = torch.zeros(batch_size, 2)
              
              coor = batch['coordinates'].cpu()
              demand = batch['demand'].cpu()
              
              for i in range(self.size - 1):
                  
                  coor_now = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size, 2))
                  dists = (coor_now - coor).norm(p=2, dim=2)
                  
                  dists.scatter_(1, selected_node, 1e5)
                  dists[~candidates] = 1e5
                  dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                  dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                  
                  next_selected_node = dists.min(-1)[1].view(-1,1)
                  selected_demand = demand.gather(1,next_selected_node)
                  cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                  cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
                  
                  rec.scatter_(1,selected_node, next_selected_node)
                  candidates.scatter_(1, next_selected_node, 0)
                  selected_node = next_selected_node                         

              return rec
          
          else:
              raise NotImplementedError()

      return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
  
    def f(self, p): # The entropy measure in Eq.(5)
        return torch.clamp(1 - 0.5 * torch.log2(2.5*np.pi*np.e*p*(1-p)+ 1e-5), 0, 1)
    
    def step(self, batch, rec, action, obj, feasible_history, t, weights = 0):
        
        bs, gs = rec.size()
        pre_bsf = obj[:,1:].clone() # batch_size, 3 (current, bsf, tsp_bsf)
        feasible_history = feasible_history.clone() # bs, total_history 
        
        # k-opt step
        next_state = self.k_opt(rec, action)
        next_obj, context = self.get_costs(batch, next_state, True)
        
        # MDP step
        non_feasible_cost_total = torch.clamp_min(context[-1] - 1.00001, 0.0).sum(-1)
        feasible = non_feasible_cost_total <= 0.0
        soft_infeasible = (non_feasible_cost_total <= self.epsilon) & (non_feasible_cost_total > 0.)
        
        now_obj = pre_bsf.clone()
        now_obj[feasible,0] = next_obj[feasible].clone()
        now_obj[soft_infeasible,1] = next_obj[soft_infeasible].clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf) #bs,2(feasible_reward,infeasible_reward) 

        # feasible history step
        feasible_history[:,1:] = feasible_history[:,:total_history-1].clone()
        feasible_history[:,0] = feasible.clone()
        
        # compute the ES features
        feasible_history_pre = feasible_history[:,1:]
        feasible_history_post = feasible_history[:,:total_history-1]
        f_to_if = ((feasible_history_pre == True) & (feasible_history_post == False)).sum(1,True) / (total_history-1)
        f_to_f = ((feasible_history_pre == True) & (feasible_history_post == True)).sum(1,True) / (total_history-1)
        if_to_f = ((feasible_history_pre == False) & (feasible_history_post == True)).sum(1,True) / (total_history-1)
        if_to_if = ((feasible_history_pre == False) & (feasible_history_post == False)).sum(1,True) / (total_history-1)
        f_to_if_2 = f_to_if / (f_to_if + f_to_f + 1e-5)
        f_to_f_2 =  f_to_f / (f_to_if + f_to_f + 1e-5)
        if_to_f_2 =  if_to_f / (if_to_f + if_to_if + 1e-5)
        if_to_if_2 =  if_to_if / (if_to_f + if_to_if + 1e-5)

        # update info to decoder
        active = (t >= (total_history - 2))
        context2 = torch.cat((
                      (if_to_if * active),
                      (if_to_if_2 * active),
                      (f_to_f * active),
                      (f_to_f_2 * active),
                      (if_to_f * active),
                      (if_to_f_2 * active),
                      (f_to_if * active),
                      (f_to_if_2 * active),
                      feasible.unsqueeze(-1).float(),
                    ),-1) # 9 ES features
        
        # update regulation
        reg = self.f(f_to_f_2) + self.f(if_to_if_2)
        
        reward = torch.cat((rewards[:,:1], # reward
                            -1 * reg * weights * 0.05 * self.with_regular, # regulation, alpha = 0.05
                            rewards[:,1:2] * 0.05 * self.with_bonus, # bonus, beta = 0.05
                           ),-1)

        out = (next_state, 
               reward,
               torch.cat((next_obj[:,None], now_bsf),-1), 
               feasible_history,
               context,
               context2,
               (if_to_if,if_to_f,f_to_if,f_to_f,if_to_if_2,if_to_f_2,f_to_if_2,f_to_f_2)
               )
        
        return out

    def k_opt(self, rec, action):
        
        # action bs * (K_index, K_from, K_to)
        selected_index = action[:,:self.k_max]
        left = action[:,self.k_max:2*self.k_max]
        right = action[:,2*self.k_max:]
        
        # prepare
        rec_next = rec.clone()
        right_nodes = rec.gather(1,selected_index)
        argsort = rec.argsort()
        
        # new rec
        rec_next.scatter_(1,left,right)
        cur = left[:,:1].clone()
        for i in range(self.size - 2): # self.size - 2 is already correct
            next_cur = rec_next.gather(1,cur)
            pre_next_wrt_old = argsort.gather(1, next_cur)
            reverse_link_condition = ((cur!=pre_next_wrt_old) & ~((next_cur==right_nodes).any(-1,True)))
            next_next_cur = rec_next.gather(1,next_cur)
            rec_next.scatter_(1,next_cur,torch.where(reverse_link_condition, pre_next_wrt_old, next_next_cur))
            # if i >= self.size - 2: assert (reverse_link_condition == False).all()
            cur = next_cur
            
        return rec_next

    def get_order(self, rec, return_solution = False):
        
        bs,p_size = rec.size()
        visited_time = torch.zeros((bs,p_size),device = rec.device)
        pre = torch.zeros((bs),device = rec.device).long()
        for i in range(p_size - 1):
            visited_time[torch.arange(bs),rec[torch.arange(bs),pre]] = i + 1
            pre = rec[torch.arange(bs),pre]
        if return_solution:
            return visited_time.argsort() # return decoded solution in sequence
        else:
            return visited_time.long() # also return visited order

    def check_feasibility(self, rec, partial_sum_wrt_route_plan, basic = False):
        p_size = self.size
        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).sum(-1),"not visiting all nodes")
        
        real_solution = self.get_order(rec, True)
            
        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            real_solution.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            real_solution.sort(1)[0]
        ).sum(-1),"not valid tour")
            
        if not basic:   
            assert (partial_sum_wrt_route_plan <= 1 + 1e-5).all(), ("not satisfying capacity constraint", partial_sum_wrt_route_plan, partial_sum_wrt_route_plan.max())
    
    def get_costs(self, batch, rec, get_context = False, check_full_feasibility = False):
        
        coor = batch['coordinates']
        coor_next = coor.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
        cost = (coor - coor_next).norm(p=2, dim=2).sum(1)
        
        # check TSP feasibility if needed
        if self.with_assert:
            self.check_feasibility(rec, None, basic = True)
        
        # check full feasibility if needed
        if check_full_feasibility or get_context:
            context = self.preprocessing(rec, batch)
            if check_full_feasibility:
                self.check_feasibility(rec, context[-1], basic = False)          
        
        # get CVRP context
        if get_context:
            return cost, context
        else:
            return cost

    def get_dynamic_feature(self, rec, batch, context):
    
        route_plan_0x, visited_time, cum_demand, partial_sum_wrt_route_plan = context
        demand = batch['demand'].unsqueeze(-1)
        cum_demand = cum_demand.unsqueeze(-1)
        route_total_demand_per_node = partial_sum_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)
        
        infeasibility_indicator_after_visit = torch.clamp_min(cum_demand - 1.00001, 0.0) > 0
        infeasibility_indicator_before_visit = torch.clamp_min((cum_demand - demand) - 1.00001, 0.0) > 0
        
        to_actor = torch.cat((
            cum_demand,  
            demand, 
            route_total_demand_per_node - cum_demand,
            (demand == 0).float(),
            infeasibility_indicator_before_visit,
            infeasibility_indicator_after_visit,
            ), -1) # the node features
        
        return visited_time, to_actor
        
    def preprocessing(self, solutions, batch):
        
        batch_size, seq_length = solutions.size()
        assert seq_length < 1000
        arange = torch.arange(batch_size)
        demand = batch['demand']
        
        pre = torch.zeros(batch_size, device = solutions.device).long()
        route = torch.zeros(batch_size, device = solutions.device).long()
        route_plan_visited_time = torch.zeros((batch_size,seq_length), device = solutions.device).long()
        cum_demand = torch.zeros((batch_size,seq_length), device = solutions.device)
        partial_sum_wrt_route_plan = torch.zeros((batch_size, self.dummy_size), device = solutions.device)
        
        for i in range(seq_length):
            next_ = solutions[arange,pre]
            next_is_dummy_node = next_ < self.dummy_size
            route[next_is_dummy_node] += 1
            route_plan_visited_time[arange,next_] = (route % self.dummy_size) * int(1e3) + (i+1) % self.size
            new_cum_demand = partial_sum_wrt_route_plan[arange,route % self.dummy_size] + demand[arange, next_]
            partial_sum_wrt_route_plan[arange,route % self.dummy_size] = new_cum_demand.clone()
            cum_demand[arange,next_] = new_cum_demand * (~next_is_dummy_node)
            
            pre = next_.clone()
    
        route_plan_0x = (route_plan_visited_time // int(1e3))
        
        out =  (route_plan_0x, # route plan 0xxxxx
                (route_plan_visited_time % int(1e3)), # visited time
                cum_demand.clone(), # cum_demand (inclusive)
                partial_sum_wrt_route_plan.clone()) # partial_sum_wrt_route_plan
        
        return out
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPDataset(*args, **kwargs)


class CVRPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE = 0.5, CVRPLib_paths=None, scale_factor=1000):
        
        super(CVRPDataset, self).__init__()
        
        self.data = []
        self.size = int(np.ceil(size * (1 + DUMMY_RATE))) # the number of real nodes plus dummy nodes in cvrp
        self.real_size = size # the number of real nodes in cvrp
        self.scale_factor = scale_factor
        
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            print(f"Loading data from {filename}...")
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Application de l'offset et de la limite num_samples
            # On gère le cas où le fichier contient moins de données que demandé
            end = min(len(data), offset + num_samples)
            data_slice = data[offset:end]

            if len(data_slice) > 0:
                # DÉTECTION AUTOMATIQUE DU FORMAT
                first_item = data_slice[0]

                if isinstance(first_item, dict):
                    # CAS A : Le dataset est déjà traité (liste de dicts {'coordinates', 'demand'})
                    # C'est le cas si vous avez fait pickle.dump(dataset .data, f)
                    self.data = data_slice
                    print(f" -> Format détecté : Dictionnaires traités. Chargé {len(self.data)} instances.")
                    
                    # Optionnel : Vérification de la cohérence des tailles
                    if self.data[0]['coordinates'].size(0) != self.size:
                        print(f"Warning: Loaded data size ({self.data[0]['coordinates'].size(0)}) does not match requested size ({self.size}). Updating self.size.")
                        self.size = self.data[0]['coordinates'].size(0)
                        self.real_size = self.size - int(self.size * (DUMMY_RATE / (1+DUMMY_RATE))) # Approximation inverse
                
                else:
                    # CAS B : Le dataset est brut (liste de tuples/listes [depot, loc, ...])
                    # C'est le format standard des datasets de validation NeuOpt/POMO
                    self.data = [self.make_instance(args) for args in data_slice]
                    print(f" -> Format détecté : Données brutes. Converti {len(self.data)} instances.")
            else:
                self.data = []
                print("Warning: No data loaded (check offset/num_samples).")

        elif CVRPLib_paths is not None:

            for instance_path in CVRPLib_paths:
                instance = vrplib.read_instance(instance_path)
                
                
                capacity = instance['capacity']
                norm_coords = torch.from_numpy(instance['node_coord']).float() / self.scale_factor
                norm_demand = torch.from_numpy(instance['demand']).float() / capacity
                
                depot_coord = norm_coords[0]        # [x, y]
                client_coords = norm_coords[1:]    # [[x1, y1], [x2, y2], ...]
                client_demands = norm_demand[1:]   # [d1, d2, ...]

                n_padding = self.size - self.real_size

                padding_coords = depot_coord.unsqueeze(0).repeat(n_padding, 1)
                full_coords = torch.cat((padding_coords, client_coords), 0)

                padding_demand = torch.zeros(n_padding)
                full_demand = torch.cat((padding_demand, client_demands), 0)

                # Ajout au dictionnaire
                self.data.append({
                    'coordinates': full_coords,
                    'demand': full_demand
                })

        elif distribution == 'centered':
            self.data = [{'coordinates': torch.cat((torch.full((self.size - self.real_size, 2), 0.5), 
                                                    torch.FloatTensor(self.real_size, 2).uniform_(0, 1)), 0),
                          'demand': torch.cat((torch.zeros(self.size - self.real_size),
                                               torch.FloatTensor(self.real_size).uniform_(1, 10).long() / get_capacity(self.real_size)), 0)
                          } for i in range(num_samples)]
            
        else:            
            self.data = [{'coordinates': torch.cat((torch.FloatTensor(1, 2).uniform_(0, 1).repeat(self.size - self.real_size,1), 
                                                    torch.FloatTensor(self.real_size, 2).uniform_(0, 1)), 0),
                          'demand': torch.cat((torch.zeros(self.size - self.real_size),
                                               torch.FloatTensor(self.real_size).uniform_(1, 10).long() / get_capacity(self.real_size)), 0)
                          } for i in range(num_samples)]
        
        self.N = len(self.data)
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        depot, loc, demand, capacity, *args = args
        
        depot = torch.FloatTensor(depot)
        loc = torch.FloatTensor(loc)
        demand = torch.FloatTensor(demand)
        
        return {'coordinates': torch.cat((depot.view(-1, 2).repeat(self.size - self.real_size,1), loc), 0),
                'demand': torch.cat((torch.zeros(self.size - self.real_size), demand / capacity), 0) }
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]
    
    def divide(self, divider):
        """
        Applique le divider à chaque instance du dataset et met à jour self.data avec les instances divisées.
        """
        new_data = []
        for instance in self.data:
            divided_instances = divider(instance)
            n_splits = divided_instances['coordinates'].size(0)
            for i in range(n_splits):
                sub_instance = {
                    key: tensor[i].clone() 
                    for key, tensor in divided_instances.items()
                }
                new_data.append(sub_instance)
        
        self.data = new_data
        self.N = len(self.data)
        if self.N > 0:
            self.size = self.data[0]['coordinates'].size(0)
            self.real_size = self.real_size // n_splits
        print(f'{self.N} instances after division.')
        
    def save(self, filename):
        output_dir = 'my_datasets'
        os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Dataset saved : {filepath}")
            print(f"   Contains {len(self.data)} instances.")
        except Exception as e:
            print(f"Error saving dataset: {e}")
        
    
class SubCVRPDataset(Dataset):
    def __init__(self, problem, divider):
        """
        data_list: Liste de dictionnaires {'coordinates': tensor, 'demand': tensor}
        """
        super(SubCVRPDataset, self).__init__()
        self.data = []
        self.N = len(self.data)
        self.problem = problem
        self.divider = divider
        self.bs = 16
        print(f'{self.N} instances modifiées chargées.')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]
    
    def save(self, filename):
        output_dir = 'my_datasets'
        os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Dataset saved : {filepath}")
            print(f"   Contains {len(self.data)} instances.")
        except Exception as e:
            print(f"Error saving dataset: {e}")
            
    def generate_centered_instances(self, num_samples):
        """
        Génère des instances synthétiques avec le dépôt AU CENTRE (0.5, 0.5).
        
        Args:
            num_samples (int): Nombre d'instances à générer
            size (int): Taille totale (Padding + Clients)
            real_size (int): Nombre réel de clients
            
        Attribue a self.data une liste de dictionnaires {'coordinates': tensor, 'demand': tensor} après application du divider.
        Génère num_samples*self.divider.n_split instances au total après le split.
    
        """
        
        # 1. Calcul des dimensions
        n_padding = self.problem.size - self.problem.real_size  # Nombre de dummies (incluant le dépôt principal)
        cap = get_capacity(self.problem.real_size)
        
        client_locs = torch.FloatTensor(num_samples, self.problem.real_size, 2).uniform_(0, 1)
        padding_locs = torch.full((num_samples, n_padding, 2), 0.5)
        
        all_locs = torch.cat((padding_locs, client_locs), dim=1)
        
        # 3. Génération de la demande
        # A. Padding : Demande 0
        padding_demands = torch.zeros(num_samples, n_padding)
        
        # B. Clients : Uniforme Entier [1, 9], normalisé par la capacité
        # Shape: [num_samples, real_size]
        client_demands = torch.FloatTensor(num_samples, self.problem.real_size).uniform_(0, 1)
        
        # C. Concaténation
        all_demands = torch.cat((padding_demands, client_demands), dim=1)
        
        # 4. Conversion en liste de dictionnaires (Dé-batching pour stockage propre)
        temp_dataset = TensorDataset(all_locs, all_demands)
        loader = DataLoader(temp_dataset, batch_size=self.bs, shuffle=False)
        
        final_data_list = []
        
        print(f"--- Application du split et formatage (Batch size: {self.bs}) ---")
        
        for batch_locs, batch_dems in loader:
            # Création du dictionnaire attendu par votre fonction de split
            # batch_locs: [B, Size, 2], batch_dems: [B, Size]
            current_batch = {'coordinates': batch_locs, 'demand': batch_dems}
            
            # Application du split (si une fonction est fournie)
            if self.divider is not None:
                # La fonction split renvoie un nouveau dictionnaire avec des tenseurs plus gros (ou plus nombreux)
                processed_batch = self.divider(current_batch)
            else:
                processed_batch = current_batch

            # Extraction des résultats
            proc_locs = processed_batch['coordinates'] # [New_B, New_Size, 2]
            proc_dems = processed_batch['demand']      # [New_B, New_Size]
            
            # --- ETAPE 3 : Dé-batching (Mise en liste) ---
            # On itère sur la dimension 0 du batch traité
            current_batch_count = proc_locs.size(0)
            
            for i in range(current_batch_count):
                final_data_list.append({
                    'coordinates': proc_locs[i].clone(), # Clone pour détacher de la mémoire du batch
                    'demand': proc_dems[i].clone()
                })

        print(f"✨ Terminé. Nombre final d'instances : {len(final_data_list)}")
        self.data = final_data_list
        self.N = len(self.data)
        print(f'{self.N} instances modifiées chargées.')
        
    def plot_instance(self, idx, show_demands=True, title=None):
        """
        Affiche l'instance à l'index donné.
        
        Args:
            idx (int): L'index de l'instance dans le dataset.
            show_demands (bool): Si True, affiche la demande à côté de chaque nœud.
            title (str): Titre optionnel du graphique.
        """
        import matplotlib.pyplot as plt

        # 1. Récupération des données (et passage sur CPU / Numpy)
        instance = self.data[idx]
        coords = instance['coordinates']
        demands = instance['demand']

        if torch.is_tensor(coords):
            coords = coords.cpu().numpy()
            demands = demands.cpu().numpy()

        # Séparation Dépôt (index 0) et Clients (index 1 à la fin)
        # Note : Si vous avez des dummies au même endroit que le dépôt, ils seront superposés
        depot = coords[0]
        clients = coords[1:]

        # 2. Configuration du Plot
        plt.figure(figsize=(8, 8))
        
        # Tracer le Dépôt (Carré Rouge)
        plt.scatter(depot[0], depot[1], c='red', marker='s', s=100, label='Dépôt', zorder=10)
        
        # Tracer les Clients (Ronds Bleus)
        plt.scatter(clients[:, 0], clients[:, 1], c='blue', s=50, alpha=0.6, label='Clients')

        # 3. Annotations (Demandes)
        if show_demands:
            # Pour le dépôt (souvent demande 0, on l'affiche quand même ou non)
            plt.text(depot[0]+0.02, depot[1]+0.02, f"D: {demands[0]:.2f}", fontsize=9, color='red')
            
            # Pour les clients
            for i, (x, y) in enumerate(clients):
                # i+1 car on a sauté le dépôt dans la liste 'clients'
                d = demands[i+1]
                if d > 0: # On n'affiche que si la demande est positive (pour éviter de surcharger avec les dummies)
                    plt.text(x+0.01, y+0.01, f"{d:.2f}", fontsize=9)

        # 4. Mise en forme
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Instance #{idx} (N={len(coords)})")
            
        plt.xlabel("Coordonnée X")
        plt.ylabel("Coordonnée Y")
        plt.show()
