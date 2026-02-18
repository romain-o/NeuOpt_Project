import torch
import numpy as np
import multiprocessing
import os
import pyvrp

# La fonction _solve_worker doit être définie au niveau module (top-level)
# pour être "picklable" par multiprocessing.
def _solve_worker(args):
    depot, coords, demands, capacity, time_limit, scale_factor = args

    # ... (le début de votre fonction reste identique) ...
    m = pyvrp.Model()
    real_size = coords.shape[0]

    # Ajout du dépôt et des clients
    depot_loc = m.add_depot(x=depot[0], y=depot[1], name="Depot")
    vehicles = m.add_vehicle_type(num_available=real_size, capacity=capacity)
    
    clients = []
    for i, demand in enumerate(demands):
        clients.append(m.add_client(
            x=float(coords[i][0]),
            y=float(coords[i][1]),
            delivery=int(demand),
            name=f"Client {i + 1}",
        ))

    # OPTIMISATION (Voir note plus bas) : 
    # PyVRP calcule automatiquement les distances euclidiennes si on ne précise rien.
    # Votre boucle manuelle est très lente. Si vous voulez forcer les distances entières :
    # Il vaut mieux le faire sans recréer des np.array à chaque itération.
    # Pour l'instant, je garde votre logique mais notez que c'est un goulot d'étranglement.
    locations = [depot_loc] + clients
    for frm in locations:
        for to in locations:
            if frm != to:
                # Optimisation légère ici : éviter np.linalg.norm sur des scalaires
                dx = frm.x - to.x
                dy = frm.y - to.y
                dist = int((dx**2 + dy**2)**0.5) 
                m.add_edge(frm, to, distance=dist)

    res = m.solve(stop=pyvrp.stop.MaxRuntime(time_limit))
    cost = res.cost() if res.is_feasible() else float('inf')
    
    return cost / scale_factor


class HGSSolver:
    def __init__(self, problem, time_limit=10.0, scale_factor=1000):
        self.time_limit = time_limit
        self.scale_factor = scale_factor
        self.size = problem.size
        self.real_size = problem.real_size
        self.dummy_size = self.size - self.real_size

    def _tensor_to_numpy(self, t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    def __call__(self, batch, capacity=1.0):
        # 1. Préparation des données (identique à votre code)
        coords_ = batch['coordinates']
        demands_ = batch['demand']
        
        depot = coords_[:, 0, :]
        coords = coords_[:, self.dummy_size:, :]
        demands = demands_[:, self.dummy_size:]

        # Conversions numpy
        depot = self._tensor_to_numpy(depot) * self.scale_factor
        depot = depot.astype(int)
        coords = self._tensor_to_numpy(coords) * self.scale_factor
        coords = coords.astype(int)
        demands = self._tensor_to_numpy(demands) * self.scale_factor
        demands = demands.astype(int)
        capacity = int(capacity * self.scale_factor)

        batch_size = depot.shape[0]
        
        # 2. Création de la liste des tâches
        tasks = []
        for i in range(batch_size):
            tasks.append((
                depot[i],
                coords[i],
                demands[i],
                capacity,
                self.time_limit,
                self.scale_factor
            ))

        # 3. PARALLÉLISATION ICI
        # On utilise le nombre de coeurs CPU disponibles, limité à la taille du batch
        num_workers = min(os.cpu_count(), batch_size)
        
        # Création du Pool et exécution map
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_solve_worker, tasks)
    
        return results