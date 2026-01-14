import os
import random
import math
import numpy as np
from tqdm import tqdm # Barre de chargement

def generate_cvrp_instance_numpy(n, root_pos, cust_pos, demand_type, avg_route_size, instance_id, seed, output_dir):
    """
    Génère une instance VRPLib optimisée avec NumPy.
    """
    max_coord = 1000
    decay = 40
    
    # Initialisation du générateur aléatoire NumPy
    rng = np.random.default_rng(seed)

    # --- 1. Paramètres ---
    # r (Average route size)
    r_ranges = np.array([
        [0,0], [3, 5], [5, 8], [8, 12], [12, 16], [16, 25], [25, 50]
    ])
    if not (1 <= avg_route_size <= 6): raise ValueError("avg_route_size out of range")
    r = rng.uniform(*r_ranges[avg_route_size])

    # --- 2. Positionnement du Dépôt ---
    if root_pos == 1:   # Random
        depot = rng.integers(0, max_coord + 1, size=2)
    elif root_pos == 2: # Centered
        center = max_coord // 2
        depot = np.array([center, center])
    elif root_pos == 3: # Cornered
        depot = np.array([0, 0])
    else:
        raise ValueError("root_pos out of range")

    # --- 3. Positionnement des Clients ---
    n_rand = 0
    n_seeds = 0
    
    if cust_pos == 1:   # Random
        n_rand = n
    elif cust_pos == 2: # Clustered
        n_seeds = rng.integers(2, 7) # [2, 6]
    elif cust_pos == 3: # Random-Clustered
        n_rand = n // 2
        n_seeds = rng.integers(2, 7)
    
    # Tableau final des coordonnées (sans le dépôt pour l'instant)
    coords = np.zeros((n, 2))
    current_idx = 0

    # A. Clients Aléatoires
    if n_rand > 0:
        # On génère un peu plus pour filtrer les doublons éventuels
        candidates = rng.integers(0, max_coord + 1, size=(n_rand + 10, 2))
        # Filtrage basique (on garde n_rand points) - Simplification pour vitesse
        # Dans l'absolu, la probabilité de collision sur 1000x1000 est faible.
        coords[:n_rand] = candidates[:n_rand]
        current_idx = n_rand

    # B. Clients Clusterisés (Le goulot d'étranglement original)
    if n_seeds > 0:
        seeds = rng.integers(0, max_coord + 1, size=(n_seeds, 2))
        
        # Calcul du facteur de normalisation
        # Matrice de distance Seeds <-> Seeds
        # diff: (n_seeds, n_seeds, 2)
        diff = seeds[:, np.newaxis, :] - seeds[np.newaxis, :, :]
        dists_seeds = np.linalg.norm(diff, axis=2)
        # Poids: exp(-dist / decay)
        weights_seeds = np.sum(2.0 ** (-dists_seeds / decay), axis=1)
        max_weight = np.max(weights_seeds)
        norm_factor = 1.0 / max_weight

        # Génération par batch (Accept-Reject Vectorisé)
        remaining = n - current_idx
        while remaining > 0:
            # On génère un gros batch de candidats (ex: 2x le besoin) pour aller vite
            batch_size = remaining * 3 
            candidates = rng.integers(0, max_coord + 1, size=(batch_size, 2))
            
            # Distances Candidats <-> Seeds
            # candidates: (B, 2) -> (B, 1, 2)
            # seeds: (S, 2)      -> (1, S, 2)
            dists_cand = np.linalg.norm(candidates[:, np.newaxis, :] - seeds[np.newaxis, :, :], axis=2)
            
            # Somme des poids pour chaque candidat
            weights = np.sum(2.0 ** (-dists_cand / decay), axis=1)
            weights *= norm_factor
            
            # Tirage aléatoire pour acceptation
            probs = rng.random(batch_size)
            accepted_mask = probs <= weights
            accepted_points = candidates[accepted_mask]
            
            # Remplissage
            take = min(len(accepted_points), remaining)
            coords[current_idx : current_idx + take] = accepted_points[:take]
            current_idx += take
            remaining -= take

    # On ajoute le dépôt en tête de liste
    V = np.vstack([depot, coords])

    # --- 4. Génération des Demandes ---
    demand_min = [1, 1, 5, 1, 50, 1, 51]
    demand_max = [10, 10, 100, 100, 50, 100, 100]
    
    # Indexation Python (0-6) vs Type (1-7)
    dt_idx = demand_type - 1
    
    demands = np.zeros(n, dtype=int)

    if demand_type == 6: # Quadrant dependent
        half = max_coord / 2.0
        # Masque booléen pour les "grands" quadrants (Bas-Gauche ou Haut-Droite)
        is_large = ((coords[:, 0] < half) & (coords[:, 1] < half)) | \
                   ((coords[:, 0] >= half) & (coords[:, 1] >= half))
        
        # Génération vectorisée
        demands[is_large] = rng.integers(51, 101, size=np.sum(is_large))
        demands[~is_large] = rng.integers(1, 11, size=np.sum(~is_large))
        
    elif demand_type == 7: # Few large, many small
        large_per_route = 1.5
        limit_idx = int((n / r) * large_per_route)
        limit_idx = min(limit_idx, n)
        
        # Les premiers sont grands, les suivants petits (avant shuffle éventuel)
        # Note: Dans le script original, c'est basé sur l'index i.
        demands[:limit_idx] = rng.integers(50, 101, size=limit_idx)
        demands[limit_idx:] = rng.integers(1, 11, size=n - limit_idx)
        
        rng.shuffle(demands) # Shuffle spécifique au type 7 souvent implicite ou explicite
        
    else: # Types 1-5 (Uniforme)
        d_min = demand_min[dt_idx]
        d_max = demand_max[dt_idx]
        demands = rng.integers(d_min, d_max + 1, size=n)
        
        if demand_type != 6:
            rng.shuffle(demands)

    # Ajout de la demande 0 pour le dépôt
    full_demands = np.concatenate(([0], demands))

    # --- 5. Capacité ---
    max_demand_val = np.max(demands)
    sum_demands = np.sum(demands)
    
    if sum_demands == n:
        capacity = math.floor(r)
    else:
        capacity = max(max_demand_val, math.ceil(r * sum_demands / n))

    # --- 6. Écriture Fichier ---
    filename = f"XML{n}_{root_pos}{cust_pos}{demand_type}{avg_route_size}_{instance_id:05d}.vrp"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"NAME : {filename[:-4]}\n")
        f.write(f"COMMENT : Generated for NeuOpt training\n")
        f.write(f"TYPE : CVRP\n")
        f.write(f"DIMENSION : {n + 1}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {int(capacity)}\n")
        f.write("NODE_COORD_SECTION\n")
        
        # Écriture formatée
        # Pour aller vite en écriture, on prépare les chaînes
        # (L'écriture ligne par ligne reste rapide pour 270 noeuds)
        for i in range(n + 1):
            f.write(f"{i + 1:<4} {int(V[i,0]):<4} {int(V[i,1]):<4}\n")
            
        f.write("DEMAND_SECTION\n")
        for i in range(n + 1):
            f.write(f"{i + 1:<4} {int(full_demands[i]):<4}\n")
            
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

def create_dataset_optimized(output_folder="data/train_270", num_instances=10000):
    """
    Crée un dataset massif avec barre de progression.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier créé : {output_folder}")
    
    print(f"🚀 Démarrage de la génération de {num_instances} instances...")
    
    # Utilisation de TQDM pour la barre de chargement
    for i in tqdm(range(num_instances), desc="Génération", unit="inst"):
        # Paramètres aléatoires (seed différente à chaque tour)
        # On utilise random standard pour les paramètres globaux, numpy pour la géométrie
        root_pos = random.randint(1, 3)
        cust_pos = random.randint(1, 3)
        demand_type = random.randint(1, 6)
        avg_route_size = random.randint(1, 6)
        
        seed = 1234 + i
        
        generate_cvrp_instance_numpy(
            n=270,
            root_pos=root_pos,
            cust_pos=cust_pos,
            demand_type=demand_type,
            avg_route_size=avg_route_size,
            instance_id=i,
            seed=seed,
            output_dir=output_folder
        )

if __name__ == "__main__":
    create_dataset_optimized(output_folder="datasets/val_270", num_instances=10000)