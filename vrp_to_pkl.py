import os
import glob
import pickle
import vrplib
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# Configuration
INPUT_FOLDER = 'datasets/val_270'   # Dossier contenant vos .vrp
OUTPUT_PKL = 'datasets/val_270.pkl' # Fichier de sortie

def normalize_coords(coords):
    """Normalise les coordonnées entre [0, 1] (Min-Max scaling)"""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # On normalise en conservant l'aspect ratio si possible, ou strictement min-max
    # Ici on fait comme NeuOpt standard : min-max strict sur chaque axe ou global
    # Pour être sûr, on utilise le max global pour éviter les déformations
    max_val = np.max(coords)
    return coords / max_val

def process_single_file(filepath):
    """Lit un fichier .vrp et retourne le tuple formaté pour NeuOpt"""
    try:
        instance = vrplib.read_instance(filepath)
        
        # 1. Récupération des données brutes
        coords = instance['node_coord']
        demand = instance['demand']
        capacity = instance['capacity']
        
        # 2. Normalisation des coordonnées [0, 1]
        # (Obligatoire car make_instance ne le fait pas)
        coords_norm = normalize_coords(coords)
        
        # 3. Séparation Dépôt / Clients
        depot_coord = coords_norm[0]       # Le dépôt est l'index 0
        customer_coords = coords_norm[1:]  # Les clients sont le reste
        
        # 4. Demandes
        # make_instance divise la demande par la capacité, donc on garde la demande BRUTE ici.
        # On ne garde que les clients (index 1 à la fin)
        customer_demands = demand[1:]
        
        # Structure attendue par make_instance : (depot, loc, demand, capacity)
        return (depot_coord, customer_coords, customer_demands, capacity)
        
    except Exception as e:
        print(f"Erreur sur {filepath}: {e}")
        return None

def main():
    # 1. Lister tous les fichiers .vrp
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.vrp"))
    print(f"Trouvé {len(files)} fichiers .vrp dans {INPUT_FOLDER}")
    
    if len(files) == 0:
        print("Aucun fichier trouvé. Vérifiez le chemin.")
        return

    # 2. Lecture en parallèle (Utilise tous les coeurs CPU)
    # C'est ici qu'on gagne énormément de temps par rapport à une boucle for
    print("Conversion en cours...")
    data = []
    with Pool() as p:
        # map_async ou imap permet d'avoir une barre de progression
        results = list(tqdm(p.imap(process_single_file, files), total=len(files)))
    
    # Filtrer les erreurs éventuelles (None)
    data = [res for res in results if res is not None]
    
    # 3. Sauvegarde en Pickle
    print(f"Sauvegarde de {len(data)} instances dans {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data, f)
    
    print("Terminé ! Vous pouvez maintenant utiliser ce fichier avec CVRPDataset.")

if __name__ == '__main__':
    main()