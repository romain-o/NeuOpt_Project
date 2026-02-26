import torch
import torch.optim as optim
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

def save_subdivision_plot(batch, assignments, epoch, save_dir, idx=0, dummy_rate=0.5):
    """
    Sauvegarde une image du graphe subdivisé.
    idx: l'index du graphe dans le batch qu'on veut dessiner (ex: 0).
    """
    # Extraction des données sur CPU pour Matplotlib
    coords = batch['coordinates'][idx].cpu().numpy()
    assigns = assignments[idx].cpu().numpy()
    
    # Calcul des Dummies
    N_Total = coords.shape[0]
    N_clients = int(N_Total // (1 + dummy_rate)) # Ajustez si votre formule est différente
    N_DUMMIES = N_Total - N_clients
    
    plt.figure(figsize=(8, 8))
    
    # 1. Tracé du/des Dépôts (Carrés rouges)
    plt.scatter(coords[:N_DUMMIES, 0], coords[:N_DUMMIES, 1], 
                c='red', marker='s', s=80, label='Dépôt', zorder=5)
    
    # 2. Tracé des Clients (Ronds colorés par cluster)
    clients_coords = coords[N_DUMMIES:]
    scatter = plt.scatter(clients_coords[:, 0], clients_coords[:, 1], 
                          c=assigns, cmap='tab10', s=40, alpha=0.8)
    
    plt.title(f"Évolution de la Subdivision - Époque {epoch}")
    
    # Création du dossier si nécessaire et sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"subdivision_epoch_{epoch:03d}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    
    # CRUCIAL : Fermer la figure pour éviter une fuite de RAM (Memory Leak)
    plt.close()

class DividerTrainer:
    def __init__(self, divider_model, agent, opts):
        """
        Trainer for the Learn-to-Divide model.
        
        Args:
            divider_model: Instance of NeuralDivider
            agent: Pre-trained NeuOpt agent (frozen) 'must have self.subproblem with greedy'
            opts: Configuration options
        """
        self.model = divider_model
        self.agent = agent
        self.opts = opts
        self.pomo_M = opts.pomo_M
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=opts.lr_divider)
     
        os.makedirs(opts.save_dir, exist_ok=True)

    def get_reward(self, batch):
        """
        Calculates reward using the pre-trained CVRP solver.
        Reward = - (Total Cost of all sub-problems combined)
        """
        with torch.no_grad():
            # sub_batch keys: 'coordinates', 'demand', 'norm_factor', ...
            # Shape: [B*K, 150, 2]
            
            # Call NeuOpt solver (Greedy decoding)
            # Ensure your agent's rollout method accepts this dict structure
            rollout_out = self.agent.rollout(
                problem=self.agent.problem, 
                batch=batch, 
                T=50,
                val_m=1,
                record=False,
                stall_limit=self.opts.stall_limit
            )
            
            # rollout_out[0] contains normalized costs: [B*K]
            real_costs = rollout_out[0] #already normalized by norm_factor in the agent's rollout
            
            # Sum costs per original instance to get global performance
            # [B*K] -> [B, K] -> [B]
            B = real_costs.size(0) // self.model.n_splits
            total_cost_per_instance = real_costs.view(B, self.model.n_splits).sum(dim=1)
            
        return -total_cost_per_instance # Maximize negative cost

    def train_one_epoch(self, dataloader):
        self.model.train()
        avg_reward = 0
        avg_loss = 0
        steps = 0
        
        M = self.pomo_M
        


        pbar = tqdm(dataloader, desc="Training Divider")
        
        for batch in pbar:
            bs = batch['coordinates'].size(0)
            Total_N = batch['coordinates'].size(1)
            N_clients = self.opts.graph_size
            N_dummies = Total_N - N_clients
            K = self.model.n_splits
            # Move to GPU if needed (si dataloader ne le fait pas)
            batch = {k: v.to(self.opts.device, non_blocking=True) for k, v in batch.items()}


            # POMO Augmentation
            batch_pomo = {}
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_pomo[k] = v.repeat_interleave(M, dim=0)
                else:
                    batch_pomo[k] = v
            random_indices = torch.argsort(torch.rand(bs * M, N_clients, device=self.opts.device), dim=1)
            pomo_starts = random_indices[:, :K] # Shape: [B*M, K]
            # --- 1. Forward Pass ---
            # assignments: [B, N]
            # log_probs_sum: [B] (C'est la somme des log_prob de tous les noeuds)
            self.optimizer.zero_grad()
            assignements, log_probs_sum = self.model(
                batch_pomo,
                greedy=False,
                pomo_starts=pomo_starts
            )
            
            sub_batch = self.model.make_sub_batch(batch_pomo, assignements) # [B*K, 150, 2]
            
            # --- 3. Compute Reward ---
            rewards = self.get_reward(sub_batch)
            rewards = rewards.to(log_probs_sum.device) # [B]

            rewards_reshaped = rewards.view(bs, M)
            baseline = rewards_reshaped.mean(dim=1, keepdim=True)
            advantage = rewards_reshaped - baseline
            advantage = advantage.view(-1)
              
            loss = -(advantage.detach() * log_probs_sum).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Logging
            avg_reward += rewards.mean().item()
            avg_loss += loss.item()
            steps += 1
            
            pbar.set_postfix({'Rw': f"{-rewards.mean().item():.2f}", 'Loss': f"{loss.item():.4f}"})
            
        return avg_loss / steps, avg_reward / steps

    def train(self, train_loader, val_loader, n_epochs, start_epoch=0): # <-- Ajout start_epoch
        
        # --- Création du dossier (Nouveau dossier pour la reprise ou suite ?) ---
        # Si on reprend, on crée quand même un nouveau dossier "run_..." pour ne pas mélanger
        # les logs, mais le modèle partira bien des poids entraînés.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = 'trained_divider/CVRP400'
        self.run_dir = os.path.join(base_dir, f"run_{timestamp}_resume_{start_epoch}")
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"📁 Dossier de sauvegarde : {self.run_dir}")
        image_save_dir = os.path.join(self.run_dir, "evolution_images")


        train_history = {'train_loss': [], 'train_reward': []}
        eval_history = {'val_reward': []}

        vis_batch = next(iter(val_loader))
        vis_batch = {k: v.to(self.opts.device) if isinstance(v, torch.Tensor) else v for k, v in vis_batch.items()}
        
        best_val_reward = -float('inf')
        # --- 3. Boucle d'entraînement ---
        try: # Début du bloc de protection
            for epoch in range(start_epoch, n_epochs):
                progress = epoch / n_epochs
                current_tau = max(0.1, 1.0 - progress) 
                self.model.tau = current_tau
                print(f"\n--- Epoch {epoch}/{n_epochs} ---")
                
                # 1. Train
                avg_loss, avg_reward = self.train_one_epoch(train_loader)
                print(f"Train | Reward: {avg_reward:.2f} | Loss: {avg_loss:.4f}")
                
                # 2. Validation (Optionnel à chaque époque si trop lent)
                val_reward = self.eval(val_loader)
                print(f"Val   | Reward: {val_reward:.2f}")
                

                self.model.eval() # On s'assure d'être en mode eval
                with torch.no_grad():
                    assignments, _ = self.model(vis_batch, greedy=True)
                
                    # On sauvegarde le graphe d'index 0
                    save_subdivision_plot(
                        batch=vis_batch, 
                        assignments=assignments, 
                        epoch=epoch, 
                        save_dir=image_save_dir,
                        idx=0,
                        dummy_rate=self.opts.dummy_rate
                    )

                # 3. Sauvegardes Stratégiques
                
                # A. Toujours sauvegarder le "latest" (pour reprise rapide)
                self.save(epoch, filename="checkpoint_latest.pt")
                
                # B. Sauvegarder l'historique tous les X époques
                if epoch % 1 == 0:
                    self.save(epoch, filename=f"checkpoint_epoch_{epoch}.pt")
                
                # C. Sauvegarder le meilleur modèle
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    self.save(epoch, is_best=True)

        except KeyboardInterrupt:
            print("\n Interruption détectée. Sauvegarde du checkpoint avant de quitter...")
            self.save(epoch, filename="checkpoint_interrupted.pt")
            print("Sauvegarde terminée. Arrêt propre.")
            return train_history, eval_history
                
        return train_history, eval_history

    def save(self, epoch, is_best=False, filename=None):
        """
        Sauvegarde un checkpoint complet.
        """
        # Si aucun nom n'est donné, on crée un nom par défaut
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
            
        path = os.path.join(self.run_dir, filename)
        
        # Le dictionnaire complet
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint_dict, path)
        print(f"💾 Checkpoint sauvegardé : {path}")
        
        # Si c'est le meilleur modèle (selon validation), on fait une copie spéciale
        if is_best:
            best_path = os.path.join(self.run_dir, "best_model.pt")
            torch.save(checkpoint_dict, best_path)
            print(f"🏆 Nouveau meilleur modèle sauvegardé !")
        
    def eval(self, val_loader):
        self.model.eval()
        avg_reward = 0
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.opts.device, non_blocking=True) for k, v in batch.items()}
                rewards = self.get_reward(batch)
                avg_reward += rewards.mean().item()
                steps += 1
        print(f"Eval Reward: {- avg_reward / steps}")
        return avg_reward / steps