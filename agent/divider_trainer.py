import torch
import torch.optim as optim
import os
from tqdm import tqdm
from datetime import datetime

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
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=opts.lr_divider)
        
        # Baseline for REINFORCE (Exponential Moving Average)
        self.baseline = None
        self.beta = 0.9 
        
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
                problem=self.agent.subproblem, 
                batch=batch, 
                T=100,
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
        
        pbar = tqdm(dataloader, desc="Training Divider")
        
        for batch in pbar:
            # Move to GPU if needed (si dataloader ne le fait pas)
            batch = {k: v.to(self.opts.device, non_blocking=True) for k, v in batch.items()}
        
            # --- 1. Forward Pass ---
            # assignments: [B, N]
            # log_probs_sum: [B] (C'est la somme des log_prob de tous les noeuds)
            assignements, log_probs_sum = self.model(
                batch,
                greedy=False
            )
            
            sub_batch = self.model.make_sub_batch(batch, assignements) # [B*K, 150, 2]
            
            # --- 3. Compute Reward ---
            rewards = self.get_reward(sub_batch)
            rewards = rewards.to(log_probs_sum.device) # [B]
            
            # --- 4. REINFORCE Loss Stabilization ---
            
            # A. Baseline Update (Moyenne mobile)
            if self.baseline is None:
                self.baseline = rewards.mean().item()
            else:
                self.baseline = self.beta * self.baseline + (1 - self.beta) * rewards.mean().item()
            
            # B. Calcul de l'Avantage Brut
            raw_advantage = rewards - self.baseline
            
            # C. Normalisation de l'Avantage (CRUCIAL POUR LA STABILITÉ)
            # On centre l'avantage sur le batch courant : (x - mean) / std
            # Cela aide si un batch contient des instances particulièrement dures ou faciles
            if raw_advantage.size(0) > 1:
                advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)
            else:
                advantage = raw_advantage

            # D. Normalisation de la Log-Probabilité (CRUCIAL POUR LA TAILLE DU GRAPHE)
            # log_probs_sum est la somme sur N noeuds (~ -500). 
            # On divise par N pour ramener à une échelle raisonnable (~ -1.5).
            N = self.opts.graph_size
            log_probs_mean = log_probs_sum / N
            
            # --- 5. Loss Calculation ---
            # Loss = - (Advantage_Norm * Log_Prob_Mean) - Entropy
            # Note : On réduit aussi le coeff d'entropie car log_probs_mean est beaucoup plus petit maintenant
            entropy_coef = 0.001 
            
            loss = -(advantage * log_probs_mean).mean()
            
            # --- 6. Optimization ---
            self.optimizer.zero_grad()
            loss.backward()
            
            # Le gradient clipping aura maintenant du sens car la loss est à une échelle normale
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
        print(f"🚀 Reprise de l'entraînement ! Logs : {self.run_dir}")

        train_history = {'train_loss': [], 'train_reward': []}
        eval_history = {'val_reward': []}
        
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
                
                # 3. Sauvegardes Stratégiques
                
                # A. Toujours sauvegarder le "latest" (pour reprise rapide)
                self.save(epoch, filename="checkpoint_latest.pt")
                
                # B. Sauvegarder l'historique tous les X époques
                if epoch % 10 == 0:
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
            'baseline': self.baseline, # <--- Très important !
            # Optionnel : RNG states pour reproductibilité exacte
            # 'rng_state': torch.get_rng_state(),
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