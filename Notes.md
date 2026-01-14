Comment est stockée (puis transmise dans le NN) l'information que la capacité max a été dépassée ? (Ou c'est dans le code ?)

Comment on backprop en RL ? La loss ne dépend pas explicitement des coefficients à apprendre (la reward est définie sur des résultats expérimentaux) ok

Comment généraliser l'algo k-Opt à un CVRP ? Il y a plusieurs cycles. (idée: tout n'est qu'un seul cycle mais le dépôt est présent avec multiplicité) l'idée est bonne

La Critic apprend a fit les rewards (présentes + futures, ie les ^R_t construits à partir de la formule explicite de reward et du coeff gamma(discount)).
La critic donne les baseline_values qui sont les v_phi^origin, v_phi^reg et v_phi^bonus
Avec ces rewards différentiables on arrive à créer des loss (différentiables en les paramètres) pour la critic et pour le PPO qui détermine la policy.


Fonctionnement général du Neural k-opt:

- les N nodes ont un embedding h (N*embed_dim) où h_k est l'embedding du node x_k (k)
    NeuOpt encoder:
        -TODO voir annexe B

- Un état s_t est un triplet {G, tau_t, tau_t^bsf} qui comprend l'instance, la solution actuelle, et la meilleur solution so far

- Une action correspond à:
    faire k exchanges d'edge dans le graph représentant la solution actuelle:
        - Passe par un starting move, des intermediate moves et un ending move (total: k edges ajoutées)
        - chacun est uniquement déterminé par un point. Celui du starting move est l'anchor node

- Cette action amène à une nouvelle solution tau_{t+1} et donc à un nouvel état s_{t+1} auquel est associé une reward
    cette reward est non-différentiable (principe du RL, et d'avoir une récompense intuitive)

- La policy utilisée pour choisir une action à partir d'un state est donnée en créant P_theta^k:
    - un dual stream composé de:
        - un move stream créé avec: q_mu_k = GRU(h_{k-1}, q_mu_{k-1}) ie. des last selected (end of added edges) nodes
        - un edge stream créé avec: q_lambda_k = GRU(h_{i_k}, q_lambda_{k-1}) ie. des current source (start of added edges) nodes
    - un méchanisme d'attention:
        - mu_k = Attention(q_mu_k)
        - lambda_k = Attention(q_lambda_k)
        - P_theta^k = Softmax(C.tanh(mu_k + lambda_k)), la distribution de proba à l'étape k, permettant de faire l'action: choisir x_k

- GIRE (Guided infeasible region exploration):
    - ajout des contraintes du CVRP (et autres!):
        - créer des boolean features qui target si une solution viole une contrainte (et quand dans sa construction)
        - compute les probas de passer d'une solution de type i à une solution de type j (i,j étant possible ou impossible) en se basant sur les T_his dernières solutions.
        - sommer la reward standard r_t avec:
            - r_t^reg qui régule les comportement d'exploration extrèmes (ie. ceux pour lesquels un type de solution mène de manière trop certaines à un autre type à l'étape suivante)
            - r_t^bonus qui encourage l'exploration de régions epsilon-impossibles
        r_t^GIRE = r_t + r_t^reg + r_t^bonus
        On peut donc la modifier en ajoutant d'autres reward, à condition de s'occuper de la création de leur loss (différentiable) associée:

- Les deux réseaux Actor/Critic:
    - Critic apprend à donner la valeur moyenne du retour attendu à chaque état s_t. ie. la valeur attendue d'un état sachant la policy actuelle. (param: phi)
    - Actor apprend à choisir la série d'actions la plus avantageuse. (param. theta)

    - Ils ont besoin de Loss pour être entrainés mais on n'a que des rewards (non-différentiables):
        - Loss_phi^Critic dépend de la différence^2 entre les retours cumulés discountés R_t et v(s_t) l'output du réseau. Ainsi le réseau Critic apprend à bien prédire le retour moyen d'un état étant donné la policy actuelle (donc il évolue pour suivre l'autre réseau)
        - Loss_theta^Actor dépend des ratios de probas de pi_theta et de l'ancienne policy pour tous les états, pondéré par l'avantage de la policy à chaque étape A_t = R_t - v(s_t) qui mesure si le retour cumulé discounté de ma série d'action faites avec ma policy pi_theta est meilleure que le retour moyen de l'état. ie. on cherche la bonne trajectoire d'action
        ie. on oriente pi_theta vers les actions qui font ont mené à un haut R_t (et donc v(s_t)) en premier lieu. Car savoir que R_t/v(s_t) est haut n'indique pas quelle action fait monter cette moyenne/espérance.