import random
import math
import torch
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import *

from local_engine import LocalGame
from local_player import LocalPlayer
from fabian_player import Player
from neural_net import DeepCFRModel
from engine import FoldAction, CheckAction, CallAction, RaiseAction, TerminalState


############################################
# Memory Reservoir for advantage data
############################################
class MemoryReservoir:
    def __init__(self, capacity):
        """
        Initialize the MemoryReservoir with a fixed capacity.
        :param capacity: The maximum number of items the reservoir can hold.
        """
        self.capacity = capacity
        self.memory = []
        self.stream_size = 0

    def add(self, item):
        """
        Add a new item to the reservoir using reservoir sampling.
        :param item: The item to be added.
        """
        self.stream_size += 1
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            replace_index = random.randint(0, self.stream_size - 1)
            if replace_index < self.capacity:
                self.memory[replace_index] = item

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)


############################################
# CFR Implementation
############################################
class CFR:
    def __init__(self, cfr_iters, mcc_iters, round_iters, reservoir_size, device):
        # Hyperparameters
        self.cfr_iters = cfr_iters           # Number of outer CFR iterations
        self.mcc_iters = mcc_iters           # Number of Monte Carlo samples per state
        self.round_iters = round_iters       # Number of states generated per iteration
        self.reservoir_size = reservoir_size
        
        # CHANGED: Lower learning rate, moderate batch size, added weight decay in training
        self.num_epochs = 25                 
        self.batch_size = 512                
        self.learning_rate = 1e-3            # was 0.1 previously; now 1e-3 for stability

        self.game_engine = LocalGame()

        # Discrete approximation for bet feats
        self.nbets = 8
        # Action space: Fold, Check, Call, Raise(1/2 pot), Raise(1.5 pot)
        self.nactions = 5

        # Advantage memory to store (state, regrets) pairs
        self.advantage_memory = [MemoryReservoir(reservoir_size) for _ in [0,1]]

        # Keep a list of trained models (advantage nets) for each player
        self.strategy_memory = [[], []]

        # Device: "cpu" or "cuda"
        self.device = device

    def generate_model(self):
        # CHANGED: Pass device explicitly to the model constructor
        return DeepCFRModel(4, self.nbets, self.nactions, self.device).to(self.device)

    def regret_matching(self, raw_regrets):
        """
        Clamps negative regrets to 0, normalizes to get a policy distribution.
        raw_regrets shape: [batch_size, nactions]
        """
        clamped = torch.clamp(raw_regrets, min=0)
        sums = torch.sum(clamped, dim=1, keepdim=True) + 1e-9
        policy = clamped / sums
        return policy

    def get_action_distribution(self, advantage_model, roundstate, active):
        """
        Returns a numpy array of action probabilities via regret matching.
        """
        with torch.no_grad():
            card_tensor, bet_tensor = advantage_model.tensorize_roundstate(roundstate, active)
            # shape: [1, nactions]
            raw_regrets = advantage_model(card_tensor, bet_tensor)
            mask = advantage_model.tensorize_mask(roundstate)  # [1, nactions]

            # Mask out illegal actions
            raw_regrets = raw_regrets * mask
            # Convert to policy
            policy = self.regret_matching(raw_regrets)
        return policy[0].cpu().numpy()

    def get_averaged_model(self, player):
        """
        Returns a new DeepCFRModel that is the average of all models in 
        self.strategy_memory[player]. If none exist, returns a fresh model.
        """
        if len(self.strategy_memory[player]) == 0:
            return self.generate_model()  # no models => random

        avg_model = self.generate_model()
        avg_model_params = dict(avg_model.named_parameters())

        n = len(self.strategy_memory[player])
        for m in self.strategy_memory[player]:
            for name, param in m.named_parameters():
                avg_model_params[name].data += param.data

        # Divide by n for the average
        for name, param in avg_model_params.items():
            param.data /= float(n)

        return avg_model

    def generate_roundstates(self, traverser):
        opp_model = self.get_averaged_model(1 - traverser)
        trav_model = self.get_averaged_model(traverser)

        player_0 = LocalPlayer("A", trav_model, self.device)
        player_1 = LocalPlayer("B", opp_model, self.device)

        players = [player_0, player_1]
        bounties = []

        roundstates = []
        for _ in range(self.round_iters):
            partial_roundstates = self.game_engine.generate_roundstates(players, bounties)
            roundstates.extend(partial_roundstates)
        return roundstates

    def compute_monte_carlo_ev(self, roundstate, traverser, t_model, o_model):
        """
        Estimate EV for the traverser by simulating self.mcc_iters times.
        """
        players = [
            LocalPlayer("A", t_model if traverser == 0 else o_model, self.device),
            LocalPlayer("B", t_model if traverser == 1 else o_model, self.device)
        ]

        total_pnls = 0.0
        for _ in range(self.mcc_iters):
            pnls = self.game_engine.simulate_round_state(players, roundstate)
            total_pnls += pnls[traverser]
        return total_pnls / float(self.mcc_iters)

    def compute_regrets_for_state(self, roundstate, traverser, t_model, o_model):
        """
        Returns a vector of regrets for each action in self.nactions
        by comparing baseline EV to EV(action -> next_state).
        """
        if isinstance(roundstate, TerminalState):
            return None

        active_player = roundstate.button % 2
        if active_player != traverser:
            return None

        # Baseline EV
        baseline_ev = self.compute_monte_carlo_ev(roundstate, traverser, t_model, o_model)

        pot = sum(STARTING_STACK - roundstate.stacks[i] for i in [0,1])
        half_pot = RaiseAction(math.ceil(pot * 0.5))
        full_pot = RaiseAction(math.ceil(pot * 1.0))
        action_space = [FoldAction(), CheckAction(), CallAction(), half_pot, full_pot]

        # Evaluate regrets
        mask = t_model.tensorize_mask(roundstate)[0]  # shape [nactions]
        regrets = torch.zeros(self.nactions, device=self.device)

        for action_idx, action in enumerate(action_space):
            if mask[action_idx] > 0.5:  # i.e. action is legal
                next_state = roundstate.proceed(action)
                action_ev = self.compute_monte_carlo_ev(next_state, traverser, t_model, o_model)
                regrets[action_idx] = action_ev - baseline_ev
        return regrets

    def gather_advantages(self, traverser, roundstates):
        """
        For each roundstate in roundstates, compute regrets if the traverser is active.
        Store (roundstate, regrets) in the memory reservoir.
        """
        t_model = self.get_averaged_model(traverser)
        o_model = self.get_averaged_model(1 - traverser)

        for rs in roundstates:
            r_vec = self.compute_regrets_for_state(rs, traverser, t_model, o_model)
            if r_vec is None:
                continue
            self.advantage_memory[traverser].add((rs, r_vec))

    def train_advantage_net(self, traverser):
        """
        Train a new advantage net for the given player using data in advantage_memory[traverser].
        """
        new_model = self.generate_model()
        dataset = list(self.advantage_memory[traverser])
        random.shuffle(dataset)

        X_cards_list = [[] for _ in range(4)]
        X_bets_list = []
        X_mask_list = []
        y_regrets_list = []

        # CHANGED: We clamp regrets to avoid extreme outliers
        for (rs, regrets) in dataset:
            card_tensors, bet_tensor = new_model.tensorize_roundstate(rs, traverser)
            mask_tensor = new_model.tensorize_mask(rs)
            
            X_bets_list.append(bet_tensor)
            for i in range(4):
                X_cards_list[i].append(card_tensors[i])
            X_mask_list.append(mask_tensor)
            
            regrets_clamped = torch.clamp(regrets, min=-20, max=20)  # CHANGED
            y_regrets_list.append(regrets_clamped.unsqueeze(0))

        if len(X_bets_list) == 0:
            # no training data => just return the new (random) model
            return new_model

        # Stack them into large tensors
        X_bets = torch.cat(X_bets_list, dim=0)
        X_cards = [torch.cat(X_cards_list[i], dim=0) for i in range(4)]
        X_masks = torch.cat(X_mask_list, dim=0)      # [N, nactions]
        y_regrets = torch.cat(y_regrets_list, dim=0) # [N, nactions]

        criterion = torch.nn.MSELoss()
        # CHANGED: lower lr, small weight_decay for better generalization
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        n_samples = X_bets.shape[0]
        indices = list(range(n_samples))

        for epoch in range(self.num_epochs):
            random.shuffle(indices)
            epoch_loss = 0.0

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_cards = [x[batch_indices] for x in X_cards]
                batch_bets = X_bets[batch_indices]
                batch_masks = X_masks[batch_indices]
                batch_targets = y_regrets[batch_indices]

                # Forward pass
                pred_regrets = new_model(batch_cards, batch_bets)
                # Mask out illegal actions
                pred_regrets = pred_regrets * batch_masks
                batch_targets = batch_targets * batch_masks

                loss = criterion(pred_regrets, batch_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # CHANGED: Avoid .item() in every loop if you want to reduce CPU-GPU sync,
                # but for debugging, it's often fine to accumulate.
                epoch_loss += loss.item() * len(batch_indices)

            avg_loss = epoch_loss / n_samples
            print(f"[AdvTrain Player {traverser}][Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        return new_model

    def run_cfr_iteration(self, iteration_id):
        """
        One "iteration" of CFR: for each player, sample states and gather advantages,
        then train a new advantage net and store it.
        """
        for player in [0,1]:
            roundstates = self.generate_roundstates(player)
            self.gather_advantages(player, roundstates)

        new_models = []
        for player in [0,1]:
            new_model = self.train_advantage_net(player)
            new_models.append(new_model)

        for player in [0,1]:
            self.strategy_memory[player].append(new_models[player])

    def cfr_training(self):
        """
        Main driver: initialize random nets, then run cfr_iters loops.
        """
        # Initialize random for each player
        for player in [0,1]:
            self.strategy_memory[player].append(self.generate_model())

        for i in range(self.cfr_iters):
            print(f"CFR Iteration {i+1} / {self.cfr_iters}")
            self.run_cfr_iteration(i)

        print("CFR Training Complete!")
