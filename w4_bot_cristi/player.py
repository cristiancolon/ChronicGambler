import random
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import eval7

from skeleton.actions import FoldAction, CheckAction, CallAction, RaiseAction
from skeleton.states import STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# Import your DeepCFRModel
from neural_net import DeepCFRModel


class Player(Bot):
    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''   
        super().__init__()          
        self.bankroll = 0

        self.idx_to_action = dict()
        
        for idx, action in enumerate(["Fold", "Check", "Call", "Raise 1/2", "Raise 3/2"]):
            self.idx_to_action[idx] = action

        # Model parameters (must match training)
        self.nbets = 8
        self.nactions = 5

        # Load your trained DeepCFR model
        self.model = DeepCFRModel(4, self.nbets, self.nactions, t.device("cpu"))
        self.model.load_state_dict(t.load("models/player_1_model3.pth"))
        self.model.eval()  # evaluation mode
        self.bets = []

    def handle_new_round(self, game_state, terminal_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")
        
    def tensorize_mask(self, legal_actions):

        mask_tensor = t.zeros(self.nactions, dtype=t.float32)

        if FoldAction in legal_actions:
            mask_tensor[0] = 1
        
        if CheckAction in legal_actions:
            mask_tensor[1] = 1
        
        if CallAction in legal_actions:
            mask_tensor[2] = 1
        
        if RaiseAction in legal_actions:

            if self.min_raise <= math.ceil(self.pot*1/2) <= self.max_raise:
                mask_tensor[3] = 1
            if self.min_raise <= math.ceil(self.pot*3/2) <= self.max_raise:
                mask_tensor[4] = 1
        
        return mask_tensor


    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        bets = self.bets
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        self.pot = my_contribution + opp_contribution
        self.my_stack = my_stack

        raise_bounds = [None, None]

        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds() # the smallest and largest numbers of chips for a legal bet/raise
           self.min_raise = min_raise
           self.max_raise = max_raise 
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip 
    
        tensorized_bets = self.model.tensorize_bets(bets)
        my_cards = [eval7.Card(c) for c in my_cards]
        board_cards = [eval7.Card(c) for c in board_cards]
        tensorized_cards = self.model.tensorize_cards(my_cards, board_cards)
        mask_tensor = self.model.tensorize_mask(round_state).squeeze(0)

        model_regrets = self.model(tensorized_cards, tensorized_bets).squeeze(0)
        model_regrets = F.relu(mask_tensor*model_regrets)

        if t.sum(model_regrets) < 0.001:
            model_regrets = mask_tensor*t.ones(self.nactions)
        
        action_probabilities = model_regrets/(t.sum(model_regrets))

        selected_idx = int(t.multinomial(action_probabilities, 1, replacement=True))
        selected_action = self.idx_to_action[selected_idx]

        if selected_action == "Fold":
            output = FoldAction()

        elif selected_action == "Check":
            output = CheckAction()
        
        elif selected_action == "Call":
            output = CallAction()
        
        elif selected_action == "Raise 1/2":
            half_pot = math.ceil(self.pot*1/2)
            output = RaiseAction(half_pot)
        
        elif selected_action == "Raise 3/2":
            three_half_pot = math.ceil(self.pot*3/2)
            output = RaiseAction(three_half_pot)
        
        if isinstance(selected_action, FoldAction):
            self.bets.append(0)
        elif isinstance(selected_action, CallAction):
            # local_game sets 0 if call (except for SB calls BB, but keep it simple)
            self.bets.append(0)
        elif isinstance(selected_action, CheckAction):
            self.bets.append(0)
        elif isinstance(selected_action, RaiseAction):
            self.bets.append(selected_action.amount)

        return output


if __name__ == '__main__':
    run_bot(Player(), parse_args())
