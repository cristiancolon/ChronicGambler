'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7
import pandas as pd
import math
import numpy as np

all_ranges = [
    "AAo", "KKo", "QQo", "JJo", "TTo", "99o", "88o", "AKs", "77o", "AQs",
    "AJs", "AKo", "ATs", "AQo", "AJo", "KQs", "66o", "A9s", "ATo", "KJs",
    "A8s", "KTs", "KQo", "A7s", "A9o", "KJo", "55o", "QJs", "K9s", "A5s",
    "A6s", "A8o", "KTo", "QTs", "A4s", "A7o", "K8s", "A3s", "QJo", "K9o",
    "A5o", "A6o", "Q9s", "K7s", "JTs", "A2s", "QTo", "44o", "A4o", "K6s",
    "K8o", "Q8s", "A3o", "K5s", "J9s", "Q9o", "JTo", "K7o", "A2o", "K4s",
    "Q7s", "K6o", "K3s", "T9s", "J8s", "33o", "Q6s", "Q8o", "K5o", "J9o",
    "K2s", "Q5s", "T8s", "K4o", "J7s", "Q4s", "Q7o", "T9o", "J8o", "K3o",
    "Q6o", "Q3s", "98s", "T7s", "J6s", "K2o", "22o", "Q2s", "Q5o", "J5s",
    "T8o", "J7o", "Q4o", "97s", "J4s", "T6s", "J3s", "Q3o", "98o", "87s",
    "T7o", "J6o", "96s", "J2s", "Q2o", "T5s", "J5o", "T4s", "97o", "86s",
    "J4o", "T6o", "95s", "T3s", "76s", "J3o", "87o", "T2s", "85s", "96o",
    "J2o", "T5o", "94s", "75s", "T4o", "93s", "86o", "65s", "84s", "95o",
    "T3o", "92s", "76o", "74s", "T2o", "54s", "85o", "64s", "83s", "94o",
    "75o", "82s", "73s", "93o", "65o", "53s", "63s", "84o", "92o", "43s",
    "74o", "72s", "54o", "64o", "52s", "62s", "83o", "42s", "82o", "73o",
    "53o", "63o", "32s", "43o", "72o", "52o", "62o", "42o", "32o"
]
epsilon = 0.0001
class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        # opponent's stats
        self.fold_to_raise_preflop = 0
        self.three_bet_freq = 0
        self.cont_bet_postflop = 0
        self.opp_aggressiveness = .50
        self.opp_tightness = .50
        self.opp_range_size = int(self.opp_tightness * len(all_ranges))
        self.opp_range = all_ranges[:self.opp_range_size]
        self.opp_contribution = 0

        # my stats
        self.am_button = False
        self.my_bankroll = 0
        
        # game stats
        self.pot = 0

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        self.my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        self.am_button = not bool(active)  # True if you are the button
        self.opp_contribution = 0

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
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        if opp_cards:
            # We track back opponent's play with his range
            # TODO: update opponent's range, tightness, aggressiveness, etc
            pass

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
        
        curr_bet = my_pip + opp_pip
        self.pot += curr_bet
        self.opp_contribution += opp_pip
        self.bounty_hit = self.has_bounty_hit(my_cards, board_cards, my_bounty)

        pot_odds = self.get_pot_odds(continue_cost, opp_pip)
        calling_ev = self.get_call_ev(equity, continue_cost, opp_pip)
        am_button = self.am_button

        if street == 0:
            equity = self.get_preflop_equity(my_cards)
            if am_button:
                


        # TODO: Handle action based on equity, pot_odds, and calling_ev
        # TODO: If street is preflop, compute equity from equity_ranges.json. Otherwise, use outs * 2% trick
        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        if RaiseAction in legal_actions:
            if random.random() < 0.5:
                if equity > 2*pot_odds:
                    raise_amount = int(min_raise + 0.1 * (max_raise - min_raise))
                    return RaiseAction(raise_amount)
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()
    
    def expand_hand_range(self, hand_range):
        suits = 'shdc'
        expanded = []
        for hand in hand_range:
            ranks = hand[:2]
            if hand[-1] == 's':
                rank1, rank2 = ranks[0], ranks[1]
                for s in suits:
                    expanded.append([eval7.Card(f"{rank1}{s}"), eval7.Card(f"{rank2}{s}")])
            elif hand[-1] == 'o':
                rank1, rank2 = ranks[0], ranks[1]
                if rank1 == rank2:
                    for s1 in suits:
                        for s2 in suits:
                            if s1 != s2:
                                expanded.append([eval7.Card(f"{rank1}{s1}"), eval7.Card(f"{rank1}{s2}")])
                else:
                    for s1 in suits:
                        for s2 in suits:
                            if s1 != s2:
                                expanded.append([eval7.Card(f"{rank1}{s1}"), eval7.Card(f"{rank2}{s2}")])
            else:
                rank1 = ranks[0]
                for s1 in suits:
                    for s2 in suits:
                        if s1 != s2:
                            expanded.append([eval7.Card(f"{rank1}{s1}"), eval7.Card(f"{rank1}{s2}")])
        return expanded
    
    def condense_hand(self, hand):
        ranks = '23456789TJQKA'
        c1, c2 = hand[0], hand[1]
        if ranks.index(c1.rank) < ranks.index(c2.rank):
            c1, c2 = c2, c1
        if c1.suit == c2.suit:
            return f"{c1.rank}{c2.rank}s"
        else:
            return f"{c1.rank}{c2.rank}o"
    
    def get_preflop_equity(self, my_cards):
        with open('equity_ranges.json', 'r') as f:
            equity_ranges = pd.read_json(f, orient='index')
        obs_tight_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        obs_equity_vals = []
        hand = self.condense_hand(my_cards)
        for tight_val in obs_tight_vals:
            obs_equity_vals.append(equity_ranges[tight_val][hand])
        return np.interp(self.opp_tightness, obs_tight_vals, obs_equity_vals)
    
    def get_call_ev(self, equity, continue_cost, opp_contribution):
        normal_ev = (equity * self.pot) - ((1-equity) * continue_cost)
        bounty_ev = (equity * (self.pot + opp_contribution * 0.5 + 10)) - ((1-equity) * continue_cost)
        return bounty_ev if self.bounty_hit else normal_ev
    
    def get_pot_odds(self, continue_cost, opp_contribution):
        normal_odds = continue_cost / (self.pot + continue_cost + epsilon)
        bounty_odds = continue_cost / (self.pot + continue_cost + opp_contribution * 0.5 + 10 + epsilon)
        return bounty_odds if self.bounty_hit else normal_odds
    
    def has_bounty_hit(self, hole_cards, board_cards, bounty_rank):
        for card in hole_cards + board_cards:
            if card[0] == bounty_rank:
                return True
        return False
    
if __name__ == '__main__':
    run_bot(Player(), parse_args())
