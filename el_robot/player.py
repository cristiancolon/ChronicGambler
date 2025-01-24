'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import pickle
import pandas as pd
import numpy as np
import eval7

# Assuming necessary classes like Bot, FoldAction, CallAction, RaiseAction, CheckAction, BoardState, TerminalState are defined elsewhere

class Player(Bot):
    '''
    A poker bot focusing on hand evaluation and betting strategy for a single board in Bounty Hold'em.
    '''

    def __init__(self):
        '''
        Initializes the poker bot.
        '''
        self.hole_strength = None
        self.our_fold_count = 0
        self.opp_fold_count = 0
        self.opp_fold_street = {0: 0, 3: 0, 4: 0, 5: 0}
        self.our_fold_street = {0: 0, 3: 0, 4: 0, 5: 0}
        self.opp_post_flop_raise_count = 0
        self.opp_post_flop_no_raise_count = 0
        self.opp_bets = []  # List to track opponent's bets
        self.opp_bets_strength = []
        self.opp_post_flop_call_count = 0
        self.our_post_flop_raise_count = 0
        self.num_showdowns = 0
        self.num_showdown_wins = 0
        self.num_showdown_losses = 0
        self.play_checkfold = False
        self.checkfold_threshold = 4000

        self.prev_opp_pips = 0
        self.prev_my_pips = 0

        # Load precomputed hand strengths
        try:
            with open('hand_strengths.p', 'rb') as fp:
                self.hand_strengths = pickle.load(fp)
        except FileNotFoundError:
            print("Error: 'hand_strengths.p' file not found. Please generate it using the provided script.")
            self.hand_strengths = {}
        except Exception as e:
            print(f"Error loading 'hand_strengths.p': {e}")
            self.hand_strengths = {}

        self.card_val_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.our_raise_rate = 0
        self.opp_raise_rate = 0
        self.our_fold_rate = 0
        self.opp_fold_rate = 0
        self.opp_bb_raise_count = 0
        self.opp_bb_raise_rate = 0
        self.opp_sb_raise_count = 0
        self.opp_sb_raise_rate = 0
        self.opp_sb_call_count = 0
        self.opp_sb_call_rate = None
        self.board_folded = False
        self.our_showdown_losses = 0
        self.our_showdown_wins = 0
        self.opp_board_strengths = []
        self.opp_board_avg_strength = None

        self.opp_bankroll = 0
        self.current_pot = 0

        # Initialize bounty tracking
        self.my_bounty = 0  # Tracking only your own bounty

        # Define tight hand range for aggressive opponents
        self.tight_hand_range = self.generate_tight_hand_range()
    
    def is_premium_hand(self, hole_cards):
        rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
        rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
        ranks = '23456789TJQKA'
        idx1 = ranks.index(rank1)
        idx2 = ranks.index(rank2)
        is_pair = rank1 == rank2
        is_suited = suit1 == suit2
        high_rank = rank1 if idx1 > idx2 else rank2
        low_rank = rank2 if idx1 > idx2 else rank1

        # All Pocket Pairs
        if is_pair:
            return True

        # All Aces
        if 'A' in [rank1, rank2]:
            return True

        # King-Ten or Better (KT+)
        

        # Suited Connectors Starting from Eight-Nine (89s+)
        if is_suited:
            gap = abs(idx1 - idx2)
            if gap == 1 and ranks.index(low_rank) >= ranks.index('K'):
                return True

        return False

    def is_strong_hand(self, hole_cards):
        rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
        rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
        ranks = '23456789TJQKA'
        idx1 = ranks.index(rank1)
        idx2 = ranks.index(rank2)
        is_pair = rank1 == rank2
        is_suited = suit1 == suit2
        high_rank = rank1 if idx1 > idx2 else rank2
        low_rank = rank2 if idx1 > idx2 else rank1

        # Suited Aces
        if is_suited and 'A' in [rank1, rank2] and ranks.index(low_rank) >= ranks.index('5'):
            return True
        
        if (high_rank == 'K' and ranks.index(low_rank) >= ranks.index('T')):
            return True

        # Queen-Jack or Better (QJ+)
        if (high_rank == 'Q' and ranks.index(low_rank) >= ranks.index('J')):
            return True

        # Suited Connectors (76s+)
        if is_suited and abs(idx1 - idx2) == 1 and ranks.index(low_rank) >= ranks.index('7'):
            return True

        return False

    def is_speculative_hand(self, hole_cards):
        rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
        rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
        ranks = '23456789TJQKA'
        idx1 = ranks.index(rank1)
        idx2 = ranks.index(rank2)
        is_pair = rank1 == rank2
        is_suited = suit1 == suit2
        high_rank = rank1 if idx1 > idx2 else rank2
        low_rank = rank2 if idx1 > idx2 else rank1

        # Suited Aces (A2s-A4s)
        if is_suited and 'A' in [rank1, rank2] and ranks.index(low_rank) <= ranks.index('5'):
            return True

        # Lower Suited Connectors (65s+, 87s+)
        if is_suited and abs(idx1 - idx2) == 1 and ranks.index(low_rank) >= ranks.index('3'):
            return True

        return False

    def is_weak_hand(self, hole_cards):
        # Any hand not categorized as Premium, Strong, or Speculative is Weak
        return not (self.is_premium_hand(hole_cards) or 
                    self.is_strong_hand(hole_cards) or 
                    self.is_speculative_hand(hole_cards))

    def categorize_hand(self, hole_cards):
        '''
        Categorizes the current hand into Premium, Strong, Speculative, or Weak.
        '''
        if self.is_premium_hand(hole_cards):
            return 'Premium'
        elif self.is_strong_hand(hole_cards):
            return 'Strong'
        elif self.is_speculative_hand(hole_cards):
            return 'Speculative'
        else:
            return 'Weak'

    def generate_tight_hand_range(self):
        '''
        Generates the tight hand range including:
        - Pairs: 22 through AA
        - Suited Connectors: 89s, 9Ts, TJs, JQs, QKs, KQs, KJs
        - All Suited Aces: A2s through AKs
        - Good Unsuited Aces: A5o through AKo
        '''
        pairs = [f"{rank}{rank}" for rank in self.card_val_order]
        
        suited_connectors = ['89s', '9Ts', 'TJs', 'JQs', 'QKs', 'KQs', 'KJs', '98s', 'T9s', 'JTs', 'QJs', 'JKs']
        
        suited_aces = [f"A{rank}s" for rank in ['2', '3', '4', '5', 'T', 'J', 'Q', 'K', 'A']] + [f"{rank}As" for rank in ['2', '3', '4', '5', 'T', 'J', 'Q', 'K', 'A']]
        
        good_unsuited_aces = ['ATo', 'AJo', 'AQo', 'AKo', 'TAo', 'JAo', 'QAo', 'KAo']
        
        return set(pairs + suited_connectors + suited_aces + good_unsuited_aces)

    def calculate_strength(self, hole, iters, board_cards, dead_cards, opp_known_cards=None):
        '''
        Estimates the win probability of a pair of hole cards using Monte Carlo simulations.
        '''
        deck = eval7.Deck()
        hole_cards = [eval7.Card(card) for card in hole]
        board_cards = [eval7.Card(card) for card in board_cards if card != '']
        dead_cards = [eval7.Card(card) for card in dead_cards]

        # Remove known cards from the deck
        for card in hole_cards + board_cards + dead_cards:
            if card in deck.cards:
                deck.cards.remove(card)

        score = 0
        if opp_known_cards is not None:
            opp_known_cards = [eval7.Card(card) for card in opp_known_cards]
            for card in opp_known_cards:
                if card in deck.cards:
                    deck.cards.remove(card)

            our_hand = hole_cards + board_cards
            opp_hand = opp_known_cards + board_cards
            our_hand_value = eval7.evaluate(our_hand)
            opp_hand_value = eval7.evaluate(opp_hand)

            if our_hand_value > opp_hand_value:
                return 1, 0  # Win
            elif our_hand_value < opp_hand_value:
                return 0, 0  # Loss
            else:
                return 0.5, 0.5  # Draw

        for _ in range(iters):
            deck.shuffle()
            _COMM = 5 - len(board_cards)
            _OPP = 2

            draw = deck.peek(_COMM + _OPP)
            opp_hole = draw[:_OPP]
            community = draw[_OPP:]
            community = community + board_cards  # Corrected Line

            our_hand = hole_cards + community
            opp_hand = opp_hole + community

            our_hand_value = eval7.evaluate(our_hand)
            opp_hand_value = eval7.evaluate(opp_hand)

            if our_hand_value > opp_hand_value:
                score += 2  # Win
            elif our_hand_value == opp_hand_value:
                score += 1  # Draw

        hand_strength = score / (2 * iters)  # Normalize to [0,1]
        draw_prob = (score % 2) / iters  # Optional: Calculate draw probability if needed

        return hand_strength, draw_prob

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Initializes variables for the round.
        '''
        self.hole_strength = {0: None, 3: None, 4: None, 5: None}
        self.opp_bets = []
        self.current_pot = 0
        self.prev_opp_pips = 0
        self.prev_my_pips = 0

        my_bankroll = game_state.bankroll
        opp_bankroll = self.opp_bankroll
        round_num = game_state.round_num
        big_blind = bool(active)
        self.current_hand_category =  self.categorize_hand(round_state.hands[active])

        # Update your own bounty from the current round
        self.my_bounty = round_state.bounties[active]

        if my_bankroll > self.checkfold_threshold + 1:
            if self.play_checkfold == False:
                print('Playing checkfold starting at round', round_num)
                print('opp post flop raise rate: ', self.opp_raise_rate)
                print('opp fold rate', self.opp_fold_rate)
                print('our fold rate', self.our_fold_rate)
                print('our showdown wins', self.our_showdown_wins)
                print('our showdown losses', self.our_showdown_losses)
            self.play_checkfold = True
        # else:
        #     self.play_checkfold = False

        if round_num >= 29 and round_num % 10 == 0:
            self.opp_raise_rate = (
                self.opp_post_flop_raise_count / 
                (self.opp_post_flop_raise_count + self.opp_post_flop_no_raise_count)
                if (self.opp_post_flop_raise_count + self.opp_post_flop_no_raise_count) > 0 
                else 0
            )
            if self.opp_post_flop_raise_count > 0:
                self.our_fold_rate = self.our_fold_count / (3 * (round_num + 1))
            if self.our_post_flop_raise_count > 0:
                self.opp_fold_rate = self.opp_fold_count / (3 * (round_num + 1))

        if round_num >= 14 and round_num % 10 == 0:
            if self.opp_board_strengths:
                self.opp_board_avg_strength = sum(self.opp_board_strengths) / len(self.opp_board_strengths)
            else:
                self.opp_board_avg_strength = 0.6

            self.opp_bb_raise_rate = (
                self.opp_bb_raise_count / 
                (3 * round_num / 2) 
                if (3 * round_num / 2) > 0 
                else 0
            )
            self.opp_sb_raise_rate = (
                self.opp_sb_raise_count / 
                (3 * round_num / 2) 
                if (3 * round_num / 2) > 0 
                else 0
            )

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Updates statistics based on the outcome.
        '''
        my_delta = terminal_state.deltas[active]
        opp_delta = terminal_state.deltas[1 - active]
        previous_state = terminal_state.previous_state
        street = previous_state.street
        opp_cards = previous_state.hands[1-active]
        self.opp_bankroll += opp_delta

        # Update bounty based on round outcome
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty

        if opp_cards != ['', ''] and opp_cards != []:
            showdown_res, _ = self.calculate_strength(
                previous_state.hands[active], 
                1, 
                previous_state.deck, 
                [], 
                opp_known_cards=opp_cards
            )
            self.num_showdowns += 1
            self.num_showdown_wins += showdown_res
            self.num_showdown_losses += (1 - showdown_res)

        if (opp_cards == ['', ''] or opp_cards == []) and not self.board_folded:
            self.opp_fold_count += 1
        elif self.board_folded:
            self.our_fold_count += 1

        if opp_cards != [] and opp_cards != ['', '']:
            card1_value = opp_cards[0][0]
            card2_value = opp_cards[1][0]
            card1_suit = opp_cards[0][1]
            card2_suit = opp_cards[1][1]
            suit_relation = 'same' if card1_suit == card2_suit else 'diff'
            if card1_value == card2_value:
                card_pair_tup = (card1_value, card2_value)
            else:
                card1_value_index = self.card_val_order.index(card1_value)
                card2_value_index = self.card_val_order.index(card2_value)
                if card1_value_index > card2_value_index:
                    card_pair_tup = (card2_value, card1_value, suit_relation)
                else:
                    card_pair_tup = (card1_value, card2_value, suit_relation)

            pair_strength = (
                self.hand_strengths.get(card_pair_tup, {}).get('win_prob', 0.5) + 
                0.5 * self.hand_strengths.get(card_pair_tup, {}).get('draw_prob', 0)
            )
            self.opp_board_strengths.append(pair_strength)

        for bet_entry in self.opp_bets:
            bet, board = bet_entry
            if opp_cards != ['', '']:
                _ITERS = 500
                opp_strength_at_bet, _ = self.calculate_strength(opp_cards, _ITERS, board, [])
                self.opp_bets_strength.append({'bet': bet, 'strength': opp_strength_at_bet})

        self.hole_strength = {0: None, 3: None, 4: None, 5: None}
        self.opp_bets = []
        self.board_folded = False

        # Assuming NUM_ROUNDS is defined somewhere in your code
        if game_state.round_num == NUM_ROUNDS:
            if not self.play_checkfold:
                print('Opponent post-flop raise rate:', self.opp_raise_rate)
                print('Opponent fold rate:', self.opp_fold_rate)
                print('Our fold rate:', self.our_fold_rate)
                print('Our showdown wins:', self.num_showdown_wins)
                print('Our showdown losses:', self.num_showdown_losses)
                print('Opponent average strength:', self.opp_board_avg_strength if self.opp_board_avg_strength else 0.6)
            print(game_state.game_clock)

    def get_bet_amount(self, strength, pot_size):
        '''
        Determines the bet amount based on hand strength and pot size using a polynomial model.
        '''
        # Polynomial model adjusted to cap at 1.3 before scaling
        bet_amount = (
            -106.02197802198144 + 
            616.7786499215267 * strength - 
            1310.2537938252638 * (strength ** 2) + 
            1206.0439560439938 * (strength ** 3) - 
            405.54683411827534 * (strength ** 4)
        )
        bet_amount = min(1.3, bet_amount)
        bet_amount = max(0, bet_amount)
        bet_amount = bet_amount * (pot_size + 25)
        
        # Ensure minimum bet is at least 12 or 20% of the pot
        # if int(bet_amount) <= 10:
        #     bet_amount = max(12, pot_size * 0.2)
        
        # Cap the bet to 398 to encourage opponents to call
        bet_amount = min(bet_amount, 398)
        #print(f"Calculated bet amount: {bet_amount}")
        return int(bet_amount)

    def tighten_hand_range(self, my_cards):
        '''
        Adjusts the bot's hand range to be tighter, focusing on pairs, suited connectors, and strong aces.
        Returns True if the hand is within the tight range, False otherwise.
        '''
        card1, card2 = my_cards
        val1, suit1 = card1[0], card1[1]
        val2, suit2 = card2[0], card2[1]
        if val1 == val2:
            hand = f"{val1}{val2}"
        elif abs(self.card_val_order.index(val1) - self.card_val_order.index(val2)) == 1 and suit1 == suit2:
            # Suited connectors are represented as 'XYs', e.g., '89s'
            high = max(val1, val2, key=lambda x: self.card_val_order.index(x))
            low = min(val1, val2, key=lambda x: self.card_val_order.index(x))
            hand = f"{high}{low}s"
        elif suit1 == suit2 and 'A' in (val1, val2):
            # All suited aces
            ace = val1 if val1 == 'A' else val2
            other = val2 if val1 == 'A' else val1
            hand = f"{ace}{other}s"
        elif 'A' in (val1, val2):
            # Good unsuited aces (A5o+)
            ace = 'A'
            other = val2 if val1 == 'A' else val1
            hand_rank = self.card_val_order.index(other)
            if hand_rank >= self.card_val_order.index('T'):
                hand = f"{ace}{other}o"
            else:
                return False
        else:
            return False  # Not in tight range
        #print(hand in self.tight_hand_range, hand)
        return hand in self.tight_hand_range

    def calculate_ev(self, hand_strength, pot, continue_cost, my_cards, board_cards):
        '''
        Calculates the Expected Value (EV) based on the current hand strength and bounty status.
        Checks if the bounty card is in the board or in the hand.
        '''
        normal_winnings = pot + continue_cost

        # Check if any of our hand cards are on the board (indicating our bounty card is exposed)
        bounty_card_in_play = self.my_bounty in board_cards or self.my_bounty in my_cards

        # Removed opponent bounty tracking

        if bounty_card_in_play:
            adjusted_win = 1.5 * normal_winnings + 10
        else:
            adjusted_win = normal_winnings

        # Removed opponent bounty influence on adjusted_lose
        adjusted_lose = continue_cost

        ev = (hand_strength * adjusted_win) - ((1 - hand_strength) * adjusted_lose)
        return ev

    def get_action(self, game_state, round_state, active):
        '''
        Determines the actions to take based on the current game state.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]  # Single board
        my_pips = round_state.pips[active]
        opp_pips = round_state.pips[1 - active]
        continue_cost = opp_pips - my_pips
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        min_raise, max_raise = round_state.raise_bounds()
        net_cost = 0

        # Adjust hand range based on opponent aggression

        if self.play_checkfold:
            # Replace CheckAction with a small bet
            if FoldAction in legal_actions:
                my_action = FoldAction()
                self.board_folded = True
            elif CheckAction in legal_actions:
                my_action = CheckAction()
            elif CallAction in legal_actions:
                my_action = CallAction()
            else:
                my_action = FoldAction()
            return my_action

        # Determine opponent's last action
        if continue_cost > 0:
            opp_last_action = 'Raise'
        elif self.prev_opp_pips == self.prev_my_pips:
            opp_last_action = 'Call'
            if game_state.round_num >= 3:
                self.opp_post_flop_call_count += 1
        else:
            opp_last_action = 'Check'

        if opp_last_action != 'Raise' and street >= 3:
            self.opp_post_flop_no_raise_count += 1

        self.prev_opp_pips = opp_pips
        self.prev_my_pips = my_pips

        # Board statistics
        self.current_pot = opp_pips + my_pips
        board_cont_cost = continue_cost
        

        if self.hole_strength[street] is None:
                NUM_ITERS = 500
                hole_cards = my_cards  # Assuming two hole cards
                dead_cards = list(set(my_cards) - set(hole_cards))
                hand_strength, _ = self.calculate_strength(hole_cards, NUM_ITERS, board_cards, dead_cards)
                self.hole_strength[street] = hand_strength
        else:
            hand_strength = self.hole_strength[street]

        if street == 0:
            # Preflop strategy
            if self.opp_bb_raise_rate >= 0.09 or self.opp_sb_raise_rate >= 0.09:
                    print(my_cards, "IN TIGHT RANGE", self.tighten_hand_range(my_cards), self.opp_bb_raise_rate, self.opp_sb_raise_rate) 
                    if not self.tighten_hand_range(my_cards):
                        if CheckAction in legal_actions:
                            return CheckAction()
                        if FoldAction in legal_actions:
                            return FoldAction()
                        else:
                            return CallAction() if CallAction in legal_actions else FoldAction()
            if active == 0:
                if board_cont_cost == 1:
                    if self.current_hand_category in ['Premium', 'Strong']:
                        # Play aggressively
                        raise_amount = self.get_bet_amount(self.hole_strength[0], self.current_pot)
                        raise_amount = max(min_raise, raise_amount)
                        raise_amount = min(max_raise, raise_amount, 398)  # Cap at 398
                        my_action = RaiseAction(int(raise_amount))
                        net_cost += raise_amount - my_pips
                        
                        return my_action
                    elif self.current_hand_category == 'Speculative':
                        # Play cautiously
                        if CallAction in legal_actions:
                            my_action = CallAction()
                            net_cost += board_cont_cost
                            
                            return my_action
                        else:
                            # If can't call, consider folding or a small raise
                            if CheckAction in legal_actions:
                                
                                return CheckAction()
                            else:
                                
                                my_action = FoldAction()
                                return my_action
                    else:
                        # Weak hands: Fold
                        if CheckAction in legal_actions:
                                
                            return CheckAction()
                        else:
                            my_action = FoldAction()
                        return my_action
                else:
                    # Opp raised from big blind
                    if continue_cost > 12:
                        self.opp_bb_raise_count += 1
                    pot_odds = board_cont_cost / (self.current_pot + board_cont_cost) if (self.current_pot + board_cont_cost) > 0 else 0
                    raw_hand_strength = self.hole_strength[0] if self.hole_strength[0] is not None else 0.5

                    # Adjust hand_strength based on opponent's raise rate
                    hand_strength = (
                        raw_hand_strength - 
                        2 * (self.opp_bb_raise_rate ** 2) 
                        if self.opp_bb_raise_rate is not None 
                        else raw_hand_strength - 2 * (0.2 ** 2)
                    )

                    if self.current_hand_category in ['Premium', 'Strong'] and hand_strength >= pot_odds:
                        if raw_hand_strength > 0.7:
                            raise_amount = min(max_raise, self.get_bet_amount(hand_strength, self.current_pot), 398)
                            raise_amount = max(min_raise, raise_amount)
                            my_action = RaiseAction(int(raise_amount))
                            net_cost += raise_amount - my_pips
                            
                        elif CallAction in legal_actions:
                            my_action = CallAction()
                            net_cost += board_cont_cost
                            
                        else:
                            if CheckAction in legal_actions:
                                
                                return CheckAction()
                            else:
                                
                                my_action = FoldAction()
                        return my_action
                    elif self.current_hand_category in ['Speculative'] and hand_strength >= pot_odds:
                        if CallAction in legal_actions:
                            my_action = CallAction()
                            net_cost += board_cont_cost
                            
                            return my_action
                        else:
                            if CheckAction in legal_actions:
                                
                                return CheckAction()
                            else:
                                
                                my_action = FoldAction()
                                return my_action
                    else:
                        if CallAction in legal_actions:
                            my_action = CallAction()
                            
                            return my_action
                        else:

                            if CheckAction in legal_actions:
                                return CheckAction()
                            else:
                                my_action = FoldAction()
                            return my_action
            else:
                # We are big blind
                if board_cont_cost > 0:
                    if continue_cost > 12:
                        self.opp_sb_raise_count += 1
                    pot_odds = board_cont_cost / (self.current_pot + board_cont_cost) if (self.current_pot + board_cont_cost) > 0 else 0
                    raw_hand_strength = self.hole_strength[0] if self.hole_strength[0] is not None else 0.5

                    # Adjust hand_strength based on opponent's raise rate
                    hand_strength = (
                        raw_hand_strength - 
                        2 * (self.opp_sb_raise_count ** 2) 
                        if self.opp_sb_raise_count is not None 
                        else raw_hand_strength - 2 * (0.2 ** 2)
                    )

                    if self.current_hand_category in ['Premium', 'Strong']:
                        print(my_cards, raw_hand_strength)
                        raise_amount = min(max_raise, self.get_bet_amount(hand_strength, self.current_pot), 398)
                        raise_amount = max(min_raise, raise_amount)
                        raise_cost = raise_amount - my_pips
                        if RaiseAction in legal_actions and (raise_cost <= my_stack - net_cost) and hand_strength > 0.7:
                            my_action = RaiseAction(int(raise_amount))
                            net_cost += raise_cost
                        elif CallAction in legal_actions:
                            my_action = CallAction()
                            net_cost += board_cont_cost
                        else:
                            if CheckAction in legal_actions:
                                
                                my_action = CheckAction()
                            else:
                                my_action = FoldAction()
                        return my_action
                    elif self.current_hand_category in ['Speculative'] and hand_strength >= pot_odds:
                        if CallAction in legal_actions:
                            my_action = CallAction()
                            net_cost += board_cont_cost
                        
                            return my_action
                        else:
                            if CheckAction in legal_actions:
                                my_action = CheckAction()
                            else:
                                my_action = FoldAction()
                            return my_action
                    else:
                        if CheckAction in legal_actions:
                                
                            return CheckAction()
                        else:
                            my_action = FoldAction()
                        return my_action
                else:
                    self.opp_sb_call_count += 1
                    pot_odds = board_cont_cost / (self.current_pot + board_cont_cost) if (self.current_pot + board_cont_cost) > 0 else 0
                    if self.current_hand_category in ['Premium', 'Strong']  and hand_strength >= pot_odds:
                        if RaiseAction in legal_actions:
                            raise_amount = min(max_raise, self.get_bet_amount(self.hole_strength[0], self.current_pot), 398)
                            raise_amount = max(min_raise, raise_amount)
                            my_action = RaiseAction(raise_amount)
                            net_cost += raise_amount - my_pips
                            
                            return my_action
                        else:
                            if CheckAction in legal_actions:
                                my_action = CheckAction()

                            else:
                                my_action = FoldAction()
                                
                            return my_action
                    elif self.current_hand_category in ['Speculative']:
                        if self.hole_strength[0] > 0.6:
                            if CallAction in legal_actions:
                                my_action = CallAction()
                                net_cost += board_cont_cost
                               
                                return my_action
                            else:
                                if CheckAction in legal_actions:
                                    my_action = CheckAction()
                                    
                                else:
                                    my_action = FoldAction()
                                    
                                return my_action
                        else:
                            if CheckAction in legal_actions:
                                
                                return CheckAction()
                            else:
                                my_action = FoldAction()
                            return my_action
                    else:
                        if CheckAction in legal_actions:
                                
                            return CheckAction()
                        else:
                            my_action = FoldAction()
                        return my_action
        # EV Calculation
        ev = self.calculate_ev(hand_strength, self.current_pot, board_cont_cost, my_cards, board_cards)
        pot_odds = board_cont_cost / (self.current_pot + board_cont_cost) if (self.current_pot + board_cont_cost) > 0 else 0

        if hand_strength > pot_odds:
            #print("ev > pot_odds", ev, pot_odds)
            # Positive EV: Decide to Raise or Call based on action availability
            raise_amount = self.get_bet_amount(hand_strength, self.current_pot)
            raise_amount = max(min_raise, raise_amount)
            raise_amount = min(max_raise, 398, raise_amount)
            raise_cost = raise_amount - my_pips

            if RaiseAction in legal_actions:
                my_action = RaiseAction(int(raise_amount))
                net_cost += raise_amount - my_pips
                self.our_post_flop_raise_count += 1
                return my_action
            elif CallAction in legal_actions:
                my_action = CallAction()
                net_cost += board_cont_cost
                return my_action
            else:
                # Replace CheckAction with a small bet
                if CheckAction in legal_actions:
                    return CheckAction()
                else:
                    my_action = FoldAction()
                return my_action
        else:
            # Negative or zero EV: Decide to Fold or Call to minimize losses
            if CheckAction in legal_actions:
                my_action = CheckAction()
                return my_action
            elif FoldAction in legal_actions:
                my_action = FoldAction()
                return my_action

        

if __name__ == '__main__':
    run_bot(Player(), parse_args())
