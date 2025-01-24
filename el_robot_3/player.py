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
    A poker-playing bot.
    '''

    def __init__(self):
        '''
        Initializes a new game. Called once at the start.

        Arguments:
        None.

        Returns:
        None.
        '''
        self.hand_rankings = {'AAo':1,'KKo':2,'QQo':3,'JJo':4,'TTo':5,'99o':6,'88o':7,'AKs':8,'77o':9,'AQs':10,'AJs':11,'AKo':12,'ATs':13,
                             'AQo':14,'AJo':15,'KQs':16,'KJs':17,'A9s':18,'ATo':19,'66o':20,'A8s':21,'KTs':22,'KQo':23,'A7s':24,'A9o':25,'KJo':26,
                             '55o':27,'QJs':28,'K9s':29,'A5s':30,'A6s':31,'A8o':32,'KTo':33,'QTs':34,'A4s':35,'A7o':36,'K8s':37,'A3s':38,'QJo':39,
                             'K9o':40,'A5o':41,'A6o':42,'Q9s':43,'K7s':44,'JTs':45,'A2s':46,'QTo':47,'44o':48,'A4o':49,'K6s':50,'K8o':51,'Q8s':52,
                             'A3o':53,'K5s':54,'J9s':55,'Q9o':56,'JTo':57,'K7o':58,'A2o':59,'K4s':60,'Q7s':61,'K6o':62,'K3s':63,'T9s':64,'J8s':65,
                             '33o':66,'Q6s':67,'Q8o':68,'K5o':69,'J9o':70,'K2s':71,'Q5s':72,'T8s':73,'K4o':74,'J7s':75,'Q4s':76,'Q7o':77,'T9o':78,
                             'J8o':79,'K3o':80,'Q6o':81,'Q3s':82,'98s':83,'T7s':84,'J6s':85,'K2o':86,'22o':87,'Q2s':87,'Q5o':89,'J5s':90,'T8o':91,
                             'J7o':92,'Q4o':93,'97s':80,'J4s':95,'T6s':96,'J3s':97,'Q3o':98,'98o':99,'87s':75,'T7o':101,'J6o':102,'96s':103,'J2s':104,
                             'Q2o':105,'T5s':106,'J5o':107,'T4s':108,'97o':109,'86s':110,'J4o':111,'T6o':112,'95s':113,'T3s':114,'76s':80,'J3o':116,'87o':117,
                             'T2s':118,'85s':119,'96o':120,'J2o':121,'T5o':122,'94s':123,'75s':124,'T4o':125,'93s':126,'86o':127,'65s':128,'84s':129,'95o':130,
                             '53s':131,'92s':132,'76o':133,'74s':134,'65o':135,'54s':87,'85o':137,'64s':138,'83s':139,'43s':140,'75o':141,'82s':142,'73s':143,
                             '93o':144,'T2o':145,'T3o':146,'63s':147,'84o':148,'92o':149,'94o':150,'74o':151,'72s':152,'54o':153,'64o':154,'52s':155,'62s':156,
                             '83o':157,'42s':158,'82o':159,'73o':160,'53o':161,'63o':162,'32s':163,'43o':164,'72o':165,'52o':166,'62o':167,'42o':168,'32o':169,
                             }

        self.simulation_trials = 125
        self.round_counter = 0
        self.game_victory = False
        self.caution_level = 0
        self.opponent_is_aggressive = False

        self.switched_to_hundred = False
        self.switched_to_fifty = False

        self.min_raise_limit = 88
        self.max_raise_limit = 32
        self.call_limit = 88

        self.current_round_bluff = False
        self.opponent_pot_bets = 0
        self.opponent_total_bets = 0

        self.raise_chance = 0.2
        self.double_raise_chance = 0.025

        self.bluff_pnl = 0

        self.bluff_current_pnl = 0
        self.bluff_wins = 0
        self.bluff_losses = 0

        self.double_bluff_pnl = 0
        self.double_bluff_wins = 0
        self.double_bluff_losses = 0

        self.single_bluff_pnl = 0
        self.single_bluff_wins = 0
        self.single_bluff_losses = 0

        self.double_bluff_multiplier = 1
        self.double_bluff_disabled = False
        self.single_bluff_multiplier = 1
        self.single_bluff_disabled = False
        self.bluff_multiplier = 1
        self.bluff_disabled_state = 1
        self.draw_bluff_multiplier = 1
        self.draw_bluff_attempts = 0
        self.draw_bluff_failures = 0
        self.draw_bluff_pnl = 0
        self.has_drawn_bluff = False

        self.bluff_attempt = 1

        self.triple_win_count = 0
        self.triple_bet_count = 0
        self.check_counter = 0
        self.opponent_check_bluff_count = 0
        self.opponent_check_is_bluffing = False

        self.opponent_total_checks = 0
        self.my_total_checks = 0
        self.last_contribution_amount = 0
        self.opponent_bluffed_this_round = False

        self.opponent_auction_wins = 0
        self.opponent_auction_bets = 0
        self.opponent_auction_is_bluffing = False

        self.reduced_nit_call = False
        self.reduced_nit_pnl = 0
        self.reduced_nit_losses = 0

        self.nit_unreliable = False

        self.player_bounty = 0

    def has_bounty_card(self, hole_cards, board_cards, target_rank):
        for card in hole_cards + board_cards:
            if card[0] == target_rank:
                return True
        return False

    def handle_new_round(self, game_state, round_state, active_player):
        '''
        Invoked at the start of each new round. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active_player: your player's index.

        Returns:
        None.
        '''
        player_balance = game_state.bankroll  # Total chips gained or lost since game start
        remaining_time = game_state.game_clock  # Total seconds left to play
        current_round_num = game_state.round_num  # Current round number
        #player_hands = round_state.hands[active_player]  # Your current hand
        is_big_blind = bool(active_player)  # True if you're the big blind

        self.opponent_total_checks = 0
        self.my_total_checks = 0
        self.last_contribution_amount = 0
        self.opponent_bluffed_this_round = False

        self.opponent_auction_won = False
        self.opponent_auction_bet_current = False
        self.reduced_nit_call = False

        self.times_preflop_bet = 0
        self.current_round_bluff = False
        self.double_check = False
        self.single_check = False
        self.is_currently_bluffing = False
        self.draw_completion = 0
        self.draw_completion_percentage = 0

        self.has_drawn_bluff = False

        if player_balance > 600:
            self.bluff_attempt = 1/4
        else:
            self.bluff_attempt = 1

        if self.bluff_disabled_state == 1:
            self.bluff_multiplier = 1
        elif self.bluff_disabled_state == 2:
            self.bluff_multiplier = 2
        else:
            self.bluff_multiplier = 1/6

        if not self.double_bluff_disabled:
            self.double_bluff_multiplier = 1
        else:
            self.double_bluff_multiplier = 1/6

        if not self.single_bluff_disabled:
            self.single_bluff_multiplier = 1
        else:
            self.single_bluff_multiplier = 1/6

        if player_balance > 2.295*(NUM_ROUNDS-current_round_num)+7.53*((NUM_ROUNDS-current_round_num)**0.5) + 50:
            self.game_victory = True

        if remaining_time < 20 and current_round_num <= 333 and not self.switched_to_hundred:
            self.simulation_trials = 100
            self.switched_to_hundred = True
            self.caution_level = 0.03
            #print('switched to 100 trials')

        elif remaining_time < 10 and current_round_num <= 666 and not self.switched_to_fifty:
            self.simulation_trials = 50
            self.switched_to_fifty = True
            self.caution_level = 0.06
            #print('switched to 50 trials')

        if self.draw_bluff_failures >= 3 and self.draw_bluff_pnl < -69:
            self.draw_bluff_multiplier = 1/4
        else:
            self.draw_bluff_multiplier = 1

    def handle_round_over(self, game_state, terminal_state, active_player):
        '''
        Invoked at the end of each round. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active_player: your player's index.

        Returns:
        None.
        '''
        balance_change = terminal_state.deltas[active_player]  # Change in your bankroll this round
        #print(balance_change)
        prior_state = terminal_state.previous_state  # RoundState before payouts
        final_street = prior_state.street  # 0, 3, 4, or 5 indicating when the round ended
        #player_hands = prior_state.hands[active_player]  # Your cards
        #opponent_hands = prior_state.hands[1-active_player]  # Opponent's cards or [] if not revealed

        if self.reduced_nit_call:
            if balance_change < 0:
                self.reduced_nit_losses += 1
            self.reduced_nit_pnl += balance_change

        if self.reduced_nit_losses >= 3 and self.reduced_nit_pnl < -69:
            self.nit_unreliable = True
            #print('Nit strategy unreliable activated')
        else:
            self.nit_unreliable = False
            #print('Nit strategy remains reliable')

        self.round_counter += 1

        if game_state.round_num == NUM_ROUNDS:
            print(game_state.game_clock)
            print(f'Opponent total bets: {self.opponent_total_bets}')
            print(f'Opponent pot bets: {self.opponent_pot_bets}')
            print(f'Bluff PnL: {self.bluff_pnl}')
            print(f'Current Bluff PnL: {self.bluff_current_pnl}')
            print(f'Bluff Wins: {self.bluff_wins}')
            print(f'Bluff Losses: {self.bluff_losses}')
            print(f'Double Bluff PnL: {self.double_bluff_pnl}')
            print(f'Double Bluff Wins: {self.double_bluff_wins}')
            print(f'Double Bluff Losses: {self.double_bluff_losses}')
            print(f'Single Bluff PnL: {self.single_bluff_pnl}')
            print(f'Single Bluff Wins: {self.single_bluff_wins}')
            print(f'Single Bluff Losses: {self.single_bluff_losses}')
            print(f'Check Count: {self.check_counter}')
            print(f'Opponent Check Bluffs: {self.opponent_check_bluff_count}')
            print(f'Draw Bluff Attempts: {self.draw_bluff_attempts}')
            print(f'Draw Bluff Failures: {self.draw_bluff_failures}')
            print(f'Draw Bluff PnL: {self.draw_bluff_pnl}')

        if self.has_drawn_bluff:
            self.draw_bluff_pnl += balance_change
            self.draw_bluff_attempts += 1
            if balance_change < 0:
                self.draw_bluff_failures += 1

        if self.opponent_total_bets >= 25 and (self.opponent_pot_bets / self.opponent_total_bets > 0.4):
            self.opponent_is_aggressive = True
            #print('Opponent is aggressive')
        else:
            self.opponent_is_aggressive = False

        if self.opponent_auction_won:
            self.opponent_auction_wins += 1

        if (self.check_counter >= 8) and (self.opponent_check_bluff_count / self.check_counter >= 0.7):
            self.opponent_check_is_bluffing = True
        else:
            self.opponent_check_is_bluffing = False

        if (self.opponent_auction_wins >= 10) and (self.opponent_auction_bets / self.opponent_auction_wins >= 0.7):
            self.opponent_auction_is_bluffing = True
        else:
            self.opponent_auction_is_bluffing = False

        if self.current_round_bluff:
            if abs(balance_change) != 400:
                self.bluff_pnl += balance_change

        if self.is_currently_bluffing:
            self.bluff_current_pnl += balance_change
            if balance_change > 0:
                self.bluff_wins += 1
            else:
                self.bluff_losses += 1
            if ((self.bluff_wins + self.bluff_losses >= 5) and (self.bluff_losses / (self.bluff_wins + self.bluff_losses) >= 0.2) and self.bluff_current_pnl < 0) or (self.bluff_current_pnl < -250):
                #print('Bluff strategy failing')
                self.bluff_disabled_state = 0
            elif (self.bluff_wins + self.bluff_losses >= 5) and (self.bluff_losses / (self.bluff_wins + self.bluff_losses) <= 0.15) and self.bluff_current_pnl > 0:
                #print('Bluff strategy succeeding')
                self.bluff_disabled_state = 2
            else:
                self.bluff_disabled_state = 1

        elif self.double_check:
            if abs(balance_change) != 400:
                self.double_bluff_pnl += balance_change
                if balance_change > 0:
                    self.double_bluff_wins += 1
                else:
                    self.double_bluff_losses += 1
            if (not self.double_bluff_disabled and (self.double_bluff_wins + self.double_bluff_losses >= 8) and (self.double_bluff_losses / (self.double_bluff_wins + self.double_bluff_losses) >= 0.3) and self.double_bluff_pnl < 0) or self.double_bluff_pnl < -250:
                #print('Double bluff strategy failing')
                self.double_bluff_disabled = True

        elif self.single_check:
            if abs(balance_change) != 400:
                self.single_bluff_pnl += balance_change
                if balance_change > 0:
                    self.single_bluff_wins += 1
                else:
                    self.single_bluff_losses += 1
            if not self.single_bluff_disabled and (self.single_bluff_wins + self.single_bluff_losses >= 8) and (self.single_bluff_losses / (self.single_bluff_wins + self.single_bluff_losses) >= 0.3) and self.single_bluff_pnl < 0:
                #print('Single bluff strategy failing')
                self.single_bluff_disabled = True

    def categorize_hand(self, cards):
        first_rank = cards[0][0]
        second_rank = cards[1][0]
        first_suit = cards[0][1]
        second_suit = cards[1][1]
        sorted_hand = ''
        suit_status = ''
        rank_priority = {'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4, '9': 5, '8': 6, '7': 7, '6': 8, '5': 9, '4': 10, '3': 11, '2': 12}

        if rank_priority[first_rank] < rank_priority[second_rank]:
            sorted_hand = first_rank + second_rank
        else:
            sorted_hand = second_rank + first_rank

        if first_suit == second_suit:
            suit_status = 's'
        else:
            suit_status = 'o'

        return (sorted_hand + suit_status)

    def adjust_raise(self, proposed_bet, round_state):
        min_raise, max_raise = round_state.raise_bounds()  # Minimum and maximum raise amounts
        if proposed_bet >= max_raise:
            return max_raise
        else:
            return proposed_bet

    def preflop_strategy(self, cards, round_state, active_player):
        allowed_moves = round_state.legal_actions()  # Allowed actions
        my_stack = round_state.stacks[active_player]  # Your remaining chips
        opponent_stack = round_state.stacks[1-active_player]  # Opponent's remaining chips
        my_pot_contrib = STARTING_STACK - my_stack  # Your contribution to the pot
        opponent_pot_contrib = STARTING_STACK - opponent_stack  # Opponent's contribution to the pot
        opponent_current_pip = round_state.pips[1-active_player]
        total_pot = my_pot_contrib + opponent_pot_contrib
        is_big_blind = bool(active_player)
        categorized = self.categorize_hand(cards)

        if not is_big_blind and self.times_preflop_bet == 0:
            if self.hand_rankings[categorized] in range(1,26):
                print(self.player_bounty)
                self.times_preflop_bet +=1
                bet_amount = 3 * total_pot
                return RaiseAction(self.adjust_raise(bet_amount, round_state))
            elif self.hand_rankings[categorized] in range(20, self.min_raise_limit) or (self.player_bounty in cards and self.hand_rankings[categorized] in range(5,60)):
                self.times_preflop_bet +=1
                bet_amount = 2 * total_pot
                return RaiseAction(self.adjust_raise(bet_amount, round_state))
            else:
                return FoldAction()
        elif is_big_blind and self.times_preflop_bet == 0:
            if self.hand_rankings[categorized] in range(1,5) or (self.hand_rankings[categorized] in range(5, self.max_raise_limit) and total_pot <= 20) or (self.player_bounty in cards and self.hand_rankings[categorized] in range(5,60)):
                self.times_preflop_bet +=1
                bet_amount = 2 * total_pot
                if RaiseAction in allowed_moves:
                    return RaiseAction(self.adjust_raise(bet_amount, round_state))
                elif CallAction in allowed_moves:
                    return CallAction()
                else:
                    print("Unexpected action scenario")
            elif opponent_current_pip == 2 and self.hand_rankings[categorized] in range(1,60) and random.random() < 0.69:
                self.times_preflop_bet +=1
                bet_amount = 2 * total_pot
                if RaiseAction in allowed_moves:
                    return RaiseAction(self.adjust_raise(bet_amount, round_state))
                return CheckAction()
            elif self.hand_rankings[categorized] in range(5, int(self.call_limit +1 - ((opponent_current_pip-2)/198)**(1/3)*(self.call_limit +1 - 5))) and opponent_current_pip <= 200:
                if CallAction in allowed_moves:
                    return CallAction()
                else:
                    return CheckAction()
            else:
                if CheckAction in allowed_moves:
                    return CheckAction()
                return FoldAction()
        else:
            if self.hand_rankings[categorized] in range(1,5) or (self.hand_rankings[categorized] in range(5, 13) and total_pot < STARTING_STACK // 2):
                self.times_preflop_bet +=1
                bet_amount = 2 * total_pot
                print("RERAISE INITIATED", cards, bet_amount)
                if RaiseAction in allowed_moves:
                    return RaiseAction(self.adjust_raise(bet_amount, round_state))
                elif CallAction in allowed_moves:
                    return CallAction()
                else:
                    print("Unexpected action scenario")
            elif self.hand_rankings[categorized] in range(5, int(67 - ((opponent_current_pip-2)/398)**(1/3)*61)):
                if CallAction in allowed_moves:
                    return CallAction()
                else:
                    return CheckAction()
            else:
                if CheckAction in allowed_moves:
                    return CheckAction()
                return FoldAction()

    def postflop_decision(self, round_state, hand_strength_score, active_player):
        allowed_moves = round_state.legal_actions()
        current_phase = round_state.street
        my_current_pip = round_state.pips[active_player]
        opponent_current_pip = round_state.pips[1-active_player]
        my_remaining_stack = round_state.stacks[active_player]
        opponent_remaining_stack = round_state.stacks[1-active_player]
        my_pot_contrib = STARTING_STACK - my_remaining_stack
        opponent_pot_contrib = STARTING_STACK - opponent_remaining_stack
        total_pot = my_pot_contrib + opponent_pot_contrib
        is_big_blind = bool(active_player)
        card_count = len(round_state.hands[active_player])
        bluff_adjustment = 0

        if current_phase == 3:
            self.opponent_auction_won = True

        if opponent_current_pip > 0:
            if self.my_total_checks > 0:
                self.opponent_check_bluff_count += 1
                self.opponent_bluffed_this_round = True
            if current_phase == 3 and self.opponent_auction_won:
                self.opponent_auction_bets += 1
                self.opponent_auction_bet_current = True
            if my_current_pip > 0:
                self.current_round_bluff = True
            self.opponent_total_bets += 1
            self.opponent_total_checks = 0
            self.last_contribution_amount = opponent_pot_contrib
        elif is_big_blind and current_phase > 3:
            if opponent_pot_contrib == self.last_contribution_amount:
                self.opponent_total_checks += 1
        elif not is_big_blind and opponent_current_pip == 0:
            self.opponent_total_checks += 1

        if opponent_current_pip > 0.8 * (total_pot - opponent_current_pip + my_current_pip):
            self.opponent_pot_bets += 1

        rand_val = random.random()
        if CheckAction in allowed_moves:  # Can check or raise
            if self.opponent_check_is_bluffing and hand_strength_score > 0.75 and current_phase != 5:
                self.check_counter += 1
                self.my_total_checks += 1
                return CheckAction, None
            if rand_val < hand_strength_score + 0.15 and hand_strength_score >= (0.5 + ((current_phase % 3) * self.raise_chance)):
                self.my_total_checks = 0
                self.opponent_total_checks = 0
                return RaiseAction, 1  # Value bet
            elif current_phase == 5 and hand_strength_score > 0.9:
                self.my_total_checks = 0
                self.opponent_total_checks = 0
                return RaiseAction, 1  # Strong hand on river
            elif self.draw_completion_percentage > 0.25 and hand_strength_score >= 0.4 and current_phase != 5 and not self.current_round_bluff and rand_val <= self.draw_bluff_multiplier:
                self.my_total_checks = 0
                self.opponent_total_checks = 0
                self.current_round_bluff = True
                self.has_drawn_bluff = True
                #print('Initiated semi-bluff')
                return RaiseAction, 0
            elif self.opponent_total_checks == 3 and rand_val < 0.8:
                #print('3-check bluff initiated')
                return RaiseAction, 0
            elif not self.current_round_bluff and not is_big_blind and (self.opponent_total_checks == 2) and (rand_val < 0.869 * self.bluff_attempt * self.double_bluff_multiplier):
                self.opponent_total_checks = 0
                self.current_round_bluff = True
                self.double_check = True
                self.my_total_checks = 0
                #print('2-check bluff as dealer')
                return RaiseAction, 0
            elif not self.current_round_bluff and is_big_blind and (self.opponent_total_checks == 2) and (rand_val < self.bluff_attempt * 0.69 * self.double_bluff_multiplier):
                self.opponent_total_checks = 0
                self.current_round_bluff = True
                self.double_check = True
                self.my_total_checks = 0
                #print('2-check bluff as big blind')
                return RaiseAction, 0
            elif not self.current_round_bluff and not is_big_blind and (self.opponent_total_checks == 1) and (rand_val < self.bluff_attempt * 0.25 * self.single_bluff_multiplier):
                self.opponent_total_checks = 0
                self.current_round_bluff = True
                self.single_check = True
                self.my_total_checks = 0
                #print('1-check bluff as dealer')
                return RaiseAction, 0
            elif not self.reduced_nit_call and not self.current_round_bluff and (rand_val < (self.bluff_attempt * self.bluff_multiplier * (1 - hand_strength_score) / (1 + (current_phase % 3)))) and (hand_strength_score < 0.65):
                self.opponent_total_checks = 0  # Bluff after winning auction
                self.current_round_bluff = True
                self.is_currently_bluffing = True
                self.my_total_checks = 0
                #print('Bluff initiated')
                return RaiseAction, 0  # Bluff
            self.check_counter += 1
            self.my_total_checks += 1
            return CheckAction, None
        else:  # Can fold, call, or raise
            pot_equity = (opponent_current_pip - my_current_pip) / (total_pot - (opponent_current_pip - my_current_pip))
            if 0.7 < pot_equity < 0.8:
                pot_equity = 0.7
            elif 0.8 <= pot_equity < 1.1:
                pot_equity = 0.8
            elif pot_equity >= 1.1:
                pot_equity = 0.85
            elif pot_equity <= 0.75:
                pot_equity = min(pot_equity + 0.0725, 0.725)
            if pot_equity <= 0.5:
                pot_equity = min(pot_equity + 0.0725, 0.5)
            if self.opponent_is_aggressive and pot_equity >= 0.8 and my_current_pip == 0:
                if card_count == 2:
                    bluff_adjustment += 0.1
                else:
                    bluff_adjustment += 0.05
                #print('Adjusted bluff due to aggressive opponent')
            if self.opponent_auction_is_bluffing and self.opponent_auction_bet_current:
                if self.opponent_is_aggressive and ((opponent_current_pip - my_current_pip) / (total_pot - (opponent_current_pip - my_current_pip)) > 0.8):
                    bluff_adjustment += 0.15
                    #print('Auction bluff adjustment')
                elif not self.opponent_is_aggressive:
                    bluff_adjustment += 0.1
                    #print('Auction bluff without aggression adjustment')
            elif self.opponent_check_is_bluffing and self.opponent_bluffed_this_round:
                if self.opponent_is_aggressive and ((opponent_current_pip - my_current_pip) / (total_pot - (opponent_current_pip - my_current_pip)) > 0.8):
                    if card_count == 2:
                        bluff_adjustment += 0.1
                    else:
                        bluff_adjustment += 0.05
                    #print('Check bluff adjustment for aggressive opponent')
                elif not self.opponent_is_aggressive and card_count == 2:
                    bluff_adjustment += 0.075
                    #print('Check bluff adjustment for non-aggressive opponent with 2 cards')
            #print(f'Bluff Adjustment: {bluff_adjustment}')

            # If Nit strategy is unreliable, halve the bluff adjustment
            if self.nit_unreliable:
                bluff_adjustment /= 2
                #print('Nit strategy unreliable, halved bluff adjustment')

            pot_equity -= bluff_adjustment
            if hand_strength_score > pot_equity and hand_strength_score < pot_equity + bluff_adjustment and hand_strength_score > 0.35:
                #print('Initiating reduced nit call')
                self.reduced_nit_call = True
            self.my_total_checks = 0
            self.opponent_bluffed_this_round = False
            self.opponent_auction_bet_current = False
            if hand_strength_score < pot_equity:  # Poor pot equity
                return FoldAction, None
            elif hand_strength_score < 0.35:
                return FoldAction, None
            else:  # Favorable pot equity
                raise_threshold = (0.9 + ((current_phase % 3) * self.double_raise_chance))
                if not self.opponent_bluffed_this_round and (hand_strength_score > raise_threshold) or (hand_strength_score - pot_equity > 0.3 and hand_strength_score > (raise_threshold - 0.05)):
                    return RaiseAction, 1  # Value raise
                return CallAction, None

    def evaluate_hand_strength(self, round_state, phase, active_player):
        board = [eval7.Card(x) for x in round_state.deck[:phase]]
        my_hole = [eval7.Card(a) for a in round_state.hands[active_player]]
        combined = board + my_hole
        remaining_board = 5 - len(board)

        if len(my_hole) == 2 and phase > 0:
            opponent_cards = 3
        elif len(my_hole) == 3 and phase > 0:
            opponent_cards = 3
        else:
            opponent_cards = 2

        deck = eval7.Deck()
        for card in combined:
            deck.cards.remove(card)

        superior_hands = 0
        trial_count = 0
        self.draw_completion = 0

        while trial_count < self.simulation_trials:
            deck.shuffle()
            sampled = deck.peek(opponent_cards + remaining_board)
            opponent_hole = sampled[:opponent_cards]
            board_extension = sampled[opponent_cards:]
            my_hand_value = eval7.evaluate(my_hole + board + board_extension)
            opponent_hand_value = eval7.evaluate(opponent_hole + board + board_extension)
            if my_hand_value > opponent_hand_value:
                superior_hands += 2
            if my_hand_value == opponent_hand_value:
                superior_hands += 1
            trial_count += 1
            if 67305472 <= my_hand_value <= 84715911:
                self.draw_completion += 1

        percentage_better = superior_hands / (2 * trial_count)
        self.draw_completion_percentage = self.draw_completion / trial_count
        return percentage_better

    def get_action(self, game_state, round_state, active_player):
        '''
        Determines the bot's action during the game.
        Invoked whenever an action is required.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active_player: your player's index.

        Returns:
        The chosen action.
        '''
        allowed_moves = round_state.legal_actions()  # Allowed actions
        current_phase = round_state.street  # Current phase: 0, 3, 4, or 5
        player_hand = round_state.hands[active_player]  # Your current hand
        #board_cards = round_state.deck[:current_phase]  # Current board
        my_current_pip = round_state.pips[active_player]  # Your current pip
        opponent_current_pip = round_state.pips[1-active_player]  # Opponent's current pip
        my_remaining_stack = round_state.stacks[active_player]  # Your remaining chips
        opponent_remaining_stack = round_state.stacks[1-active_player]  # Opponent's remaining chips
        #continue_cost = opponent_current_pip - my_current_pip  # Chips needed to stay in pot
        my_pot_contrib = STARTING_STACK - my_remaining_stack  # Your contribution to pot
        opponent_pot_contrib = STARTING_STACK - opponent_remaining_stack  # Opponent's contribution to pot
        self.draw_completion = 0
        self.draw_completion_percentage = 0

        self.player_bounty = round_state.bounties[active_player]

        if self.game_victory:
            if FoldAction in allowed_moves:
                return FoldAction()
            else:
                return CheckAction()

        total_pot = my_pot_contrib + opponent_pot_contrib
        min_raise, max_raise = round_state.raise_bounds()
        strength = self.evaluate_hand_strength(round_state, current_phase, active_player) - self.caution_level
        # print(self.draw_completion_percentage)
        #if self.draw_hit_pct > .25 and self.draw_hit_pct < 1:
            #print('DRAWWWWWWWWWW')

        if my_pot_contrib > 100:
            strength -= 0.03

        if current_phase == 0:
            return self.preflop_strategy(player_hand, round_state, active_player)
        else:
            if current_phase == 3:
                self.last_contribution_amount = opponent_pot_contrib
            decision, confidence = self.postflop_decision(round_state, strength, active_player)

        rand_val = random.random()
        if decision == RaiseAction and RaiseAction in allowed_moves:
            strength_limit = 0.8 + 0.05 * (current_phase % 3)
            if confidence != 0 and strength < strength_limit:
                max_bet = int((1 + (2 * (strength ** 2) * rand_val)) * 3 * total_pot / 8)
                final_max = min(max_raise, max_bet)
                final_min = max(min_raise, total_pot / 4)
            else:
                final_max = min(max_raise, 7 * total_pot / 4)
                final_min = max(min_raise, 1.10 * total_pot)
            if final_max <= final_min:
                bet_amount = int(min_raise)
            else:
                bet_amount = int(rand_val * (final_max - final_min) + final_min)
            return RaiseAction(bet_amount)
        if decision == RaiseAction and RaiseAction not in allowed_moves:
            if CallAction in allowed_moves:
                return CallAction()
            self.check_counter += 1
            self.my_total_checks += 1
            return CheckAction()
        return decision()

        

if __name__ == '__main__':
    run_bot(Player(), parse_args())
