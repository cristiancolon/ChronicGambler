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
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.preflop_dict = {'AAo':1,'KKo':2,'QQo':3,'JJo':4,'TTo':5,'99o':6,'88o':7,'AKs':8,'77o':9,'AQs':10,'AJs':11,'AKo':12,'ATs':13,
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
        
        self.trials = 125
        self.total_rounds = 0
        self.already_won = False
        self.nit = 0
        self.opp_aggressive = False

        self.switched_to_100 = False
        self.switched_to_50 = False



        self.small_blind_raise = 88
        self.big_blind_raise = 32
        self.big_blind_call = 88

        self.bluffed_this_round = False
        self.num_opp_potbets = 0
        self.num_opp_bets = 0

        self.raise_fact = .2
        self.reraise_fact = .025

        self.bluff_pm = 0

        self.bluffed_pm = 0
        self.bluff_numwins = 0
        self.bluff_numlosses = 0

        self.twobluff_pm = 0
        self.twonumwins = 0
        self.twonumlosses = 0

        self.onebluff_pm = 0
        self.onenumwins = 0
        self.onenumlosses = 0

        self.twobluff_fact = 1
        self.twobluff_not_working = False
        self.onebluff_fact = 1
        self.onebluff_not_working = False
        self.bluff_fact = 1
        self.bluff_not_working = 1
        self.draw_bluff_fact = 1
        self.draw_bluff_games = 0
        self.draw_bluff_losses = 0
        self.draw_bluff_pm = 0
        self.draw_bluff_this_round = False
        
        self.try_bluff = 1

        self.three_card_win = 0
        self.three_card_bet = 0
        self.check = 0
        self.opp_check_bluffs = 0
        self.opp_check_bluffing = False

        self.opp_checks = 0
        self.my_checks = 0
        self.last_cont = 0
        self.opp_check_bluff_this_round = False

        self.opp_auction_wins = 0
        self.opp_auction_bets = 0
        self.opp_auction_bluffing = False

        self.less_nit_call = False
        self.less_nit_call_pm = 0
        self.less_nit_call_losses = 0

        self.unnit_not_working = False

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
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        #print(f'round_num: {round_num}')
        self.opp_checks = 0
        self.my_checks = 0
        self.last_cont = 0
        self.opp_check_bluff_this_round = False
        
        self.opp_won_auction = False
        self.opp_auction_bet_this_round = False
        self.less_nit_call = False

        self.times_bet_preflop = 0
        self.bluffed_this_round = False
        self.twocheck = False
        self.onecheck = False
        self.bluff = False
        self.draw_hit = 0
        self.draw_hit_pct = 0

        self.draw_bluff_this_round = False

        if my_bankroll > 600:
            self.try_bluff = 1/4
        else:
            self.try_bluff = 1

        if self.bluff_not_working == 1:
            self.bluff_fact = 1
        elif self.bluff_not_working == 2:
            self.bluff_fact = 2
        else:
            self.bluff_fact = 1/6

        if not self.twobluff_not_working:
            self.twobluff_fact = 1
        else:
            self.twobluff_fact = 1/6

        if not self.onebluff_not_working:
            self.onebluff_fact = 1
        else:
            self.onebluff_fact = 1/6

        if my_bankroll > 2.295*(NUM_ROUNDS-round_num)+7.53*((NUM_ROUNDS-round_num)**(1/2)) + 50:
            self.already_won = True

        if game_clock < 20 and round_num <= 333 and not self.switched_to_100:
            self.trials = 100
            self.switched_to_100 = True
            self.nit = .03
            #print('switch to 100')

        elif game_clock < 10 and round_num <= 666 and not self.switched_to_50:
            self.trials = 50
            self.switched_to_50 = True
            self.nit = .06
            #print('switch to 50')

        if self.draw_bluff_losses >= 3 and self.draw_bluff_pm < -69:
            self.draw_bluff_fact = 1/4
        else:
            self.draw_bluff_fact = 1
        
        

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
        #print(my_delta)
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed

        if self.less_nit_call:
            if my_delta < 0:
                self.less_nit_call_losses += 1
            self.less_nit_call_pm += my_delta

        if self.less_nit_call_losses >= 3 and self.less_nit_call_pm < -69:
            self.unnit_not_working = True
            #print('unnit not working turned on True')
        else:
            self.unnit_not_working = False
            # print('unnit not working turned on False')

        self.total_rounds += 1

        if game_state.round_num == NUM_ROUNDS:
            print(game_state.game_clock)
            print(f'num opp bets {self.num_opp_bets}')
            print(f'num opp 80%: {self.num_opp_potbets}')
            print(f'bluff pm: {self.bluff_pm}')
            print(f'normal bluff pm: {self.bluffed_pm}')
            print(f'normal bluff wins: {self.bluff_numwins}')
            print(f'normal bluff losses: {self.bluff_numlosses}')
            print(f'two check bluff pm: {self.twobluff_pm}')
            print(f'two check wins: {self.twonumwins}')
            print(f'two check losses: {self.twonumlosses}')
            print(f'one check bluff pm: {self.onebluff_pm}')
            print(f'one check wins: {self.onenumwins}')
            print(f'one check losses: {self.onenumlosses}')
            print(f'checks {self.check}')
            print(f'opp check bets {self.opp_check_bluffs}')
            print(f'opp auction wins {self.opp_auction_wins}')
            print(f'opp auction flop bets {self.opp_auction_bets}')
            print(f'draw bluffs {self.draw_bluff_games}')
            print(f'draw losses {self.draw_bluff_losses}')
            print(f'draw pm {self.draw_bluff_pm}')

        if self.draw_bluff_this_round:
            self.draw_bluff_pm += my_delta
            self.draw_bluff_games += 1
            if my_delta < 0:
                self.draw_bluff_losses += 1
            

        if self.num_opp_bets >= 25 and (self.num_opp_potbets / self.num_opp_bets > .4):
            self.opp_aggressive = True
            #print('aggressive')
        else:
            self.opp_aggressive = False

        if self.opp_won_auction:
            self.opp_auction_wins += 1

        if (self.check >= 8) and (self.opp_check_bluffs / self.check >= .7):
            self.opp_check_bluffing = True
        else:
            self.opp_check_bluffing = False

        if (self.opp_auction_wins >= 10) and (self.opp_auction_bets / self.opp_auction_wins >= .7):
            self.opp_auction_bluffing = True
        else:
            self.opp_auction_bluffing = False

        if self.bluffed_this_round:
            if abs(my_delta) != 400:
                self.bluff_pm += my_delta

        if self.bluff:
            self.bluffed_pm += my_delta
            if my_delta > 0:
                self.bluff_numwins += 1
            else:
                self.bluff_numlosses += 1
            if ((self.bluff_numwins + self.bluff_numlosses >= 5) and (self.bluff_numlosses / (self.bluff_numwins + self.bluff_numlosses) >= .2) and self.bluffed_pm < 0) or (self.bluffed_pm < -250):
                #print('bluff not working!!')
                self.bluff_not_working = 0
            elif (self.bluff_numwins + self.bluff_numlosses >= 5) and (self.bluff_numlosses / (self.bluff_numwins + self.bluff_numlosses) <= .15) and self.bluffed_pm > 0:
                #print('bluff is working!!')
                self.bluff_not_working = 2
            else:
                self.bluff_not_working = 1

        elif self.twocheck:
            if abs(my_delta) != 400:
                self.twobluff_pm += my_delta
                if my_delta > 0:
                    self.twonumwins += 1
                else:
                    self.twonumlosses += 1
            if (not self.twobluff_not_working and (self.twonumwins + self.twonumlosses >= 8) and (self.twonumlosses / (self.twonumwins + self.twonumlosses) >= .3) and self.twobluff_pm < 0) or self.twobluff_pm < -250:
                #print('two bluff not working!!')
                self.twobluff_not_working = True

        elif self.onecheck:
            if abs(my_delta) != 400:
                self.onebluff_pm += my_delta
                if my_delta > 0:
                    self.onenumwins += 1
                else:
                    self.onenumlosses += 1
            if not self.onebluff_not_working and (self.onenumwins + self.onenumlosses >= 8) and (self.onenumlosses / (self.onenumwins + self.onenumlosses) >= .3) and self.onebluff_pm < 0:
                #print('one bluff not working!!')
                self.onebluff_not_working = True

            

    def categorize_cards(self,cards):
        rank1 = cards[0][0]
        rank2 = cards[1][0]
        suit1 = cards[0][1]
        suit2 = cards[1][1]
        hpair = ''
        onsuit = ''
        ranking = {'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4, '9': 5, '8': 6, '7': 7, '6': 8, '5': 9, '4': 10, '3': 11, '2': 12}

        if ranking[rank1]<ranking[rank2]:
            hpair = rank1+rank2
        else:
            hpair = rank2+rank1
        
        if suit1 == suit2:
            onsuit = 's'
        else:
            onsuit = 'o'
        
        return (hpair+onsuit)

    def no_illegal_raises(self,bet,round_state):
        min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise        
        if bet >= max_raise:
            return max_raise
        else:
            return bet

    def get_preflop_action(self,cards,round_state,active):
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        opp_pip = round_state.pips[1-active]
        pot = my_contribution+opp_contribution
        big_blind = bool(active)
        new_cards = self.categorize_cards(cards)
        if big_blind == False and self.times_bet_preflop == 0:
            if self.preflop_dict[new_cards] in range(1,26):
                self.times_bet_preflop +=1
                my_bet = 3*pot
                return RaiseAction(self.no_illegal_raises(my_bet,round_state))
            elif self.preflop_dict[new_cards] in range(20,self.small_blind_raise):
                self.times_bet_preflop +=1
                my_bet = 2*pot
                return RaiseAction(self.no_illegal_raises(my_bet,round_state))
            else:
                return FoldAction()
        elif big_blind == True and self.times_bet_preflop ==0:
            if self.preflop_dict[new_cards] in range(1,5) or (self.preflop_dict[new_cards] in range(5,self.big_blind_raise) and pot <= 20):
                self.times_bet_preflop +=1
                my_bet = 2*pot
                if RaiseAction in legal_actions:
                    return RaiseAction(self.no_illegal_raises(my_bet,round_state))
                elif CallAction in legal_actions:
                    return CallAction()
                else:
                    print("this shouldn't ever happen")
            elif opp_pip == 2 and self.preflop_dict[new_cards] in range(1,60) and random.random() < .69:
                self.times_bet_preflop +=1
                my_bet = 2*pot
                if RaiseAction in legal_actions:
                    return RaiseAction(self.no_illegal_raises(my_bet,round_state))
                return CheckAction()
            elif self.preflop_dict[new_cards] in range(5,int(self.big_blind_call+1-((opp_pip-2)/198)**(1/3)*(self.big_blind_call+1-5))) and opp_pip <= 200:
                if CallAction in legal_actions:
                    return CallAction()
                else:
                    return CheckAction()
            else:
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()
        else:
            if self.preflop_dict[new_cards] in range(1,5):
                self.times_bet_preflop +=1
                my_bet = 2*pot
                if RaiseAction in legal_actions:
                    return RaiseAction(self.no_illegal_raises(my_bet,round_state))
                elif CallAction in legal_actions:
                    return CallAction()
                else:
                    print("this shouldn't ever happen")
            elif self.preflop_dict[new_cards] in range(5, int(67-((opp_pip-2)/398)**(1/3)*61)):
                if CallAction in legal_actions:
                    return CallAction()
                else:
                    return CheckAction()
            else:
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()


    def decide_action_postflop(self, round_state, hand_strength, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_pip = round_state.pips[active]  
        opp_pip = round_state.pips[1-active]  
        my_stack = round_state.stacks[active] 
        opp_stack = round_state.stacks[1-active]  
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot = my_contribution + opp_contribution
        big_blind = bool(active)
        num_cards = len(round_state.hands[active])
        unnit = 0

        if street == 3:
            self.opp_won_auction = True

        if opp_pip > 0:
            if self.my_checks > 0:
                self.opp_check_bluffs += 1
                self.opp_check_bluff_this_round = True
            if street == 3 and self.opp_won_auction:
                self.opp_auction_bets += 1
                self.opp_auction_bet_this_round = True
            if my_pip > 0:
                self.bluffed_this_round = True
            self.num_opp_bets += 1
            self.opp_checks = 0
            self.last_cont = opp_contribution
        elif big_blind and street > 3:
            if opp_contribution == self.last_cont:
                self.opp_checks += 1
        elif not big_blind and opp_pip == 0:
            self.opp_checks += 1

        if opp_pip > .8*(pot - opp_pip + my_pip):
            self.num_opp_potbets += 1

        rand = random.random()
        if CheckAction in legal_actions: #Check, raise
            if self.opp_check_bluffing and hand_strength > .75 and street != 5: #let them bet into us, free dough
                self.check += 1
                self.my_checks += 1
                return CheckAction, None
            if rand < hand_strength + 0.15 and hand_strength >= (.5 + ((street % 3) * self.raise_fact)): # value raises, need stronger hands as street increases
                self.my_checks = 0
                self.opp_checks = 0
                return RaiseAction, 1 #value bet
            elif street == 5 and hand_strength > .9:
                self.my_checks = 0
                self.opp_checks = 0
                return RaiseAction, 1  #no checks on river with super strong hands
            elif self.draw_hit_pct > .25 and hand_strength >= .4 and street != 5 and not self.bluffed_this_round and rand <= self.draw_bluff_fact:
                self.my_checks = 0
                self.opp_checks = 0
                self.bluffed_this_round = True
                self.draw_bluff_this_round = True
                #print('semi bluff')
                return RaiseAction, 0
            elif self.opp_checks == 3 and rand < .8:
                #print('3 check bluff')
                return RaiseAction, 0
            elif not self.bluffed_this_round and not big_blind and (self.opp_checks == 2) and (rand < .869*self.try_bluff*self.twobluff_fact): #2 check bluff as dealer
                self.opp_checks = 0
                self.bluffed_this_round = True
                self.twocheck = True
                self.my_checks = 0
                #print('2 check bluff')
                return RaiseAction, 0
            elif not self.bluffed_this_round and big_blind and (self.opp_checks == 2) and (rand < self.try_bluff*.69*self.twobluff_fact): #2 check bluff as big blind
                self.opp_checks = 0
                self.bluffed_this_round = True
                self.twocheck = True
                self.my_checks = 0
                #print('2 check bluff')
                return RaiseAction, 0
            elif not self.bluffed_this_round and not big_blind and (self.opp_checks == 1) and (rand < self.try_bluff*.25*self.onebluff_fact): #1 check bluff as dealer
                self.opp_checks = 0
                self.bluffed_this_round = True
                self.onecheck = True
                self.my_checks = 0
                #print('1 check bluff')
                return RaiseAction, 0
            elif not self.less_nit_call and not self.bluffed_this_round and (rand < (self.try_bluff*self.bluff_fact*(1-hand_strength)/(1+(street%3)))) and (hand_strength < 0.65):
                self.opp_checks = 0  #3 card bluff after winning auction
                self.bluffed_this_round = True
                self.bluff = True
                self.my_checks = 0
                #print('bluffed')
                return RaiseAction, 0 #bluff
            self.check += 1
            self.my_checks += 1
            return CheckAction, None
        else: #Fold, Call, Raise
            pot_equity = (opp_pip-my_pip) / (pot - (opp_pip - my_pip))
            if pot_equity > .7 and pot_equity < .8:
                pot_equity = .7
            elif pot_equity >= .8 and pot_equity < 1.1:
                pot_equity = .8
            elif pot_equity >= 1.1:
                pot_equity = .85
            elif pot_equity <= .75:
                pot_equity = min(pot_equity + 0.0725,0.725)
            if pot_equity <= .5:
                pot_equity = min(pot_equity + 0.0725, .5)
            if self.opp_aggressive and pot_equity >= .8 and my_pip == 0:
                if num_cards == 2:
                    unnit += .1
                else:
                    unnit += .05
                #print('added less nit, bc aggressive')
            if self.opp_auction_bluffing and self.opp_auction_bet_this_round:
                if self.opp_aggressive and ((opp_pip-my_pip) / (pot - (opp_pip - my_pip)) > .8):
                    unnit += .15
                    #print('auction less nit')
                elif not self.opp_aggressive:
                    unnit += .1
                    #print('auction less nit')
            elif self.opp_check_bluffing and self.opp_check_bluff_this_round:
                if self.opp_aggressive and ((opp_pip-my_pip) / (pot - (opp_pip - my_pip)) > .8):
                    if num_cards == 2:
                        unnit += .1
                    else:
                        unnit += .05
                    #print('check less nit')
                elif not self.opp_aggressive and num_cards == 2:
                    unnit += .075
                    #print('check less nit')
            #print(f'unnit {unnit}')

            # if unnit not working, divide by two
            if self.unnit_not_working:
                unnit = unnit / 2
                #print('unnit not working, divided by two', f'unnit after divide {unnit}')

            pot_equity -= unnit
            if hand_strength > pot_equity and hand_strength < pot_equity + unnit and hand_strength > .35:
                #print('less nit call')
                self.less_nit_call = True
            self.my_checks = 0
            self.opp_check_bluff_this_round = False
            self.opp_auction_bet_this_round = False
            if hand_strength < pot_equity: #bad pot equity
                return FoldAction, None
            elif hand_strength < .35:
                return FoldAction, None
            else: #good pot equity
                reraise_strength = (.9 + ((street % 3) * self.reraise_fact)) 
                if not self.opp_check_bluff_this_round and (hand_strength > reraise_strength) or (hand_strength - pot_equity > .3 and hand_strength > (reraise_strength - .05)):
                    return RaiseAction, 1 #value raise
                return CallAction, None

    def hand_strength(self, round_state, street, active):
        board = [eval7.Card(x) for x in round_state.deck[:street]]
        my_hole = [eval7.Card(a) for a in round_state.hands[active]]
        comb = board + my_hole
        num_more_board = 5 - len(board)

        if len(my_hole) == 2 and street > 0:
            opp_num = 3
        elif len(my_hole) == 3 and street > 0:
            opp_num = 3
        else:
            opp_num = 2

        deck = eval7.Deck()
        for card in comb:
            deck.cards.remove(card)

        num_better = 0
        trials = 0
        self.draw_hit = 0

        while trials < self.trials:
            deck.shuffle()
            cards = deck.peek(opp_num + num_more_board)
            opp_hole = cards[:opp_num]
            board_rest = cards[opp_num:]
            my_val = eval7.evaluate(my_hole+board+board_rest)
            opp_value = eval7.evaluate(opp_hole+board+board_rest)
            if my_val > opp_value:
                num_better += 2
            if my_val == opp_value:
                num_better += 1
            trials += 1
            if my_val >= 67305472 and my_val <= 84715911:
                self.draw_hit += 1

        percent_better_than = num_better/(2*trials)
        self.draw_hit_pct = self.draw_hit/trials
        return percent_better_than

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
        # May be useful, but you may choose to not use.
        legal_actions = round_state.legal_actions() # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        #board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        #continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        self.draw_hit = 0
        self.draw_hit_pct = 0

        if self.already_won:
            if FoldAction in legal_actions:
                return FoldAction()
            else:
                return CheckAction()

        pot = my_contribution + opp_contribution
        min_raise, max_raise = round_state.raise_bounds()
        hand_strength = self.hand_strength(round_state, street, active) - self.nit
        # print(self.draw_hit_pct)
        #if self.draw_hit_pct > .25 and self.draw_hit_pct < 1:
            #print('DRAWWWWWWWWWW')

        if my_contribution > 100:
            hand_strength -= 0.03

        if street == 0:       
            return self.get_preflop_action(my_cards,round_state,active)
        else:
            if street == 3:
                self.last_cont = opp_contribution
            decision, conf = self.decide_action_postflop(round_state, hand_strength, active)

        rand = random.random()
        if decision == RaiseAction and RaiseAction in legal_actions:
            hand_strength_threshold = 0.8+0.05*(street % 3)
            if conf != 0 and hand_strength < hand_strength_threshold:
                bet_max = int((1+(2*(hand_strength**2)*rand)) * 3 * pot / 8)
                maximum = min(max_raise, bet_max)
                minimum = max(min_raise, pot / 4)
            else:
                maximum = min(max_raise, 7*pot/4)
                minimum = max(min_raise, 1.10*pot)
            if maximum <= minimum:
                amount = int(min_raise)
            else:
                amount = int(rand * (maximum - minimum) + minimum)
            return RaiseAction(amount)
        if decision == RaiseAction and RaiseAction not in legal_actions:
            if CallAction in legal_actions:
                return CallAction()
            self.check += 1
            self.my_checks += 1
            return CheckAction()
        return decision()

        

if __name__ == '__main__':
    run_bot(Player(), parse_args())
