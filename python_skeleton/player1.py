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
        self.has_checked = False

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
        # my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        # round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        # my_cards = round_state.hands[active]  # your cards
        # big_blind = bool(active)  # True if you are the big blind
        # my_bounty = round_state.bounties[active]  # your current bounty rank
        self.has_checked = False

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
        # my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        # previous_state = terminal_state.previous_state  # RoundState before payoffs
        # street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        # my_cards = previous_state.hands[active]  # your cards
        # opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        # my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        # opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        pass

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
        if (high_rank == 'K' and ranks.index(low_rank) >= ranks.index('T')):
            return True

        # Queen-Jack or Better (QJ+)
        if (high_rank == 'Q' and ranks.index(low_rank) >= ranks.index('J')):
            return True

        # Suited Connectors Starting from Eight-Nine (89s+)
        if is_suited:
            gap = abs(idx1 - idx2)
            if gap == 1 and ranks.index(low_rank) >= ranks.index('8'):
                return True

        return False
    
    def has_bounty_hit(self, hole_cards, board_cards, bounty_rank):
        for card in hole_cards + board_cards:
            if card[0] == bounty_rank:
                return True
        return False
    
    def evaluate_hand_strength(self, hole_cards, board_cards):
        deck = eval7.Deck()
        known_cards = [eval7.Card(card) for card in hole_cards + board_cards]
        deck.cards = [card for card in deck.cards if card not in known_cards]

        simulations = 100
        wins = 0
        ties = 0

        my_hand = [eval7.Card(card) for card in hole_cards]
        community = [eval7.Card(card) for card in board_cards]

        for _ in range(simulations):
            deck.shuffle()
            needed = 5 - len(community)
            simulated_board = community + deck.peek(needed)
            my_full_hand = my_hand + simulated_board

            opp_hole = deck.peek(2)
            opp_full_hand = [opp_hole[0], opp_hole[1]] + simulated_board

            score_my_hand = eval7.evaluate(my_full_hand)
            score_opp_hand = eval7.evaluate(opp_full_hand)

            if score_my_hand > score_opp_hand:
                wins += 1
            elif score_my_hand == score_opp_hand:
                ties += 1

        return (wins + 0.5 * ties) / simulations

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
        # legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        # street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        # my_cards = round_state.hands[active]  # your cards
        # board_cards = round_state.deck[:street]  # the board cards
        # my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        # opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        # my_stack = round_state.stacks[active]  # the number of chips you have remaining
        # opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        # continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        # my_bounty = round_state.bounties[active]  # your current bounty rank
        # my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        # opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        # if RaiseAction in legal_actions:
        #    min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
        #    min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
        #    max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        # if RaiseAction in legal_actions:
        #     if random.random() < 0.5:
        #         return RaiseAction(min_raise)
        # if CheckAction in legal_actions:  # check-call
        #     return CheckAction()
        # if random.random() < 0.25:
        #     return FoldAction()
        # return CallAction()
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:round_state.street]
        my_bounty = round_state.bounties[active]

        premium = self.is_premium_hand(my_cards)
        bounty_hit = self.has_bounty_hit(my_cards, board_cards, my_bounty)

        # All-In with Premium Hand
        if premium:
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(max_raise)
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()
            return FoldAction()

        # Call if holding a bounty card and cannot check
        if bounty_hit:
            can_check = CheckAction in legal_actions
            if not can_check and CallAction in legal_actions:
                return CallAction()
            if can_check and not self.has_checked:
                self.has_checked = True
                return CheckAction()
            if self.has_checked:
                hand_strength = self.evaluate_hand_strength(my_cards, board_cards)
                hand_strength_threshold = 0.6
                if hand_strength > hand_strength_threshold and RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(max_raise)
                if hand_strength > hand_strength_threshold and CallAction in legal_actions:
                    return CallAction()
                return FoldAction()

        # Non-Premium, Non-Bounty Handling
        can_check = CheckAction in legal_actions
        if can_check and not self.has_checked:
            self.has_checked = True
            return CheckAction()
        if self.has_checked:
            hand_strength = self.evaluate_hand_strength(my_cards, board_cards)
            hand_strength_threshold = 0.7
            if hand_strength > hand_strength_threshold and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(max_raise)
            return FoldAction()
        if can_check:
            return CheckAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
