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
        self.preflop_raiser = False 
        self.opponent_actions = {
            'raises': 0,
            'bets': 0,
            'calls': 0,
            'folds': 0
        }
        self.total_opponent_actions = 0
        self.opponent_af = 0

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
        self.preflop_raiser = False

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
        # Update Aggression Factor after each round
        if self.total_opponent_actions > 0:
            self.opponent_af = (self.opponent_actions['raises'] + self.opponent_actions['bets']) / self.total_opponent_actions
        else:
            self.opponent_af = 0
        self.opponent_actions = {
            'raises': 0,
            'bets': 0,
            'calls': 0,
            'folds': 0
        }
        self.total_opponent_actions = 0
    
    def is_premium_hand(self, hole_cards):
        '''
        Determines if the given hole cards are premium for head-up play.

        Arguments:
        hole_cards: List of two strings representing the hole cards (e.g., ['Ts', '3d'])

        Returns:
        Boolean indicating if the hand is premium.
        '''
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
        '''
        Determines if a bounty has been hit based on hole cards and board cards.

        Arguments:
        hole_cards: List of two strings representing the hole cards.
        board_cards: List of strings representing the board cards.
        bounty_rank: String representing the bounty rank.

        Returns:
        Boolean indicating if a bounty has been hit.
        '''
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
            opp_hole = deck.peek(2)
            simulated_board = community + deck.peek(needed)
            my_full_hand = my_hand + simulated_board

            opp_full_hand = [opp_hole[0], opp_hole[1]] + simulated_board

            score_my_hand = eval7.evaluate(my_full_hand)
            score_opp_hand = eval7.evaluate(opp_full_hand)

            if score_my_hand > score_opp_hand:
                wins += 1
            elif score_my_hand == score_opp_hand:
                ties += 1

        return (wins + 0.5 * ties) / simulations

    def calculate_pot_odds(self, continue_cost, current_pot):
        '''
        Calculates the pot odds.

        Arguments:
        continue_cost: The amount needed to call.
        current_pot: The current size of the pot.

        Returns:
        Float representing the pot odds.
        '''
        if (current_pot + continue_cost) == 0:
            return 0  # Avoid division by zero
        return continue_cost / (current_pot + continue_cost)
    
    def categorize_opponent(self):
        '''
        Categorizes the opponent based on the Aggression Factor (AF).

        Returns:
        String indicating opponent type: 'TAG', 'LAG', or 'Overly Aggressive'
        '''
        if self.opponent_af > 2:
            return 'Overly Aggressive'
        elif self.opponent_af >= 1:
            return 'LAG'
        else:
            return 'TAG'

    def get_expected_value(self, hand_strength, pot, continue_cost, bounty_hit):
        '''
        Calculates the expected value (EV) of calling.

        Arguments:
        hand_strength: Probability of winning the hand.
        pot: Current pot size.
        continue_cost: The amount needed to call.
        bounty_hit: Boolean indicating if a bounty has been hit.

        Returns:
        Float representing the expected value.
        '''
        normal_winnings = pot + continue_cost

        if bounty_hit:
            adjusted_winnings = 1.5 * normal_winnings + 10
        else:
            adjusted_winnings = normal_winnings

        ev = (hand_strength * adjusted_winnings) - ((1 - hand_strength) * continue_cost)
        return ev


    def calculate_bet_size(self, hand_strength, street, current_pot):
        '''
        Calculates bet size dynamically based on hand strength and game stage.

        Arguments:
        hand_strength: Probability of winning.
        street: Current street (3: flop, 4: turn, 5: river).
        current_pot: Current pot size.

        Returns:
        Integer representing the bet amount.
        '''
        if street == 3:  # Flop
            if hand_strength > 0.8:
                multiplier = 1.0
            elif hand_strength > 0.6:
                multiplier = 0.75
            else:
                multiplier = 0.4
        elif street == 4:  # Turn
            if hand_strength > 0.8:
                multiplier = 1.0
            elif hand_strength > 0.6:
                multiplier = 0.75
            else:
                multiplier = 0.5
        elif street == 5:  # River
            if hand_strength > 0.8:
                multiplier = 1.2
            elif hand_strength > 0.6:
                multiplier = 1.0
            else:
                multiplier = 0.6
        else:
            multiplier = 0.5

        bet_size = int(multiplier * current_pot)
        return bet_size


    def validate_raise_amount(self, desired_raise, min_raise, max_raise, my_stack):
        '''
        Validates and adjusts the desired raise amount to ensure it's legal.

        Arguments:
        desired_raise: The initial desired raise amount.
        min_raise: The minimum raise allowed.
        max_raise: The maximum raise allowed.
        my_stack: The bot's current stack.

        Returns:
        A legal raise amount.
        '''
        raise_amount = max(desired_raise, min_raise)
        raise_amount = min(raise_amount, max_raise, my_stack)
        return raise_amount

    def get_action(self, game_state, round_state, active):
        '''
        Determines the bot's action based on opponent profiling, pot odds, EV, dynamic bet sizing, and an aggressive three-bet strategy.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action (FoldAction, CheckAction, CallAction, or RaiseAction).
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip
        my_bounty = round_state.bounties[active]

        bounty_hit = self.has_bounty_hit(my_cards, board_cards, my_bounty)
        hand_strength = self.evaluate_hand_strength(my_cards, board_cards)
        current_pot = my_pip + opp_pip
        pot_odds = self.calculate_pot_odds(continue_cost, current_pot)
        ev_call = self.get_expected_value(hand_strength, current_pot, continue_cost, bounty_hit)

        if street == 0:
            return self.handle_preflop(
                legal_actions, my_cards, round_state, active,
                ev_call, pot_odds, current_pot, hand_strength,
                opp_pip, my_stack, my_pip
            )
        else:
            return self.handle_postflop(
                legal_actions, my_cards, board_cards, round_state, active,
                hand_strength, ev_call, pot_odds, current_pot, my_stack
            )

    def handle_preflop(
    self, legal_actions, my_cards, round_state, active,
    ev_call, pot_odds, current_pot, hand_strength,
    opp_pip, my_stack, my_pip
    ):
        '''
        Handles pre-flop decision-making, including aggressive three-bet strategy.

        Arguments:
        legal_actions: List of legal actions.
        my_cards: List of two hole cards.
        round_state: Current RoundState object.
        active: Player index.
        ev_call: Expected value of calling.
        pot_odds: Calculated pot odds.
        current_pot: Current pot size.
        hand_strength: Probability of winning.
        opp_pip: Opponent's pips in this round.
        my_stack: Bot's current stack.
        my_pip: Bot's pips in this round.

        Returns:
        Action to take.
        '''
        preflop_bet = (round_state.street == 0) and (round_state.pips[active] > 0 or round_state.pips[1 - active] > 0)
        if preflop_bet:
            if self.is_premium_hand(my_cards):
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    desired_raise = 3 * opp_pip
                    three_bet_amount = self.validate_raise_amount(desired_raise, min_raise, max_raise, my_stack)
                    if three_bet_amount >= my_stack:
                        three_bet_amount = my_stack
                    if three_bet_amount > my_pip:
                        self.preflop_raiser = True
                        return RaiseAction(three_bet_amount)
            if hand_strength > pot_odds and CallAction in legal_actions:
                return CallAction()
            else:
                if FoldAction in legal_actions:
                    return FoldAction()
        else:
            if self.is_premium_hand(my_cards):
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    desired_raise = 2 * min_raise
                    open_raise_amount = self.validate_raise_amount(desired_raise, min_raise, max_raise, my_stack)
                    if open_raise_amount >= my_stack:
                        open_raise_amount = my_stack
                    if open_raise_amount > my_pip:
                        self.preflop_raiser = True
                        return RaiseAction(open_raise_amount)
            if CallAction in legal_actions:
                return CallAction()
            elif CheckAction in legal_actions:
                return CheckAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()

    def handle_postflop(
    self, legal_actions, my_cards, board_cards, round_state, active,
    hand_strength, ev_call, pot_odds, current_pot, my_stack
    ):
        '''
        Handles post-flop decision-making, including dynamic bet sizing.

        Arguments:
        legal_actions: List of legal actions.
        my_cards: List of two hole cards.
        board_cards: List of community cards.
        round_state: Current RoundState object.
        active: Player index.
        hand_strength: Probability of winning.
        ev_call: Expected value of calling.
        pot_odds: Calculated pot odds.
        current_pot: Current pot size.
        my_stack: Bot's current stack.

        Returns:
        Action to take.
        '''
        opponent_type = self.categorize_opponent()

        if self.preflop_raiser:
            if hand_strength > pot_odds:
                bet_amount = self.calculate_bet_size(hand_strength, round_state.street, current_pot)
                if RaiseAction in legal_actions and bet_amount > 0:
                    bet_amount = self.validate_raise_amount(bet_amount, *round_state.raise_bounds(), my_stack=my_stack)
                    if bet_amount >= my_stack:
                        bet_amount = my_stack
                    return RaiseAction(bet_amount)
                elif CallAction in legal_actions:
                    return CallAction()
            else:
                if CheckAction in legal_actions:
                    return CheckAction()
        else:
            if opponent_type == 'Overly Aggressive':
                if hand_strength > 0.5:
                    if CallAction in legal_actions:
                        return CallAction()
                else:
                    if ev_call > pot_odds and CallAction in legal_actions:
                        return CallAction()
                    elif FoldAction in legal_actions:
                        return FoldAction()
            elif opponent_type == 'LAG':
                if hand_strength > 0.7:
                    bet_amount = self.calculate_bet_size(hand_strength, round_state.street, current_pot)
                    if RaiseAction in legal_actions and bet_amount > 0:
                        bet_amount = self.validate_raise_amount(bet_amount, *round_state.raise_bounds(), my_stack=my_stack)
                        if bet_amount >= my_stack:
                            bet_amount = my_stack
                        return RaiseAction(bet_amount)
                elif hand_strength > 0.4 and random.random() < 0.3:
                    bet_amount = self.calculate_bet_size(hand_strength, round_state.street, current_pot)
                    if RaiseAction in legal_actions and bet_amount > 0:
                        bet_amount = self.validate_raise_amount(bet_amount, *round_state.raise_bounds(), my_stack=my_stack)
                        if bet_amount >= my_stack:
                            bet_amount = my_stack
                        return RaiseAction(bet_amount)
                else:
                    if ev_call > pot_odds and CallAction in legal_actions:
                        return CallAction()
                    elif FoldAction in legal_actions:
                        return FoldAction()
            elif opponent_type == 'TAG':
                if hand_strength > 0.6:
                    bet_amount = self.calculate_bet_size(hand_strength, round_state.street, current_pot)
                    if RaiseAction in legal_actions and bet_amount > 0:
                        bet_amount = self.validate_raise_amount(bet_amount, *round_state.raise_bounds(), my_stack=my_stack)
                        if bet_amount >= my_stack:
                            bet_amount = my_stack
                        return RaiseAction(bet_amount)
                elif ev_call > pot_odds and CallAction in legal_actions:
                    return CallAction()
                else:
                    if FoldAction in legal_actions:
                        return FoldAction()

        if hand_strength > pot_odds:
            if RaiseAction in legal_actions:
                bet_amount = self.calculate_bet_size(hand_strength, round_state.street, current_pot)
                bet_amount = self.validate_raise_amount(bet_amount, *round_state.raise_bounds(), my_stack=my_stack)
                if bet_amount >= my_stack:
                    bet_amount = my_stack
                return RaiseAction(bet_amount)
            elif CallAction in legal_actions:
                return CallAction()
        else:
            if FoldAction in legal_actions:
                return FoldAction()
            elif CallAction in legal_actions:
                return CallAction()

        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()



if __name__ == '__main__':
    run_bot(Player(), parse_args())
