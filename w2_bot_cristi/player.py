from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7


class Player(Bot):
    """
    Improved version of a poker bot for an advanced poker course at MIT.
    Incorporates:
      - Reduced overfolding preflop & postflop.
      - Clear 3-bet & 4-bet logic (less raise-fold).
      - Semi-bluff & double-barrel potential postflop.
      - Deeper usage of pot odds & EV.
      - Opponent AF-based style adaptation.
    """

    def __init__(self):
        self.has_checked = False
        self.preflop_raiser = False
        
        # Opponent action stats
        self.opponent_actions = {
            'raises': 0,
            'bets': 0,
            'calls': 0,
            'folds': 0
        }
        self.total_opponent_actions = 0
        self.opponent_af = 0
        
        # Street/pot tracking
        self.street = 0
        self.pot = 0
        self.raised_this_street = 0
    def handle_new_round(self, game_state, round_state, active):
        self.has_checked = False
        self.preflop_raiser = False
        self.street = 0
        self.pot = 0
        self.raised_this_street = 0

    def handle_round_over(self, game_state, terminal_state, active):
        # Compute Opponent Aggression Factor
        if self.total_opponent_actions > 0:
            self.opponent_af = (
                (self.opponent_actions['raises'] + self.opponent_actions['bets'])
                / self.total_opponent_actions
            )
        else:
            self.opponent_af = 0

        # Reset stats
        self.opponent_actions = {'raises': 0, 'bets': 0, 'calls': 0, 'folds': 0}
        self.total_opponent_actions = 0

    def categorize_preflop_hand(self, hole_cards):
        """
        Categorize starting hand into a simple tier: 'premium','strong','medium','weak'
        """
        ranks = '23456789TJQKA'
        rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
        rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
        idx1, idx2 = ranks.index(rank1), ranks.index(rank2)

        is_pair = (rank1 == rank2)
        is_suited = (suit1 == suit2)

        # Pair logic
        if is_pair:
            if idx1 >= ranks.index('J'):  # JJ, QQ, KK, AA
                return 'premium'
            elif idx1 >= ranks.index('7'):  # 77 - TT
                return 'strong'
            else:
                return 'medium'

        # A-x logic
        if 'A' in (rank1, rank2):
            other_rank = rank2 if rank1 == 'A' else rank1
            if ranks.index(other_rank) >= ranks.index('Q'):  # AQ or AK
                return 'premium'
            elif is_suited:
                return 'strong'
            else:
                return 'medium'

        # broadways
        broadways = {'T', 'J', 'Q', 'K', 'A'}
        if rank1 in broadways and rank2 in broadways:
            return 'strong'

        # suited connectors (small example)
        gap = abs(idx1 - idx2)
        if is_suited and gap == 1 and min(idx1, idx2) >= ranks.index('8'):
            return 'strong'

        # any card >=9
        if idx1 >= ranks.index('9') or idx2 >= ranks.index('9'):
            return 'medium'

        return 'weak'

    def has_bounty_hit(self, hole_cards, board_cards, bounty_rank):
        """
        Check if we or board contain the bounty card rank
        """
        for card in hole_cards + board_cards:
            if card[0] == bounty_rank:
                return True
        return False

    def evaluate_hand_strength(self, hole_cards, board_cards):
        """
        Monte Carlo simulation to approximate strength. 
        """
        deck = eval7.Deck()
        known_cards = [eval7.Card(c) for c in hole_cards + board_cards]
        deck.cards = [c for c in deck.cards if c not in known_cards]

        simulations = 700
        wins = 0
        ties = 0

        my_hand = [eval7.Card(c) for c in hole_cards]
        community = [eval7.Card(c) for c in board_cards]

        for _ in range(simulations):
            deck.shuffle()
            needed = 5 - len(community)
            opp_hole = deck.peek(2)
            sim_board = community + deck.peek(needed)

            my_score = eval7.evaluate(my_hand + sim_board)
            opp_score = eval7.evaluate(list(opp_hole) + sim_board)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1

        return (wins + 0.5 * ties) / simulations

    def evaluate_board_texture(self, board_cards):
        """
        Simple dryness/wetness check
        """
        if len(board_cards) < 3:
            return "unknown"
        ranks = '23456789TJQKA'
        suits = [c[1] for c in board_cards]
        unique_suits = len(set(suits))
        indices = sorted(ranks.index(c[0]) for c in board_cards)
        if len(indices) < 2:
            return "unknown"
        max_gap = max(indices[i+1] - indices[i] for i in range(len(indices) - 1))

        if unique_suits <= 2 or max_gap <= 2:
            return "wet"
        return "dry"

    def calculate_pot_odds(self, continue_cost, current_pot):
        """
        pot_odds = cost to call / (final pot)
        """
        if (current_pot + continue_cost) == 0:
            return 0
        return continue_cost / (current_pot + continue_cost)

    def get_expected_value(self, hand_strength, pot, continue_cost, bounty_hit):
        """
        EV = (HS * adjusted_winnings) - ((1 - HS) * cost_when_lose)
        """
        normal_winnings = pot + continue_cost
        if bounty_hit:
            adjusted_winnings = 1.5 * normal_winnings + 10
        else:
            adjusted_winnings = normal_winnings

        return (hand_strength * adjusted_winnings) - ((1 - hand_strength) * continue_cost)

    def calculate_bet_size(self, hand_strength, street, current_pot, board_texture="dry"):
        """
        Basic dynamic bet sizing approach
        """
        if street == 1:  # flop
            if hand_strength > 0.8:
                multiplier = 0.65
            elif hand_strength > 0.6:
                multiplier = 0.45
            else:
                multiplier = 0.25
        elif street == 2:  # turn
            if hand_strength > 0.8:
                multiplier = 0.65
            elif hand_strength > 0.6:
                multiplier = 0.45
            else:
                multiplier = 0.25
        elif street == 3:  # river
            if hand_strength > 0.8:
                multiplier = 0.7
            elif hand_strength > 0.6:
                multiplier = 0.45
            else:
                multiplier = 0.25
        else:
            multiplier = 0.25

        if board_texture == "wet":
            multiplier += 0.1

        base_bet = int(multiplier * current_pot)
        max_factor = 1.0
        max_bet = int(min(max_factor * current_pot, STARTING_STACK))
        final_bet = min(base_bet, max_bet)
        return max(final_bet, 1)

    def validate_raise_amount(self, desired_raise, min_raise, max_raise, my_stack):
        raise_amount = max(desired_raise, min_raise)
        return min(raise_amount, max_raise, my_stack)

    def safe_raise(self, round_state, desired_raise, my_pip, my_stack):
        legal_actions = round_state.legal_actions()
        if RaiseAction not in legal_actions:
            # fallback
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        min_raise, max_raise = round_state.raise_bounds()
        final_amount = self.validate_raise_amount(desired_raise, min_raise, max_raise, my_stack)
        if final_amount <= my_pip:
            # can't validly raise
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        return RaiseAction(final_amount)

    def categorize_opponent(self):
        if self.opponent_af > 2:
            return 'Overly Aggressive'
        elif self.opponent_af >= 1:
            return 'LAG'
        else:
            return 'TAG'

    def get_action(self, game_state, round_state, active):
        """
        Main decision function
        """
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:round_state.street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip

        current_bet_total = my_pip + opp_pip
        approximate_total_pot = self.pot + current_bet_total

        my_bounty = round_state.bounties[active]
        bounty_hit = self.has_bounty_hit(my_cards, board_cards, my_bounty)

        # Evaluate hand strength
        hand_strength = self.evaluate_hand_strength(my_cards, board_cards)

        pot_odds = self.calculate_pot_odds(continue_cost, approximate_total_pot)
        ev_call = self.get_expected_value(hand_strength, approximate_total_pot, continue_cost, bounty_hit)

        # track street changes
        if round_state.street != self.street:
            self.street = round_state.street
            self.raised_this_street = 0

        # decide: preflop vs postflop
        if self.street == 0:
            return self.handle_preflop(
                legal_actions,
                my_cards,
                round_state,
                active,
                ev_call,
                pot_odds,
                approximate_total_pot,
                hand_strength,
                opp_pip,
                my_stack,
                my_pip
            )
        else:
            return self.handle_postflop(
                legal_actions,
                my_cards,
                board_cards,
                round_state,
                active,
                hand_strength,
                ev_call,
                pot_odds,
                approximate_total_pot,
                my_stack
            )

    # -------------------------------
    # PRE-FLOP LOGIC
    # -------------------------------
    def handle_preflop(
        self,
        legal_actions,
        my_cards,
        round_state,
        active,
        ev_call,
        pot_odds,
        current_pot,
        hand_strength,
        opp_pip,
        my_stack,
        my_pip
    ):
        preflop_bet = (round_state.pips[active] > 0 or round_state.pips[1 - active] > 0)
        my_hand_tier = self.categorize_preflop_hand(my_cards)
        continue_cost = opp_pip - my_pip

        # If continue_cost <= 2 and we can call cheaply in BB, loosen up a bit
        if continue_cost <= 2 and CallAction in legal_actions:
            if random.random() < 0.35:  # 35% chance to loosen and call
                self.pot += continue_cost
                return CallAction()

        # If there's a raise
        if preflop_bet:
            if self.preflop_raiser:
                # We already raised
                # Possibly 3-bet or 4-bet logic: if strong enough, raise again
                if my_hand_tier in ['premium', 'strong'] and self.raised_this_street < 2:
                    # Instead of just 3x the opp_pip, consider if opp_pip is huge
                    # We'll do a simpler approach: 2.5-3x
                    multiplier = 2.5 if opp_pip < 10 else 2
                    desired_raise = int(multiplier * opp_pip)
                    action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                    if isinstance(action, RaiseAction):
                        self.raised_this_street += 1
                        self.pot += (action.amount - my_pip)
                        return action

                # Otherwise we can call more widely
                # especially if hand_strength > pot_odds - small buffer
                # or EV is positive
                if (hand_strength > pot_odds - 0.05) and (ev_call > 0):
                    if CallAction in legal_actions:
                        self.pot += (opp_pip - my_pip)
                        return CallAction()

                # fallback
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

            else:
                # We haven't raised yet, facing a bet or raise
                if my_hand_tier in ['premium', 'strong']:
                    if self.raised_this_street < 1:
                        # 3-bet
                        desired_raise = int(3.0 * opp_pip)
                        action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.preflop_raiser = True
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action

                    # If not raising further, consider calling
                    if ev_call > 0 and pot_odds < 0.45 and CallAction in legal_actions:
                        self.pot += (opp_pip - my_pip)
                        return CallAction()

                elif my_hand_tier == 'medium':
                    # We can call if the pot odds are good enough
                    if ev_call > 0 and pot_odds < 0.35 and CallAction in legal_actions:
                        self.pot += (opp_pip - my_pip)
                        return CallAction()

                # fallback
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

        else:
            # No one has raised yet
            if my_hand_tier in ['premium', 'strong']:
                # open-raise
                desired_raise = int(2.5 * BIG_BLIND)
                action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.preflop_raiser = True
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action

                # fallback
                if hand_strength > 0.15 and CallAction in legal_actions:
                    return CallAction()

            elif my_hand_tier == 'medium':
                # just check
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

            else:  # weak
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

    # -------------------------------
    # POST-FLOP LOGIC
    # -------------------------------
    def handle_postflop(
        self,
        legal_actions,
        my_cards,
        board_cards,
        round_state,
        active,
        hand_strength,
        ev_call,
        pot_odds,
        current_pot,
        my_stack
    ):
        opponent_type = self.categorize_opponent()
        board_texture = self.evaluate_board_texture(board_cards)
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip


        print(my_cards, round_state.deck[:round_state.street], hand_strength, pot_odds+.1)
        # OCCASIONAL SEMI-BLUFF if hand_strength < ~0.4
        # and we haven't raised yet
        if hand_strength < 0.40 and random.random() < 0.10 and RaiseAction in legal_actions:
            desired_bet = int(current_pot * 0.4)  # ~ 40% pot
            if self.raised_this_street < 2:
                action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action

        # If we were preflop raiser, we do c-bets
        if self.preflop_raiser:
            # Double-barrel logic: if hand_strength > pot_odds + small buffer
            if hand_strength > (pot_odds + 0.1):
                desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                if self.raised_this_street < 2:
                    action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                    if isinstance(action, RaiseAction):
                        self.raised_this_street += 1
                        self.pot += (action.amount - my_pip)
                        return action
                    elif CallAction in legal_actions and ev_call > 0:
                        self.pot += continue_cost
                        return CallAction()
                else:
                    if ev_call > 0 and CallAction in legal_actions:
                        self.pot += continue_cost
                        return CallAction()
                    if CheckAction in legal_actions:
                        return CheckAction()
                    return FoldAction()
            else:
                # Check if we can cheaply see next card
                if CheckAction in legal_actions:
                    return CheckAction()
                # If pot odds are still favorable
                if ev_call > (pot_odds + 0.05) and CallAction in legal_actions:
                    self.pot += continue_cost
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

        # if we didn't raise preflop
        else:
            if opponent_type == 'Overly Aggressive':
                # we call down a bit wider
                if hand_strength > 0.45 and ev_call > 0 and (CallAction in legal_actions):
                    self.pot += continue_cost
                    return CallAction()
                if ev_call > max(pot_odds, 0.15) and CallAction in legal_actions:
                    self.pot += continue_cost
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

            elif opponent_type == 'LAG':
                # if strong, raise
                if hand_strength > 0.65:
                    desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                    if self.raised_this_street < 2 and RaiseAction in legal_actions:
                        action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action
                # occasional bluff
                if hand_strength > 0.4 and random.random() < 0.05:
                    desired_bet = int(current_pot * 0.3)
                    if self.raised_this_street < 2 and RaiseAction in legal_actions:
                        action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action

                if ev_call > max(pot_odds, 0.15) and CallAction in legal_actions:
                    self.pot += continue_cost
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

            else:  # 'TAG'
                if hand_strength > 0.55:
                    desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                    if self.raised_this_street < 2 and RaiseAction in legal_actions:
                        action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action
                if ev_call > max(pot_odds, 0.15) and CallAction in legal_actions:
                    self.pot += continue_cost
                    return CallAction()
                print("BAD FOLD")
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

        # fallback logic
        if hand_strength > (pot_odds + 0.1):
            if self.raised_this_street < 2 and RaiseAction in legal_actions:
                desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action
            if CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()
        else:
            if FoldAction in legal_actions:
                return FoldAction()
            elif CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()

        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
