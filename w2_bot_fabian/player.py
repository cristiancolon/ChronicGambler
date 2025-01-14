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
        # Street/pot tracking
        self.street = 0
        self.pot = 0
        self.raised_this_street = 0
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

        # Track opponent all-in frequency
        self.opponent_allin_count = 0
        self.opponent_total_hands = 0

        # Track opponent shown hole cards
        self.opponent_shown_cards = []

        # CHANGE: Track the opponent's bounty rank
        self.opponent_bounty_rank = None

        # BLUFF PREDICTOR: Track how many showdowns we see
        # and how many times the opponent had "trash" after big bets
        self.opponent_showdowns = 0
        self.opponent_bluff_reveals = 0

    # -------------------------------
    # ROUND MANAGEMENT
    # -------------------------------
    def handle_new_round(self, game_state, round_state, active):
        self.street = 0
        self.pot = 0
        self.raised_this_street = 0
        self.has_checked = False
        self.preflop_raiser = False

        self.opponent_total_hands += 1
        self.opponent_bounty_rank = round_state.bounties[1 - active]

    def handle_round_over(self, game_state, terminal_state, active):
        # Update Opponent Aggression Factor
        if self.total_opponent_actions > 0:
            self.opponent_af = (
                (self.opponent_actions['raises'] + self.opponent_actions['bets'])
                / self.total_opponent_actions
            )
        else:
            self.opponent_af = 0

        self.opponent_actions = {'raises': 0, 'bets': 0, 'calls': 0, 'folds': 0}
        self.total_opponent_actions = 0

        # See if there's a showdown with opp cards
        if hasattr(terminal_state, 'previous_state'):
            opp_cards_shown = terminal_state.previous_state.hands[1 - active]
            if opp_cards_shown:
                self.opponent_shown_cards.append(opp_cards_shown)
                # BLUFF PREDICTOR: Evaluate if this was a "trash" showdown after big bets
                # (Simplified: if opp's final board eq is < 0.05, we consider it trash.)
                # We only do this if the pot was big or if the opp made big bets.
                # For illustration, let's do a trivial approach:
                self.opponent_showdowns += 1
                # Evaluate opp final strength
                # This is a bit more involved in real code, but let's do a quick check:
                final_board = terminal_state.previous_state.deck[:terminal_state.previous_state.street]
                opp_eq = self.evaluate_hand_strength(opp_cards_shown, final_board)
                # if eq < 0.05 => "trash"
                if opp_eq < 0.05:
                    self.opponent_bluff_reveals += 1

        # Check if opponent went all-in
        if getattr(self, 'opponent_went_allin_this_hand', False):
            self.opponent_allin_count += 1
        self.opponent_went_allin_this_hand = False

    # -------------------------------
    # BLUFF PREDICTOR
    # -------------------------------
    def predict_bluff_chance(self):
        """
        Return a float from 0..1 representing how likely we think
        the opponent is to be bluffing big bets.
        """
        if self.opponent_showdowns <= 0:
            # No data => default mid estimate
            return 0.2
        return self.opponent_bluff_reveals / float(self.opponent_showdowns)

    # -------------------------------
    # OPPONENT MODEL
    # -------------------------------
    def infer_opponent_range(self):
        if len(self.opponent_shown_cards) == 0:
            return "unknown"
        premium_count = 0
        total_seen = len(self.opponent_shown_cards)
        for opp_hand in self.opponent_shown_cards:
            tier = self.categorize_preflop_hand(opp_hand)
            if tier == 'premium':
                premium_count += 1
        premium_ratio = premium_count / total_seen
        if premium_ratio > 0.5:
            return "very_tight"
        elif premium_ratio < 0.2:
            return "loose"
        else:
            return "mixed"

    def is_allin_spammer(self, threshold=0.30):
        if self.opponent_total_hands <= 0:
            return False
        allin_freq = self.opponent_allin_count / float(self.opponent_total_hands)
        return allin_freq >= threshold
    def opponent_bounty_in_play(self, board_cards):
        """
        Returns True if the opponent's bounty rank appears on the board,
        meaning they might get a bounty advantage.
        """
        if not self.opponent_bounty_rank:
            return False

        for card in board_cards:
            if card[0] == self.opponent_bounty_rank:
                return True
        return False
    # -------------------------------
    # HAND CATEGORIZATION
    # -------------------------------
    def categorize_preflop_hand(self, hole_cards):
        ranks = '23456789TJQKA'
        rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
        rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
        idx1, idx2 = ranks.index(rank1), ranks.index(rank2)

        is_pair = (rank1 == rank2)
        is_suited = (suit1 == suit2)

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

        # suited connectors
        gap = abs(idx1 - idx2)
        if is_suited and gap == 1 and min(idx1, idx2) >= ranks.index('8'):
            return 'strong'

        # any card >=9
        if idx1 >= ranks.index('9') or idx2 >= ranks.index('9'):
            return 'medium'

        return 'weak'

    # -------------------------------
    # HAND EVALUATION
    # -------------------------------
    def evaluate_hand_strength(self, hole_cards, board_cards):
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

    # -------------------------------
    # HELPER: board_has_paired
    # -------------------------------
    def board_has_paired(self, board_cards):
        ranks_count = {}
        for c in board_cards:
            r = c[0]
            ranks_count[r] = ranks_count.get(r, 0) + 1
            if ranks_count[r] >= 2:
                return True
        return False

    # -------------------------------
    # POT ODDS & EV
    # -------------------------------
    def calculate_pot_odds(self, continue_cost, current_pot):
        if (current_pot + continue_cost) == 0:
            return 0
        return continue_cost / (current_pot + continue_cost)

    def get_expected_value(self, hand_strength, pot, continue_cost, bounty_hit=False, opp_bounty_in_play=False):
        normal_winnings = pot + continue_cost
        if bounty_hit:
            adjusted_winnings = 1.5 * normal_winnings + 10
        else:
            adjusted_winnings = normal_winnings

        if opp_bounty_in_play:
            adjusted_winnings -= 10

        return (hand_strength * adjusted_winnings) - ((1 - hand_strength) * continue_cost)

    # -------------------------------
    # RAISE / OVERBET LOGIC
    # -------------------------------
    def validate_raise_amount(self, desired_raise, min_raise, max_raise, my_stack):
        raise_amount = max(desired_raise, min_raise)
        return min(raise_amount, max_raise, my_stack)

    def safe_raise(self, round_state, desired_raise, my_pip, my_stack):
        legal_actions = round_state.legal_actions()
        if RaiseAction not in legal_actions:
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        min_raise, max_raise = round_state.raise_bounds()
        final_amount = self.validate_raise_amount(desired_raise, min_raise, max_raise, my_stack)
        if final_amount <= my_pip:
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()
        return RaiseAction(min(final_amount, max_raise))

    def calculate_bet_size(self, hand_strength, street, current_pot, board_texture="dry"):
        # Preflop vs. postflop logic ...
        if street == 0:
            # bigger opens preflop
            if hand_strength > 0.8:
                multiplier = 3.0
            elif hand_strength > 0.6:
                multiplier = 2.5
            else:
                multiplier = 2.0
        elif street == 3:  # flop
            if hand_strength > 0.8:
                multiplier = 0.7
            elif hand_strength > 0.6:
                multiplier = 0.6
            else:
                multiplier = 0.5
        elif street == 4:  # turn
            if hand_strength > 0.8:
                multiplier = 0.7
            elif hand_strength > 0.6:
                multiplier = 0.6
            else:
                multiplier = 0.5
        elif street == 5:  # river
            if hand_strength > 0.8:
                multiplier = 0.7
            elif hand_strength > 0.6:
                multiplier = 0.6
            else:
                multiplier = 0.5
        else:
            multiplier = 4.0 / 3.0

        if board_texture == "wet":
            multiplier += 0.1

        base_bet = int(multiplier * current_pot)
        max_factor = 1.0
        max_bet = int(min(max_factor * current_pot, STARTING_STACK))
        final_bet = min(base_bet, max_bet)
        return max(final_bet, 1)

    def evaluate_board_texture(self, board_cards):
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

    def calculate_overbet_size(self, hand_strength, current_pot, my_stack, board_texture):
        max_overbet = my_stack
        base_overbet = current_pot * 2.0
        if hand_strength > 0.90:
            desired = int(base_overbet)
        elif hand_strength > 0.85:
            desired = int(current_pot * 1.5)
        else:
            desired = int(current_pot * 1.2)
        desired = min(desired, max_overbet)
        return max(desired, 1)

    # -------------------------------
    # OPPONENT CATEGORIZATION
    # -------------------------------
    def categorize_opponent(self):
        if self.opponent_af > 3:
            return 'Overly Aggressive'
        elif self.opponent_af >= 1.5:
            return 'LAG'
        else:
            return 'TAG'

    # -------------------------------
    # MAIN DECISION POINT
    # -------------------------------
    def get_action(self, game_state, round_state, active):
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

        # Track all-in
        if continue_cost >= my_stack:
            self.opponent_went_allin_this_hand = True

        board_texture = self.evaluate_board_texture(board_cards)
        hand_strength = self.evaluate_hand_strength(my_cards, board_cards)
        opp_bounty_play = self.opponent_bounty_in_play(board_cards)
        pot_odds = self.calculate_pot_odds(continue_cost, approximate_total_pot)

        # BLUFF PROB
        bluff_prob = self.predict_bluff_chance()

        # If the board is paired, the opponent is making a big bet, and bluff_prob is very low,
        # we fold if we don't have near-nut hands
        if self.board_has_paired(board_cards) and continue_cost > 0.5 * my_stack and bluff_prob < 0.1:
            # we only continue if hand_strength is super high
            if hand_strength < 0.85:
                # fold
                if FoldAction in legal_actions:
                    return FoldAction()
                # fallback
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction()

        # Also if we see a re-raise after we've raised => if hand_strength is < 0.03 => fold
        if self.raised_this_street >= 1 and opp_pip > my_pip:
            if hand_strength < 0.03:
                # no further bluff re-raise
                if CallAction in legal_actions and continue_cost < 0.05 * my_stack:
                    # tiny call
                    return CallAction()
                return FoldAction()

        # existing all-in logic
        if self.is_allin_spammer() and continue_cost >= my_stack:
            if hand_strength >= 0.25 and CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        if continue_cost >= my_stack:
            opp_range_est = self.infer_opponent_range()
            allin_freq = self.opponent_allin_count / max(self.opponent_total_hands, 1)
            if opp_range_est == "very_tight" and allin_freq < 0.05:
                required_eq = 0.45
            elif opp_range_est == "loose" or allin_freq > 0.30:
                required_eq = 0.30
            else:
                required_eq = 0.35

            if hand_strength >= required_eq and CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # track street changes
        if round_state.street != self.street:
            self.street = round_state.street
            self.raised_this_street = 0

        # preflop vs postflop calls
        if self.street == 0:
            return self.handle_preflop(
                legal_actions,
                my_cards,
                round_state,
                active,
                hand_strength,
                pot_odds,
                approximate_total_pot,
                continue_cost,
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
                pot_odds,
                approximate_total_pot,
                continue_cost,
                my_stack,
                board_texture,
                opp_bounty_play
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
        hand_strength,
        pot_odds,
        current_pot,
        continue_cost,
        my_stack,
        my_pip
    ):
        """
        Minimal-calling approach:
         - Rarely just call; mostly raise or fold.
        """
        my_hand_tier = self.categorize_preflop_hand(my_cards)
        preflop_bet = (round_state.pips[active] > 0 or round_state.pips[1 - active] > 0)
        is_button = (round_state.button == active)

        if preflop_bet:
            # If we can still raise => do so
            if continue_cost < 0.8 * my_stack and RaiseAction in legal_actions:
                desired_raise = int(2.2 * round_state.pips[1 - active])
                action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.preflop_raiser = True
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action
            # fallback: fold if cost is not tiny
            if continue_cost > 0.05 * my_stack:
                if FoldAction in legal_actions:
                    return FoldAction()
            # else maybe call if it's cheap
            if CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()
            return FoldAction()
        else:
            # no raise yet => open
            if is_button:
                # open with bigger raise if strong, smaller if weaker
                if my_hand_tier in ['premium', 'strong']:
                    desired_raise = int(3.0 * BIG_BLIND)
                elif my_hand_tier == 'medium':
                    desired_raise = int(2.5 * BIG_BLIND)
                else:
                    # occasionally raise with 'weak' anyway
                    if random.random() < 0.3:
                        desired_raise = int(2.2 * BIG_BLIND)
                    else:
                        if CheckAction in legal_actions:
                            return CheckAction()
                        return FoldAction()

                action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.preflop_raiser = True
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action
                # fallback
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()
            else:
                # not button => also raise often
                if my_hand_tier in ['premium', 'strong', 'medium']:
                    desired_raise = int(2.5 * BIG_BLIND)
                    action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                    if isinstance(action, RaiseAction):
                        self.preflop_raiser = True
                        self.raised_this_street += 1
                        self.pot += (action.amount - my_pip)
                        return action
                    # fallback
                    if CheckAction in legal_actions:
                        return CheckAction()
                    return FoldAction()
                else:
                    # 'weak' => fold or small chance to raise
                    if random.random() < 0.2 and RaiseAction in legal_actions:
                        desired_raise = int(2.2 * BIG_BLIND)
                        action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.preflop_raiser = True
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action
                    # fallback
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
        pot_odds,
        current_pot,
        continue_cost,
        my_stack,
        board_texture,
        opp_bounty_play
    ):
        """
        Illustrative postflop code. We add:
         - Avoid re-re-raising with pure air (hand_strength < 0.03)
         - If board is paired & big bets => fold if not near-nuts & opp seldom bluffs
        """
        opponent_type = self.categorize_opponent()
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        bluff_prob = self.predict_bluff_chance()

        # If we face big bet on a paired board & bluff prob is small => fold unless strong
        if self.board_has_paired(board_cards) and continue_cost > 0.5 * my_stack and bluff_prob < 0.1:
            if hand_strength < 0.85:
                # fold
                if FoldAction in legal_actions:
                    return FoldAction()
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction()

        # If re-raised after we've raised
        if self.raised_this_street >= 1 and (opp_pip > my_pip):
            if hand_strength < 0.03:
                # don't keep re-bluffing
                if continue_cost < 0.05 * my_stack and CallAction in legal_actions:
                    return CallAction()
                return FoldAction()

        # Normal c-bet logic or calls
        # e.g., if self.preflop_raiser, do a standard bet...
        if self.preflop_raiser:
            # c-bet if hand_strength > pot_odds
            if hand_strength > pot_odds:
                desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                if self.raised_this_street < 2 and RaiseAction in legal_actions:
                    action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                    if isinstance(action, RaiseAction):
                        self.raised_this_street += 1
                        self.pot += (action.amount - my_pip)
                        return action
                # fallback call
                if CallAction in legal_actions and hand_strength > (pot_odds - 0.05):
                    self.pot += (opp_pip - my_pip)
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()
            else:
                if CheckAction in legal_actions:
                    return CheckAction()
                if CallAction in legal_actions and hand_strength > (pot_odds - 0.1):
                    return CallAction()
                return FoldAction()
        else:
            # If we didn't raise preflop => adapt
            if opponent_type == 'Overly Aggressive':
                if hand_strength > 0.4 and CallAction in legal_actions:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()
            elif opponent_type == 'LAG':
                if hand_strength > 0.6 and RaiseAction in legal_actions:
                    desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                    if self.raised_this_street < 2:
                        action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action
                if hand_strength > (pot_odds - 0.05) and CallAction in legal_actions:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()
            else:  # 'TAG'
                if hand_strength > 0.55 and RaiseAction in legal_actions:
                    desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                    if self.raised_this_street < 2:
                        action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                        if isinstance(action, RaiseAction):
                            self.raised_this_street += 1
                            self.pot += (action.amount - my_pip)
                            return action
                if hand_strength > (pot_odds - 0.1) and CallAction in legal_actions:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
