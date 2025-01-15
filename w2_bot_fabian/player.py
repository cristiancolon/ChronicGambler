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

        # Opponent bounty rank (a single character rank, e.g. 'Q' or '7')
        self.opponent_bounty_rank = None

        # Bluff predictor stats
        self.opponent_showdowns = 0
        self.opponent_bluff_reveals = 0

        # Basic range tracking
        self.opponent_range = None

        # "ultra_premium" hands
        self.ultra_premium_hands = {
            ('A','A'), ('K','K'), ('Q','Q'),
            ('A','K')
        }

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
        self.opponent_range = self.generate_full_preflop_range()

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

        # Showdown
        if hasattr(terminal_state, 'previous_state'):
            opp_cards_shown = terminal_state.previous_state.hands[1 - active]
            if opp_cards_shown:
                self.opponent_shown_cards.append(opp_cards_shown)
                self.opponent_showdowns += 1
                final_board = terminal_state.previous_state.deck[:terminal_state.previous_state.street]
                opp_eq = self.evaluate_hand_strength(opp_cards_shown, final_board)
                if opp_eq < 0.05:
                    self.opponent_bluff_reveals += 1

        # Track all-in
        if getattr(self, 'opponent_went_allin_this_hand', False):
            self.opponent_allin_count += 1
        self.opponent_went_allin_this_hand = False

    # -------------------------------
    # BLUFF PREDICTOR
    # -------------------------------
    def predict_bluff_chance(self):
        if self.opponent_showdowns <= 0:
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
        Returns True if the opponent's bounty rank is on the board.
        If we lose the hand and the opponent's bounty is in play,
        that means they get the 1.5x + 10 outcome. 
        """
        if not self.opponent_bounty_rank:
            return False
        for card in board_cards:
            if card[0] == self.opponent_bounty_rank:
                return True
        return False

    def hero_bounty_in_play(self, board_cards, hero_bounty_rank='A'):
        # If we had a bounty rank ourselves, we'd check similarly.
        for card in board_cards:
            if card[0] == hero_bounty_rank:
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

        big_pairs = {'AA','KK','QQ'}
        combo = rank1 + rank2
        sorted_2cards = ''.join(sorted([rank1, rank2], key=lambda r: ranks.index(r)))
        if sorted_2cards == 'AK':
            return 'premium'
        if combo in big_pairs or combo[::-1] in big_pairs:
            return 'premium'

        if is_pair:
            if idx1 >= ranks.index('J'):
                return 'premium'
            elif idx1 >= ranks.index('7'):
                return 'strong'
            else:
                return 'medium'

        # A-x
        if 'A' in (rank1, rank2):
            other_rank = rank2 if rank1 == 'A' else rank1
            if ranks.index(other_rank) >= ranks.index('Q'):
                return 'premium'
            elif is_suited:
                return 'strong'
            else:
                return 'medium'

        # broadways
        broadways = {'T','J','Q','K','A'}
        if rank1 in broadways and rank2 in broadways:
            return 'strong'

        # suited connectors 8+
        gap = abs(idx1 - idx2)
        if is_suited and gap == 1 and min(idx1, idx2) >= ranks.index('8'):
            return 'strong'

        # else >= 9
        if idx1 >= ranks.index('9') or idx2 >= ranks.index('9'):
            return 'medium'

        return 'weak'

    def categorize_made_hand(self, hole_cards, board_cards):
        strength = self.evaluate_hand_strength(hole_cards, board_cards)
        if strength >= 0.8:
            return 'nuts'
        elif strength >= 0.55:
            return 'strong_made'
        elif strength >= 0.35:
            return 'medium_made'
        else:
            return 'weak_made'
        
    def is_ultra_premium(self, hole_cards):
        """
        For example:
        Return True if the sorted ranks of the hole_cards 
        is in ['AA','KK','QQ','AK'], else False.
        """
        ranks = [c[0] for c in hole_cards]
        rank_str_sorted = ''.join(sorted(ranks))
        if rank_str_sorted in ['AA','KK','QQ','AK']:
            return True
        return False

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
    # BOARD TEXTURE
    # -------------------------------
    def board_has_paired(self, board_cards):
        ranks_count = {}
        for c in board_cards:
            r = c[0]
            ranks_count[r] = ranks_count.get(r, 0) + 1
            if ranks_count[r] >= 2:
                return True
        return False

    def evaluate_board_texture(self, board_cards):
        if len(board_cards) < 3:
            return "unknown"
        ranks = '23456789TJQKA'
        suits = [c[1] for c in board_cards]
        unique_suits = len(set(suits))
        indices = sorted(ranks.index(c[0]) for c in board_cards)
        if len(indices) < 2:
            return "unknown"
        max_gap = max(indices[i+1] - indices[i] for i in range(len(indices)-1))
        if unique_suits <= 2 or max_gap <= 2:
            return "wet"
        return "dry"

    # -------------------------------
    # POT ODDS & EV
    # -------------------------------
    def calculate_pot_odds(self, continue_cost, current_pot):
        if (current_pot + continue_cost) == 0:
            return 0
        return continue_cost / (current_pot + continue_cost)

    def get_expected_value(
        self,
        hand_strength,
        pot,
        continue_cost,
        hero_bounty_in_play=False,
        opp_bounty_in_play=False
    ):
        """
        If hero's bounty is on the board and hero WINS => 1.5*(pot+continue_cost) +10
        If opponent's bounty is on the board and hero LOSES => hero loses 1.5*(pot+continue_cost) +10
        Otherwise, we get/lose normal pot + continue_cost.

        So final EV = eq * hero_win_amount - (1 - eq)* hero_lose_amount
        """
        normal_winnings = pot + continue_cost

        if hero_bounty_in_play:
            adjusted_win = 1.5 * normal_winnings + 10
        else:
            adjusted_win = normal_winnings

        if opp_bounty_in_play:
            adjusted_lose = 1.5 * normal_winnings + 10
        else:
            adjusted_lose = continue_cost

        return (hand_strength * adjusted_win) - ((1 - hand_strength) * adjusted_lose)

    # -------------------------------
    # RAISE / BET SIZING
    # -------------------------------
    def validate_raise_amount(self, desired_raise, min_raise, max_raise, my_stack):
        raise_amount = max(desired_raise, min_raise)
        return min(raise_amount, max_raise, my_stack)

    def safe_raise(self, round_state, desired_raise, my_pip, my_stack):
        """
        Updated formula using my_pip:
         cost_to_raise = final_amount - my_pip
         If cost_to_raise >= my_stack => jam
        """
        legal_actions = round_state.legal_actions()
        if RaiseAction not in legal_actions:
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        min_raise, max_raise = round_state.raise_bounds()

        # clamp desired between min_raise and max_raise
        final_amount = max(desired_raise, min_raise)
        final_amount = min(final_amount, max_raise)

        cost_to_raise = final_amount - my_pip
        if cost_to_raise >= my_stack:
            final_amount = my_pip + my_stack
            final_amount = min(final_amount, max_raise)

        if final_amount <= my_pip:
            # fallback to call if possible
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        return RaiseAction(final_amount)

    def calculate_bet_size(self, hand_strength, street, current_pot, board_texture="dry"):
        if street == 0:
            multiplier = 10.0
        else:
            if hand_strength > 0.8:
                multiplier = 1.0
            elif hand_strength > 0.6:
                multiplier = 0.8
            else:
                multiplier = 0.7

        if board_texture == "wet":
            multiplier += 0.1

        base_bet = int(multiplier * current_pot)
        max_bet = int(1.2 * current_pot)
        final_bet = min(base_bet, max_bet)
        return max(final_bet, 1)

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
    # RANGE GENERATION & UPDATE
    # -------------------------------
    def generate_full_preflop_range(self):
        return set(['premium', 'strong', 'medium', 'weak'])

    def range_reduction(self, action, board_cards):
        updated_range = set(self.opponent_range)
        if action in ['raise', '3bet', '4bet']:
            if 'weak' in updated_range:
                updated_range.remove('weak')
        self.opponent_range = updated_range

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

        if continue_cost >= my_stack:
            self.opponent_went_allin_this_hand = True

        board_texture = self.evaluate_board_texture(board_cards)
        hand_strength = self.evaluate_hand_strength(my_cards, board_cards)
        opp_bounty_play = self.opponent_bounty_in_play(board_cards)
        hero_bounty_play = False  # if we had a hero bounty, we'd set this appropriately

        pot_odds = self.calculate_pot_odds(continue_cost, approximate_total_pot)
        bluff_prob = self.predict_bluff_chance()

        # If later street, we might do special folds if we have a weak hand vs big bet, etc.
        if round_state.street >= 4:
            my_made_category = self.categorize_made_hand(my_cards, board_cards)
            if my_made_category == 'weak_made' and opp_pip > my_pip:
                if continue_cost > 0.3 * my_stack and bluff_prob < 0.25:
                    if FoldAction in legal_actions:
                        return FoldAction()

        # If board is paired & big bet & small bluff prob => nuanced approach
        if self.board_has_paired(board_cards) and continue_cost > 0.5 * my_stack and bluff_prob < 0.1:
            my_made_category = self.categorize_made_hand(my_cards, board_cards)
            if my_made_category == 'weak_made':
                if FoldAction in legal_actions:
                    return FoldAction()
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction()
            if hand_strength < 0.55:
                if FoldAction in legal_actions:
                    return FoldAction()
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction()

        # If re-raised after we raised => if eq < 3%, fold
        if self.raised_this_street >= 1 and opp_pip > my_pip:
            if hand_strength < 0.03:
                if CallAction in legal_actions and continue_cost < 0.05 * my_stack:
                    return CallAction()
                return FoldAction()

        # If opp is an all-in spammer
        if self.is_allin_spammer() and continue_cost >= my_stack:
            if hand_strength >= 0.25 and CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # Normal call-allin logic
        if continue_cost >= my_stack:
            opp_range_est = self.infer_opponent_range()
            allin_freq = self.opponent_allin_count / max(self.opponent_total_hands, 1)
            if opp_range_est == "very_tight" and allin_freq < 0.05:
                required_eq = 0.45
            elif opp_range_est == "loose" or allin_freq > 0.30:
                required_eq = 0.30
            else:
                required_eq = 0.35

            if self.is_ultra_premium(my_cards):
                if CallAction in legal_actions:
                    self.pot += continue_cost
                    return CallAction()
                if RaiseAction in legal_actions:
                    return RaiseAction(my_stack)  # jam
                return FoldAction()

            if hand_strength >= required_eq and CallAction in legal_actions:
                self.pot += continue_cost
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # Track street changes
        if round_state.street != self.street:
            self.street = round_state.street
            self.raised_this_street = 0

        # Preflop vs. postflop
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
                my_pip,
                opp_pip
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
                opp_bounty_play,
                opp_pip
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
        my_pip,
        opp_pip
    ):
        """
        Includes fix for if opp just completes the SB, we don't check/fold illegally;
        we raise to ~10. Then, no small calls approach still applies.
        """
        my_hand_tier = self.categorize_preflop_hand(my_cards)
        preflop_bet = (round_state.pips[active] > 0 or round_state.pips[1 - active] > 0)

        # CASE 1: Opponent has posted something (SB) but not actually raised above our BB.
        # i.e., continue_cost <= 0 => they limped.
        if preflop_bet and continue_cost <= 0:
            # Opponent limp => we want to raise to punish
            if RaiseAction in legal_actions:
                desired_raise = 10  # you can pick 10, 12, or bigger
                action = self.safe_raise(round_state, desired_raise, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.preflop_raiser = True
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action
            # fallback: call if available, else check/fold
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # CASE 2: Opponent actually raises above our BB
        if preflop_bet and continue_cost > 0:
            # jam/fold logic if cost >= 40%, etc.
            # same code from above
            self.range_reduction('raise', [])

            if continue_cost >= 0.4 * my_stack:
                if self.is_ultra_premium(my_cards) or hand_strength >= 0.40:
                    if RaiseAction in legal_actions:
                        return self.safe_raise(round_state, my_stack, my_pip, my_stack)
                    if CallAction in legal_actions:
                        return CallAction()
                    return FoldAction()
                else:
                    return FoldAction()

            if self.is_ultra_premium(my_cards):
                if RaiseAction in legal_actions:
                    return self.safe_raise(round_state, my_stack, my_pip, my_stack)
                return FoldAction()

            if my_hand_tier in ['premium', 'strong']:
                if RaiseAction in legal_actions:
                    opp_raise = round_state.pips[1 - active]
                    desired = max(10, int(3.0 * opp_raise))
                    action = self.safe_raise(round_state, desired, my_pip, my_stack)
                    if isinstance(action, RaiseAction):
                        self.preflop_raiser = True
                        self.raised_this_street += 1
                        self.pot += (action.amount - my_pip)
                        return action
                return FoldAction()

            if continue_cost <= 4:
                if RaiseAction in legal_actions:
                    return self.safe_raise(round_state, 10, my_pip, my_stack)
            return FoldAction()

        # CASE 3: No raise at all => we open
        else:
            if my_hand_tier == 'premium':
                desired_open = 20
            elif my_hand_tier == 'strong':
                desired_open = 15
            elif my_hand_tier == 'medium':
                desired_open = 10
            else:
                return FoldAction()

            if RaiseAction in legal_actions:
                action = self.safe_raise(round_state, desired_open, my_pip, my_stack)
                if isinstance(action, RaiseAction):
                    self.preflop_raiser = True
                    self.raised_this_street += 1
                    self.pot += (action.amount - my_pip)
                    return action
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
        opp_bounty_play,
        opp_pip
    ):
        """
        Example c-bet approach if we were preflop raiser, 
        plus usage of opp_bounty_play to bump bet sizes 10%.
        """
        opponent_type = self.categorize_opponent()
        my_pip = round_state.pips[active]
        bluff_prob = self.predict_bluff_chance()

        last_action = self.detect_opponent_action(round_state, active)
        if last_action in ['raise', 'bet']:
            self.range_reduction('raise', board_cards)

        # If turn/river & board is wet => fold medium/weak to big aggression
        if round_state.street >= 4:
            my_made_category = self.categorize_made_hand(my_cards, board_cards)
            if (opp_pip > my_pip) and (my_made_category in ['weak_made', 'medium_made']):
                if board_texture == 'wet' and continue_cost > 0.3 * my_stack and bluff_prob < 0.2:
                    if FoldAction in legal_actions:
                        return FoldAction()

        if self.preflop_raiser:
            if hand_strength > pot_odds:
                desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                if opp_bounty_play:
                    desired_bet = int(desired_bet * 1.1)  # 10% bigger if their bounty rank is out

                if self.raised_this_street < 2 and RaiseAction in legal_actions:
                    action = self.safe_raise(round_state, desired_bet, my_pip, my_stack)
                    if isinstance(action, RaiseAction):
                        self.raised_this_street += 1
                        self.pot += (action.amount - my_pip)
                        return action
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
            # adapt to villain type
            if opponent_type == 'Overly Aggressive':
                if hand_strength > 0.4 and CallAction in legal_actions:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()
            elif opponent_type == 'LAG':
                if hand_strength > 0.6 and RaiseAction in legal_actions:
                    desired_bet = self.calculate_bet_size(hand_strength, self.street, current_pot, board_texture)
                    if opp_bounty_play:
                        desired_bet = int(desired_bet * 1.1)
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
                    if opp_bounty_play:
                        desired_bet = int(desired_bet * 1.1)
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

    # -------------------------------
    # DETECT OPPONENT ACTION
    # -------------------------------
    def detect_opponent_action(self, round_state, active):
        opp_pip = round_state.pips[1 - active]
        my_pip = round_state.pips[active]
        if opp_pip > self.raised_this_street:
            if opp_pip > my_pip:
                return 'raise'
            else:
                return 'bet'
        if opp_pip == my_pip and opp_pip != 0:
            return 'call'
        return 'check'


if __name__ == '__main__':
    run_bot(Player(), parse_args())
