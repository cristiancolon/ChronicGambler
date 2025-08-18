import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# from config import *

from skeleton.actions import FoldAction, CheckAction, CallAction, RaiseAction

##############################################################################
# 1. CardEmbedding
##############################################################################
class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()

        # CHANGED: Removed console prints, same embedding logic
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, card_indices):
        B, num_cards = card_indices.shape

        flat_cards = card_indices.view(-1)
        valid = flat_cards.ge(0).float()  # 1 if card>=0 else 0
        clamped_cards = flat_cards.clamp(min=0)

        card_emb = self.card(clamped_cards)
        rank_emb = self.rank(clamped_cards // 4)
        suit_emb = self.suit(clamped_cards % 4)

        embs = card_emb + rank_emb + suit_emb
        embs = embs * valid.unsqueeze(1)

        embs = embs.view(B, num_cards, -1)
        embs = embs.sum(dim=1)  # sum across cards
        return embs

##############################################################################
# 2. DeepCFRModel
##############################################################################
class DeepCFRModel(nn.Module):
    def __init__(self, ncardtypes, nbets, nactions, device, dim=256):
        """
        ncardtypes: number of card groups (e.g. 4 => [hole, flop, turn, river])
        nbets: how many bet "slots" we track
        nactions: how many discrete actions we output regrets for
        device: torch device
        dim: hidden dimension
        """
        super(DeepCFRModel, self).__init__()

        self.nbets = nbets
        self.nactions = nactions
        self.device = device

        # Build multiple CardEmbedding modules, one per card group
        self.card_embeddings = nn.ModuleList([
            CardEmbedding(dim) for _ in range(ncardtypes)
        ])

        # Card trunk
        self.card1 = nn.Linear(dim * ncardtypes, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        # Bet trunk
        self.bet1 = nn.Linear(nbets * 2, dim)
        self.bet2 = nn.Linear(dim, dim)

        # Combined trunk
        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)

        self.action_head = nn.Linear(dim, nactions)

    def forward(self, cards, bets):
        """
        cards: list/tuple of length ncardtypes, each is (B, #cards_in_group)
               Example: [hole_tensor, flop_tensor, turn_tensor, river_tensor]
        bets: (B, nbets) float
        Returns:
          (B, nactions) unmasked regrets
        """
        # 1) embed card groups
        emb_list = []
        for embedding, card_group in zip(self.card_embeddings, cards):
            emb_list.append(embedding(card_group))  # (B, dim)
        cat_cards = torch.cat(emb_list, dim=1)      # (B, dim*ncardtypes)

        x = F.relu(self.card1(cat_cards))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # 2) bet features
        # CHANGED: Example log transform to mitigate large bet values
        clamped_bets = bets.clamp(min=0)  # negative => 0
        log_bet_size = torch.log1p(clamped_bets)   # log(1 + bet_size)
        bet_occurred = bets.ge(0).float()          # 1 if valid, else 0
        bet_feats = torch.cat([log_bet_size, bet_occurred], dim=1)

        y = F.relu(self.bet1(bet_feats))
        y = F.relu(self.bet2(y) + y)

        # 3) Combine
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)

        # CHANGED: We keep a normalization to prevent outlier expansions
        z = F.normalize(z, p=2, dim=1)

        return self.action_head(z)

    def get_card_num(self, card):
        """
        Convert an engine.Card object -> integer [0..51].
        rank in [1..13], suit in [0..3].
        """
        rank_id = card.rank - 1
        suit_id = card.suit
        return rank_id + 13 * suit_id

    def tensorize_cards(self, my_hand, board):
        hole_ids = [self.get_card_num(c) for c in my_hand]
        hole_tensor = torch.tensor(hole_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        board_ids = [self.get_card_num(c) for c in board]
        while len(board_ids) < 5:
            board_ids.append(-1)

        flop_tensor = torch.tensor(board_ids[:3], dtype=torch.long, device=self.device).unsqueeze(0)
        turn_tensor = torch.tensor([board_ids[3]], dtype=torch.long, device=self.device).unsqueeze(0)
        river_tensor = torch.tensor([board_ids[4]], dtype=torch.long, device=self.device).unsqueeze(0)

        return [hole_tensor, flop_tensor, turn_tensor, river_tensor]

    def tensorize_bets(self, bets):
        last_bets = list(bets[-self.nbets:])
        while len(last_bets) < self.nbets:
            last_bets.insert(0, -1)

        bet_tensor = torch.tensor(last_bets, dtype=torch.float32, device=self.device).unsqueeze(0)
        return bet_tensor

    def tensorize_roundstate(self, roundstate, active):
        bet_tensor = self.tensorize_bets(roundstate.bets)
        my_cards = roundstate.hands[active]
        board_cards = roundstate.deck[:roundstate.street]
        card_tensor_list = self.tensorize_cards(my_cards, board_cards)
        return (card_tensor_list, bet_tensor)

    def tensorize_mask(self, round_state):
        legal_actions = round_state.legal_actions()
        pot = sum(400 - s for s in round_state.stacks)

        mask_tensor = torch.zeros(self.nactions, dtype=torch.float32, device=self.device)

        if FoldAction in legal_actions:
            mask_tensor[0] = 1
        if CheckAction in legal_actions:
            mask_tensor[1] = 1
        if CallAction in legal_actions:
            mask_tensor[2] = 1

        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            half_pot = math.ceil(pot * 0.5)
            three_half_pot = math.ceil(pot * 1.5)
            if min_raise <= half_pot <= max_raise:
                mask_tensor[3] = 1
            if min_raise <= three_half_pot <= max_raise:
                mask_tensor[4] = 1

        return mask_tensor.unsqueeze(0)