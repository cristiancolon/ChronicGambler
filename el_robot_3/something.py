import pickle
import eval7
import random
import itertools
from collections import defaultdict

# Define the order of card ranks
card_val_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def generate_hand_categories():
    """
    Generate a list of hand categories.
    Each category is a tuple representing a unique hand type.
    """
    categories = []

    # Pocket pairs
    for rank in card_val_order:
        categories.append((rank, rank))

    # Suited and offsuit connectors
    for r1, r2 in itertools.combinations(card_val_order, 2):
        categories.append((r1, r2, 'same'))  # Suited
        categories.append((r1, r2, 'diff'))  # Offsuit

    return categories

def simulate_hand_strength(hand, num_iters=1000, board_cards=None, dead_cards=None, opp_known_cards=None):
    '''
    Estimates the win probability of a pair of hole cards using Monte Carlo simulations.
    
    Arguments:
    hand: list of two strings representing your hole cards (e.g., ['As', 'Kd'])
    num_iters: number of Monte Carlo iterations
    board_cards: list of strings representing community cards (default: None)
    dead_cards: list of strings representing dead cards (default: None)
    opp_known_cards: list of two strings representing opponent's hole cards (optional)
    
    Returns:
    Tuple of (win_prob, draw_prob)
    '''
    deck = eval7.Deck()
    
    # Convert hole cards to eval7.Card objects and remove them from the deck
    hole_cards = [eval7.Card(card) for card in hand]
    for card in hole_cards:
        if card in deck.cards:
            deck.cards.remove(card)
    
    # Handle board_cards
    if board_cards is None:
        board_cards = []
    else:
        board_cards = [eval7.Card(card) for card in board_cards if card != '']
        for card in board_cards:
            if card in deck.cards:
                deck.cards.remove(card)
    
    # Handle dead_cards
    if dead_cards is None:
        dead_cards = []
    else:
        dead_cards = [eval7.Card(card) for card in dead_cards]
        for card in dead_cards:
            if card in deck.cards:
                deck.cards.remove(card)
    
    score = 0
    if opp_known_cards is not None:
        # Convert opponent's known cards to eval7.Card objects and remove them from the deck
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
    
    for _ in range(num_iters):
        deck.shuffle()
        
        _COMM = 5 - len(board_cards)
        _OPP = 2
        
        # Peek cards for opponent and community
        draw = deck.peek(_COMM + _OPP)
        opp_hole = draw[:_OPP]  # List of eval7.Card objects
        community = draw[_OPP:]  # List of eval7.Card objects
        
        # Combine with existing board cards
        full_community = community + board_cards
        
        our_hand = hole_cards + full_community
        opp_hand = opp_hole + full_community
        
        our_hand_value = eval7.evaluate(our_hand)
        opp_hand_value = eval7.evaluate(opp_hand)
        
        if our_hand_value > opp_hand_value:
            score += 2  # Win
        elif our_hand_value == opp_hand_value:
            score += 1  # Draw
    
    hand_strength = score / (2 * num_iters)  # Normalize to [0,1]
    draw_prob = (score % 2) / num_iters  # Optional: Calculate draw probability if needed
    
    return hand_strength, draw_prob

def main():
    categories = generate_hand_categories()
    hand_strengths = {}

    total_categories = len(categories)
    print(f"Total hand categories to simulate: {total_categories}")
    for idx, hand in enumerate(categories, 1):
        print(f"Simulating {idx}/{total_categories}: Hand = {hand}")
        # Prepare the hand representation
        if len(hand) == 2:
            # Pocket pair
            rank = hand[0]
            # Assign random suits
            suits = random.sample(['s', 'h', 'd', 'c'], 2)
            hole = [rank + suits[0], rank + suits[1]]
        elif len(hand) == 3:
            # Connectors with suit relation
            rank1, rank2, suit_rel = hand
            if suit_rel == 'same':
                suit = random.choice(['s', 'h', 'd', 'c'])
                hole = [rank1 + suit, rank2 + suit]
            else:
                suit1, suit2 = random.sample(['s', 'h', 'd', 'c'], 2)
                hole = [rank1 + suit1, rank2 + suit2]
        else:
            print(f"Invalid hand category: {hand}")
            continue  # Skip invalid categories

        win_prob, draw_prob = simulate_hand_strength(hole, num_iters=1000)
        hand_strengths[hand] = {'win_prob': win_prob, 'draw_prob': draw_prob}

    # Save to pickle file
    with open('hand_strengths.p', 'wb') as fp:
        pickle.dump(hand_strengths, fp)

    print("Hand strengths have been computed and saved to 'hand_strengths.p'.")

if __name__ == "__main__":
    main()