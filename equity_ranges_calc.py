import eval7
import itertools
import json
import random

def expand_hand_range(hand_range):
    suits = 'shdc'
    expanded = []
    for hand in hand_range:
        ranks = hand[:2]
        if hand[-1] == 's':
            rank1, rank2 = ranks[0], ranks[1]
            for s in suits:
                expanded.append([eval7.Card(f"{rank1}{s}"), eval7.Card(f"{rank2}{s}")])
        elif hand[-1] == 'o':
            rank1, rank2 = ranks[0], ranks[1]
            if rank1 == rank2:
                for s1 in suits:
                    for s2 in suits:
                        if s1 != s2:
                            expanded.append([eval7.Card(f"{rank1}{s1}"), eval7.Card(f"{rank1}{s2}")])
            else:
                for s1 in suits:
                    for s2 in suits:
                        if s1 != s2:
                            expanded.append([eval7.Card(f"{rank1}{s1}"), eval7.Card(f"{rank2}{s2}")])
        else:
            rank1 = ranks[0]
            for s1 in suits:
                for s2 in suits:
                    if s1 != s2:
                        expanded.append([eval7.Card(f"{rank1}{s1}"), eval7.Card(f"{rank1}{s2}")])
    return expanded

def get_pair_equity(my_pair, opp_2card_combos):
    hero_combos = expand_hand_range([my_pair])
    MC_ITER = 50000
    hero_score = 0.0
    total_combos = 0.0
    
    base_deck = eval7.Deck().cards
    for _ in range(MC_ITER):
        hero_combo = random.choice(hero_combos)
        opp_combo = random.choice(opp_2card_combos)
        while opp_combo[0] in hero_combo or opp_combo[1] in hero_combo:
            hero_combo = random.choice(hero_combos)
            opp_combo = random.choice(opp_2card_combos)
            
        used_cards = set(hero_combo + opp_combo)
        
        remaining_cards = [c for c in base_deck if c not in used_cards]
        
        board_5 = random.sample(remaining_cards, 5)
        hero_value = eval7.evaluate(hero_combo + list(board_5))
        opp_value = eval7.evaluate(opp_combo + list(board_5))
        if hero_value > opp_value:
            hero_score += 1
        elif hero_value == opp_value:
            hero_score += 0.5
        
        total_combos += 1
    
    if total_combos == 0:
        return 0
    return hero_score / total_combos


all_ranges = [
    "AAo", "KKo", "QQo", "JJo", "TTo", "99o", "88o", "AKs", "77o", "AQs",
    "AJs", "AKo", "ATs", "AQo", "AJo", "KQs", "66o", "A9s", "ATo", "KJs",
    "A8s", "KTs", "KQo", "A7s", "A9o", "KJo", "55o", "QJs", "K9s", "A5s",
    "A6s", "A8o", "KTo", "QTs", "A4s", "A7o", "K8s", "A3s", "QJo", "K9o",
    "A5o", "A6o", "Q9s", "K7s", "JTs", "A2s", "QTo", "44o", "A4o", "K6s",
    "K8o", "Q8s", "A3o", "K5s", "J9s", "Q9o", "JTo", "K7o", "A2o", "K4s",
    "Q7s", "K6o", "K3s", "T9s", "J8s", "33o", "Q6s", "Q8o", "K5o", "J9o",
    "K2s", "Q5s", "T8s", "K4o", "J7s", "Q4s", "Q7o", "T9o", "J8o", "K3o",
    "Q6o", "Q3s", "98s", "T7s", "J6s", "K2o", "22o", "Q2s", "Q5o", "J5s",
    "T8o", "J7o", "Q4o", "97s", "J4s", "T6s", "J3s", "Q3o", "98o", "87s",
    "T7o", "J6o", "96s", "J2s", "Q2o", "T5s", "J5o", "T4s", "97o", "86s",
    "J4o", "T6o", "95s", "T3s", "76s", "J3o", "87o", "T2s", "85s", "96o",
    "J2o", "T5o", "94s", "75s", "T4o", "93s", "86o", "65s", "84s", "95o",
    "T3o", "92s", "76o", "74s", "T2o", "54s", "85o", "64s", "83s", "94o",
    "75o", "82s", "73s", "93o", "65o", "53s", "63s", "84o", "92o", "43s",
    "74o", "72s", "54o", "64o", "52s", "62s", "83o", "42s", "82o", "73o",
    "53o", "63o", "32s", "43o", "72o", "52o", "62o", "42o", "32o"
]
epsilon = 0.0001

tightness_buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

equities = {}

for opp_tightness in tightness_buckets:
    print("Calculating equity for tightness value", opp_tightness)
    opp_range_size = int(opp_tightness * len(all_ranges))
    opp_range = all_ranges[:opp_range_size]
    
    opp_2card_combos = expand_hand_range(opp_range)

    equities[opp_tightness] = {}
    
    for hero_label in all_ranges:
        print("Calculating equity for hero label", hero_label)
        eq = get_pair_equity(hero_label, opp_2card_combos)
        equities[opp_tightness][hero_label] = eq

with open("equity_ranges.json", "w") as f:
    json.dump(equities, f)