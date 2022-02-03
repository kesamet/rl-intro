# TODO
import numpy as np
import matplotlib.pyplot as plt

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  # "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# policy for dealer
POLICY_DEALER = np.zeros(22, dtype=np.int)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND


def target_policy_player(usable_ace_player, player_sum, dealer_card):
    """Player target policy."""
    return POLICY_PLAYER[player_sum]


def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    """Player behavior policy."""
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT


def get_card() -> int:
    card = np.random.randint(1, 14)
    if card == 1:
        return 11
    return min(card, 10)


def play():
    player_score = 0
    player_traj = []
    player_usable_ace = False

    dealer_usable_ace = False

    while player_score < 12:
        card = get_card()
        player_score += card

        if player_score > 21:
            assert player_score == 22  # last card must be ace
            player_score -= 10
        else:
            player_usable_ace |= (card == 11)
