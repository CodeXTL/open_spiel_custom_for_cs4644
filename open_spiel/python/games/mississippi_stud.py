# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Mississippi Stud implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum, collections

import numpy as np

import pyspiel


class Action(enum.IntEnum):
  FOLD = 0
  RAISE_1 = 1   # Raise 1x Ante
  RAISE_2 = 2   # Raise 2x Ante
  RAISE_3 = 3   # Raise 3x Ante


_NUM_PLAYERS = 1
_DECK = list(range(52))
_ANTE_AMOUNT = 1.0


# Define hand ranks
class HandRank(enum.IntEnum):
  HIGH_CARD = 0
  ONE_PAIR = 1
  TWO_PAIR = 2
  THREE_OF_A_KIND = 3
  STRAIGHT = 4
  FLUSH = 5
  FULL_HOUSE = 6
  FOUR_OF_A_KIND = 7
  STRAIGHT_FLUSH = 8
  ROYAL_FLUSH = 9


_GAME_TYPE = pyspiel.GameType(
  short_name="python_mississippi_stud",
  long_name="Python Mississippi Stud",
  dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
  chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
  information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
  utility=pyspiel.GameType.Utility.GENERAL_SUM,
  reward_model=pyspiel.GameType.RewardModel.TERMINAL,
  max_num_players=_NUM_PLAYERS,
  min_num_players=_NUM_PLAYERS,
  provides_information_state_string=True,
  provides_information_state_tensor=True,
  provides_observation_string=True,
  provides_observation_tensor=True,
  provides_factored_observation_string=True
)

# Max bet: 1 (Ante) + 3 (3rd street) + 3 (4th street) + 3 (5th street) = 10 units
# Max utility: Royal Flush (500:1 win ratio) on 10 units = +5000.0
# Min utility: Bet 10 units and lose everything = -10.0
# Max game length: deal 5 cards (2 player cards, 3 community cards) + 3 player decisions = 8 plays
_GAME_INFO = pyspiel.GameInfo(
  num_distinct_actions=len(Action),
  max_chance_outcomes=len(_DECK),
  num_players=_NUM_PLAYERS,
  min_utility=-10.0,
  max_utility=5000.0,
  utility_sum=0.0,
  max_game_length=8
)


class MississippiStudGame(pyspiel.Game):
  """A Python version of Mississippi Stud."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MississippiStudState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return MississippiStudObserver(
      iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
      params
    )


class MississippiStudState(pyspiel.State):
  """A python version of the Mississippi Stud state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.player_cards = []
    self.community_cards = []

    # List of actions taken by the player
    self.player_actions = []

    self.total_wager = _ANTE_AMOUNT
    self.current_round = 0  # 0: 3rt St, 1: 4th St, 2: 5th St
    self._game_over = False

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif len(self.player_cards) < 2:
      # First 2 chance nodes are player cards
      return pyspiel.PlayerId.CHANCE
    elif len(self.community_cards) < 3:
      # Next 3 chance nodes are community cards
      return pyspiel.PlayerId.CHANCE
    else:
      # Player turn
      return 0


  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player == 0
    return [Action.FOLD, Action.RAISE_1, Action.RAISE_2, Action.RAISE_3]


  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    dealt_cards = set(self.player_cards + self.community_cards)
    outcomes = [card for card in _DECK if card not in dealt_cards]
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]


  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      if len(self.player_cards) < 2:
        self.player_cards.append(action)
      else:
        self.community_cards.append(action)
      return
    
    # Player action
    self.player_actions.append(action)
    if action == Action.FOLD:
      self._game_over = True
    else:
      bet_multiplier = int(action)  # Works because each bet enum correspond to its multiplier
      self.total_wager += (_ANTE_AMOUNT * bet_multiplier)

    # Shift round counter
    self.current_round += 1
    if self.current_round == 3:
      self._game_over = True


  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.FOLD:
      return "Fold"
    elif action == Action.RAISE_1:
      return "Raise 1x"
    elif action == Action.RAISE_2:
      return "Raise 2x"
    else:
      return "Raise 3x"


  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over


  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self._game_over:
      return [0.0]
    
    # If player folded, they lose all bets wagered thus far
    if Action.FOLD in self.player_actions:
      return [-self.total_wager]
    
    # If game is over and player never folded, evaluate the hand
    hand_rank, pair_rank = self._evaluate_hand(self.player_cards + self.community_cards)

    # Mississippi Stud Paytable
    payout_multiplier = -1
    if hand_rank == HandRank.ROYAL_FLUSH:
      payout_multiplier = 500
    elif hand_rank == HandRank.STRAIGHT_FLUSH:
      payout_multiplier = 100
    elif hand_rank == HandRank.FOUR_OF_A_KIND:
      payout_multiplier = 40
    elif hand_rank == HandRank.FULL_HOUSE:
      payout_multiplier = 10
    elif hand_rank == HandRank.FLUSH:
      payout_multiplier = 6
    elif hand_rank == HandRank.STRAIGHT:
      payout_multiplier = 4
    elif hand_rank == HandRank.THREE_OF_A_KIND:
      payout_multiplier = 3
    elif hand_rank == HandRank.TWO_PAIR:
      payout_multiplier = 2
    elif hand_rank == HandRank.ONE_PAIR:
      if pair_rank >= 9:  # J or better
        payout_multiplier = 1
      elif pair_rank >= 4:  # 6s - 10s
        payout_multiplier = 0

    return [self.total_wager * payout_multiplier]


  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return (
      f"P:{self.player_cards} C:{self.community_cards} "
      f"Actions:{[a.name for a in self.player_actions]} "
      f"Wager:{self.total_wager}"        
    )


  # Internal helper methods
  def _get_rank(self, card):
    """Returns the rank (0-12) of a card (0-51)"""
    return card % 13
  

  def _get_suit(self, card):
    """Returns the suit (0-3) of a card (0-51)"""
    return card // 13
  

  def _evaluate_hand(self, cards):
    """
    Evaluate a 5-card hand.
    Returns:
      (hand_rank, relevant_pair_rank): the relevant_pair_rank is only used in the case of a single pair, -1 otherwise
    """
    assert len(cards) == 5
    ranks = sorted([self._get_rank(c) for c in cards])
    suits = [self._get_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # check for wheel straight (A-2-3-4-5)
    is_wheel_straight = (ranks == [0, 1, 2, 3, 12]) # Ranks: 2, 3, 4, 5, A
    # check for normal straight
    is_normal_straight = (len(set(ranks)) == 5 and (ranks[-1] - ranks[0] == 4))
    is_straight = is_wheel_straight or is_normal_straight

    if is_straight and is_flush:
      if ranks == [8, 9, 10, 11, 12]: # Ranks: 10, J, Q, K, A
        return (HandRank.ROYAL_FLUSH, -1)
      return (HandRank.STRAIGHT_FLUSH, -1)
    
    rank_counts = collections.Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    if counts == [4, 1]:
      return (HandRank.FOUR_OF_A_KIND, -1)
    if counts == [3, 2]:
      return (HandRank.FULL_HOUSE, -1)
    if is_flush:
      return (HandRank.FLUSH, -1)
    if is_straight:
      return (HandRank.STRAIGHT, -1)
    if counts == [3, 1, 1]:
      return (HandRank.THREE_OF_A_KIND, -1)
    if counts == [2, 2, 1]:
      return (HandRank.TWO_PAIR, -1)
    if counts == [2, 1, 1, 1]:
      for r, c in rank_counts.items():
        if c == 2:
          pair_rank = r
      return (HandRank.ONE_PAIR, pair_rank)
    
    return (HandRank.HIGH_CARD, -1)
      

class MississippiStudObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [
      ("player_cards", 104, (2, 52)),     # 2 player cards (2 * 52 bits)
      ("community_cards", 156, (3, 52)),  # 3 comm cards (3 * 52 bits)
      ("betting_history", 12, (3, 4))     # 3 rounds, 4 actions
    ]

    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size


  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)

    # Player knows their own cards
    if len(state.player_cards) == 2:
      self.dict["player_cards"][0, state.player_cards[0]] = 1
      self.dict["player_cards"][1, state.player_cards[1]] = 1

    # Player knows revealed community cards
    # Round 0 (3rd St): Sees 0 comm cards
    # Round 1 (4th St): Sees 1 comm cards
    # Round 2 (5th St): Sees 2 comm cards
    if state.current_round > 0 and len(state.community_cards) >= 1:
      self.dict["community_cards"][0, state.community_cards[0]] = 1
    if state.current_round > 1 and len(state.community_cards) >= 2:
      self.dict["community_cards"][0, state.community_cards[0]] = 1
      self.dict["community_cards"][1, state.community_cards[1]] = 1

    # At the end, reveal all cards
    if state.is_terminal() and len(state.community_cards) == 3:
      self.dict["community_cards"][0, state.community_cards[0]] = 1
      self.dict["community_cards"][1, state.community_cards[1]] = 1
      self.dict["community_cards"][2, state.community_cards[2]] = 1

    # Player knows their past actions
    for round_num, action in enumerate(state.player_actions):
      self.dict["betting_history"][round_num, action] = 1


  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []

    # Show player card
    if state.player_cards:
      pieces.append(f"Player Cards: {state.player_cards}")

    # Show revealed community cards
    revealed_cards = []
    if state.current_round > 0 and len(state.community_cards) >= 1:
      revealed_cards = state.community_cards[:1]
    if state.current_round > 1 and len(state.community_cards) >= 2:
      revealed_cards = state.community_cards[:2]
    if state.is_terminal() and len(state.community_cards) == 3:
      revealed_cards = state.community_cards

    pieces.append(f"Community Cards: {revealed_cards}")

    actions_str = [a.name for a in state.player_actions]
    pieces.append(f"Bets: {actions_str}")

    return " ".join(pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, MississippiStudGame)
