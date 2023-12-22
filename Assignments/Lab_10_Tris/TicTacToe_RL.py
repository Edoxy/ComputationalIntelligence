"""
Copyright **`(c)`** 2023 Edoardo Vay  `<vay.edoardo@gmail.com>`
<https://github.com/Edoxy>
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
from enum import Enum
from tqdm.auto import tqdm
from IPython.display import clear_output
from itertools import combinations
from collections import defaultdict


class StateType(Enum):
    """Enum class to classify a state"""

    WIN = 2  # means the X player wins (first Player)
    LOSS = -2  # means the O player wins (second Player)
    DRAW = 1
    IN_PROGRESS = 0


@dataclass
class ttt_node:
    """
    DataClass defining a node in the Tic-tac-toe Tree structure: all the possible positions are encoded in a tuple
    """

    state: tuple  # This tuple encodes the position: -1 for a free space, 0 for X and 1 for O
    childs: Dict[
        tuple, any
    ]  # Dictionary with the position reachable from this position: keys are the tuple encoding the position
    state_type: StateType = StateType.DRAW  # State type of the node: default is Draw


def print_state(game_state: tuple, optional: any = None) -> None:
    """
    Pretty Print of the Tic Tac Toe board in a given state
    game_state: tuple -> board representations as a tuple
    optional: Any -> any type of information that you what to display next to the board
    """
    string = ""
    for i in range(3):
        row = ""
        for j in range(3):
            index = j + 3 * i
            if game_state[index] == 0:
                row += "X "
            elif game_state[index] == 1:
                row += "O "
            else:
                row += "• "

        if i == 1 and not optional == None:
            row += "\t" + str(
                optional
            )  # gives the option to print additional info nex to the board
        string += row + "\n"
    print(string)


def print_states(game_state_list: List[Tuple[int]]) -> None:
    """
    Pretty print of multiple board in a row
    game_state_list: List -> is a list of board rappresentation as a tuple
    """
    string = ""
    for i in range(3):
        row = ""
        for game_state in game_state_list:
            for j in range(3):
                if game_state[j + 3 * i] == 0:
                    row += "X "
                elif game_state[j + 3 * i] == 1:
                    row += "O "
                else:
                    row += "• "
            row += "\t\t"

        string += row + "\n"
    print(string)


def check_same_state(
    game_state1: Tuple[int],
    game_state2: Tuple[int],
    n_rotations: int = 0,
    n_symmetry: int = 0,
) -> bool:
    """
    Recursive function that checks if two board positions are the same up to rotation or symmetry (does all the combination to generate all the element in the D4 Group)
    game_state1: tuple -> board representations of the first position
    game_state2: tuple -> board representations of the second position
    n_rotations: int -> number of rotation already checked to the second game_state
    n_symmetry: int -> number of symmetry already checked to the second game_state
    """

    if game_state1 == game_state2:  # check if they are the same
        return True

    # in the D4 group we have a finite number of operation between rotation and symmetry to reach every element of the group
    elif n_rotations == 3:  # more than 3 rotations are not useful
        return False
    elif n_symmetry == 2:  # more than 2 symmetrys are not useful
        return False
    else:
        game_state2_rotated = rotate(game_state2)  # create a rotated version
        game_state2_symmetric = symmetry(game_state2)  # create a symmetric version
        ROTATION = check_same_state(  # check the rotated version; recursive call
            game_state1,
            game_state2_rotated,
            n_rotations=n_rotations + 1,
            n_symmetry=n_symmetry,
        )
        if ROTATION:
            return True  # if the Rotated version are the same
        SYMMETRY = check_same_state(  # check the symmetric version; recursive call
            game_state1,
            game_state2_symmetric,
            n_rotations=n_rotations,
            n_symmetry=n_symmetry + 1,
        )
        return SYMMETRY  # if the Symmetric version are the same


def state_in_list(
    game_state: Tuple[int], list_game_states: Iterable[Tuple[int]]
) -> bool:
    """
    Given a List of boards representations, check if any of the is the same of the given one (up to rotation and symmetries)
    game_state: tuple -> board representation to check
    list_game_states: List/Set... -> iteratable that contains the board representations to check against
    """
    result = False
    for game in list_game_states:
        result = result or check_same_state(
            game, game_state
        )  # calls the check for each element of the list
        if result:
            break  # breaks the loop if found one
    return result


def ply(index: int, state: List[int], player_id: bool) -> bool:
    """
    Function that execute a move on a state modifing the board
    index: int -> board position to change
    state: List -> board representation as a list !must be MUTABLE to be changed inplace!
    player_id: bool {0, 1} -> player 0 plays X and player 1 plays O
    """
    if state[index] == -1:  # can only choose an empty space
        is_valid = True
        state[index] = player_id  # changes in place the board
    else:
        is_valid = False  # returns false if the move is not valid
    return is_valid


def is_terminal(game_state: Tuple[int], player_id: bool) -> bool:
    """
    checks if the game_state reached is terminal (meaning someone won)
    game_state: tuple -> board representation to check
    player_id: bool {0, 1} -> player that we want to check
    """
    VALUES = (4, 9, 2, 3, 5, 7, 8, 1, 6)  # magic square values
    score = [
        VALUES[i] for i in range(9) if game_state[i] == player_id
    ]  # magic square values that the player has control of
    n_moves = len(score)

    if n_moves < 3:  # if less than 3 moves were played it can't be terminal
        return False

    else:
        return any(
            sum(c) == 15 for c in combinations(score, 3)
        )  # returns true if there is a combination that sums to 15


def rotate(game_state: Tuple[int]) -> Tuple[int]:
    """
    Applies a clockwise rotation to the board
    game_state: tuple -> board representation to rotate
    """
    #  1 2 3      7 4 1
    #  4 5 6  ->  8 5 2
    #  7 8 9      9 6 3
    # Clock wise
    INDEX_PERMUTATION = (
        np.array([7, 4, 1, 8, 5, 2, 9, 6, 3]) - 1
    )  # indexes of the new position on the board
    return tuple(
        game_state[INDEX_PERMUTATION[i]] for i in range(9)
    )  # new board position after rotation


def symmetry(game_state: Tuple[int]) -> Tuple[int]:
    """
    Applies a vertical symmetry to the board
    game_state: tuple -> board representation to reflect
    """
    #  1 2 3      3 2 1
    #  4 5 6  ->  6 5 4
    #  7 8 9      9 8 7
    # Vertical
    INDEX_PERMUTATION = (
        np.array([3, 2, 1, 6, 5, 4, 9, 8, 7]) - 1
    )  # indexes of the new position on the board
    return tuple(
        game_state[INDEX_PERMUTATION[i]] for i in range(9)
    )  # new board position after symmetry


def create_tree(root_node: ttt_node, player_id: bool = 0) -> None:
    """
    Recursive function that create a tree of all possible position in tic-tac-toe up to rotations and symmetries
    root_node: ttt_node -> node from which we expand the tree; to generate all the positions start with the node containing an empty board
    player_id: bool -> which player is going to play from the current position. Default to the first player (X)

    """
    if is_terminal(
        root_node.state, not player_id
    ):  # checks if the current position is winning for the player that played before
        if not player_id == 0:
            root_node.state_type = StateType.WIN  # marks as a win if X wins
        else:
            root_node.state_type = StateType.LOSS  # marks as a loss if O wins
        return  # ends the call if the state is terminal
    for i in range(9):  # iterates for each square
        tmp_game = list(
            root_node.state
        )  # create a temporary board that can be modified
        if ply(
            i, tmp_game, player_id
        ) and not state_in_list(  # checks if the move is allowed and if the position as not been checked before
            tmp_game,
            root_node.childs.keys(),  # this is the list of position reached from the current one
        ):
            root_node.state_type = (
                StateType.IN_PROGRESS
            )  # if we reached any position, the game is not finished
            child = tuple(tmp_game)  # create a child immutable
            root_node.childs[child] = ttt_node(
                child, dict()
            )  # add the child to the current node dictionary
            create_tree(
                root_node.childs[child], not player_id
            )  # recursively call the fuction on the child
    return


class TicTacToe_Env:
    """
    Class that simulate the environment of a tic tac toe board with a player
    """

    def __init__(self, init_node: ttt_node) -> None:
        self.root = init_node
        self.CurrentState = init_node
        self.payoff_win = 150  # payoff for a win
        self.payoff_loss = -100  # payoff for a win
        self.payoff_draw = -50  # payoff for a win
        self.player_id = 0  # starting player

    def game_reset(self, Agent=None, agent_randomnes: float = 0):
        """
        Resets the game state for the next episode to the empty board.

        Parameters:
        - Agent: The agent making moves in the game; if None it makes a random move.
        - agent_randomness: The probability of the agent making a random move.

        Returns:
        - The player ID for the next move.
        """
        if self.player_id == 1:
            self.CurrentState = self.root
        else:
            if Agent == None or random.random() < agent_randomnes:
                # Randomly select a child node
                self.CurrentState = self.root.childs[
                    random.choice(list(self.root.childs.keys()))
                ]
            else:
                # Select a child node based on the agent's policy
                self.CurrentState = self.root.childs[Agent.policy(self.root)]

        # Switch player ID for the next move
        self.player_id = not self.player_id
        return self.player_id

    def get_FeasibleAction(self):
        """
        Returns the list of feasible actions (child nodes) for the current game state.

        Returns:
        - List of feasible actions.
        """
        return list(self.CurrentState.childs.keys())

    def TakeAction(
        self, Action: Tuple[int], Agent: "Agent" = None, RANDOM_RESPONSE=0
    ) -> Tuple[float, ttt_node, bool]:
        """
        Simulates taking an action in the Tic Tac Toe game.

        Parameters:
        - Action: The action to be taken expressed as a tuple (board representation).
        - Agent: The agent making moves the response to this move.
        - RANDOM_RESPONSE: The probability of the opponent agent making a random move.

        Returns:
        - Tuple containing the payoff, the new game state, and a flag indicating if the game ended.
        """
        GAME_ENDED = True  # Flag indicating if the game ended
        self.CurrentState = self.CurrentState.childs[
            Action
        ]  # changes the state to the action made

        state = self.CurrentState.state_type

        # Check if the game ended in the current move and calculate payoff
        if state == StateType.WIN or state == StateType.LOSS:
            payoff = self.payoff_win
        elif state == StateType.DRAW:
            payoff = self.payoff_draw
        else:
            # Simulate opponent's move
            if Agent == None or random.random() < RANDOM_RESPONSE:
                Action2 = random.choice(self.get_FeasibleAction())
            else:
                Action2 = Agent.policy(self.CurrentState)

            # Update game state based on opponent's move
            self.CurrentState = self.CurrentState.childs[Action2]
            state = self.CurrentState.state_type

            # Calculate payoff based on the new state
            if state == StateType.WIN or state == StateType.LOSS:
                payoff = self.payoff_loss
                # Update QFactor and Visits for the learning agent
                if not Agent == None:
                    Agent.QFactor[(Action, Action2)] = self.payoff_win
                    Agent.Visits[(Action, Action2)] += 1
            elif state == StateType.DRAW:
                payoff = self.payoff_draw
                # Update QFactor and Visits for the learning agent
                if not Agent == None:
                    Agent.QFactor[(Action, Action2)] = self.payoff_draw
                    Agent.Visits[(Action, Action2)] += 1
            else:
                payoff = (
                    -5
                )  # negative payoff for any move to stimulate finding a win in less moves
                GAME_ENDED = False

        return payoff, self.CurrentState, GAME_ENDED


class Agent:
    def __init__(self, env: TicTacToe_Env) -> None:
        """
        Initializes the Agent with the Tic Tac Toe environment.

        Parameters:
        - env: An instance of the TicTacToe_Env class.
        """
        self.Env = env
        self.QFactor = defaultdict(float)  # Dictionary to store Q-values
        self.Visits = defaultdict(int)  # Dictionary to store visit counts
        return

    def QLearning(
        self,
        discount: float,
        alpha: float,
        epsilon: float = 0.2,
        n_step: int = 100,
        Agent: "Agent" = None,
        agent_randomness: float = 0,
        DEBUG: bool = False,
    ) -> None:
        """
        Implements the Q-learning algorithm.

        Parameters:
        - discount: Discount factor for future rewards ([0, 1]).
        - alpha: Learning rate ([0, 1]).
        - epsilon: Exploration-exploitation trade-off parameter ([0, 1]).
        - n_step: Number of steps for Q-learning.
        - Agent: Agent that responts to our move (optional); could be itself.
        - agent_randomness: Probability of the opponent agent making a random move ([0, 1]).
        - DEBUG: Flag for debugging purposes.

        Returns:
        - None
        """
        Action = None
        n_moves = 0
        self.Env.game_reset()

        for _ in tqdm(range(n_step)):
            # Current state of the environment
            current_state = self.Env.CurrentState.state
            # Exploitation: Choose action based on greedy strategy
            qf, Action = self.FindBest(self.Env.CurrentState, DEBUG)

            # DEBUG: Display chosen action
            if DEBUG:
                print("--CHOICE----------------")
                print_state(Action, qf)
                _ = input()
                clear_output(wait=True)

            if (
                n_moves >= 2 and qf > 0 and np.random.rand() < epsilon
            ):  # Exploration: Random action if conditions are met
                Action = random.choice(self.Env.get_FeasibleAction())

            # Take action and receive immediate payoff, new state, and game end flag
            if Agent == None:
                instant_payoff, newState, GAME_END = self.Env.TakeAction(Action)
            else:
                instant_payoff, newState, GAME_END = self.Env.TakeAction(
                    Action, Agent, RANDOM_RESPONSE=agent_randomness
                )
            n_moves += 1

            # Handle game end conditions
            if GAME_END:
                self.Env.game_reset(self, agent_randomness)
                n_moves = 0
                qtilde = instant_payoff
                if instant_payoff == self.Env.payoff_win:
                    # Update QFactor and Visits for winning condition
                    self.QFactor[(Old_Action, current_state)] = self.Env.payoff_loss
                    self.Visits[(Old_Action, current_state)] += 1
                elif instant_payoff == self.Env.payoff_draw:
                    # Update QFactor and Visits for draw condition
                    self.QFactor[(Old_Action, current_state)] = self.Env.payoff_draw
                    self.Visits[(Old_Action, current_state)] += 1
                Old_Action = None
            else:
                # Continue game: Calculate Q-value based on the next state
                next_payoff, _ = self.FindBest(newState)
                qtilde = instant_payoff + discount * next_payoff
                Old_Action = Action

            # Update QFactor based on Q-learning
            self.QFactor[(current_state, Action)] = (
                alpha * qtilde + (1 - alpha) * self.QFactor[(current_state, Action)]
            )
            self.Visits[(current_state, Action)] += 1

        return

    def FindBest(
        self, State: ttt_node, DEBUG: bool = False
    ) -> Tuple[float, Tuple[int]]:
        """
        Finds the best action based on Q-values.

        Parameters:
        - State: The current state of the Tic Tac Toe game.
        - DEBUG: Flag for debugging purposes.

        Returns:
        - Tuple containing the Q-value contribution of the best action and the best action itself in the form of board representation.
        """
        ActionList = list(State.childs.keys())
        # initialize vectors
        qf = np.zeros((len(ActionList)))
        v = np.zeros((len(ActionList)))

        # Calculate Q-values and visit counts for each action
        for i, a in enumerate(ActionList):
            qf[i] = self.QFactor[(State.state, a)]
            v[i] = self.Visits[(State.state, a)]

        # DEBUG: Display Actions List an Q-factors and Visits
        if DEBUG:
            print("Current State")
            print_state(State.state)
            print("---OPTIONS-----------------")
            print_states(ActionList)
            print(qf)
            print(v)

        # Select the action with the highest Q-value
        best_index = np.argmax(qf)
        contribution = qf[best_index]
        Action = ActionList[best_index]

        return contribution, Action

    def policy(self, State: ttt_node) -> Tuple[int]:
        """
        Determines the agent's policy for choosing actions.

        Parameters:
        - State: The current state of the Tic Tac Toe game.

        Returns:
        - The action selected based on the agent's policy.
        """
        _, Action = self.FindBest(State)
        return Action

    def play_game(self, flag_agent: bool = False):
        """
        Simulates playing a Tic Tac Toe game.

        Parameters:
        - flag_agent: Flag indicating whether the agent should play both sides.

        Returns:
        - None
        """

        # Reset the board according to who is playing
        if flag_agent:
            self.Env.game_reset(self)
        else:
            self.Env.game_reset()
        GAME_END = False
        print('---Starting Position---')
        print_state(self.Env.CurrentState.state)
        print('---Agent To Move---')
        # Play the game
        while not GAME_END:
            _, Action = self.FindBest(self.Env.CurrentState)

            # Prints number of visits of each couple (State, Action)
            # print(f"{self.Visits[(self.Env.CurrentState.state, Action)]}", end="\t\t")

            # Acts on the environment
            if flag_agent:
                _, NewState, GAME_END = self.Env.TakeAction(Action, self)
            else:
                _, NewState, GAME_END = self.Env.TakeAction(Action)

            # print(f"{self.Visits[(Action, NewState.state)]}")

            # prints the board on each move
            print_states([Action, NewState.state])

        # Prints how the game ended
        print(self.Env.CurrentState.state_type)

    def play_games(
        self, n_games: int, flag_agent: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Plays multiple Tic Tac Toe games and tracks statistics.

        Parameters:
        - n_games: Number of games to play.
        - flag_agent: Flag indicating whether the agent should play against itself.

        Returns:
        - Tuple containing win rate, draw rate, loss rate, and average game length.
        """
        n_wins = 0
        n_losses = 0
        n_draws = 0

        game_lenght = 0

        for _ in range(n_games):
            player_id = 0
            if flag_agent:
                player_id = self.Env.game_reset(self)
            else:
                player_id = self.Env.game_reset()

            END = False
            n_moves = 0
            while not END:
                n_moves += 1
                _, Action = self.FindBest(self.Env.CurrentState)
                if flag_agent:
                    _, _, END = self.Env.TakeAction(Action, self)
                else:
                    _, _, END = self.Env.TakeAction(Action)
            result = self.Env.CurrentState.state_type

            if (result == StateType.WIN and player_id == 0) or (
                result == StateType.LOSS and player_id == 1
            ):
                n_wins += 1
            elif result == StateType.DRAW:
                n_draws += 1
            else:
                n_losses += 1
            game_lenght += n_moves

        return (
            n_wins / n_games,
            n_draws / n_games,
            n_losses / n_games,
            game_lenght / n_games,
        )

    def play_game_Human(self):
        """
        Play a Tic Tac Toe game against a human player interactively.

        Returns:
        - None
        """

        self.Env.game_reset(self)

        END = False
        while not END:
            print_state(self.Env.CurrentState.state)
            print("-YOUR OPTIONS-")
            Actions = self.Env.get_FeasibleAction()
            for i, s in enumerate(Actions):
                print_state(s, i)
            Action_index = int(input("Choose your move index: "))
            Action = Actions[Action_index]
            _, _, END = self.Env.TakeAction(Action, self)
            clear_output(wait=True)
        print_state(self.Env.CurrentState.state)
        print(self.Env.CurrentState.state_type)


if __name__ == "__main__":
    # For testing purpose
    StartingPosition = ttt_node(tuple(-1 for _ in range(9)), dict())
    create_tree(StartingPosition)
    env = TicTacToe_Env(StartingPosition)
    agente = Agent(env)
    agente.QLearning(discount=0.8, alpha=.01, epsilon=0.15, n_step=500_000, Agent=agente)

    WR, DR, LR, GL = agente.play_games(100_000, flag_agent = False)
    print(f'Against Random Player:\t WinRate: {WR*100:.4}% \tDrawRate: {DR*100:.4}% \tLossRate: {LR*100:.4}% \nAverageLenght: {GL:.3}')
