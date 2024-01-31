import numpy as np
from typing import Callable
from itertools import combinations
import time
from game import Move

# --------Constants Definition------------

_3X3MagicSquare = (
    0,
    0,
    0,
    0,
    0,
    0,
    4,
    9,
    2,
    0,
    0,
    3,
    5,
    7,
    0,
    0,
    8,
    1,
    6,
    0,
    0,
    0,
    0,
    0,
    0,
)

BORDER = (0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24)

MOVESET = (Move.BOTTOM, Move.TOP, Move.RIGHT, Move.LEFT)

INIT = tuple(np.int64(-1) for _ in range(25))  # Initial state

# --------Functions/Classes Definition------------


def print_state(game_state: tuple[int, ...], optional=None) -> None:
    """
    Prints the Quixo board in a given state
    """
    string = ""
    for i in range(5):
        row = ""
        for j in range(5):
            index = j + 5 * i
            if game_state[index] == 0:
                row += "X "
            elif game_state[index] == 1:
                row += "O "
            else:
                row += "• "

        if i == 2 and not optional == None:
            row += "\t" + str(optional)
        string += row + "\n"
    print(string)


def print_states(game_state_list: list[tuple[int, ...]]):
    string = ""
    for i in range(5):
        row = ""
        for game_state in game_state_list:
            for j in range(5):
                if game_state[j + 5 * i] == 0:
                    row += "X "
                elif game_state[j + 5 * i] == 1:
                    row += "O "
                else:
                    row += "• "
            row += "\t\t"

        string += row + "\n"
    print(string)


def simple_slide(state: np.array, index: int, slide: Move):
    """
    This method is borrowed from the professsor code
    """

    from_pos = (index // 5, index % 5)
    Moves = set(MOVESET)

    if from_pos[0] == 0:
        Moves.remove(Move.TOP)
    elif from_pos[0] == 4:
        Moves.remove(Move.BOTTOM)
    if from_pos[1] == 0:
        Moves.remove(Move.LEFT)
    elif from_pos[1] == 4:
        Moves.remove(Move.RIGHT)

    if not slide in Moves:
        return False
    # take the piece
    piece = state[from_pos]
    # if the player wants to slide it to the left
    if slide == Move.LEFT:
        # for each column starting from the column of the piece and moving to the left
        for i in range(from_pos[1], 0, -1):
            # copy the value contained in the same row and the previous column
            state[(from_pos[0], i)] = state[(from_pos[0], i - 1)]
        # move the piece to the left
        state[(from_pos[0], 0)] = piece
    # if the player wants to slide it to the right
    elif slide == Move.RIGHT:
        # for each column starting from the column of the piece and moving to the right
        for i in range(from_pos[1], state.shape[1] - 1, 1):
            # copy the value contained in the same row and the following column
            state[(from_pos[0], i)] = state[(from_pos[0], i + 1)]
        # move the piece to the right
        state[(from_pos[0], state.shape[1] - 1)] = piece
    # if the player wants to slide it upward
    elif slide == Move.TOP:
        # for each row starting from the row of the piece and going upward
        for i in range(from_pos[0], 0, -1):
            # copy the value contained in the same column and the previous row
            state[(i, from_pos[1])] = state[(i - 1, from_pos[1])]
        # move the piece up
        state[(0, from_pos[1])] = piece
    # if the player wants to slide it downward
    elif slide == Move.BOTTOM:
        # for each row starting from the row of the piece and going downward
        for i in range(from_pos[0], state.shape[0] - 1, 1):
            # copy the value contained in the same column and the following row
            state[(i, from_pos[1])] = state[(i + 1, from_pos[1])]
        # move the piece down
        state[(state.shape[0] - 1, from_pos[1])] = piece
    return True


def check_winner(state: tuple[int, ...]) -> bool:
    """ """
    # check rows
    for row in range(5):
        if not state[row * 5] == -1:
            if (
                state[row * 5] == state[row * 5 + 1]
                and state[row * 5] == state[row * 5 + 2]
                and state[row * 5] == state[row * 5 + 3]
                and state[row * 5] == state[row * 5 + 4]
            ):
                return state[row * 5]
    # check columns
    for col in range(5):
        if not state[col] == -1:
            if (
                state[col] == state[5 + col]
                and state[col] == state[10 + col]
                and state[col] == state[15 + col]
                and state[col] == state[20 + col]
            ):
                return state[col]
    # if a player has completed the principal diagonal
    if not state[0] == -1:
        if (
            state[0] == state[6]
            and state[0] == state[12]
            and state[0] == state[18]
            and state[0] == state[24]
        ):
            return state[0]
    if not state[4] == -1:
        if (
            state[4] == state[8]
            and state[4] == state[12]
            and state[4] == state[16]
            and state[4] == state[20]
        ):
            return state[4]
    return -1


def rotate(s: tuple[int, ...]) -> tuple[int, ...]:
    """
    return a new tuple representing a clock-wise rotation of the board
    """
    return (
        s[20],
        s[15],
        s[10],
        s[5],
        s[0],
        s[21],
        s[16],
        s[11],
        s[6],
        s[1],
        s[22],
        s[17],
        s[12],
        s[7],
        s[2],
        s[23],
        s[18],
        s[13],
        s[8],
        s[3],
        s[24],
        s[19],
        s[14],
        s[9],
        s[4],
    )


def symmetry(s: tuple[int, ...]) -> tuple[int, ...]:
    """
    return a new tuple representing a vertical symmetry of the board
    """
    return (
        s[4],
        s[3],
        s[2],
        s[1],
        s[0],
        s[9],
        s[8],
        s[7],
        s[6],
        s[5],
        s[14],
        s[13],
        s[12],
        s[11],
        s[10],
        s[19],
        s[18],
        s[17],
        s[16],
        s[15],
        s[24],
        s[23],
        s[22],
        s[21],
        s[20],
    )



#--- STATIC EVALUATION FUNCTIONS ---#

def StaticEval1(state: tuple[int, ...], player_id: bool) -> float:
    """ """
    Eval = 0

    for player in (0, 1):
        sign = +1
        if player == 1:
            sign = -1
        Eval += sum([1 for i in range(25) if state[i] == player]) * sign
        Eval += sum([1 for i in BORDER if state[i] == player]) * sign

        score = [_3X3MagicSquare[i] for i in range(25) if state[i] == player]
        if any(sum(c) == 15 for c in combinations(score, 3)):
            Eval += 50 * sign

    return Eval


def StaticEval2(state: tuple[int, ...], player_id: bool) -> float:
    """
    Static Evaluation of a position:
    - Bonus for symbols in the border;
    - Bonus for symbols in the corners;
    - Bonus for controlling both sides of a column or row
    - Bonus for alligned symbols in the central 3x3
    - Bonus for every symbol aligned
    """
    Eval = 0

    for player in (0, 1):
        sign = +1
        if player == 1:
            sign = -1

        for i in range(25):
            if state[i] == player:
                if i in BORDER:
                    Eval += 3 * sign
                if i // 5 == 0 and state[4 * 5 + i] == player:
                    Eval += 6 * sign
                if i % 5 == 0 and state[i + 4] == player:
                    Eval += 6 * sign

                else:
                    Eval += 1 * sign
                if i in {0, 4, 20, 24}:
                    Eval += 5 * sign

        score = [_3X3MagicSquare[i] for i in range(25) if state[i] == player]

        score_len = len(score)

        if score_len > 2:
            for a in range(score_len):
                for b in range(a + 1, score_len):
                    for c in range(b + 1, score_len):
                        if score[a] + score[b] + score[c] == 15:
                            Eval += 50 * sign

    return Eval


def StaticEval3(state: tuple[int, ...], player_id: bool = True) -> float:
    """ """
    Eval = 0

    for player in (0, 1):
        row = np.zeros((5), dtype=np.int64)
        column = np.zeros((5), dtype=np.int64)
        diagonal = np.zeros((2), dtype=np.int64)

        sign = +1
        if player == 1:
            sign = -1

        for i in range(25):
            if state[i] == player:
                row[i // 5] += 1
                column[i % 5] += 1

                if i == 0:
                    diagonal[0] += 1
                elif i % 6 == 0:
                    diagonal[0] += 1
                elif i % 4 == 0:
                    diagonal[1] += 1

        if player_id == player:
            row = row + 1
            column = column + 1
            diagonal = diagonal + 1

        Eval += np.sum(row**3) * sign
        Eval += np.sum(column**3) * sign
        Eval += np.sum(diagonal**3) * sign

    return Eval


def StaticEval4(state: tuple[int, ...], player_id: bool = True) -> float:
    """ 
    This approach keeps a score for each row, column and diagonal and uses exponentiation to give a non linear increse in bonus for occuping the same row/column/diagonal
    """
    Eval = 0

    row = np.zeros((5), dtype=np.int64)
    column = np.zeros((5), dtype=np.int64)
    diagonal = np.zeros((2), dtype=np.int64)

    bonus = (1, -1)

    for i in range(25):
        if state[i] == -1:
            continue
        elif state[i] == player_id:
            current_point = bonus[player_id] * 1.5 # if the position has the current player type we add a smal bonus
        else:
            current_point = bonus[not player_id]

        row[i // 5] += current_point # point to current row
        column[i % 5] += current_point # point to current column

        if i == 0: # this square is in the first diagonal
            diagonal[0] += current_point
        elif i % 6 == 0: # if not the top-left corner and this condition is met, we are in the second diagonal
            diagonal[0] += current_point
        elif i % 4 == 0: # the rest of the first diagonal
            diagonal[1] += current_point

    Eval += np.sum(row**3)
    Eval += np.sum(column**3)
    Eval += np.sum(diagonal**3)

    return Eval

#--- OLD ITERATION OF MIN MAX ---#

def min_max_v2(
    state: tuple[int, ...],
    depth: int,
    player: bool,
    alpha: float = -np.Inf,
    beta: float = +np.Inf,
) -> float:
    state_type = check_winner(state)

    if state_type == 0:
        return +1000
    elif state_type == 1:
        return -1000
    if depth == 0:
        return StaticEval1(state)
    else:
        child_set = set()

        Eval = -np.Inf
        if player == 1:
            Eval = np.Inf
        for i in BORDER:
            row = i // 5
            column = i % 5
            if state[i] == -1 or state[i] == player:
                for slide in MOVESET:
                    tmp_game = np.array(state).reshape((5, 5))
                    tmp_game[row, column] = player
                    if simple_slide(tmp_game, i, slide):
                        child = tuple(tmp_game.reshape((25)))

                        if not child in child_set:
                            child_set = child_set.union(
                                {
                                    child,
                                    rotate(child),
                                    rotate(rotate(child)),
                                    rotate(rotate(rotate(child))),
                                    symmetry(child),
                                    symmetry(rotate(child)),
                                    symmetry(rotate(rotate(child))),
                                    rotate(symmetry(child)),
                                }
                            )

                            child_eval = min_max_v2(
                                child, depth - 1, not player, alpha, beta
                            )
                            if player == 0:
                                Eval = max(Eval, child_eval)
                                alpha = max(alpha, child_eval)
                            else:
                                Eval = min(Eval, child_eval)
                                beta = min(beta, child_eval)

                            if beta <= alpha:
                                break
    return Eval


def min_max_v3(
    state: tuple[int, ...],
    depth: int,
    player: bool,
    static_eval_func: Callable,
    alpha: float = -np.Inf,
    beta: float = +np.Inf,
) -> tuple[float, tuple[int, int], int]:
    # ----# First Init #----#

    state_type = check_winner(state)
    best_pos = (0, 0)
    best_slide = 0

    # ----# Checks if its a leaf #----#
    if state_type == 0:
        return (+1000, best_pos, best_slide)
    elif state_type == 1:
        return (-1000, best_pos, best_slide)
    if depth == 0:
        eval = static_eval_func(state, player)
        return (eval, best_pos, best_slide)
    else:
        # -----# INITIALIZATION #----#

        child_set = set()
        best_Eval = -np.Inf
        if player == 1:
            best_Eval = np.Inf

        # ----# Cycle through the moves #----#
        for i in BORDER:
            row = i // 5
            column = i % 5

            if (
                state[i] == -1 or state[i] == player
            ):  # check if its possible to take the piece
                for slide in MOVESET:  # cycle for every possible slide
                    tmp_game = np.array(state).reshape((5, 5))
                    tmp_game[row, column] = player
                    if simple_slide(
                        tmp_game, i, slide
                    ):  # checks if it possible to do this move
                        # ----# Possible ply found #----#

                        child = tuple(
                            tmp_game.reshape((25))
                        )  # create standard representation

                        if (
                            not child in child_set
                        ):  # checks if the position is already in the set
                            # ----# Diedral Group #----#
                            child_set = child_set.union(
                                {
                                    child,
                                    rotate(child),
                                    rotate(rotate(child)),
                                    rotate(rotate(rotate(child))),
                                    symmetry(child),
                                    symmetry(rotate(child)),
                                    symmetry(rotate(rotate(child))),
                                    rotate(symmetry(child)),
                                }
                            )

                            # ----# Recursive Call #----#
                            child_eval, _, _ = min_max_v3(
                                child,
                                depth - 1,
                                not player,
                                static_eval_func,
                                alpha,
                                beta,
                            )
                            if player == 0:
                                if best_Eval < child_eval:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                alpha = max(alpha, child_eval)
                            else:
                                if best_Eval > child_eval:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                beta = min(beta, child_eval)

                            if beta <= alpha:
                                return (best_Eval, best_pos, best_slide)
    return (best_Eval, best_pos, best_slide)


#--- LAST VERSION MIN MAX ---#

MAX = 10000
MIN = -10000


def min_max_v4(
    state: tuple[int, ...],
    depth: int,
    player: bool,
    static_eval_func: Callable,
    parent_set=set(),
    alpha: float = MIN,
    beta: float = MAX,
) -> tuple[float, tuple[int, int], int, int]:
    
    '''
    Last iteration of the MinMax algorithm.
    This version includes the best improvements in terms of efficiency and also improves the alpha-beta pruning -> see readme.
    '''
    # ----# First Init #----#

    state_type = check_winner(state)
    best_pos = (0, 0)
    best_slide = 0
    best_depth = -1

    # ----# Checks if its a leaf #----#

    if state_type == 0:
        # print_state(state, depth)
        return (MAX, best_pos, best_slide, depth)
    elif state_type == 1:
        return (MIN, best_pos, best_slide, depth)
    if depth == 0:
        eval = static_eval_func(state, player)
        return (eval, best_pos, best_slide, depth)
    else:
        # -----# INITIALIZATION #----#

        # parent_set = parent_set.union({state})
        # child_set = parent_set
        child_set = set()
        best_Eval = -np.Inf
        if player == 1:
            best_Eval = np.Inf

        # ----# Cycle through the moves #----#
        for i in BORDER:
            row = i // 5
            column = i % 5

            if (
                state[i] == -1 or state[i] == player
            ):  # check if its possible to take the piece
                for slide in MOVESET:  # cycle for every possible slide
                    tmp_game = np.array(state).reshape((5, 5))
                    tmp_game[row, column] = player
                    if simple_slide(
                        tmp_game, i, slide
                    ):  # checks if it possible to do this move
                        # ----# Possible ply found #----#

                        child = tuple(
                            tmp_game.reshape((25))
                        )  # create standard representation

                        if (
                            not child in child_set
                        ):  # checks if the position is already in the set
                            # ----# Diedral Group #----#
                            child_set = child_set.union(
                                {
                                    child,
                                    rotate(child),
                                    rotate(rotate(child)),
                                    rotate(rotate(rotate(child))),
                                    symmetry(child),
                                    symmetry(rotate(child)),
                                    symmetry(rotate(rotate(child))),
                                    rotate(symmetry(child)),
                                }
                            )

                            # ----# Recursive Call #----#

                            child_eval, _, _, child_depth = min_max_v4(
                                child,
                                depth - 1,
                                not player,
                                static_eval_func,
                                parent_set,
                                alpha,
                                beta,
                            )

                            if (
                                player == 0
                            ):  # positive evaluation favours the first player (0)
                                if best_Eval < child_eval or (
                                    (not best_Eval == MIN)
                                    and best_Eval == child_eval
                                    and best_depth < child_depth
                                ):
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth
                                    alpha = child_eval

                                elif (
                                    best_Eval == MIN
                                    and best_Eval == child_eval
                                    and best_depth > child_depth
                                ):  # if he knows that he has lost, choose the longest path
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth

                            else:  # negative evaluation favours the second player (1)
                                if best_Eval > child_eval or (
                                    (not best_Eval == MAX)
                                    and best_Eval == child_eval
                                    and best_depth < child_depth
                                ):
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth
                                    beta = child_eval

                                elif (
                                    best_Eval == child_eval
                                    and best_Eval == MAX
                                    and best_depth > child_depth
                                ):
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth

                            if beta <= alpha:
                                return (best_Eval, best_pos, best_slide, best_depth)

    return (best_Eval, best_pos, best_slide, best_depth)


if __name__ == "__main__":
    # test on a random state
    state = (0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, 0, 1, 1, 1, 1, 0)
    print_state(state)
    start = time.time()
    print(min_max_v4(state, 5, 0, StaticEval4))
    print(time.time() - start)
