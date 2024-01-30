import numpy as np
import numba as nb
from numba import jit
from typing import List, Tuple
import time


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

INIT = tuple(nb.int64(-1) for _ in range(25))

# --------Functions/Classes Definition------------


def print_state(game_state: tuple, optional=None) -> None:
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


def print_states(game_state_list: List[tuple]):
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


@jit(nopython=True)
def simple_slide(state: np.array, index: int, slide: np.uint):
    from_pos = (index // 5, index % 5)
    Moves = {0, 1, 2, 3}

    if from_pos[0] == 0:
        Moves.remove(0)
    elif from_pos[0] == 4:
        Moves.remove(1)
    if from_pos[1] == 0:
        Moves.remove(2)
    elif from_pos[1] == 4:
        Moves.remove(3)

    if not slide in Moves:
        return False
    # take the piece
    piece = state[from_pos]
    # if the player wants to slide it to the left
    if slide == 2:
        # for each column starting from the column of the piece and moving to the left
        for i in range(from_pos[1], 0, -1):
            # copy the value contained in the same row and the previous column
            state[(from_pos[0], i)] = state[(from_pos[0], i - 1)]
        # move the piece to the left
        state[(from_pos[0], 0)] = piece
    # if the player wants to slide it to the right
    elif slide == 3:
        # for each column starting from the column of the piece and moving to the right
        for i in range(from_pos[1], state.shape[1] - 1, 1):
            # copy the value contained in the same row and the following column
            state[(from_pos[0], i)] = state[(from_pos[0], i + 1)]
        # move the piece to the right
        state[(from_pos[0], state.shape[1] - 1)] = piece
    # if the player wants to slide it upward
    elif slide == 0:
        # for each row starting from the row of the piece and going upward
        for i in range(from_pos[0], 0, -1):
            # copy the value contained in the same column and the previous row
            state[(i, from_pos[1])] = state[(i - 1, from_pos[1])]
        # move the piece up
        state[(0, from_pos[1])] = piece
    # if the player wants to slide it downward
    elif slide == 1:
        # for each row starting from the row of the piece and going downward
        for i in range(from_pos[0], state.shape[0] - 1, 1):
            # copy the value contained in the same column and the following row
            state[(i, from_pos[1])] = state[(i + 1, from_pos[1])]
        # move the piece down
        state[(state.shape[0] - 1, from_pos[1])] = piece
    return True


@jit(nopython=True)
def check_winner(state: tuple) -> bool:
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


@jit(nopython=True)
def rotate(s: tuple):
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


@jit(nopython=True)
def symmetry(s: tuple):
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


@jit(nopython=True)
def StaticEval1(state, player_id = True):
    '''
    Static Evaluation of a position:
    - Bonus for symbols in the border;
    - Bonus for symbols in the corners;
    - Bonus for controlling both sides of a column or row
    - Bonus for alligned symbols in the central 3x3
    '''
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


@jit(nopython=True)
def StaticEval(state, player_id = True):
    '''
    Static Evaluation of a position:
    - Bonus for symbols in the border;
    - Bonus for symbols in the corners;
    - Bonus for controlling both sides of a column or row
    - Bonus for alligned symbols in the central 3x3
    - Bonus for every symbol aligned
    '''
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
                
                row[i//5] += 1
                column[i%5] += 1

                if i == 0:
                    diagonal[0] += 1
                elif i%6 == 0:
                    diagonal[0] += 1
                elif i%4 == 0:
                    diagonal[1] += 1
        
        if player_id == player:
            row = row + 1
            column = column + 1
            diagonal = diagonal + 1

        Eval += np.sum(row**3)*sign
        Eval += np.sum(column**3)*sign
        Eval += np.sum(diagonal**3)*sign

    return Eval


@jit(nopython=True)
def min_max_v3(
    state: Tuple[int],
    depth: int,
    player: bool,
    alpha: float = -10_000,
    beta: float = +10_000,
) -> Tuple[float, Tuple[int, int], int]:
    state_type = check_winner(state)

    if state_type == 0:
        return (+10_000, (0, 0), 0)
    elif state_type == 1:
        return (-10_000, (0, 0), 0)
    if depth == 0:
        eval = StaticEval(state, player)
        return (eval, (0, 0), 0)
    else:
        DISCOUNT = 0.999
        child_set = {INIT}  # to tell the Numba package the types

        best_pos = (0, 0)
        best_slide = 0

        best_Eval = -np.Inf
        if player == 1:
            best_Eval = np.Inf
        for i in BORDER:
            row = i // 5
            column = i % 5
            if state[i] == -1 or state[i] == player:
                for slide in range(4):
                    tmp_game = np.array(state).reshape((5, 5))
                    tmp_game[row, column] = player
                    if simple_slide(tmp_game, i, slide):
                        child = (
                            tmp_game[0, 0],
                            tmp_game[0, 1],
                            tmp_game[0, 2],
                            tmp_game[0, 3],
                            tmp_game[0, 4],
                            tmp_game[1, 0],
                            tmp_game[1, 1],
                            tmp_game[1, 2],
                            tmp_game[1, 3],
                            tmp_game[1, 4],
                            tmp_game[2, 0],
                            tmp_game[2, 1],
                            tmp_game[2, 2],
                            tmp_game[2, 3],
                            tmp_game[2, 4],
                            tmp_game[3, 0],
                            tmp_game[3, 1],
                            tmp_game[3, 2],
                            tmp_game[3, 3],
                            tmp_game[3, 4],
                            tmp_game[4, 0],
                            tmp_game[4, 1],
                            tmp_game[4, 2],
                            tmp_game[4, 3],
                            tmp_game[4, 4],
                        )

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
                            child_eval, _, _ = min_max_v3(
                                child, depth - 1, not player, alpha, beta
                            )
                            # child_eval *= DISCOUNT
                            if player == 0:
                                if best_Eval < child_eval:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    alpha = child_eval
                            else:
                                if best_Eval > child_eval:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    beta = child_eval

                            if beta <= alpha:
                                return (best_Eval, best_pos, best_slide)
    
    return (best_Eval, best_pos, best_slide)

@jit(nopython=True)
def min_max_v3_1(
    state: Tuple[int],
    depth: int,
    player: bool,
    alpha: float = -np.Inf,
    beta: float = +np.Inf,
) -> Tuple[float, Tuple[int, int], int]:
    state_type = check_winner(state)


    if state_type == 0:
        return (+10_000, (0, 0), 0)
    elif state_type == 1:
        return (-10_000, (0, 0), 0)
    if depth == 0:
        eval = StaticEval1(state)
        return (eval, (0, 0), 0)
    else:
        DISCOUNT = 0.999
        K = 10_000 * (DISCOUNT**depth)
        child_set = {INIT}  # to tell the Numba package the types

        best_pos = (0, 0)
        best_slide = 0

        best_Eval = -np.Inf
        if player == 1:
            best_Eval = np.Inf
        for i in BORDER:
            row = i // 5
            column = i % 5
            if state[i] == -1 or state[i] == player:
                for slide in range(4):
                    tmp_game = np.array(state).reshape((5, 5))
                    tmp_game[row, column] = player
                    if simple_slide(tmp_game, i, slide):
                        child = (
                            tmp_game[0, 0],
                            tmp_game[0, 1],
                            tmp_game[0, 2],
                            tmp_game[0, 3],
                            tmp_game[0, 4],
                            tmp_game[1, 0],
                            tmp_game[1, 1],
                            tmp_game[1, 2],
                            tmp_game[1, 3],
                            tmp_game[1, 4],
                            tmp_game[2, 0],
                            tmp_game[2, 1],
                            tmp_game[2, 2],
                            tmp_game[2, 3],
                            tmp_game[2, 4],
                            tmp_game[3, 0],
                            tmp_game[3, 1],
                            tmp_game[3, 2],
                            tmp_game[3, 3],
                            tmp_game[3, 4],
                            tmp_game[4, 0],
                            tmp_game[4, 1],
                            tmp_game[4, 2],
                            tmp_game[4, 3],
                            tmp_game[4, 4],
                        )

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
                            child_eval, _, _ = min_max_v3_1(
                                child, depth - 1, not player, alpha, beta
                            )
                            # child_eval -= 1
                            if player == 0:
                                if best_Eval < child_eval:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    alpha = child_eval
                            else:
                                if best_Eval > child_eval:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    beta = child_eval

                            if beta <= alpha or alpha >= K or beta <= -K:
                                return (best_Eval, best_pos, best_slide)
    
    return (best_Eval, best_pos, best_slide)


@jit(nopython=True)
def min_max_v4(
    state: Tuple[int],
    depth: int,
    player: bool,
    alpha: float = -np.Inf,
    beta: float = +np.Inf,
) -> Tuple[float, Tuple[int, int], int]:
    MAX = 10_000
    MIN = -MAX
    
    state_type = check_winner(state)


    if state_type == 0:
        return (MAX, (0, 0), 0, depth)
    elif state_type == 1:
        return (MIN, (0, 0), 0, depth)
    if depth == 0:
        eval = StaticEval(state)
        return (eval, (0, 0), 0, depth)
    else:

        child_set = {INIT}  # to tell the Numba package the types

        best_pos = (0, 0)
        best_slide = 0
        best_depth = -1

        best_Eval = -np.Inf
        if player == 1:
            best_Eval = np.Inf
        for i in BORDER:
            row = i // 5
            column = i % 5
            if state[i] == -1 or state[i] == player:
                for slide in range(4):
                    tmp_game = np.array(state).reshape((5, 5))
                    tmp_game[row, column] = player
                    if simple_slide(tmp_game, i, slide):
                        child = (
                            tmp_game[0, 0],
                            tmp_game[0, 1],
                            tmp_game[0, 2],
                            tmp_game[0, 3],
                            tmp_game[0, 4],
                            tmp_game[1, 0],
                            tmp_game[1, 1],
                            tmp_game[1, 2],
                            tmp_game[1, 3],
                            tmp_game[1, 4],
                            tmp_game[2, 0],
                            tmp_game[2, 1],
                            tmp_game[2, 2],
                            tmp_game[2, 3],
                            tmp_game[2, 4],
                            tmp_game[3, 0],
                            tmp_game[3, 1],
                            tmp_game[3, 2],
                            tmp_game[3, 3],
                            tmp_game[3, 4],
                            tmp_game[4, 0],
                            tmp_game[4, 1],
                            tmp_game[4, 2],
                            tmp_game[4, 3],
                            tmp_game[4, 4],
                        )

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
                            child_eval, _, _, child_depth = min_max_v4(
                                child, depth - 1, not player, alpha, beta
                            )

                            if player == 0: # positive evaluation favours the first player (0)
                                
                                if best_Eval < child_eval or ((not best_Eval == MIN) and best_Eval == child_eval and best_depth < child_depth):
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth
                                    alpha = child_eval
                                
                                elif best_Eval == MIN and best_Eval == child_eval and best_depth > child_depth: # if he knows that he has lost, choose the longest path
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth

                            else: # negative evaluation favours the second player (1)
                                if best_Eval > child_eval or ((not best_Eval == MAX) and best_Eval == child_eval and best_depth < child_depth):
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth
                                    beta = child_eval
                                    
                                elif best_Eval == child_eval and best_Eval == MAX and best_depth > child_depth:
                                    best_Eval = child_eval
                                    best_pos = (row, column)
                                    best_slide = slide
                                    best_depth = child_depth

                            if beta < alpha:
                                return (best_Eval, best_pos, best_slide, best_depth)
                            
    return (best_Eval, best_pos, best_slide, best_depth)




if __name__ == "__main__":

    test_state = (
        1, 1, 1, 1, -1,
        -1,-1,-1,-1,-1,
        -1,-1,-1, 0,-1,
        -1,-1,-1,-1,-1,
        -1,-1, 0, 0, 0
    )
    
    print_state(test_state)

    print(min_max_v3(INIT, 0, 0))
    # print(min_max_v3_1(INIT, 0, 0))

    

    start = time.time()
    print(min_max_v3(INIT, 5, 1))
    print(time.time() - start)

    # start = time.time()
    # print(min_max_v3_1(test_state, 5, 1))
    # print(time.time() - start)