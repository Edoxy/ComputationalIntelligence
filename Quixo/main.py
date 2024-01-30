import random
from Quixo import min_max_v4, print_state, StaticEval4
from game import Game, Move, Player
from tqdm.auto import tqdm
from itertools import combinations

# --- IMPORT for the NUMBA version ---#
# from Quixo_numba import min_max_v4 as min_max_v4_numba
# import numpy as np


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    """
    This class uses the Min Max algorithm to play choose a move of a Quixo game. this is the best player we were able to create
    Other version of this player can be created changing the StaticEval fuction used in the min-max
    """

    def __init__(self, depth=2, DEBUG=False) -> None:
        super().__init__()
        self.depth = depth
        self.DEBUG = DEBUG

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        player_id = game.get_current_player()
        state = game.get_board().reshape((25))

        eval, pos, slide, depth = min_max_v4(
            tuple(state), self.depth, player_id, static_eval_func=StaticEval4
        )
        if self.DEBUG:
            print_state(state)
            print("Player", player_id)
            print(pos, slide, eval, self.depth - depth, sep="\t")
        return (pos[1], pos[0]), slide


# --- NUMBA PLAYER ---#
"""
    This player class requires the numba package installed
    It playes as the previous class but has much better performance (10x better in my system),
    if you wish to use this class uncomment it and uncomment also the import statement of the Quixo_numba package
"""

# class MyPlayer_numba(Player):

#     def __init__(self, depth=5) -> None:
#         super().__init__()
#         self.depth = depth

#     def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
#         player_id = game.get_current_player()
#         state = game.get_board().reshape((25))
#         state = np.asarray(state, dtype=np.int64)
#         eval, pos, slide, _ = min_max_v4_numba(tuple(state), self.depth, player_id)
#         return (pos[1], pos[0]), Move(slide)


def Random_vs_player(n_games: int, player: Player):
    """
    function to estimate the performance of a Player against a Random one

    n_games: int -> number of games played for both starting first and second
    player: Player -> Player class
    """
    n_wins = 0
    n_loss = 0
    random = RandomPlayer()
    for game in tqdm(range(n_games)):
        g = Game()
        winner = g.play(random, player)
        if winner == 0:
            n_loss += 1
        else:
            n_wins += 1
    for game in tqdm(range(n_games)):
        g = Game()
        winner = g.play(player, random)
        if winner == 1:
            n_loss += 1
        else:
            n_wins += 1

    return n_wins / (n_games * 2), n_loss / (n_games * 2)


def Player_vs_Player(
    n_games, list_player: tuple[Player, Player]
) -> tuple[float, float]:
    '''Method used to test different player that we wrote during the development'''

    n_wins = 0
    n_loss = 0
    n_draws = 0
    for _ in tqdm(range(n_games)):
        g = Game()
        winner = g.play(list_player[0], list_player[1])
        if winner == 1:
            n_loss += 1
        elif winner == 0:
            n_wins += 1
        else:
            n_draws += 1
    for _ in tqdm(range(n_games)):
        g = Game()
        winner = g.play(list_player[1], list_player[0])
        if winner == 0:
            n_loss += 1
        elif winner == 1:
            n_wins += 1
        else:
            n_draws += 1

    return n_wins / (n_games * 2), n_loss / (n_games * 2), n_draws / (n_games * 2)


def Player_vs_Player_depth(n_games, classes):

    ''' this method was used in test to try different combination of static evaluation, players and depth'''

    n_wins = 0
    n_loss = 0
    n_draws = 0

    for c in combinations([1, 2, 3, 4], 2):
        print(c)
        for _ in tqdm(range(n_games)):
            g = Game()

            winner = g.play(classes[0](c[0]), classes[1](c[1]))
            if winner == 1:
                n_loss += 1
            elif winner == 0:
                n_wins += 1
            else:
                n_draws += 1
        for _ in tqdm(range(n_games)):
            g = Game()
            winner = g.play(classes[1](c[0]), classes[0](c[1]))
            if winner == 0:
                n_loss += 1
            elif winner == 1:
                n_wins += 1
            else:
                n_draws += 1

    return (
        n_wins / (n_games * 4 * 3),
        n_loss / (n_games * 4 * 3),
        n_draws / (n_games * 4 * 3),
    )


if __name__ == "__main__":
    g = Game()
    g.print()
    player1 = MyPlayer(2)
    player2 = RandomPlayer()
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")

    # --- TEST ON MULTIPLE GAMES ---#

    w, l = Random_vs_player(50, MyPlayer(2))
    print(f"Wins: {w*100:.4}%\tLoss: {l*100:.4}%")
