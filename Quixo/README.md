# README Quixo project

## Credits

This work has been developed and worked on by myself in strict collaboration with Abdelouahab Moubane (S305716).

## Code Organisation

All the relevant code of the project is collected in the Quixo.py and Quixo_numba.py file. In the main.py file is only implemented the MyPlayer class and some boring method to test the player against other players and the random one.

## Quixo_numba.py

In this file you'll find the same method that is written in Quixo.py with the difference that, it's only compatible with Python3.11 because I used the Numba package(Unfortunately at this time there isn't a version compatible with Python3.12) to compile part of the code and further optimise it's execution. I included this file just in case someone is interested in running the MinMax algorithm with depths grater that 5 and still being able to observe the complete game in a very reasonable time (On my PC the performance improved by an order of magnitude).

To use the Player Class using this methods you will have to uncommon the NumbaPlayer class; in this way you can use it only if you intend to.

## Logic

The work revolves around the Min-Max Algorithm. The function calls itself recursively, progressively increasing the depth of the search. The main focus of this work is concentrated on the best way to adapt the algorithm to the Quixo game making it as efficient as possible. For this reason we implemented a way of avoiding the search in a state that was already searched or just a rotation/symmetry of an already seen position. We also implemented Min-Max with Alpha-Beta Pruning increasing the efficiency of the search. An other area of focus is the Static Evaluation: this function allows us to evaluate a state even if we did not reach a terminal state; moreover from this function also depends the ability of the algorithm to prune part of the tree search more effectively. In the Quixo file there are different version of StaticEvaluation functions but in the final Player I only used the one that proved to be the most effective (The testing of the evaluation function was done with depth = 1 in the Min Max algorithm: in this way the game has to rely much more on the static evaluation.)

### Alpha-Beta Pruning

In the MinMax_v4 I changed the standard alpha-beta pruning. The change that I made were necessary to ensure that the algorithm chooses always the fastest way to win; but using the standard alpha-beta pruning this doesn't always occur; the problem rises because we are not only interested on the static evaluation of a position but also in the move that leads us to 'nearest' leaf with that evaluation. If, for example, exploring the tree we find that the game is won in 5 moves, if we are only interested in the evaluation we can stop the search here (the position is winning even if we didn't find the most efficient way of playing), but if we want to win fast it's an other story. In the algorithm we keep track of the depth at which the recursion stop and when we find a other position with the same static evaluation we keep the one that received the evaluation at the earliest recursion depth. This small change makes the algorithm slower but assure us that the fastest way to win will be played.

## Results

We achieved great results against the random player and also in term of performance and efficiency. Playing 200 games against the RandomPlayer using depth = 1 (100 playing first and 100 playing second) the winning rate is 99.5\%. Increasing the depth search to 2 completely removes any winning chance for the RandomPlayer. To test further we also trying our self to beat it but with our little knowledge of the game and experience we where easily beaten even with depth = 3.
