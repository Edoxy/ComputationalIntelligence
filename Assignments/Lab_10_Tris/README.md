# README LAB 10

## Contributions

All the code generating the tree structure accounting for the rotations and the symmetry was developed with Abdelouahab Moubane (S305716); the remaning parts were written in close collaboration too.

## LAB 10 Outline

In the TicTaToe_RL.py file I wrote all of the classes and methods that I used in the lab 10.
All the board representations are done using a tuple of length 9 that encodes the whole square of tic-tac-toe:

- -1: means that the space is empty;

- 0: means that the space is occupied by a X;

- 1: means that the space is occupied by a O.

For example an empty board is represented by: ``tuple([-1, -1, -1, -1, -1, -1, -1, -1, -1])``

The choice of using a tuple is to make python able to hash the variable using it as a key for a dictionary. This is useful when dealing with the QFactors that are stored using a ``defaultdict``.

The 2 main classes are:

- TicTacToe_Env: this class acts as the Environment that and Agent interacts with, keeping things separated; the rewards are given to the agent from this class.
- Agent: this class is the main agent that is able to learn from is actions. After having played in the Q-Learning phase, is able to apply a policy and play games instantly. There is also a method that lets you play against the agent to test its ability to win.

## Disclamer

If a state is classified as a Win it doesn't directly mean that the Agent has won, but the State_Type only refers to the player playing first (X). In a Game visualised by Agent.play_game() the first move after the starting position is always one by the agent, regardless of the (X or O): it automatically understand what move it has to do based on the position it was passed to it.

## Result

The result that we can see in the Python Notebook are reached very easily in just 20 seconds of training. To understand the results we need to note that in this version the Random Player is a bit less likely to lose: in fact if the first move done by the Agent is:

    X • •
    • • •  
    • • •

The Random Player can only reply in 5 ways instead of 8 (because it's restricted to the tree structure):

    X O •   X • O   X • •   X • •   X • •
    • • •   • • •   • • O   • O •   • • •
    • • •   • • •   • • •   • • •   • • O

Given that the only winning move in the position is the central one, we shift the probability of winning against a Random Player from $7/8$ to $4/5$ meaning that is circa 9\% less likely to loose on the spot.
