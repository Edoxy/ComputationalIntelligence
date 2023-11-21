# LAB 2 Nim Game

## Contributions

The initial logic of the policy based on a set of Floating Numbers was developed in collaboration with Abdelouahab Moubane. The code implementing the evolution strategies is heavily inspired by the code of Prof. Squillero published in the course repository.

## Logic

This project optimises uses Evolution Strategies to optimise a parametric strategy to play the game Nim. Each individual as a genome that encodes a different strategy that is an array of weights associated to each possible nim-sum that can be obtain in the game; for example \[ -0.1 , 0.34 , 2 , ... ], means that:

- -0.1 is the weight associated to the nim-sum == 0;
- 0.34 is the weight associated to the nim-sum == 1;
- 2 is the weight associated to the nim-sum == 2;
- ...

The strategy now chooses between all the possible move that we can make from a position, what move is associated to the maximum weight using the nim-sum of the move.
