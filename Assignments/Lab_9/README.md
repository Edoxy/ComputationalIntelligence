# README LAB9

## Contributions

The logic of the code and the inial structure of the EA algorithm was developed in strict collaboration with Abdelouahab Moubane (S305716).

## LAB 9 Outline

All my Evolution Algorithm code is organized inside the EvolutionAlgorithm.py module. Inside it there are all functions definition that I used and some more that I wrote but didn't use in the end; for example:

- EA(): this is the first version of the algorithm that I used but it's much too slow to be used so I ended up improving it with a heap queu
- EA_heap(): second iteration of the algorithm that implements the heap queue
- EA_Extinction(): my implementation of a genetic algorithm that has periodic extintions: I did not use it in the final notebook becuse in this case it didn0t improve the situation as much as the Islands do.
...

I kept this function declarations to make for an easier task when tring to understand the logic of the latest version of the same functions (EA_v2 or EA_Islands).

Moreover in the same module are also declared all the different genetic operators that I tried even if in the final analisys I did not use them all.
