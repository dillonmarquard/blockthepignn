## blockthepignn - Graph-based BFS to beat Block the Pig on Cool Math Games
### Dependencies
```
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
```
### Description
This project was an effort to apply graph based algorithms to solve for an optimal solution.
As of 2/7/2021 there is a greedy approach that will solve around 70% of gamestates only considering the cost of a move one step in the future. While fast, it does not guarentee a solution, even if one exists.
As of 2/8/2021 a bruteforce was made as a proof of concept and while there are methods of reducing the search space, ultimately an alternative approach is necessary because O(12^n) is not computationally viable. This problem in particular has a very large search space, so pursuing an alternative method seems to be a reasonable direction for this project.
### Goals
* Design algorithm to solve for any gameboard in polynomial time.
* Model the problem using NN architecture and solve a gamestate using machine learning.
