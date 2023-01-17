# Projet IA E3 Adrien PELFRESNE - Alexis VAPAILLE

## MonteCarlo Algorithms
In this first section, we wil introduce two differents monte carlo algorithms,

### Classic monte carlo 

### MCTS (Monte carlo tree search)
the Upper confidence Trees (UCT) formula:  
$$
S_i = x_i + c \sqrt{\frac{ln(n)}{n_i}}
$$
**Where**
- $S_i$ is the value of the node $i$
- $x_i$ is the mean of all the game passing through this node.
In our case
- $n_i$ is the number of simulation that passed through this node
- $c$ is a a coefficient that allows us to adjust the balance between exploration and exploitation 
- $n$ number of simulation passed by the parent node $x_i$  

the Monte carlo tree search algorithm can be broken down into 4 steps:

