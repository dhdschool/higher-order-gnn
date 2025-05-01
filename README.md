# Topological Neural Network for Social Recommendation
Based on [this paper](https://arxiv.org/abs/1902.07243) by Fan et al.

The dataset used by this paper can be found [here](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm).

We propose multiple potential architectures to port the graph-based methods of the original paper into a topological framework. The first of these involves a transformer-like vector embedding of item vertice and edge data, as well as defining our combinitorical cells as follows:
- 0-cell: Person vertices
- 1-cell: Person-Person edges
- 2-cell: Person cliques of size $i$, $\forall i \in \{3, 4, ..., n\}\ \exists n \geq 3$

Where a clique is defined as a set of vertices that have mutual edges between all vertices