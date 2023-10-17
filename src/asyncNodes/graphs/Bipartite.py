import math

from decentralizepy.graphs.Graph import Graph


class Bipartite(Graph):
    """
    The class for generating a Bipartite Graph Topology with passive (even rank) and active (odd rank) nodes.
    Used in ADPSGD.

    """

    def __init__(self, n_procs, exponential=False):
        """
        Constructor. Generates a Bipartite graph with passive and active nodes

        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph
        
        exponential : bool
            defines if the graph is exponential or not

        """
        super().__init__(n_procs)


        if exponential:

            for node in range(n_procs):

                for i in range(0, int(math.log(n_procs - 1, 2)) + 1):
                    if i == 0:
                        f_peer = self._rotate_forward(node, 1)
                        b_peer = self._rotate_backward(node, 1)
                    else:
                        f_peer = self._rotate_forward(node, 1 + 2 ** i)
                        b_peer = self._rotate_backward(node, 1 + 2 ** i)

                    if (not self.is_passive(node) and (self.is_passive(f_peer) and self.is_passive(b_peer))) or (
                        self.is_passive(node) and not (self.is_passive(f_peer) or self.is_passive(b_peer))):

                        self.adj_list[node].add(f_peer)
                        self.adj_list[node].add(b_peer)
        else:

            for node in range(n_procs):

                for i in range(1, n_procs):
                    f_peer = self._rotate_forward(node, i)
                    b_peer = self._rotate_backward(node, i)

                    if (not self.is_passive(node) and self.is_passive(f_peer)) or (
                        self.is_passive(node) and not self.is_passive(f_peer)):

                        self.adj_list[node].add(f_peer)

                    if (not self.is_passive(node) and self.is_passive(b_peer)) or (
                        self.is_passive(node) and not self.is_passive(b_peer)):

                        self.adj_list[node].add(b_peer)



    def _rotate_forward(self, rank, hops):
        return (rank + hops) % self.n_procs
    
    def _rotate_backward(self, rank, hops):
        tmp = rank - hops
        while(tmp<0):
            tmp += self.n_procs

        return tmp


    def is_passive(self, rank):
        """
        Returns True if the node is passive, False otherwise
        """        

        return (rank % 2) == 0
