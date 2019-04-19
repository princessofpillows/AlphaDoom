import numpy as np
import networkx as nx


class MCTS(object):

    class Node(object):
        # s is state, a is action on edge, p is prior probability of selecting, n is number of visits, q is mean value, w is total value
        def __init__(self, parent=None, s=None, a=None, p=0, Cpuct=0.99):
            self.parent = parent
            self.children = []
            self.Cpuct = Cpuct
            self.s = s
            self.a = a
            self.p = p
            self.n = 0
            self.w = 0
            self.q = 0
        
        def add_child(self, child):
            self.children.append(child)
        
        # v is valuation from network in set [-1, 1]
        def update(self, v):
            self.w = self.w + v
            self.q = self.w / self.n
        
        # Add visit to node
        def increment(self):
            self.n += 1
        
        # V(s, a) = Q(s, a) + U(s, a)
        def value(self, sum_n):
            if self.n > 0:
                u = self.Cpuct * self.p * (np.sqrt(sum_n) / (1 + self.n))
            else:
                u = 0
            return self.q + u

    # Create root of tree
    def __init__(self, cfg):
        self.actions = cfg.actions
        self.eps = cfg.eps
        self.T = cfg.T
        self.Cpuct = cfg.Cpuct
        self.d_noise = cfg.d_noise

        self.root = MCTS.Node()
        # Add children
        for i in range(len(self.actions)):
            self.expand(self.root, self.actions[i])
    
    # Add edge (action) and empty child node (state) from parent
    def expand(self, parent, a, p=0):
        child = MCTS.Node(parent, None, a, p, self.Cpuct)
        parent.add_child(child)

    # Add new edges to tree, backprop through current path
    def update(self, curr, v, p_set):
        # Add dirichlet noise
        noise = np.random.dirichlet(self.d_noise * np.ones(len(self.actions)))
        p_set = p_set.numpy()[0]
        # Adds edge and empty node for each action
        for i in range(len(self.actions)):
            p = (1 - self.eps) * p_set[i] + self.eps * noise[i]
            self.expand(curr, self.actions[i], p)

        # Backpropogation
        while curr.parent != None:
            curr.increment()
            curr.update(v.numpy()[0])
            curr = curr.parent

    # Selects best path from current state
    def select(self, parent):
        sum_n = 0
        for child in parent.children:
            sum_n += child.n

        v = []
        # Gets value for each child of parent (N children == N actions)
        for child in parent.children:
            v.append(child.value(sum_n))

        v = np.asarray(v)
        # Gets index of best child node
        v_max = np.where(v == np.max(v))[0]
        idx = v_max[0]
        # Random select if tie
        if len(v_max) > 1:
            idx = np.random.choice(v_max)

        # Return best node
        return parent.children[idx]
    
    # Finds leaf node while taking best path
    def search(self):
        curr = self.root
        while(curr.children != []):
            curr = self.select(curr)
        
        return curr
    
    # From current root node, iterate through children and select one based on exploration heuristic and visit count
    def select_action(self):
        best = 0
        selection = None
        # Exploration based expansion; when T is 0, becomes deterministic instead of stochastic
        for child in self.root.children:
            value = np.power(child.n, 1 / self.T) / np.power(self.root.n, 1 / self.T)
            if value > best:
                best = value
                selection = child

        # Update root
        self.root = selection
        pi = []
        # Compute visit probs of each child of root
        for child in self.root.children:
            pi.append(np.power(child.n, 1 / self.T))

        return selection.a, pi
    
    # Game tree visualization
    def visualize(self):

        def iterate_children(self, G, parent):
            # Recursively add children to graph
            for child in parent.children:
                G.add_node([child.w, child.n])
                G.add_edge(parent.w, child.w, object=child.p)
                G = iterate_children(G, child)
            return G

        tree = nx.Graph()
        tree.add_node(self.root.w)
        tree = iterate_children(tree, self.root)
        nx.draw(tree, with_labels=True)
        return tree
