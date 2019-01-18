import math

import numpy as np

from six.moves import xrange


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        # Maximum reward of child, see http://www.cs.du.edu/~sturtevant/papers/im-mcts.pdf
        self.Vsa = {}
        self.Ns = {}        # stores #times board s was visited
        self.Qn = {}        # stores Q values for number of samples
        self.Vn = {}        # stores V values for number of samples
        self.Nn = {}        # stores #times a sample was draw with N samples
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Valids = {}        # stores game.getValidMoves for board s
        # self.nodes = 0

        self.best_v = -np.inf
        self.best_board = None

    def getActionProb(self, canonicalBoard, idx_eps_th, temp, numMCTSSims,
                      avg_pi):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # If num search is zero, simply do policy
        if numMCTSSims == 0:
            probs, _ = self.nnet.predict(canonicalBoard, idx_eps_th, avg_pi)
            self.Valids[s] = self.game.getValidMoves(canonicalBoard)
            probs *= self.Valids[s]
            # Check if all are masked
            if not (probs.sum() > 0):
                # Simply set a uniform random probability
                probs = self.Valids[s].astype(np.float)
            # Normalize to do probabilistic selection
            probs /= probs.sum()
            return probs

        for i in xrange(self.args.numMCTSSims):
            self.search(canonicalBoard, idx_eps_th, avg_pi)

        counts = np.zeros(self.game.getActionSize())
        for a in xrange(len(counts)):
            if self.Valids[s][a]:
                next_s, next_player = self.game.getNextState(
                    canonicalBoard, 1, a)
                next_s = self.game.getCanonicalForm(next_s, next_player)
                next_s = self.game.stringRepresentation(next_s)
                if next_s in self.Ns:
                    counts[a] = self.Ns[next_s]

        if temp == 0:
            assert counts.sum() > 0
            # find the maximum count
            max_count = counts.max()
            probs = (counts == max_count).astype(np.float)
            # Again mask with valid move
            probs *= self.Valids[s]
            # Normalize to do probabilistic selection
            probs /= probs.sum()
            return probs

        # print('Ps',self.Ps)
        counts = [x**(1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, canonicalBoard, idx_eps_th, avg_pi):
        """This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Wsa,
        Qsa are updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other
        player.

        alpha is the dirclet prior, eps is the amount we are going to mix

        Returns:
            v: the negative of the value of the current canonicalBoard

        """
        # print("size of canonical board", canonicalBoard.shape)
        s = self.game.stringRepresentation(canonicalBoard)
        # print('canonical board',s)
        n = np.sum(canonicalBoard)

        # --------------------------------------------------
        # 1. If a new leaf node
        is_new_leaf = s not in self.Es
        if is_new_leaf:
            # 1.1 Get the game state and check if we've reached the end of the
            # game.  Since we don't want to slow down and it does not make
            # sense to have this step in actual testing scenario, we don'e
            # reconstruct, and we just use the stored value of this node, which
            # would be the V value from the network.
            bUseActualReward = True
            res = self.game.getGameEnded(
                canonicalBoard, idx_eps_th, recon=bUseActualReward)
            self.Es[s] = res["ended"]
            # Also store the reward. This reward can be None
            v = res["reward"]

            # 1.2. If our v is None, or if we are not at the end of the game,
            # compute policy and value. We consider v == None case, since we
            # may opt to not return the actual reward. Otherwise, i.e., if we
            # are at the end of the node, and we do have a reward, we replace
            # Ps and Valids to zeros, and keep using the v from the reward.
            if v is None or not self.Es[s]:

                # If s is not in self.Ps, we reached a leaf node. Note that
                # this coudl also mean that we reached the end state.
                self.Ps[s], v = self.nnet.predict(
                    canonicalBoard, idx_eps_th, avg_pi)
                # mask invalid moves
                self.Valids[s] = self.game.getValidMoves(canonicalBoard)
                self.Ps[s] = self.Ps[s] * self.Valids[s]
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s    # renormalize
                else:
                    # if all valid moves were masked make all valid moves
                    # equally probable.
                    print("All valid moves were masked, do workaround.")
                    self.Ps[s] = self.Ps[s] + self.Valids[s]
                    self.Ps[s] /= np.sum(self.Ps[s])

                # Simulate actual play -- Note that this will visit nodes and
                # save the actual state values!
                if self.args.mcts_actual_v:
                    # Select the best action using np.choice
                    a = np.random.choice(len(self.Ps[s]), p=self.Ps[s])
                    # Get the value from the subtree
                    next_s, next_player = self.game.getNextState(
                        canonicalBoard, 1, a)
                    next_s = self.game.getCanonicalForm(next_s, next_player)
                    # Recurse down
                    v = self.search(next_s, idx_eps_th, avg_pi)

            else:
                # Put both to zeros to cause crashes as sanity check
                self.Ps[s] = np.zeros(self.game.getActionSize())
                self.Valids[s] = np.zeros(self.game.getActionSize())

            # 1.3. Now, also store our visits, and return. At this point, we
            # should not have v = None.
            assert v is not None
            self.Ns[s] = 1
            self.Qsa[s] = v
            self.Vsa[s] = v

            # If we are at the end state, remember the best value
            if self.Es[s]:
                if v > self.best_v:
                    self.best_v = v
                    self.best_board = np.copy(canonicalBoard)

            # 1.4 Also, Update the Q, V, and N state for the number of samples
            # that we did.
            if n not in self.Qn:
                self.Nn[n] = 1
                self.Qn[n] = v
                self.Vn[n] = v
            else:
                self.Qn[n] = (self.Qn[n] * self.Nn[n] + v) / (self.Nn[n] + 1.0)
                self.Vn[n] = max(self.Vn[n], v)
                self.Nn[n] += 1

            return v

        # --------------------------------------------------
        # 2. If an end node. This is not a newly generated leaf node, but we
        # have arrived at the end, where we do not have to do more searches. We
        # simply update our visit count and return. We use stored results for
        # v.
        if self.Es[s]:
            self.Ns[s] += 1
            self.Nn[n] += 1
            v = self.Qsa[s]     # Q, V are both fine
            return v            # negate if needed

        # --------------------------------------------------
        # 3. All other cases. Continue on the search down using recursion,
        # after selecting the child with hichest UCB

        # 3.1. Compute UCB for all items
        # Start with - inf to ignore that place
        na = self.game.getActionSize()
        # Apply dirichlet noise
        noise = np.random.dirichlet([self.args.mcts_alpha] * na)
        noise *= self.Valids[s]
        noise /= noise.sum()
        Ps = (
            (1.0 - self.args.mcts_eps) * self.Ps[s] +
            self.args.mcts_eps * noise
        )
        us = -np.inf * np.ones(na)
        for a in xrange(self.game.getActionSize()):
            if self.Valids[s][a]:
                # Use only the state for book-keeping!
                next_s, next_player = self.game.getNextState(
                    canonicalBoard, 1, a)
                next_s = self.game.getCanonicalForm(next_s, next_player)
                next_s = self.game.stringRepresentation(next_s)
                if next_s in self.Qsa:
                    us[a] = (
                        (1.0 - self.args.mcts_im_alpha) * self.Qsa[next_s] +
                        self.args.mcts_im_alpha * self.Vsa[next_s] +
                        self.args.cpuct * Ps[a] *
                        math.sqrt(self.Ns[s]) / (1.0 + self.Ns[next_s])
                    )
                else:
                    if self.args.mcts_heuristic == "prev":
                        Q_heuristic = self.Qsa[s]
                    elif self.args.mcts_heuristic == "nsamp":
                        # Visit might not have happened! In that case, we
                        # simply follow the policy, because we don't have
                        # anything for any of the next actions. We have no diea
                        # what comes next
                        if n + 1 in self.Qn:
                            Q_heuristic = self.Qn[n + 1]
                        else:
                            Q_heuristic = 0

                    us[a] = (
                        (1.0 - self.args.mcts_im_alpha) * Q_heuristic +
                        self.args.mcts_im_alpha * self.Vsa[s] +
                        self.args.cpuct * Ps[a] *
                        math.sqrt(self.Ns[s]) / (1.0)
                    )

        # 3.2. Find the next state that gives the highest UCB
        u_best = us.max()
        us = (us == u_best).astype(np.float)
        # Again mask with valid move
        us *= self.Valids[s]
        if sum(np.isnan(us)):
            print("nan in ucb!")
            import IPython
            IPython.embed()
        # Normalize to do probabilistic selection
        us /= us.sum()
        # Select the best action using np.choice
        a = np.random.choice(len(us), p=us)
        # Get the value from the subtree
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # 3.3. Recurse down
        v = self.search(next_s, idx_eps_th, avg_pi)

        # 3.4. Update self with updated statistics
        self.Qsa[s] = (self.Qsa[s] * self.Ns[s] + v) / (self.Ns[s] + 1.0)
        self.Vsa[s] = max(self.Vsa[s], v)
        self.Ns[s] += 1
        self.Qn[n] = (self.Qn[n] * self.Nn[n] + v) / (self.Nn[n] + 1.0)
        self.Vn[n] = max(self.Vn[n], v)
        self.Nn[n] += 1

        # 3.5 Keep propagating up
        return v

    def getBestLeaf(self):

        return np.copy(self.best_board)
