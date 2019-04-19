import tensorflow as tf
import numpy as np
import pickle, os
from pathlib import Path
from tqdm import trange
from alphadoom_cfg import get_cfg
from classes.mcts import MCTS
from classes.replay import Replay
from classes.models import AutoEncoder
from utils.vizdoom_api import VizDoom
from utils.writer import Writer


# Remove logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

tf.enable_eager_execution()
cfg = get_cfg()

class AlphaDoom(object):

    def __init__(self):
        super(AlphaDoom, self).__init__()

        self.vizdoom = VizDoom(cfg)
        self.mcts = MCTS(cfg)
        self.replay = Replay(cfg)
        self.autoencoder = AutoEncoder(cfg)
        tf.train.Checkpoint(model=self.autoencoder).restore(tf.train.latest_checkpoint('./simulator_saves/best'))

        # Load selected model
        self.model = cfg.model(cfg)
        self.loss1 = cfg.loss1
        self.loss2 = cfg.loss2
        self.optim = cfg.optim(cfg.learning_rate)
        self.epoch = tf.Variable(0)

        self.writer = Writer(cfg)
        # Restore if save exists
        self.model, self.optim, self.epoch = self.writer.restore(self.model, self.optim, self.epoch)

    def update(self):
        # Fetch batch of experiences
        s0, pi, z, s1 = self.replay.fetch()
        z = np.array(z).reshape((len(z), 1))
        pi = np.array(pi)
        # Construct graph
        with tf.GradientTape() as tape:
            p, v = self.model(s0)
            loss = tf.reduce_mean(self.loss1(z, v) - self.loss2(tf.pow(pi, cfg.T), p)) # + cfg.c * tf.nn.l2_loss(self.model.weights))
        
        self.writer.log(self.optim, tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optim.apply_gradients(grads_and_vars)

        self.writer.global_step.assign_add(1)
    
    # Runs N simulations, where each sim reaches a leaf node in MCTS tree
    def simulate(self):
        for i in range(cfg.num_sims):
            # Find leaf
            leaf = self.mcts.search()
            # Simulate leaf's state
            action = np.reshape(leaf.a, [1, 1, len(cfg.actions)]).astype(np.float32)
            leaf.s = self.autoencoder.predict(leaf.parent.s, action)
            # Get p, the prior probability set of all actions (edges) from current leaf node, and v, the value of current leaf node
            p, v = self.model(leaf.s)
            # Backprop through MCTS tree
            self.mcts.update(leaf, v, p)

    # Returns best action
    def perform_action(self, s0):
        # Shape (H, W, 1) to (1, H, W, 1)
        self.mcts.root.s = s0[None]
        self.simulate()
        action, pi = self.mcts.select_action()
        # Take action
        reward = self.vizdoom.make_action(action)
        # Reward of -1 if bad, +1 if good
        if reward > 0:
            reward = 1
        else:
            reward = -1
        return pi, reward
    
    def train(self):
        if Path('/replay.pkl').is_file():
            with open('/replay.pkl', 'rb') as f:
                self.replay.memory = pickle.load(f)

        for epoch in trange(self.epoch.numpy(), cfg.epochs):
            self.vizdoom.new_episode()
            frame = self.vizdoom.get_preprocessed_state()
            frames = []
            # Init stack of n frames
            for i in range(cfg.num_frames):
                frames.append(frame)
            
            while not self.vizdoom.is_episode_finished():
                s0 = tf.concat(frames, axis=-1)
                pi, z = self.perform_action(s0)

                # Update frames with latest image
                frames.pop(0)
                frames.append(self.vizdoom.get_preprocessed_state())

                s1 = tf.concat(frames, axis=-1)
                self.replay.push([s0, pi, z, s1])
                
            # Train on experiences from memory
            self.update()

            # Save model
            if epoch % cfg.save_freq == 0:
                self.writer.save(self.model, self.optim, self.epoch)
                with open('./replay.pkl', 'wb') as f:
                    pickle.dump(self.replay.memory, f)
        
        self.writer.save(self.model, self.optim, self.epoch)

def main():
    model = AlphaDoom()
    model.train()

if __name__ == "__main__":
    main()
    #cProfile.run('main(cfg)', 'prof')
    #p = pstats.Stats('prof')
    #p.strip_dirs().sort_stats('cumulative').print_stats(50)
