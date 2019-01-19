import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, os
from datetime import datetime
from pathlib import Path
from tqdm import trange
from config import get_config
from models import AlphaGoZero


tf.enable_eager_execution()
cfg = get_config()

class MCTS(object):

    class Node(object):
        # p is prior probability of selecting, n is number of visits, q is mean value, w is total value
        def __init__(self, parent=None, s=None, p=0):
            self.parent = parent
            self.children = []
            self.s = s
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
            
        def increment(self):
            self.n += 1
        
        # V(s, a) = Q(s, a) + U(s, a)
        def value(self):
            return self.q + cfg.Cpuct * self.p * (np.sqrt(self.parent.n) / (1 + self.n))

    def __init__(self):
        self.root = MCTS.Node()
        self.curr_root = self.root

    def select(self, parent):
        v = []
        for child in parent.children:
            v.append(child.value())

        v = np.asarray(v)
        v_max = np.where(v == np.max(v))
        if len(v_max) > 1:
            return np.random.choice(v_max)
        return v_max[0]

    def expand(self, parent, p):
        child = MCTS.Node(parent, None, p)
        parent.add_child(child)
    
    def search(self):
        # Find leaf
        curr = self.curr_root
        while(curr.children != []):
            curr = select(curr)
        
        return curr
    
    def update(self, leaf, s, v, p_set):
        # Expand node and evaluate
        for p in p_set:
            # Add dirichlet noise
            p = (1 - cfg.eps) * p + cfg.eps * np.random.dirichlet(cfg.d_noise)
            expand(leaf, p)

        # Backpropogation
        curr = leaf
        while curr.parent != None:
            curr.increment()
            curr.update(v)
            curr = curr.parent
        
    def select(self):
        # Play
        best_s = 0
        selection = None
        # When T is 0, becomes deterministic instead of stochastic
        for child in self.curr_root.children:
            s = np.pow(child.n, 1 / cfg.T) / np.pow(self.root.n, 1 / cfg.T)
            if s > best_s:
                best_s = s
                selection = child

        self.curr_root = selection
        return selection, np.pow(selection.n, 1 / cfg.T)


class replay_memory(object):

    def __init__(self):
        self.memory = []

    def push(self, exp):
        size = len(self.memory)
        # Remove oldest memory first
        if size == cfg.cap:
            self.memory.pop(random.randint(0, size-1))
        self.memory.append(exp)
    
    def fetch(self):
        size = len(self.memory)
        # Select batch
        if size < cfg.batch_size:
            batch = random.sample(self.memory, size)
        else:
            batch = random.sample(self.memory, cfg.batch_size)
        # Return batch
        batch = np.asarray(batch, dtype=object)
        return zip(*batch)


class AlphaDoom(object):

    def __init__(self):
        super(AlphaDoom, self).__init__()

        # Create game instance
        self.game = vzd.DoomGame()
        self.game.load_config('vizdoom/scenarios/basic.cfg')
        self.game.init()

        self.global_step = tf.train.get_or_create_global_step()
        self.terminal = tf.zeros([84, 84, 1])

        # Assign mcts, memory to CPU due to GPU memory limitations
        with tf.device('CPU:0'):
            self.mcts = MCTS()
            self.replay_memory = replay_memory()

        # Load selected model
        self.model = cfg.models[cfg.model](cfg, len(cfg.actions))
        self.model.build((None,) + self.model.shape + (cfg.num_frames,))
        self.loss1 = cfg.losses[cfg.loss1]
        self.loss2 = cfg.losses[cfg.loss2]
        self.learning_rate = cfg.lr_schedule[str(cfg.learning_rate)]
        self.optimizer = cfg.optims[cfg.optim](self.learning_rate)

        self.build_writers()

    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)
        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = cfg.log_dir + cfg.extension
        self.writer = tf.contrib.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, optimizer_step=self.global_step)

    def logger(self, tape, loss):
        with tf.contrib.summary.record_summaries_every_n_global_steps(cfg.log_freq, self.global_step):
            # Log vars
            tf.contrib.summary.scalar('loss', loss)

            # Log weights
            slots = self.optimizer.get_slot_names()
            for variable in tape.watched_variables():
                    tf.contrib.summary.scalar(variable.name, tf.nn.l2_loss(variable))

                    for slot in slots:
                        slotvar = self.optimizer.get_slot(variable, slot)
                        if slotvar is not None:
                            tf.contrib.summary.scalar(variable.name + '/' + slot, tf.nn.l2_loss(slotvar))

    def update(self):
        # Fetch batch of experiences
        s, pi, z = self.replay_memory.fetch()
        
        # Construct graph
        with tf.GradientTape() as tape:
            p, v = self.model(s)
            loss = tf.reduce_mean(self.loss1(z, v) - self.loss2(tf.pow(pi, cfg.T), p) + cfg.c * tf.nn.l2_loss(self.model.weights))
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)
    
    def simulation(self, s):
        for i in range(cfg.num_sims):
            leaf = self.mcts.search()
            # TODO ********* need some way of acquiring s for a simulated sequence of actions *********
            leaf.s = s
            p, v = self.model(s)
            self.mcts.update(leaf, v, p_set)

    def perform_action(self, s):
        simulation(s)
        selection, pi = self.mcts.select()
        # One-hot action
        choice = np.zeros(len(cfg.actions))
        choice[tuple(a)] += 1
        # Take action
        reward = self.game.make_action(choice, cfg.skiprate)
        if reward > 0:
            return selection.s, pi, 1
        else:
            return selection.s, pi, -1
    
    def preprocess(self):
        screen = self.game.get_state().screen_buffer
        frame = np.multiply(screen, 255.0/screen.max())
        frame = tf.image.rgb_to_grayscale(frame)
        frame = tf.image.resize_images(frame, self.model.shape)
        return frame
    
    def evaluate(self):
        old_T = cfg.T
        cfg.T = 0
        p1_wins = 0
        p2_wins = 0
        for game in range(cfg.num_eval):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            frames = []
            # Init stack of 4 frames
            for i in range(cfg.num_frames):
                frames.append(frame)
            
            p1_frames = frames
            p2_frames = frames
            
            # Alternate between players
            while True:
                # P1
                curr = tf.zeros([84, 84, 1])
                s = tf.reshape(p1_frames, [1, self.model.shape[0], self.model.shape[1], cfg.num_frames, curr])
                _, _, _ = self.perform_action(s)

                if self.game.is_episode_finished():
                    p1_wins += 1
                    break

                p1_frames.pop(0)
                # TODO ******** Add support for multiple players in preprocess *********
                p1_frames.append(self.preprocess())

                # P2
                curr = tf.ones([84, 84, 1])
                s = tf.reshape(p2_frames, [1, self.model.shape[0], self.model.shape[1], cfg.num_frames, curr])
                _, _, _ = self.perform_action(s)

                if self.game.is_episode_finished():
                    p2_wins += 1
                    break

                p2_frames.pop(0)
                p2_frames.append(self.preprocess())
        
        cfg.T = old_T
        if (p1_wins / p2_wins) > 0.55 :
            return True
        else:
            return False
    
    def train(self):
        self.saver.restore(tf.train.latest_checkpoint(cfg.save_dir))
        for episode in trange(cfg.episodes):
            # Update learning rate at scheduled times
            if episode % cfg.lr_schedule[str(cfg.learning_rate)][1] == 0:
                cfg.learning_rate += 1
                self.learning_rate = cfg.lr_schedule[str(cfg.learning_rate)][0]

            # Save model
            if episode % cfg.save_freq == 0:
                # Check if new model is improvement over current best
                if self.evaluate():
                    self.saver.save(file_prefix=self.ckpt_prefix)

            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            frames = []
            # Init stack of 4 frames
            for i in range(cfg.num_frames):
                frames.append(frame)

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], cfg.num_frames])
                s, pi, z = self.perform_action(s)

                # Update frames with latest image
                if self.game.get_state() is not None:
                    frames.pop(0)
                    frames.append(self.preprocess())

                self.replay_memory.push([s, pi, z])
                
            # Train on experiences from memory
            self.update()
        
        self.saver.save(file_prefix=self.ckpt_prefix)

    def test(self):
        rewards = []
        self.saver.restore(tf.train.latest_checkpoint(cfg.save_dir))
        for _ in trange(cfg.test_episodes):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            frames = []
            # Init stack of 4 frames
            for i in cfg.num_frames:
                frames.append(frame)

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], cfg.num_frames])
                z = self.perform_action(a)
                
                # Update frames with latest image
                if self.game.get_state() is not None:
                    frames.pop(0)
                    frames.append(self.preprocess())

            rewards.append(self.game.get_total_reward())
        print("Average Reward: ", sum(rewards)/cfg.test_episodes)

def main():
    model = AlphaDoom()

if __name__ == "__main__":
    main()
    #cProfile.run('main(cfg)', 'prof')
    #p = pstats.Stats('prof')
    #p.strip_dirs().sort_stats('cumulative').print_stats(50)
