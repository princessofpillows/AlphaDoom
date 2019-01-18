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
        self.curr = self.root

    def select(self, parent):
        v = []
        for child in parent.children:
            v.append(child.value())

        v = np.asarray(v)
        v_max = np.where(v == np.max(v))
        if len(v_max) > 0:
            return np.random.choice(v_max)
        return v_max

    def expand(self, parent, p):
        child = MCTS.Node(parent, None, p)
        parent.add_child(child)
    
    def search(self):
        # Select
        while(self.curr.children != []):
            self.curr = select(self.curr)
        
        return self.curr
    
    def update(self, s, v, p_set):
        # Expand and Evaluate
        self.curr.s = s
        for p in p_set:
            # Add Dirichlet noise
            p = (1 - cfg.eps) * p + cfg.eps * np.random.dirichlet(cfg.d_noise)
            expand(self.curr, p)

        # Backpropogation
        while self.curr.parent != None:
            self.curr.increment()
            self.curr.update(v)
            self.curr = self.curr.parent
        
        # Play
        best_s = 0
        most_n = 0
        selection = None
        # Deterministic selection
        if cfg.eval:
            for child in self.curr.children:
                if child.n > most_n:
                    most_n = child.n
                    selection = child
        # Stochastic selection
        else:
            for child in self.curr.children:
                s = np.power(child.n, 1 / cfg.T) / np.power(self.root.n, 1 / cfg.T)
                if s > best_s:
                    best_s = s
                    selection = child

        self.curr = selection


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

        self.game = vzd.DoomGame()
        self.game.load_config('vizdoom/scenarios/basic.cfg')
        self.game.init()

        self.global_step = tf.train.get_or_create_global_step()
        self.terminal = tf.zeros([84, 84, 1])

        # Assign CPU due to GPU memory limitations
        with tf.device('CPU:0'):
            self.mcts = MCTS()
            self.replay_memory = replay_memory()

        # Load selected model
        self.model = cfg.models[cfg.model](cfg, len(cfg.actions))
        self.model.build((None,) + self.model.shape + (4,))
        self.loss = cfg.losses[cfg.loss]
        self.optimizer = cfg.optims[cfg.optim](cfg.learning_rate)

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
        self.saver = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, optimizer_step=self.global_step)

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
        with tf.device('CPU:0'):
            # Fetch batch of experiences
            s, p, z = self.replay_memory.fetch()
        
        # Construct graph
        with tf.GradientTape() as tape:
            loss = 1/2 * tf.reduce_mean(tf.losses.huber_loss(rewards + cfg.discount * target_q, q)) - entropy * cfg.entropy_rate
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)

    def simulation(self):
        curr = self.mcts.search()
        s, v, p_set = self.model(curr)
        self.mcts.update(s, v, p_set)

    def perform_action(self, frames):
        p = self.model(frames)
        # One-hot action
        choice = np.zeros(len(cfg.actions))
        choice[tuple(cols)] += 1
        # Take action
        z = self.game.make_action(choice, cfg.skiprate)
        return z
    
    def preprocess(self):
        screen = self.game.get_state().screen_buffer
        frame = np.multiply(screen, 255.0/screen.max())
        frame = tf.image.rgb_to_grayscale(frame)
        frame = tf.image.resize_images(frame, self.model.shape)
        return frame
    
    def train(self):
        self.saver.restore(tf.train.latest_checkpoint(cfg.save_dir))
        for episode in trange(cfg.episodes):
            # Save model
            if episode % cfg.save_freq == 0:
                self.saver.save(file_prefix=self.ckpt_prefix)

            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            # Init stack of 4 frames
            frames = [frame, frame, frame, frame]

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], cfg.num_frames])
                z = self.perform_action(s)

                # Update frames with latest image
                prev_frames = frames[:]
                frames.pop(0)
                # Reached terminal state, kill gradient
                if self.game.get_state() is None:
                    frames = [self.terminal, self.terminal, self.terminal, self.terminal]
                    logits = tf.zeros(logits.shape)
                else:
                    frames.append(self.preprocess())

                self.replay_memory.push([s, p, z])
                
            # Train on experiences from memory
            self.update()
        
        self.saver.save(file_prefix=self.ckpt_prefix)

    def test(self):
        self.saver.restore(tf.train.latest_checkpoint(cfg.save_dir))
        rewards = []
        for _ in trange(cfg.test_episodes):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            # Init stack of 4 frames
            frames = [frame, frame, frame, frame]

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], cfg.num_frames])
                _ = self.perform_action(s)
                
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
