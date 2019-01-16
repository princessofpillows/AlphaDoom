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
        
        # N is the summation of all visits from root to curr node
        # V(s, a) = Q(s, a) + U(s, a)
        def value(self, N):
            return self.q + cfg.Cpuct * self.p * (np.sqrt(N + self.n) / (1 + self.n))

    def __init__(self):
        self.root = Node()
        self.alphadoom = alphadoom()

    def select(self, parent, N):
        max_v = 0
        for child in parent.children:
            v = child.value(N)
            if v > max_v:
                max_v = v
                selected = child

        return selected

    def expand(self, parent, p):
        child = Node(parent, None, p)
        parent.add_child(child)
    
    def search(self):
        # Select
        N = 0
        curr = self.root
        while(curr.children != [])
            curr = select(curr, N)
            N += curr.n
        
        # Expand and Evaluate
        s, v, p_set = self.alphadoom.eval(curr)
        curr.s = s
        for p in p_set:
            expand(curr, p)

        # Backpropogation
        while curr.parent != None:
            curr.increment()
            curr.update(v)
            curr = curr.parent
        
        # Play
        best_s = 0
        for child in self.root:
            s = np.power(child.n, 1 / cfg.T) / np.power(child.n + self.root.n, 1 / cfg.T)
            if s > best_s:
                best_s = s

        self.root = best_s
        self.alphadoom.move(best_s)


class alphadoom(object):

    def __init__(self):
        super(alphadoom, self).__init__()

        self.game = vzd.DoomGame()
        self.game.load_config('vizdoom/scenarios/basic.cfg')
        self.game.init()

        self.global_step = tf.train.get_or_create_global_step()
        self.terminal = tf.zeros([84, 84, 1])

        # Create network
        self.model = AlphaGoZero(self.cfg, len(cfg.actions))
        self.model.build((None,) + self.model.shape + (4,))
        self.optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)

        self.build_writers()

    def build_writers(self):
        if not Path(self.cfg.save_dir).is_dir():
            os.mkdir(self.cfg.save_dir)
        if self.cfg.extension is None:
            self.cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = self.cfg.log_dir + self.cfg.extension
        self.writer = tf.contrib.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()

        self.save_path = self.cfg.save_dir + self.cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, optimizer_step=self.global_step)

    def logger(self, tape, loss):
        with tf.contrib.summary.record_summaries_every_n_global_steps(self.cfg.log_freq, self.global_step):
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
            s0, logits, rewards, s1 = self.replay_memory.fetch()
        
        # Get entropy
        probs = tf.nn.softmax(logits)
        entropy = -1 * tf.reduce_sum(probs*tf.math.log(probs + 1e-20))
        # Construct graph
        with tf.GradientTape() as tape:
            # Get predicted q values
            logits, _ = self.model(s0)
            # Choose max q values for all batches
            rows = tf.range(tf.shape(logits)[0])
            cols = tf.argmax(logits, 1, output_type=tf.int32)
            rows_cols = tf.stack([rows, cols], axis=1)
            q = tf.gather_nd(logits, rows_cols)

            # Get target q values
            target_logits, _ = self.target(s1)
            rows = tf.range(tf.shape(target_logits)[0])
            # Using columns of selected actions from prediction, stack with target rows
            rows_cols = tf.stack([rows, cols], axis=1)
            # Slice Q values, with actions chosen by prediction
            target_q = tf.gather_nd(logits, rows_cols)
            # Kill target network gradient
            target_q = tf.stop_gradient(target_q)

            # Compare target q and predicted q (q = 0 on terminal state)
            loss = 1/2 * tf.reduce_mean(tf.losses.huber_loss(rewards + self.cfg.discount * target_q, q)) - entropy * self.cfg.entropy_rate
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)

    def perform_action(self, frames):
        logits, _ = self.model(frames)
        # One-hot action
        choice = np.zeros(len(self.cfg.actions))
        choice[tuple(cols)] += 1
        # Take action
        z = self.game.make_action(choice, self.cfg.skiprate)
        # Modify rewards (game is blackbox)
        '''
        if reward > 50:
            reward = 10
        elif reward < -6:
            reward = -3
        else:
            reward = -1
        '''
        return z
    
    def preprocess(self):
        screen = self.game.get_state().screen_buffer
        frame = np.multiply(screen, 255.0/screen.max())
        frame = tf.image.rgb_to_grayscale(frame)
        frame = tf.image.resize_images(frame, self.model.shape)
        return frame
    
    def train(self):
        self.saver.restore(tf.train.latest_checkpoint(self.cfg.save_dir))
        for episode in trange(self.cfg.episodes):
            # Save model
            if episode % self.cfg.save_freq == 0:
                self.saver.save(file_prefix=self.ckpt_prefix)

            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            # Init stack of 4 frames
            frames = [frame, frame, frame, frame]

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], self.cfg.num_frames])
                z = self.perform_action(s)

                # Update frames with latest image
                prev_frames = frames[:]
                frames.pop(0)
                # Reached terminal state, kill q values
                if self.game.get_state() is None:
                    frames = [self.terminal, self.terminal, self.terminal, self.terminal]
                    logits = tf.zeros(logits.shape)
                else:
                    frames.append(self.preprocess())
                
            # Train on experiences from memory
            self.update()
        
        self.saver.save(file_prefix=self.ckpt_prefix)

    def test(self):
        self.saver.restore(tf.train.latest_checkpoint(self.cfg.save_dir))
        rewards = []
        for _ in trange(self.cfg.test_episodes):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            # Init stack of 4 frames
            frames = [frame, frame, frame, frame]

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], self.cfg.num_frames])
                _ = self.perform_action(s)
                
                # Update frames with latest image
                if self.game.get_state() is not None:
                    frames.pop(0)
                    frames.append(self.preprocess())

            rewards.append(self.game.get_total_reward())
        print("Average Reward: ", sum(rewards)/self.cfg.test_episodes)

def main(cfg):
    mcts = MCTS()

if __name__ == "__main__":
    main()
    #cProfile.run('main(cfg)', 'prof')
    #p = pstats.Stats('prof')
    #p.strip_dirs().sort_stats('cumulative').print_stats(50)
