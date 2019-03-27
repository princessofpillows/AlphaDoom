import vizdoom as vzd
import tensorflow as tf
import numpy as np
import networkx as nx
import random, os, cv2, pickle
from datetime import datetime
from pathlib import Path
from tqdm import trange
from config import get_config
from models import Simulator
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb


tf.enable_eager_execution()
cfg = get_config()

class MCTS(object):

    class Node(object):
        # s is state, a is action on edge, p is prior probability of selecting, n is number of visits, q is mean value, w is total value
        def __init__(self, parent=None, s=None, a=None, p=0):
            self.parent = parent
            self.children = []
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
        def value(self):
            return self.q + cfg.Cpuct * self.p * (np.sqrt(self.parent.n) / (1 + self.n))

    def __init__(self):
        self.root = MCTS.Node()
        self.curr_root = self.root

    # Selects best path from current state
    def select(self, parent):
        v = []
        # Gets value for each child of parent (N children == N actions)
        for child in parent.children:
            v.append(child.value())

        v = np.asarray(v)
        # Gets index of best child node
        v_max = np.where(v == np.max(v))[0]
        idx = v_max[0]
        # Random select if tie
        if len(v_max) > 1:
            idx = np.random.choice(v_max)

        return parent.children[idx]
    
    # Add edge and empty child node to parent node
    def expand(self, parent, p):
        child = MCTS.Node(parent, None, None, p)
        parent.add_child(child)
    
    # Finds leaf from best current path
    def search(self):
        curr = self.curr_root
        while(curr.children != []):
            curr = self.select(curr)
        
        return curr
    
    # Add new edges to tree, backprop through current path
    def update(self, leaf, v, p_set):
        # Adds edge and empty node for each action
        noise = np.random.dirichlet(cfg.d_noise * np.ones(len(cfg.actions)))
        p_set = p_set.numpy()[0]
        for i in range(len(cfg.actions)):
            # Add dirichlet noise
            p = (1 - cfg.eps) * p_set[i] + cfg.eps * noise[i]
            self.expand(leaf, p)

        # Backpropogation
        curr = leaf
        while curr.parent != None:
            curr.increment()
            curr.update(v.numpy()[0])
            curr = curr.parent
    
    # From current root node, iterate through children and select one based on exploration heuristic and vist count
    def select_action(self):
        best = 0
        selection = None
        # Exploration based expansion; when T is 0, becomes deterministic instead of stochastic
        for child in self.curr_root.children:
            curr = np.power(child.n, 1 / cfg.T) / np.power(self.root.n, 1 / cfg.T)
            if curr > best:
                best = curr
                selection = child
 
        self.curr_root = selection
        return selection.a, np.power(selection.n, 1 / cfg.T)
    
    def visualize_tree(self):

        def iterate_children(self, G, parent):
            # Recursively add children to graph
            for child in parent.children:
                G.add_node([child.w, child.n])
                G.add_edge(parent.w, child.w, object=child.p)
                G = iterate_children(G, child)
            return G

        # Entire game tree visualization
        tree = nx.Graph()
        tree.add_node(self.root.w)
        tree = iterate_children(tree, self.root)
        nx.draw(tree, with_labels=True)
        return tree

    def visualize(self):
        # Local action tree visualization
        G = nx.Graph()
        G.add_node(self.curr_root.w)
        for child in self.curr_root.children:
            G.add_node(child.w)
            G.add_edge(self.curr_root.w, child.w, object=child.p)
        nx.draw(G, with_labels=True)
        return tree


class Replay(object):

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
            
        return zip(*batch)


class AlphaDoom(object):

    def __init__(self):
        super(AlphaDoom, self).__init__()

        # Create game instance
        self.game = vzd.DoomGame()
        # Scenario
        self.game.set_doom_scenario_path("./vizdoom/scenarios/basic.wad")
        self.game.set_doom_map("map01")
        # Screen
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(True)
        #self.game.set_sound_enabled(True)
        # Buffers
        #self.game.set_depth_buffer_enabled(True)
        #self.game.set_labels_buffer_enabled(True)
        #self.game.set_automap_buffer_enabled(True)
        # Rendering
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False) # If HUD enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(False)
        self.game.set_render_decals(False)  # Bullet holes and blood on the walls
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)  # Smoke and blood
        self.game.set_render_messages(False)  # In-game messages
        self.game.set_render_corpses(False)
        self.game.set_render_screen_flashes(True) # Effect upon taking damage or picking up items
        # Actions
        self.game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
        # Variables included in state
        self.game.set_available_game_variables([vzd.GameVariable.AMMO2])
        # Start/end time
        self.game.set_episode_start_time(14)
        self.game.set_episode_timeout(300)
        # Reward
        self.game.set_living_reward(-1)
        self.game.set_doom_skill(5)
        # Game mode
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.init()

        self.global_step = tf.train.get_or_create_global_step()
        # Assign mcts, replay to CPU due to GPU memory limitations
        with tf.device('CPU:0'):
            self.mcts = MCTS()
            self.replay = Replay()

        # Init next state simulator
        self.simulator = Simulator(cfg)
        self.sim_saver = tf.train.Checkpoint(model=self.simulator)
        self.sim_saver.restore(tf.train.latest_checkpoint("./simulator/saves/best"))

        # Load selected model
        self.model = cfg.models[cfg.model](cfg)
        #self.model.build((None,) + self.model.shape + (cfg.num_frames*cfg.num_channels,))
        self.loss1 = cfg.losses[cfg.loss1]
        self.loss2 = cfg.losses[cfg.loss2]
        self.learning_rate = cfg.lr_schedule[str(cfg.learning_rate)]
        self.optimizer = cfg.optims[cfg.optim](self.learning_rate)

        self.resolution = (cfg.resolutions[cfg.model])
        self.terminal = tf.zeros([self.model.shape[0], self.model.shape[1], cfg.num_channels])
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
        self.saver = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, global_step=self.global_step)
        
    def logger(self, tape, loss):
        if self.global_step.numpy() % cfg.log_freq == 0:
            with tf.contrib.summary.always_record_summaries():
                # Log vars
                tf.contrib.summary.scalar('loss', loss)
                
                if self.global_step.numpy() % (cfg.log_freq * 100) == 0:
                    # Log weights
                    slots = self.optimizer.get_slot_names()
                    for variable in tape.watched_variables():
                            tf.contrib.summary.histogram(variable.name, variable)
                            for slot in slots:
                                slotvar = self.optimizer.get_slot(variable, slot)
                                if slotvar is not None:
                                    tf.contrib.summary.histogram(variable.name + '/' + slot, slotvar)
    
    def log_state(self, state, name):
        if self.global_step.numpy() % (cfg.log_freq * 100) == 0:
            with tf.contrib.summary.always_record_summaries():
                state = tf.cast(state, tf.float32)
                tf.contrib.summary.image(name, state, max_images=3)
    
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
                s = tf.concat(frames, axis=-1)
                _, _, _ = self.perform_action(s)

                if self.game.is_episode_finished():
                    p1_wins += 1
                    break

                p1_frames.pop(0)
                # TODO ******** Add support for multiple players in preprocess *********
                p1_frames.append(self.preprocess())

                # P2
                curr = tf.ones([84, 84, 1])
                s = tf.concat(frames, axis=-1)
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

    def update(self):
        # Fetch batch of experiences
        s0, pi, z, s1 = self.replay.fetch()
        
        # Construct graph
        with tf.GradientTape() as tape:
            p, v = self.model(s0)
            loss = tf.reduce_mean(self.loss1(z, v) - self.loss2(tf.pow(pi, cfg.T), p) + cfg.c * tf.nn.l2_loss(self.model.weights))
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)
    
    # Runs N simulations, where each sim reaches a leaf node in MCTS tree
    def simulate(self, s0):
        for i in range(cfg.num_sims):
            # Find leaf
            leaf = self.mcts.search()
            leaf.a = random.choice(cfg.actions)
            # Simulate leaf's state
            leaf.s = self.simulator.predict(s0, np.reshape(leaf.a, [1, 1, len(cfg.actions)]).astype(np.float32))
            # Get p, the prior probability set of all actions (edges) from current leaf node, and v, the value of current leaf node
            p, v = self.model(leaf.s)
            # Backprop through MCTS tree
            self.mcts.update(leaf, v, p)

    # Returns best action
    def perform_action(self, s0):
        self.simulate(s0)
        action, pi = self.mcts.select_action()
        # Take action
        reward = self.game.make_action(action, cfg.skiprate)
        # Reward of -1 if bad, +1 if good
        if reward > 0:
            return pi, 1
        else:
            return pi, -1
    
    def preprocess(self):
        frame = self.game.get_state().screen_buffer
        # Blur, crop, resize
        frame = cv2.GaussianBlur(frame, (39,39), 0, 0)
        frame = tf.image.central_crop(frame, 0.5)
        frame = tf.image.resize(frame, self.resolution, align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
        # Kmeans clustering
        frame = rgb2hsv(frame)
        kmeans = KMeans(n_clusters=4).fit(frame.reshape((-1, 3)))
        frame = kmeans.cluster_centers_[kmeans.labels_].reshape(frame.shape)
        frame = hsv2rgb(frame)
        # Greyscale
        if cfg.num_channels == 1:
            frame = tf.image.rgb_to_grayscale(frame).numpy()
        return frame
    
    def train(self):
        if Path(self.save_path).is_dir():
            self.saver.restore(tf.train.latest_checkpoint(self.save_path))
        if Path(self.save_path + '/replay.pkl').is_file():
            with open(self.save_path + '/replay.pkl', 'rb') as f:
                self.replay.memory = pickle.load(f)
        for episode in trange(cfg.episodes):
            # Update learning rate at scheduled times
            if episode % cfg.lr_schedule[str(cfg.learning_rate)][1] == 0:
                cfg.learning_rate += 1
                self.learning_rate = cfg.lr_schedule[str(cfg.learning_rate)][0]

            # Save model
            if episode % cfg.save_freq == 0:
                # Check if new model is improvement over current best
                #if self.evaluate():
                self.saver.save(file_prefix=self.ckpt_prefix)
                with open(self.save_path + 'replay.pkl', 'wb') as f:
                    pickle.dump(self.replay.memory, f)

            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            frames = []
            # Init stack of n frames
            for i in range(cfg.num_frames):
                frames.append(frame)
            
            while not self.game.is_episode_finished():
                s0 = tf.concat(frames, axis=-1)
                pi, z = self.perform_action(s0)

                # Update frames with latest image
                if self.game.get_state() is not None:
                    frames.pop(0)
                    frames.append(self.preprocess())

                s1 = tf.concat(frames, axis=-1)
                self.replay.push([s0, pi, z, s1])
                
            # Train on experiences from memory
            self.update()
        
        self.saver.save(file_prefix=self.ckpt_prefix)

def main():
    model = AlphaDoom()
    model.train()

if __name__ == "__main__":
    main()
    #cProfile.run('main(cfg)', 'prof')
    #p = pstats.Stats('prof')
    #p.strip_dirs().sort_stats('cumulative').print_stats(50)
