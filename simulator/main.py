import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, os
from tensorboardX import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import trange
from config import get_config
from models import AutoEncoder


tf.enable_eager_execution()
cfg = get_config()

class Simulator(object):

    def __init__(self):
        super(Simulator, self).__init__()

        self.game = vzd.DoomGame()
        self.game.load_config("vizdoom/scenarios/basic.cfg")
        self.game.init()

        self.model = AutoEncoder(cfg)
        self.optimizer = tf.train.AdamOptimizer(cfg.learning_rate)

        self.global_step = tf.train.get_or_create_global_step()
        self.terminal = tf.zeros([self.model.shape[0], self.model.shape[1], cfg.num_channels])

        self.build_writers()

    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)
        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, optimizer_step=self.global_step)

        log_path = cfg.log_dir + cfg.extension
        self.writer = SummaryWriter(log_path)

    def logger(self, tape, loss):
        if self.global_step.numpy() % cfg.log_freq == 0:
            # Log scalars
            self.writer.add_scalar('loss', loss.numpy(), self.global_step)
            
            # Log weight scalars
            slots = self.optimizer.get_slot_names()
            for variable in tape.watched_variables():
                    self.writer.add_scalar(variable.name, tf.nn.l2_loss(variable).numpy(), self.global_step)

                    for slot in slots:
                        slotvar = self.optimizer.get_slot(variable, slot)
                        if slotvar is not None:
                            self.writer.add_scalar(variable.name + '/' + slot, tf.nn.l2_loss(slotvar).numpy(), self.global_step)
    
    def log_state(self, frames):
        for i in range(len(frames)):
            s = np.transpose(frames[i], [2,0,1]).astype(np.uint8)
            self.writer.add_image('state ' + 'n-' + str(i), s, self.global_step)

    def update(self, s):
        # Construct graph
        with tf.GradientTape() as tape:
            logits, truth = self.forward(s)
            # Compare logits with ground truth
            loss = tf.reduce_mean(tf.losses.mean_squared_error(truth, logits))
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads_and_vars = zip(grads, self.model.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)

        
        self.global_step.assign_add(1)
    
    def forward(self, s):
        # Take action
        action = random.choice(cfg.actions)
        self.game.make_action(action)
        
        # Approximate next frame
        action = tf.reshape(tf.constant(action, tf.float32), [1, 1, 1, len(cfg.actions)])
        logits = self.model(s[None], action)
        print(logits)

        # Get next frame
        if self.game.get_state() is not None:
            truth = self.preprocess()
        else:
            truth = self.terminal
        truth = tf.expand_dims(truth, 0)

        return logits, truth

    def preprocess(self):
        screen = self.game.get_state().screen_buffer
        frame = np.multiply(screen, 255.0/screen.max())
        if cfg.num_channels == 1:
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
            frames = []
            # Init stack of n frames
            for i in range(cfg.num_frames):
                frames.append(frame)

            while not self.game.is_episode_finished():
                s0 = tf.concat(frames, axis=-1)
                self.update(s0)

                # Update frames with latest image
                frames.pop(0)
                if self.game.get_state() is not None:
                    frames.append(self.preprocess())
                
        self.saver.save(file_prefix=self.ckpt_prefix)

    def test(self):
        self.saver.restore(tf.train.latest_checkpoint(cfg.save_dir))
        for _ in trange(cfg.test_episodes):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            frames = []
            # Init stack of n frames
            for i in range(cfg.num_frames):
                frames.append(frame)

            count = 0
            while not self.game.is_episode_finished():
                s = tf.concat(frames, axis=-1)
                logits, _ = self.forward(s)

                if count % cfg.num_frames == 0:
                    self.log_state(logits)
                
                # Update frames with latest image
                if self.game.get_state() is not None:
                    frames.pop(0)
                    frames.append(self.preprocess())
                
                count += 1

def main():
    model = Simulator()

    model.train()

    model.test()

if __name__ == "__main__":
    main()
