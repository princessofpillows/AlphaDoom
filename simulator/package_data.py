import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, pickle
from tqdm import trange
from config import get_config


tf.enable_eager_execution()
cfg = get_config()

class Gatherer(object):

    def __init__(self):
        super(Gatherer, self).__init__()

        self.game = vzd.DoomGame()
        self.game.load_config("vizdoom/scenarios/basic.cfg")
        self.game.init()

        self.resolution = cfg.resolutions[cfg.model]
        self.terminal = tf.zeros([self.resolution[0], self.resolution[1], cfg.num_channels])
    
    def preprocess(self):
        frame = self.game.get_state().screen_buffer
        frame = tf.image.resize_images(frame, self.resolution)
        #frame = tf.image.per_image_standardization(frame)
        if cfg.num_channels == 1:
            # Convert rgb to greyscale
            frame = tf.image.rgb_to_grayscale(frame)
        return frame
    
    def run(self):
        memory = []
        for episode in trange(10):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            frames = []
            # Init stack of n frames
            for i in range(cfg.num_frames):
                frames.append(frame)

            while not self.game.is_episode_finished():
                #s0 = tf.concat(frames, axis=-1)
                s0 = frames[-1]
                action = random.choice(cfg.actions)
                self.game.make_action(action)

                # Update frames with latest image
                frames.pop(0)
                if self.game.get_state() is not None:
                    frames.append(self.preprocess())
                else:
                    frames.append(self.terminal)

                action = tf.reshape(tf.constant(action, tf.float32), [1, 1, len(cfg.actions)])
                #s1 = tf.concat(frames, axis=-1)
                s1 = frames[-1]
                memory.append([s0.numpy(), action.numpy(), s1.numpy()])
        
        with open('data.pkl', 'wb') as f:
            pickle.dump(memory, f)

def main():
    gatherer = Gatherer()
    gatherer.run()

if __name__ == "__main__":
    main()
