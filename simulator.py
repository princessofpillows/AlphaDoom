import tensorflow as tf
import numpy as np
import random, pickle, os
from pathlib import Path
from tqdm import trange
from simulator_cfg import get_cfg
from utils.vizdoom_api import VizDoom
from utils.writer import Writer


# Remove logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

tf.enable_eager_execution()
cfg = get_cfg()

class Simulator(object):

    def __init__(self):
        super(Simulator, self).__init__()
        
        self.model = cfg.model(cfg)
        self.optim = cfg.optim(cfg.learning_rate)
        self.loss = cfg.loss
        self.epoch = tf.Variable(0)

        self.writer = Writer(cfg)
        # Restore if save exists
        if Path('./simulator_saves/best').is_dir():
            self.model, self.optim, self.epoch = self.writer.restore(model=self.model, optim=self.optim, epoch=self.epoch)

        self.preprocessing()

    def preprocessing(self):
        if cfg.package_data or not Path('./data.pkl').is_file():
            vizdoom = VizDoom(cfg)
            memory = []
            for episode in trange(cfg.gather_epochs):
                vizdoom.new_episode()
                s0 = vizdoom.get_preprocessed_state()

                while not vizdoom.is_episode_finished():
                    action = random.choice(cfg.actions)
                    vizdoom.make_action(action)

                    s1 = vizdoom.get_preprocessed_state()
                    action = np.reshape(action, [1, 1, len(cfg.actions)]).astype(np.float32)

                    memory.append([s0, action, s1])
                    s0 = s1

            with open('data.pkl', 'wb') as f:
                pickle.dump(memory, f)

        # Load data
        with open(cfg.data_dir, 'rb') as f:
            s0, action, s1 = zip(*pickle.load(f))

        self.size = len(s0)
        self.data = tf.data.Dataset.from_tensor_slices((np.array(s0), np.array(action), np.array(s1)))

    def update(self, s0, action, s1):
        # Normalize
        s0_n = tf.image.per_image_standardization(s0)
        truth = tf.image.per_image_standardization(s1) - s0_n
        # Construct graph
        with tf.GradientTape() as tape:
            # Approximate next frame
            logits = self.model(s0_n, action)
            # Compare generated transformation matrix with truth
            loss = tf.reduce_mean(self.loss(truth, logits))

        # Log stats, images
        self.writer.log(self.optim, tape, loss)
        self.writer.log_state("logits", logits)
        self.writer.log_state("truth_logits", truth)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads_and_vars = zip(grads, self.model.trainable_weights)
        self.optim.apply_gradients(grads_and_vars)
        
        self.writer.global_step.assign_add(1)
    
    def train(self):
        for epoch in trange(self.epoch.numpy(), cfg.epochs):
            # Uniform shuffle
            batch = self.data.shuffle(self.size).batch(cfg.batch_size)
            for s0, action, s1 in batch:
                self.update(s0, action, s1)
            self.epoch.assign_add(1)
        self.writer.save(self.model, self.optim, self.epoch)

    def predict(self, s0, action):
        s0_n = tf.image.per_image_standardization(s0)
        logits = self.model(s0_n, action[None])
        return logits + s0_n

def main():
    model = Simulator()
    model.train()

if __name__ == "__main__":
    main()
