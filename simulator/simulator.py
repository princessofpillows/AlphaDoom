import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, os, pickle, cv2
from datetime import datetime
from pathlib import Path
from tqdm import trange
from config import get_config
from package_data import Gatherer


tf.enable_eager_execution()
cfg = get_config()

class Simulator(object):

    def __init__(self):
        super(Simulator, self).__init__()

        self.global_step = tf.train.get_or_create_global_step()
        self.epoch = tf.Variable(0)
        self.val_acc = 0.0
        self.best_acc = tf.Variable(0.0)

        self.model = cfg.models[cfg.model](cfg)
        self.optimizer = cfg.optims[cfg.optim](cfg.learning_rate)
        self.loss = cfg.losses[cfg.loss]

        self.preprocessing()
        self.build_writers()

    def preprocessing(self):
        if cfg.package_data:
            Gatherer().run()

        # Load data
        with open(cfg.data_dir, 'rb') as f:
            s0, action, s1 = zip(*pickle.load(f))
    
        # 70% train, 20% val, 10% test split
        size = len(s0)
        split_tr = int(size*0.7)
        split_va = int(size*0.2) + split_tr

        s0 = np.array(s0)
        action = np.array(action)
        s1 = np.array(s1)

        self.data_tr = tf.data.Dataset.from_tensor_slices((s0[:split_tr], action[:split_tr], s1[:split_tr]))
        self.data_va = tf.data.Dataset.from_tensor_slices((s0[split_tr:split_va], action[split_tr:split_va], s1[split_tr:split_va]))
        self.data_ts = tf.data.Dataset.from_tensor_slices((s0[split_va:], action[split_va:], s1[split_va:]))
        self.size = size

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
        self.saver = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, global_step=self.global_step, best_acc=self.best_acc, epoch=self.epoch)

    def logger(self, tape, loss, logits, s0, s1):
        if self.global_step.numpy() % cfg.log_freq == 0:
            with tf.contrib.summary.always_record_summaries():
                # Log vars
                tf.contrib.summary.scalar('loss', loss)
                
                if self.global_step.numpy() % (cfg.log_freq * 100) == 0:
                    tf.contrib.summary.histogram('logits', logits)
                    tf.contrib.summary.histogram('s0', s0)
                    tf.contrib.summary.histogram('s1', s1)

                    # Log weights
                    slots = self.optimizer.get_slot_names()
                    for variable in tape.watched_variables():
                            tf.contrib.summary.histogram(variable.name, variable)
                            for slot in slots:
                                slotvar = self.optimizer.get_slot(variable, slot)
                                if slotvar is not None:
                                    tf.contrib.summary.histogram(variable.name + '/' + slot, slotvar)
    
    def log_state(self, state, name):
        if self.global_step.numpy() % (cfg.log_freq * 10) == 0:
            with tf.contrib.summary.always_record_summaries():
                state = tf.cast(state, tf.float32)
                tf.contrib.summary.image(name, state, max_images=3)

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
        
        self.logger(tape, loss, logits, s0, s1)
        self.log_state(s0, "s0")
        self.log_state(logits, "logits")
        self.log_state(logits + s0_n, "logits_s0")
        self.log_state(s1, "truth")
        self.log_state(truth, "truth_logits")
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads_and_vars = zip(grads, self.model.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)
        
        self.global_step.assign_add(1)
    
    def train(self):
        if Path(self.save_path).is_dir():
            self.saver.restore(tf.train.latest_checkpoint(self.save_path))
        epoch = self.epoch.numpy()
        for epoch in trange(epoch, cfg.epochs):
            # Uniform shuffle
            batch = self.data_tr.shuffle(self.size).batch(cfg.batch_size)
            for s0, action, s1 in batch:
                self.update(s0, action, s1)
            self.epoch.assign_add(1)
        self.saver.save(file_prefix=self.ckpt_prefix)

    def predict(self, s0, action):
        self.saver.restore(tf.train.latest_checkpoint(self.save_path))
        s0_n = tf.image.per_image_standardization(s0)
        logits = self.model(s0_n, action)
        return logits + s0_n

def main():
    model = Simulator()
    model.train()

if __name__ == "__main__":
    main()
