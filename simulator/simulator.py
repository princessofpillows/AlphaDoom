import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, os, pickle, cv2
from tensorboardX import SummaryWriter
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

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, optimizer_step=self.global_step)

        log_path = cfg.log_dir + cfg.extension
        self.writer = SummaryWriter(log_path, max_queue=0)

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
    
    def log_state(self, logits, count):
        s = np.transpose(logits[0], [2,0,1]).astype(np.uint8)
        self.writer.add_image('logits', s, count)
        

    def update(self, s0, action, s1):
        # Construct graph
        with tf.GradientTape() as tape:
            # Approximate next frame
            logits = self.model(s0, action)
            # Compare logits with ground truth
            loss = tf.reduce_mean(self.loss(s1, logits))
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads_and_vars = zip(grads, self.model.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.writer.add_histogram('logits', logits.numpy(), self.global_step)
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
            # Perform validation
            if epoch % cfg.val_freq == 0:
                batch = self.data_va.shuffle(self.size).batch(cfg.batch_size)
                acc_total = []
                for s0, action, s1 in batch:
                    logits = self.model(s0, action)
                    acc = tf.reduce_mean(tf.cast(tf.equal(logits, s1), 'float32'))
                    acc_total.append(acc)

                self.val_acc = np.sum(acc_total) / len(acc_total)
                if self.best_acc < self.val_acc:
                    self.best_acc.assign(self.val_acc)
                    self.saver.save(file_prefix=self.ckpt_prefix)
                else:
                    cfg.val_freq //= 2
                    if cfg.val_freq == 0:
                        print("Stopping training early at epoch " + str(epoch) + " due to overfitting.")
                        return

    def test(self):
        self.saver.restore(tf.train.latest_checkpoint(self.save_path))
        batch = self.data_ts.shuffle(self.size).batch(cfg.batch_size)
        acc_total = []
        count = 0
        for s0, action, s1 in batch:
            logits = self.model(s0, action)
            self.log_state(logits, count)
            acc = tf.reduce_mean(tf.cast(tf.equal(logits, s1), 'float32'))
            acc_total.append(acc)
            count += 1
        
        acc = np.sum(acc_total) / len(acc_total)
        message = 'Accuracy: ' + str(acc) + '. Number of epochs where overfitting occured: ' + str(cfg.epochs - self.epoch.numpy()) + '. Note: Model stops saving when overfitting occurs.'
        print(message)

    def predict(self, s0, action):
        self.saver.restore(tf.train.latest_checkpoint(self.save_path))
        logits = self.model(s0, action)
        return logits

def main():
    model = Simulator()
    model.train()
    model.test()

if __name__ == "__main__":
    main()
