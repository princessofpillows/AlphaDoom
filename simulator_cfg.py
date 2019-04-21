
import argparse
import tensorflow as tf
from classes.models import AutoEncoder

# ----------------------------------------
# Global variables
arg_lists = []
parser = argparse.ArgumentParser()

# ----------------------------------------
# Macro for arparse
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# ----------------------------------------
# Arguments for preprocessing
pre_arg = add_argument_group("Preprocessing")

pre_arg.add_argument("--data_dir", type=str,
                       default="data.pkl",
                       help="Location to save and load image data")

pre_arg.add_argument("--vizdoom_dir", type=str,
                       default="./ViZDoom",
                       help="Location of vizdoom engine")

pre_arg.add_argument("--package_data", type=bool,
                       default=True,
                       help="Whether or not to gather and save new frames")

pre_arg.add_argument("--gather_epochs", type=int,
                       default=20,
                       help="Number of epochs to gather data with")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Number of images in each forward pass")

train_arg.add_argument("--epochs", type=int,
                       default=500,
                       help="Number of epochs to train on")

train_arg.add_argument("--val_freq", type=int,
                       default=10,
                       help="Validation interval in epochs")

train_arg.add_argument("--log_dir", type=str,
                       default="./simulator_logs/",
                       help="Directory to save logs")

train_arg.add_argument("--log_freq", type=int,
                       default=10,
                       help="Number of steps before logging weights")

train_arg.add_argument("--save_dir", type=str,
                       default="./simulator_saves/",
                       help="Directory to save current model")

train_arg.add_argument("--save_freq", type=int,
                       default=100,
                       help="Number of episodes before saving model")

train_arg.add_argument("-f", "--extension", type=str,
                       default="best",
                       help="Specific name to save training session or restore from")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--model", type=str,
                       default=AutoEncoder,
                       choices=[AutoEncoder],
                       help="Chosen architecture")

model_arg.add_argument("--optim", type=str,
                       default=tf.train.AdamOptimizer,
                       choices=[tf.train.AdamOptimizer],
                       help="Chosen optimizer")

model_arg.add_argument("--loss", type=str,
                       default=tf.losses.mean_squared_error,
                       choices=[tf.losses.mean_squared_error],
                       help="Chosen loss")

model_arg.add_argument("--resolution",
                       default=(32,32),
                       choices=[(32,32)],
                       help="Chosen resolution")

model_arg.add_argument("--activ", type=str,
                       default="relu",
                       choices=["relu", "elu", "selu", "tanh", "sigmoid"],
                       help="Activation function to use")

model_arg.add_argument("--init", type=str,
                       default="glorot_normal",
                       choices=["glorot_normal", "glorot_uniform", "random_normal", "random_uniform", "truncated_normal"],
                       help="Initialization function to use")

# Possible actions
shoot = [1, 0, 0]
left = [0, 1, 0]
right = [0, 0, 1]

model_arg.add_argument("--actions", type=int,
                       default=[shoot, left, right],
                       help="Possible actions to take")

model_arg.add_argument("--skiprate", type=int,
                       default=5,
                       help="Number of frames to skip during each action. Current action will be repeated for duration of skip")

model_arg.add_argument("--num_channels", type=int,
                       default=1,
                       help="Number of colour channels in frame [1, 3]")

train_arg.add_argument("--num_blks",
                       default=1,
                       help="Residual blocks in autoencoder")

train_arg.add_argument("--min_filters",
                       default=16,
                       help="Minimum number of filters in network")

train_arg.add_argument("--max_filters",
                       default=128,
                       help="Maximum number of filters in network")         

# ----------------------------------------
# Function to be called externally
def get_cfg():
    config, unparsed = parser.parse_known_args()

    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    return config
