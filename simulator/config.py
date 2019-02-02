
import argparse
import tensorflow as tf
from models import AutoEncoder

# ----------------------------------------
# Global variables
arg_lists = []
parser = argparse.ArgumentParser()

# Possible actions
shoot = [1, 0, 0]
left = [0, 1, 0]
right = [0, 0, 1]

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
                       help="Location of image data")

pre_arg.add_argument("--package_data", type=bool,
                       default=True,
                       help="Whether or not to regather frames from package_data.py")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=32,
                       help="Number of images in each forward pass")

train_arg.add_argument("--epochs", type=int,
                       default=1,
                       help="Number of epochs to train on")

train_arg.add_argument("--val_freq", type=int,
                       default=10,
                       help="Validation interval in epochs")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs/",
                       help="Directory to save logs")

train_arg.add_argument("--log_freq", type=int,
                       default=10,
                       help="Number of steps before logging weights")

train_arg.add_argument("--save_dir", type=str,
                       default="./saves/",
                       help="Directory to save current model")

train_arg.add_argument("--save_freq", type=int,
                       default=1,
                       help="Number of episodes before saving model")

train_arg.add_argument("-f", "--extension", type=str,
                       default=None,
                       help="Specific name to save training session or restore from")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--model", type=str,
                       default="autoencoder",
                       choices=["autoencoder"],
                       help="Chosen architecture")

model_arg.add_argument("--models",
                       default={"autoencoder":AutoEncoder},
                       help="Architecture options")

model_arg.add_argument("--resolutions",
                       default={"autoencoder":(32,32)},
                       help="Resolution for chosen architecture")

model_arg.add_argument("--activ", type=str,
                       default="relu",
                       choices=["relu", "elu", "selu", "tanh", "sigmoid"],
                       help="Activation function to use")

model_arg.add_argument("--init", type=str,
                       default="glorot_normal",
                       choices=["glorot_normal", "glorot_uniform", "random_normal", "random_uniform", "truncated_normal"],
                       help="Initialization function to use")

model_arg.add_argument("--optim", type=str,
                       default="adam",
                       choices=["adam"],
                       help="Chosen optimizer")

model_arg.add_argument("--optims",
                       default={"adam":tf.train.AdamOptimizer},
                       help="Optimizer options")

model_arg.add_argument("--loss", type=str,
                       default="mse",
                       choices=["mse"],
                       help="Chosen loss")

model_arg.add_argument("--losses",
                       default={"mse":tf.losses.mean_squared_error},
                       help="Loss options")

model_arg.add_argument("--actions", type=int,
                       default=[shoot, left, right],
                       help="Possible actions to take")

model_arg.add_argument("--skiprate", type=int,
                       default=3,
                       help="Number of frames to skip during each action. Current action will be repeated for duration of skip")

model_arg.add_argument("--num_frames", type=int,
                       default=4,
                       help="Number of stacked frames to send to CNN, depicting motion")

model_arg.add_argument("--num_channels", type=int,
                       default=3,
                       help="Number of colour channels in frame [1, 3]")

model_arg.add_argument("--output_channels", type=int,
                       default=3,
                       help="Size of last dimension of autoencoder")

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
def get_config():
    config, unparsed = parser.parse_known_args()

    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    return config
