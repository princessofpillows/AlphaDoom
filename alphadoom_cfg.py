
import argparse
import tensorflow as tf
from classes.models import AlphaGoZero

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
                       default="replay.pkl",
                       help="Location to save and load replay data")

pre_arg.add_argument("--vizdoom_dir", type=str,
                       default="./ViZDoom",
                       help="Location of vizdoom engine")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=8,
                       help="Positions in queue to be evaluated at a time")

train_arg.add_argument("--mini_batch_size", type=int,
                       default=2048,
                       help="States to sample from replay memory")

train_arg.add_argument("--num_sims", type=int,
                       default=4,
                       help="Number of simulations to run before selecting action")

train_arg.add_argument("--epochs", type=int,
                       default=100,
                       help="Number of episodes to train on")

train_arg.add_argument("--momentum", type=float,
                       default=0.9,
                       help="Hyperparameter for momentum")

train_arg.add_argument("--eps", type=float,
                       default=0.25,
                       help="Epsilon for Dirichlet noise equation")

train_arg.add_argument("--d_noise", type=float,
                       default=0.03,
                       help="Strength of Dirichlet noise function")

train_arg.add_argument("--Cpuct", type=float,
                       default=0.99,
                       help="Constant for determing exploration rate")

train_arg.add_argument("--T", type=float,
                       default=1.0,
                       help="Temperature for exploration. Starts at 1 for 30n, then T -> 0")

train_arg.add_argument("--c", type=float,
                       default=1e-4,
                       help="Hyperparameter for L2 weight regularization")

train_arg.add_argument("--log_dir", type=str,
                       default="./alphadoom_logs/",
                       help="Directory to save logs")

train_arg.add_argument("--log_freq", type=int,
                       default=1,
                       help="Number of steps before logging weights")

train_arg.add_argument("--save_dir", type=str,
                       default="./alphadoom_saves/",
                       help="Directory to save current model")

train_arg.add_argument("--save_freq", type=int,
                       default=10,
                       help="Number of episodes before saving / evaluating current model")

train_arg.add_argument("-f", "--extension", type=str,
                       default=None,
                       help="Specific name to save training session or restore from")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--model", type=str,
                       default=AlphaGoZero,
                       choices=[AlphaGoZero],
                       help="Chosen architecture")

model_arg.add_argument("--optim", type=str,
                       default=tf.train.AdamOptimizer,
                       choices=[tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer],
                       help="Chosen optimizer")

model_arg.add_argument("--loss1", type=str,
                       default=tf.losses.mean_squared_error,
                       choices=[tf.losses.huber_loss, tf.losses.mean_squared_error],
                       help="Chosen loss for z and reward")

model_arg.add_argument("--loss2", type=str,
                       default=tf.losses.softmax_cross_entropy,
                       choices=[tf.losses.softmax_cross_entropy, tf.losses.sigmoid_cross_entropy, tf.losses.sparse_softmax_cross_entropy],
                       help="Chosen loss for probabilities and mcts edges")

model_arg.add_argument("--resolution",
                       default=(32,32),
                       help="Resolution for chosen architecture")

model_arg.add_argument("--activ", type=str,
                       default="relu",
                       choices=["relu", "elu", "selu", "tanh", "sigmoid"],
                       help="Activation function to use")

model_arg.add_argument("--init", type=str,
                       default="random_normal",
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

model_arg.add_argument("--num_frames", type=int,
                       default=1,
                       help="Number of stacked frames to send to CNN, depicting history")

model_arg.add_argument("--num_channels", type=int,
                       default=1,
                       choices=[1, 3],
                       help="Number of colour channels in frame")   

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
# Arguments for memory
mem_arg = add_argument_group("Memory")

mem_arg.add_argument("--cap", type=int,
                       default=5000,
                       help="Maximum number of transitions in replay memory")

# ----------------------------------------
# Function to be called externally
def get_cfg():
    config, unparsed = parser.parse_known_args()

    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    return config
