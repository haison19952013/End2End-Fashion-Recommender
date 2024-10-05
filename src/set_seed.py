import os
import random
import numpy as np
import tensorflow as tf
import config

config_ = config.Config()
# Set seed for built-in random module
random.seed(config_.seed)

# Set seed for NumPy
np.random.seed(config_.seed)

# Set seed for TensorFlow
tf.random.set_seed(config_.seed)

# Set PYTHONHASHSEED for reproducibility
os.environ['PYTHONHASHSEED'] = str(config_.seed)
