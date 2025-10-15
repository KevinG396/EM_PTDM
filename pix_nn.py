import os
from PIL import Image
import random
import argparse
import numpy as np
import tensorflow as tf


import tensorflow as tf

class TargetConvResizer(tf.Module):
    def __init__(self, input_size, in_channels):
        super(TargetConvResizer, self).__init__()
        #print(f"targ inp {input_size}")
        
        self.conv1 = tf.Variable(tf.random.normal([3, 3, in_channels, 1]), name='conv1_kernel')
        self.bias1 = tf.Variable(tf.zeros([1]), name='conv1_bias')
        
        self.fc1_weights = tf.Variable(tf.random.normal([input_size * input_size // 4, 4]), name='fc1_weights')
        self.fc1_bias = tf.Variable(tf.zeros([4]), name='fc1_bias')

        self.fc2_weights = tf.Variable(tf.random.normal([4, 32]), name='fc2_weights') # modeified
        self.fc2_bias = tf.Variable(tf.zeros([32]), name='fc2_bias')

        self.fc3_weights = tf.Variable(tf.random.normal([32, 16]), name='fc3_weights')
        self.fc3_bias = tf.Variable(tf.zeros([16]), name='fc3_bias')

        self.fc4_weights = tf.Variable(tf.random.normal([16, 8]), name='fc4_weights')
        self.fc4_bias = tf.Variable(tf.zeros([8]), name='fc4_bias')

        self.fc5_weights = tf.Variable(tf.random.normal([8, 2]), name='fc5_weights')
        self.fc5_bias = tf.Variable(tf.zeros([2]), name='fc5_bias')

    def leaky_relu(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def __call__(self, x):
        x = tf.nn.conv2d(x, self.conv1, strides=2, padding='SAME')
        x = x + self.bias1
        x = self.leaky_relu(x)

        x = tf.reshape(x, [x.shape[0], -1]) 

        x = tf.matmul(x, self.fc1_weights) + self.fc1_bias
        x = self.leaky_relu(x)

        x = tf.matmul(x, self.fc2_weights) + self.fc2_bias
        x = self.leaky_relu(x)

        x = tf.matmul(x, self.fc3_weights) + self.fc3_bias
        x = self.leaky_relu(x)

        x = tf.matmul(x, self.fc4_weights) + self.fc4_bias
        x = self.leaky_relu(x)

        x = tf.matmul(x, self.fc5_weights) + self.fc5_bias
        x = self.leaky_relu(x)

        return x



