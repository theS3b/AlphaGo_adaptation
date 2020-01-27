# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:24:19 2019

@author: sebastien
"""

from keras import layers
from keras import models
from keras import optimizers
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
from math import sqrt
from collections import deque
import os

nbr_shapes = 8
shape_length = 9
input_shape = (nbr_shapes * shape_length, 8, 8)
policy_output = 4098  # 8*8*(7*8 + 8) + 2 3
n_residual_blocks = 19
n_hidden_layers = 256
filters = 256
kernel = (3,3)
stride = 1
c = 0.01
# we try with l2 regularization, we sould also try with weight drop out

with tf.device('/GPU:0'):
    
    def policy_loss(y_true, y_pred):
        return - K.dot(K.transpose(y_true), K.log(y_pred))
    
    def res_block(first_layer):
        tmp_layer = first_layer
        first_layer = layers.Conv2D(filters, kernel, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(first_layer)
        first_layer = layers.BatchNormalization()(first_layer)
        first_layer = layers.LeakyReLU()(first_layer)
        first_layer = layers.Conv2D(filters, kernel, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(first_layer)
        first_layer = layers.BatchNormalization()(first_layer)
        first_layer = layers.Add()([first_layer, tmp_layer])  # SKIP connection
        first_layer = layers.LeakyReLU()(first_layer)
        return first_layer
        
            
    class Agent:
        def __init__(self, load_weights):
            self.weights_backup = 'alpha_weights.h5'
            self.EPOCHS = 50
            self.LR = 0.2
            self.BS = 32
            self.cpuct = sqrt(2)
            self.nbr_simulations = 800  # nbr simulations per move
            self.t = 0.5              # temperature parameter
            self.memory = deque(maxlen=2000)
            self.last_nn_output = []
            self.brain = self.build_RNN(load_weights)
        
        def save_brain(self):
            self.brain.save_weights(self.weights_backup)
        
        # action needs to be an index of an element of the policy output
        def append_memory(self, state, player, pi):
            self.memory.append((state, player, pi))
           
        def build_RNN(self, load_weights):
            print("Building RNN...")
            inputs = layers.Input(input_shape)
            
            y = layers.Conv2D(filters, kernel, padding='same')(inputs)
            y = layers.BatchNormalization()(y)
            y = layers.LeakyReLU()(y)

            for _ in range(n_residual_blocks):
                y = res_block(y)
                
            # policy head
            policy = layers.Conv2D(filters=2, kernel_size=(1,1), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(y)
            policy = layers.BatchNormalization()(policy)
            policy = layers.LeakyReLU()(policy)
            policy = layers.Flatten()(policy)
            policy = layers.Dense(policy_output, name="policy_output")(policy)
            
            # value head
            value = layers.Conv2D(filters=1, kernel_size=(1,1), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(y)
            value = layers.BatchNormalization()(value)
            value = layers.Flatten()(value)
            value = layers.LeakyReLU()(value)
            value = layers.Dense(n_hidden_layers)(value)
            value = layers.LeakyReLU()(value)
            value = layers.Dense(1)(value)
            value = layers.Activation('tanh', name="value_output")(value)
            
            model = models.Model(inputs=inputs, outputs=[policy, value])
            
            losses = {
            	"policy_output": policy_loss,
            	"value_output": "MSE"
            }
            lossWeights = {"policy_output": 0.5, "value_output": 0.5}
            opt = optimizers.Adam(lr = self.LR, decay=self.LR / self.EPOCHS)
            
            model.compile(opt, loss=losses, loss_weights=lossWeights, metrics={"policy_output" : 'accuracy', "value_output" : 'mse'})
            
            # if we want to load the weights, we load them
            if load_weights and os.path.isfile(self.weights_backup):
                model.load_weights(self.weights_backup)

            return model
