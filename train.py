# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:42:50 2019

@author: sebastien
"""

import numpy as np
import random
from chess_env import *
from model import Agent
from tree import Node
from simulation import select_move
from time import time

white = 1
black = -1

def train(agent):
    
    print("Training agent...")
    agent.EPOCHS = 1
    agent.nbr_simulations = 1
    
    # each epoch represent a game
    for epoch in range(agent.EPOCHS):
        print("Epoch", str(epoch))
        
        # start from init position
        root = Node(Board(), None)
        
        # Simulate a whole game
        end = False
        c = 0
        while not end:
            root, end = select_move(root, agent)
            print(c)
            c += 1
        agent.brain.summary()
            
        # Evaluate who won
        z = root.state.get_result(white)
        print(z)
        
        # Learn from this game
        minibatch = random.sample(agent.memory, agent.BS)
        training_state = np.array([elem[0][0] for elem in minibatch])  # 0 for state
        training_target = {'policy_output' : np.array([elem[2] for elem in minibatch]),  # 2 for pi
                           'value_output' : np.array([z * elem[1] for elem in minibatch])}  # 1 for player
        
        agent.brain.fit(training_state, training_target, epochs = 1, batch_size = agent.BS, verbose = 1, validation_split = 0)
        
        if epoch % 20 == 0:
            agent.save_brain()
        
a = Agent(False)
train(a)


# Learning to do