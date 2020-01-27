# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:54:34 2019

@author: seb
"""

from tree import Edge
from tree import Node
from model import Agent
from tree import encode_input
from model import policy_output
import numpy as np

from time import time
from chess_env import *

def select_move(root, agent):
    
    if not root.state.get_bitboard_children():
        return root, True
        
    for _ in range(agent.nbr_simulations):
        run_simulation(root, agent)  # root passed by reference
    
    pi = np.zeros(policy_output)
    
    for i in range(len(root.edges)):
        pi[root.edges[i].case_nb] = root.edges[i].c.n ** agent.t

    maxi = np.argmax(edge.n for edge in root.edges)
    agent.append_memory(encode_input(root), root.state.player, pi)
    return root.edges[maxi].c, False


def run_simulation(root_node, agent):
    cu = root_node
    
    # simulation
    while not cu.is_leaf():
        cu = select_ucb(cu, agent)
        
    # expand leaf node
    v = cu.expand(agent)
    
    # Backpropagation
    while True:
        cu.n += 1
        cu.q = (cu.q + v) / cu.n
        
        if cu == root_node:
            break
            
        cu = cu.parent
        
def select_ucb(root, agent):
    tab_root_edges = range(len(root.edges))
    nsum = sum(st.n for st in (root.edges[i].c for i in tab_root_edges))
    maxucb = -999
    maxi = 0
    
    for i in tab_root_edges:
        # cpuct * p * sum(n)/(1 + n) + q
        ucb = agent.cpuct * root.edges[i].p * (nsum /(1+ root.edges[i].c.n)) + root.edges[i].c.q
        if ucb > maxucb:
            maxucb = ucb
            maxi = i
    
    return root.edges[maxi].c