# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:42 2019

@author: seb
"""

import numpy as np
from chess_env import *
from model import Agent
from model import nbr_shapes
from model import input_shape
from model import shape_length

class Edge:
    def __init__(self, prob, child, case_index):
        self.c = child
        self.p = prob
        self.case_nb = case_index  # used to easily create the pi vector

class Node:
    def __init__(self, state, parent):
        self.n = 0
        self.q = 0
        self.state = state      # numpy array
        self.parent = parent    # Node
        self.edges = []         # arr of numpy arrays, float values and Nodes
        
    def is_leaf(self):
        return (self.edges == [])
    
    def print(self):
        print('n : ' + str(self.n) + '  q : ' + str(self.q))
        print('player :', self.state.player)
        printb(self.state)
        print('nbr children : ' + str(len(self.edges)) + '\n')
    
    def expand(self, agent):
        children = np.array(self.state.get_bitboard_children())
        
        if len(children) == 0:
            return self.state.get_result(self.state.player)
            
        # v is the value, and p is the probabilities. We remeber them to then train the network
        v, self.edges, p = decode_output(agent.brain.predict(encode_input(self)), self, children)
        return v
        
def encode_input(node):
    arr = np.zeros(input_shape)
    n = node
    inversion = False
    
    for nbrnode in range(nbr_shapes):
        state = n.state
        bitfilter = 1
        
        for i in range(8):
            for j in range(8):
                shift = shape_length * nbrnode
                counter = 0
                player = state.player * -1 # we start with the opposite player to easily have the previous moves
                for ptype in range(colorToIndex(player), 12, 2):
                    arr[shift + counter][i][j] = min(1, bitfilter & state.bitboard[ptype])  # 0 or 1 not 255 or smg
                    counter += 1
                    
                arr[shift + 6][i][j] = player
                arr[shift + 7][i][j] = state.check_type[colorToIndex(player)]
                arr[shift + 8][i][j] = state.counter
                bitfilter = bitfilter << 1
        

        if n.parent != None:
            n = n.parent
        # if we are at the begining of the game, we still have to process one more layer to have both colors
        elif nbrnode == 0:
            n.state.player *= -1
            inversion = True
        else:
            break
    
    if inversion:
        n.state.player *= -1  # we invert a second time so it doesn't change because object are passed by reference
        
    arr = arr.reshape((1, 72, 8, 8))
    return arr
            
def decode_output(nn_out, node, children):
    v = nn_out[1][0][0]  # table inside table inside big table
    p = nn_out[0][0]     # table inside table
    e = []
    s0 = node.state
    shift = colorToIndex(s0.player)  # player gives the next player but we want the actual player
    
    for child in children:
        cnode = Node(child, node)
        
        # Promotion
        if check_promotion(s0.bitboard[int(wpawn) + shift], child.bitboard[int(wpawn) + shift]):
            
            before = s0.bitboard[int(wpawn) + shift] ^ child.bitboard[int(wpawn) + shift]
            # queen
            if s0.bitboard[int(wqueen) + shift] ^ child.bitboard[int(wqueen) + shift] > 0:
                after = s0.bitboard[int(wqueen) + shift] ^ child.bitboard[int(wqueen) + shift]
            # knight
            elif s0.bitboard[int(wknight) + shift] ^ child.bitboard[int(wknight) + shift] > 0:
                after = s0.bitboard[int(wknight) + shift] ^ child.bitboard[int(wknight) + shift]
            # rook
            elif s0.bitboard[int(wrook) + shift] ^ child.bitboard[int(wrook) + shift] > 0:
                after = s0.bitboard[int(wrook) + shift] ^ child.bitboard[int(wrook) + shift]
            # bishop
            elif s0.bitboard[int(wbishop) + shift] ^ child.bitboard[int(wbishop) + shift] > 0:
                after = s0.bitboard[int(wbishop) + shift] ^ child.bitboard[int(wbishop) + shift]
            else:
                raise Exception('[-] Decode output promotion')
                
            i = encode_Q_move(before, after)
            e.append(Edge( p[i], cnode, i ))
            continue
        
        # Roque
        if s0.bitboard[int(wking) + shift] != child.bitboard[int(wking) + shift] and s0.bitboard[int(wrook) + shift] != child.bitboard[int(wrook) + shift]:
            if child.bitboard[int(wking) + shift] < 16:  # Right roque
                e.append(Edge( p[4096], cnode, 4096 ))
            else:
                e.append(Edge( p[4097], cnode, 4097 ))   # Left roque
            continue
        
        # Normal moves
        for t in range(shift, 12, 2):
            if s0.bitboard[t] != child.bitboard[t]:
                if t == int(wknight) or t == int(bknight):
                    i = encode_K_move(s0.bitboard[t], child.bitboard[t])
                    e.append(Edge( p[i], cnode, i ))
                else:
                    i = encode_Q_move(s0.bitboard[t], child.bitboard[t])
                    e.append(Edge( p[i], cnode, i ))
    
    return v, e, p

def colorToIndex(color):
    if color == 1:
        return 0
    else:
        return 1
    
def printb(b):
    s = b.string_board()
    c = 0
    for i in range(8):
        for j in range(8):
            print(s[c], end="")
            c += 1
        print("")
        
def printu(nb):
    s = str(bin(nb))[2:]
    c = 0
    s = '0' * (64 - len(s)) + s
    for i in range(8):
        for j in range(8):
            print(s[c], end="")
            c += 1
        print()