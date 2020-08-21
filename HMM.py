#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  17 10:20:15 2019

@authors: ana fisher
"""

import numpy as np
import pandas as pd
import os
from collections import Counter

class HMM:
    
    
    def __init__(self, data):
        
        assert isinstance(data, pd.DataFrame), 'Pandas DataFrame obeject required'
        
        self.data = data
        self.data.columns = ["state", "observation"]
        self.a_ij = self.get_transition()
        self.b_ij = self.get_emission()
        n_states = self.a_ij.shape[0]
        self.pi = np.ones(n_states)/n_states #set initial probabilities to equal values
    
    def create_matrix(self, row, col):
        
        counts = Counter(tuple(zip(row, col)))
        mtx = pd.Series(counts).unstack()
        mtx = mtx.apply(lambda x: x/mtx.sum(axis=1), axis = 0)
        
        return mtx
            
    def get_transition(self):

        a_i = self.data.state[:-1]
        a_j = self.data.state[1:]

        return self.create_matrix(a_i, a_j)
    
    
    def get_emission(self):
        
        b_i = self.data.state
        b_j = self.data.observation

        return self.create_matrix(b_i, b_j)
    
    def set_pi(self, init_state):
        
        self.pi = init_state
        return self

    
    def forward(self, oserv_sequence):
        
        fwd = np.zeros((self.a_ij.shape[0], len(oserv_sequence)))
                
        for i, state in enumerate(self.a_ij.index):
            fwd[i, 0] = self.pi[i]*self.b_ij.at[state, oserv_sequence[0]]
        
        for t in range(1, len(oserv_sequence)):
            for i, new_state in enumerate(self.a_ij.index):
                transit = self.a_ij.loc[:, new_state]
                emit = self.b_ij.at[new_state, oserv_sequence[t]]
                fwd[i, t] = sum(fwd[:, t-1]*transit*emit)

        forwardprob = np.sum(fwd[:, -1])

        return forwardprob
    
    def decoding(self, oserv_sequence):
        
        viterbi = np.zeros((self.a_ij.shape[0], len(oserv_sequence)))
        backpointer = np.zeros((self.a_ij.shape[0], len(oserv_sequence)), dtype = int)
        
        for i, state in enumerate(self.a_ij.index):
            viterbi[i, 0] = self.pi[i]*self.b_ij.at[state, oserv_sequence[0]]
            backpointer[i, 0] = 0
            
        for t in range(1, len(oserv_sequence)):
            for i, new_state in enumerate(self.a_ij.index):
                transit = self.a_ij.loc[:, new_state]
                emit = self.b_ij.at[new_state, oserv_sequence[t]]
                viterbi[i, t] = np.max(viterbi[:, t-1]*transit*emit)
                backpointer[i, t] = np.argmax((viterbi[:, t-1]*transit*emit).values)
                
        bestprob = np.max(viterbi[:, -1])
        
        #backtracing the matrix of backpointers
        ptr = np.argmax(viterbi[:, -1])
        bestpath = list()
        bestpath.append(self.a_ij.index.values[ptr])
        
        for i in range(backpointer.shape[1]-1, 0, -1):
            beststate = backpointer[ptr, i]
            bestpath.append(self.a_ij.index.values[beststate])
            ptr = backpointer[ptr, i]
        
        bestpath.reverse()
        
        return bestprob, bestpath
            
    
if __name__ == "__main__":
    
    def data_loader(path):
        
        data = pd.read_csv(path, sep = ",", header = None)
        data.columns = ["state", "observation"]
        
        return data


    path = os.getcwd()
    
    data = data_loader(os.path.join(path, "HMMData.txt"))
    hmm = HMM(data)
    
    obs_sequence = ['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
    init_state = [0, 0, 1]
    hmm = hmm.set_pi(init_state)
        
    print(hmm.a_ij)
    print(hmm.b_ij)
    print('Initial State: ', hmm.pi)
    
    print('Sequence probability: ', hmm.forward(obs_sequence))
    prob, seq = hmm.decoding(obs_sequence)
    print('Most probable sequence is: ', seq)
    print('It has probability of: ', prob)
    


    

