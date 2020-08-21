# Hidden Markov Model for a discrete dataset

Implementation of a simple HMM model for a discrete dataset in Python from scratch

## Dependancies
Python > 2.7  
numpy toolkit
pandas toolkit

## Dataset
The dataset (HMMData.txt) contains 1000 rows of past weather observations. The states (ω) are “sunny”, “rainy” and “foggy”. The emission states are “yes”, “no” indicating if an umbrella was observed.

## Code overview
HMM class takes an observation sequence <img src="https://render.githubusercontent.com/render/math?math=V^T"> as an input and outputs
- The state matrix and all needed probabilities such as 𝑎𝑖𝑗 𝑎𝑛𝑑 𝑏𝑗𝑘 need to be calculated from given data
- The probability of the given observation using Viterbi algorithm.
- The most probable path to generate the given observations using Viterbi algorithm
