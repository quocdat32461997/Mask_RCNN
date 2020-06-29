"""
utils.py - module to implement helper functions for trainig and inference
"""

""" import dependenceis """
import numpy as np

def rand(a=0, b=1):
    """
    rand - method to randomly generate float number between a and b
    Inputs:
        a - lower bound
        b - upper bound
    Outputs:
        __ - random float number between a and b
    """
    return np.random.rand()*(b-a) + a
