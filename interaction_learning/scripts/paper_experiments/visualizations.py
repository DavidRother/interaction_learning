import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pickle

with open("stats/training_impact_learners.stats", 'rb') as outp:  # Overwrites any existing file.
    obj = pickle.load(outp)


print("done")
