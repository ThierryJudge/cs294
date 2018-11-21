import numpy as np
import pickle
import os

env = "Ant-v2"

with open(os.path.join('experts',env + ".pkl"), 'rb') as f:
    expert = pickle.load(f)


print(expert)
