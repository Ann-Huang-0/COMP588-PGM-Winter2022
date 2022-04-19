import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pomegranate.BayesianNetwork import *

def inference_in_symptom_net(model):
    X = np.loadtxt("X.csv")
