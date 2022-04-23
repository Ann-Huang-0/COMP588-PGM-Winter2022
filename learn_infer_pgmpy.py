import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import BeliefPropagation
from helper import *
from notears.linear import *

# controller
structure_learning = "HillClimbing"

# split train and test data
X = np.loadtxt("data/X.csv")
train_ratio = 0.97
n = X.shape[0]
num_nodes = X.shape[1]
idx_shuffle = np.random.permutation(n)
X_train = X[idx_shuffle[:int(n*train_ratio)], :]
X_test = X[idx_shuffle[int(n*train_ratio):], :]
'''
node_names = load_node_names()
data_train = pd.DataFrame(X_train, columns=node_names)

# structure learning
if structure_learning == "HillClimbing":
    est = HillClimbSearch(data_train)
    model = est.estimate(scoring_method=BicScore(data_train))
    edges = model.edges()
elif structure_learning == "NoTears":
    edges = convert_adj_to_edge_list()
    
model = BayesianModel(edges)
print("Structure learning finished.")   
    
# parameter learning via Bayesian Estimation
model.fit(data=data_train, estimator=BayesianEstimator, prior_type="BDeu")
print("Parameter learning finished.")
'''
# MAP inference
#bp = BeliefPropagation(model)
#MAP_inference_11_evidence(bp, X_train)
inference_discriminative(X)
        

# predict the probability


