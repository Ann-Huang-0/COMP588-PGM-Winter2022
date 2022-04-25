import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import BeliefPropagation
from helper import *
from notears.linear import *
from linclab_utils import plot_utils
plot_utils.linclab_plt_defaults()

# controller
structure_learning = "NoTears"

# split train and test data
X = np.loadtxt("data/X.csv")
train_ratio = 0.99
n = X.shape[0]
num_nodes = X.shape[1]
idx_shuffle = np.random.permutation(n)
X_train = X[idx_shuffle[:int(n*train_ratio)], :]
X_test = X[idx_shuffle[int(n*train_ratio):], :]

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

# MAP inference
bp = BeliefPropagation(model)
MAP_inference_1_evidence(bp, X_test)

# plot discriminative accuracy
#inference_discriminative(X, method="HillClimbing")
#accuracy = np.load("data/discrim_accuracy.npy")
#plot_prediction_accuracy(accuracy, ["BayesianNetwork", "NaiveBayes", "LogisticReg", "MLP"])

'''
accuracy = np.load("data_pgmpy/MAP_1_evi.npy")
plt.imshow(accuracy[:,:,0])
plt.show()

A = np.loadtxt("data/dag.csv", delimiter=",")
plt.imshow(A)
plt.show()

MAP_inference_increasing_evidence(bp, X_test, n_trial=5)
plot_log_prob([1,5,10])
'''