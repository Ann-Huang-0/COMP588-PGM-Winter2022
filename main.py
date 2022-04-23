import pandas as pd
import torch
from notears.linear import notears_linear
from notears.nonlinear import *
import numpy as np
from notears import *
import networkx as nx
import matplotlib.pyplot as plt
from helper import *
from pomegranate import *
from inference import *
from pomegranate.BayesianNetwork import *


# Hyperparameters
relearn_structure = True
linear, nonlinear = [True, False]
recalculate_accuracy = False
num_var = 12
inference_discrim = False
inference_missing_data = True

# Strcuture learning via NoTears
if relearn_structure:
    X = np.loadtxt("data/X.csv")
    if linear:
        dag = notears_linear(X, lambda1=0.01, loss_type="logistic")
        np.savetxt("data/dag.csv", dag, delimiter=',')
    elif nonlinear:
        torch.set_default_dtype(torch.double)
        np.set_printoptions(precision=3)
        mlp = NotearsMLP(dims=[num_var, 50, 1], bias=True)
        dag = notears_nonlinear(mlp, X, lambda1=0.001, lambda2=0.001)
        np.savetxt("data/dag.csv", dag, delimiter=',')
     
# Visualize the learned Bayesian network 
visualize_symptom_network(disorder="anxiety")

'''
# Inference
MAP_varying_num_evidence("notears", [10])
#predict_prob_varying_num_evidence("chowliu", [8])
#discriminative_notears()

X = np.loadtxt("data/X.csv")
BayesNet = symptom_net_parameter_learning(X)
#print(BayesNet.log_probability(X))
print(BayesNet)


if inference_discrim:
    all_methods = ["BayesNet_notears", "BayesNet_chowliu", "NaiveBayes", "LogisticRegression", "MLP"]
    if recalculate_accuracy:
        num_repeat = 10
        accuracy = np.zeros((len(all_methods), num_var, num_repeat))
        for i, method in enumerate(all_methods):
            for repeat in range(num_repeat):
                accuracy[i,:,repeat] = discriminative(method)
        np.save("data/discrim_accuracy", accuracy)
    else:
        accuracy = np.load("data/discrim_accuracy.npy")
    plot_prediction_accuracy(accuracy, all_methods)
'''