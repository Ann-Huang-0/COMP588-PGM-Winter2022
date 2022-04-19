import pandas as pd
from notears.notears.linear import notears_linear
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
recalculate_accuracy = False
num_var = 12


# Strcuture learning via NoTears
if relearn_structure:
    X = np.loadtxt("data/X.csv")
    dag = notears_linear(X, lambda1=0.02, loss_type="logistic")
    np.savetxt("data/dag.csv", dag, delimiter=',')

# Visualize the learned Bayesian network 
visualize_symptom_network(disorder="anxiety")
'''
# MAP inference
#all_methods = ["BayesNet_notears", "BayesNet_chowliu", "NaiveBayes", "LogisticRegression", "MLP", "NeuralNet"]
all_methods = ["BayesNet_notears", "BayesNet_chowliu", "NaiveBayes", "LogisticRegression", "MLP"]
if recalculate_accuracy:
    num_repeat = 10
    accuracy = np.zeros((len(all_methods), num_var, num_repeat))
    for i, method in enumerate(all_methods):
        for repeat in range(num_repeat):
            accuracy[i,:,repeat] = inference_single_missing_var(method)
    np.save("data/prediction_accuracy", accuracy)
else:
    accuracy = np.load("data/prediction_accuracy.npy")
#plot_prediction_accuracy(accuracy, all_methods)

logp, p = calculate_log_probability()
print(logp)

plot_symptom_stats()
'''