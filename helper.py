import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pomegranate import *
from pomegranate.BayesianNetwork import *
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import tensorflow as tf
import copy


# ============================================================
# Helper Functions for Visualizing Data Statistics
# ============================================================

def plot_symptom_stats():
    anx_freqs = pd.read_csv('NCSR/anx_freqs.csv').to_numpy()
    anx_SOI_idx = anx_freqs[2,1:]<0.8
    freqs = copy.deepcopy(anx_freqs[:2, 1:])
    freqs = freqs[:, anx_SOI_idx]
    freqs_sum = np.sum(freqs, axis=0)
    freqs = freqs / freqs_sum
    node_names = ["worry", "restless", "fatigue", "irritable", "distract", "tense", 
             "sleep", "heart", "sweat", "tremble", "dryMouth", "sad"]
    width = 0.5

    fig, ax = plt.subplots()
    ax.bar(node_names, freqs[0,:], width,  label='Presence', color="indianred")
    ax.bar(node_names, freqs[1,:], width, bottom=freqs[0,:], label='Absence', color="steelblue")
    ax.set_xticks(np.arange(len(node_names)))
    ax.set_xticklabels(node_names, fontsize=11)
    ax.set_yticks(np.linspace(0,10,num=6)/10)
    ax.set_yticklabels(np.linspace(0,10,num=6)/10, fontsize=12)
    ax.set_xlabel('Symptom', fontsize=13)
    #ax.set_xtitle()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)
    remove_figure_border(ax)
    plt.show()


# ============================================================
# Helper Functions for Bayesian Network Visualization
# ============================================================

def get_edge_color(G, A):
    edgelist = list(G.edges())
    edgeweight = [A[edge[0], edge[1]] for edge in edgelist]
    edgesign = [weight>0 for weight in edgeweight] * 1
    edgecolor = []
    for sign in edgesign:
        if sign == 1:
            edgecolor.append('indianred')
        else:
            edgecolor.append('steelblue')
    return edgecolor

def get_edge_width(G, A, gain=4):
    edgelist = list(G.edges())
    edgewidth = np.array([abs(A[edge[0], edge[1]]) for edge in edgelist])
    edgewidth *= gain
    return edgewidth

def get_node_size(disorder='anxiety', gain=20000):
    if disorder == 'anxiety':
        anx_freqs = pd.read_csv('NCSR/anx_freqs.csv').to_numpy()
        anx_SOI_idx = anx_freqs[2,1:]<0.8
        freq_vector = anx_freqs[0,1:]
        node_size = np.array(freq_vector[anx_SOI_idx]) * gain
        node_size_int = [int(x) for x in node_size]
        return node_size_int
   
def load_node_names():
    return ["worry", "restless", "fatigue", "irritable", "distract", "tense muscles", 
             "sleep problems", "heart racing", "sweat", "tremble", "dry mouth", "sad"]
        
def get_node_name(disorder='anxiety'):
    if disorder == 'anxiety':
        node_name = load_node_names()
        node_name_dict = {}
        for i, name in enumerate(node_name):
            node_name_dict[i] = name
        return node_name_dict
    
def visualize_symptom_network(disorder="anxiety"):
    if disorder == "anxiety":
        A = np.loadtxt("data/dag.csv", delimiter=",")
        G = nx.DiGraph(incoming_graph_data=A)
            
        edge_color = get_edge_color(G, A)
        edge_width = get_edge_width(G, A)
        node_name = get_node_name(disorder='anxiety')
        node_size = get_node_size(disorder='anxiety')    

        fig, axs = plt.subplots()
        nx.draw_shell(G, with_labels=True, labels=node_name, node_size=node_size, width=edge_width, edge_color=edge_color,
                    arrowsize=20, arrowstyle="->", node_color="lightgray")
        plt.show()
        
        
# ==================================================================================
# Helper Functions for Import Learned Bayesian Network to Pomogranate
# ==================================================================================

def get_Bayesian_net_structure(disorder="anxiety"):
    if disorder == "anxiety":
        A = np.loadtxt("data/dag.csv", delimiter=",")
        num_nodes = A.shape[0]
        structure = []
        for node in range(num_nodes):
            parents = np.nonzero(A[:,node])[0]
            structure.append(tuple(parents))
        return tuple(structure)

def symptom_net_parameter_learning(X, disorder="anxiety", count=5):
    node_names = load_node_names()
    structure = get_Bayesian_net_structure(disorder)
    model = BayesianNetwork.from_structure(X=X, structure=structure, pseudocount=count, name="Anxiety Symptom Network", 
                           state_names=node_names)
    return model
    

# ============================================================
# Helper Functions for Inference
# ============================================================    

def make_masked_matrix_for_prediction(X, target, n_evidence):
    num_nodes = X.shape[1]
    remaining = list(np.arange(num_nodes))
    remaining.remove(target)
    mask_nodes = np.random.permutation(remaining)[:n_evidence]
    mask_nodes = np.append(mask_nodes, target)
    X_test, y_test = mask_samples(X, mask_nodes, del_var_column=False)
    return X_test

def mask_samples(X, var_masked, del_var_column=True):
    y = X[:, var_masked]
    if del_var_column:  
        X_masked = [X[:,col] for col in range(X.shape[1]) if col not in var_masked]
        X_masked = np.transpose(np.array(X_masked))
    else:
        X_masked = copy.deepcopy(X)
        X_masked[:, var_masked] = np.nan
    return X_masked, y

def remove_figure_border(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True) 

def plot_prediction_accuracy(accuracy, all_methods):
    node_names = ["worry", "restless", "fatigue", "irritable", "distract", "tense", 
             "sleep", "heart", "sweat", "tremble", "dryMouth", "sad"]
    X = np.arange(len(node_names))
    fig, axs = plt.subplots()
    for i, method in enumerate(all_methods):
        axs.plot(X, np.mean(accuracy[i,:,:], axis=1), label=method, linewidth=1.5)
        axs.fill_between(X, np.mean(accuracy[i,:,:], axis=1)+np.std(accuracy[i,:,:], axis=1), 
                         np.mean(accuracy[i,:,:], axis=1)-np.std(accuracy[i,:,:], axis=1), alpha=0.25)
    remove_figure_border(axs)
    axs.legend(frameon= False, fontsize=12)
    axs.set_xticks(np.arange(len(node_names)))
    axs.set_xticklabels(node_names, fontsize=10)
    axs.set_xlabel("Symptom", fontsize=14)
    axs.set_ylabel("Prediction accuracy", fontsize=14)
    plt.show()
         
def calculate_log_probability():
    X = np.loadtxt("data/X.csv")
    BayesNet = symptom_net_parameter_learning(X)
    logp = BayesNet.log_probability(X)
    p = np.power(2,logp)
    return logp, p
    
    

def plot_acc_single_evidence():
    acc = np.load("data/121_accuracy.npy")
    anx_freqs = pd.read_csv('NCSR/anx_freqs.csv').to_numpy()
    anx_SOI_idx = anx_freqs[2,1:]<0.8
    freqs = copy.deepcopy(anx_freqs[:2, 1:])
    freqs = freqs[:, anx_SOI_idx]
    freqs_sum = np.sum(freqs, axis=0)
    freqs = freqs / freqs_sum
    max_likely_freq = np.array([np.max(freqs, axis=0) for i in range(acc.shape[0])])
    np.fill_diagonal(max_likely_freq, 0)
    marg_acc = acc[:,:,0] - max_likely_freq
    print(marg_acc)
    
    fig, ax = plt.subplots()
    im = ax.imshow(marg_acc)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(acc.shape[0]))
    ax.set_yticks(np.arange(acc.shape[0]))
    ax.set_xticklabels(load_node_names())
    ax.set_yticklabels(load_node_names())
    ax.set_xlabel("Target")
    ax.set_ylabel("Evidence")
    plt.show()
