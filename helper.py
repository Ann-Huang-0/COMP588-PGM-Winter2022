from json import load
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pomegranate.BayesianNetwork import *
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import BeliefPropagation
import tensorflow as tf
from notears.linear import notears_linear
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
    print(freqs)
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
    return ["worry", "restless", "fatigue", "irritable", "distract", "tense", 
             "sleep", "heart", "sweat", "tremble", "dryMouth", "sad"]
        
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
    print(structure)
    model = BayesianNetwork.from_structure(X=X, structure=structure, pseudocount=0, name="Anxiety Symptom Network", 
                           state_names=node_names)
    return model
    

# ============================================================
# Helper Functions for Inference (Pomogranate)
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


def calculate_thresholded_accuracy(y_pred, y_true, target, thresh_high=0.5):
    y_pred_prob = extract_prob(y_pred, target)
    #print(y_pred_prob)
    above = y_pred_prob > thresh_high
    below = y_pred_prob < 1-thresh_high
    confident_idx = np.logical_or(above, below)
    #print(y_pred_prob[confident_idx])
    prediction = np.zeros(y_pred_prob.shape[0])
    prediction[above] = 1
    correct_idx = y_true == prediction
    #print(y_pred_prob[np.logical_and(confident_idx, correct_idx)])
    print("Correct confidence: ", np.sum(y_pred_prob[np.logical_and(confident_idx, correct_idx)]))
    return np.mean(y_true[confident_idx] == prediction[confident_idx])
    
    
def extract_prob(y_pred, target):
    y_pred_prob = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        dist = y_pred[i, target].values()
        y_pred_prob[i] = dist[1]
    return y_pred_prob


def calc_correct_confidence(corr_idx, prob_pred):
    return np.sum(prob_pred[corr_idx])




# ============================================================
# Helper Functions (Pgmpy)
# ============================================================    

def convert_adj_to_edge_list(A=[]):
    node_names = load_node_names()
    if A == []:
        A = np.loadtxt("data/dag.csv", delimiter=",")
    G = nx.DiGraph(incoming_graph_data=A)
    edgelist = list(G.edges())
    return [(node_names[edge[0]], node_names[edge[1]]) for edge in edgelist]


def MAP_inference_1_evidence(bp, data, n_repeat=5):
    n, num_nodes = data.shape
    node_names = load_node_names()
    accuracy = np.zeros((num_nodes, num_nodes, n_repeat)) 
    for rep in range(n_repeat):
        for evi in range(num_nodes):
            pred = np.zeros((n, num_nodes-1))
            target = copy.deepcopy(node_names)
            target.remove(node_names[evi])
            target_idx = list(np.arange(num_nodes))
            target_idx.remove(evi)
            for i in range(n):
                # create a dictionary representing the evidence
                evidence = {node_names[evi]: data[i, evi]}
                map_pred = bp.map_query(variables=target, evidence=evidence, show_progress=False)
                pred[i, :] = np.array(list(map_pred.values()))
            accuracy[evi, target_idx, rep] = np.mean(data[:, target_idx] == pred, axis=0)
            print(accuracy[evi, target_idx, rep])
    np.save("data_pgmpy/MAP_1_evi", accuracy)         
                
                
def MAP_inference_11_evidence(bp, data, n_repeat=5):
    n, num_nodes = data.shape
    node_names = load_node_names()
    accuracy = np.zeros((num_nodes, n_repeat)) 
    for rep in range(n_repeat):
        for target in range(num_nodes):
            pred = np.zeros(n)
            for i in range(n):
                evidence = {}
                for evi in range(num_nodes):
                    if evi != target:
                        evidence[node_names[evi]] = data[i, evi]
                map_pred = bp.map_query(variables=[node_names[target]], evidence=evidence, show_progress=False)
                pred[i] = np.array(list(map_pred.values()))
            accuracy[target, rep] = np.mean(data[:, target] == pred, axis=0)
            print(accuracy[target, rep])
    print("Overall accuracy:", np.mean(accuracy))
    np.save("data_pgmpy/MAP_11_evi", accuracy)       
    
 
def inference_discriminative(X, n_repeat=5):
    train_ratio = 0.95
    n = X.shape[0]
    num_nodes = X.shape[1]
    node_names = load_node_names()
    accuracy = np.zeros((n_repeat, num_nodes))
    for rep in range(n_repeat):
        idx_shuffle = np.random.permutation(n)
        X_train = X[idx_shuffle[:int(n*train_ratio)], :]
        X_test = X[idx_shuffle[int(n*train_ratio):], :] 
        X_test_pred = np.zeros(X_test.shape[0])
        for target in range(num_nodes):
            # seperate positive and negative cases
            X_pos = X_train[X_train[:,target]==1, :]
            X_neg = X_train[X_train[:,target]==0, :]
            # mix one case of the other class
            X_pos = np.vstack((X_pos, X_neg[-1,:]))
            X_neg = np.vstack((X_neg, X_pos[0,:]))
            # train two seperate Bayesian network
            bp_pos = create_bp_object(X_pos)
            bp_neg = create_bp_object(X_neg)
            # inference
            for i in range(X_test.shape[0]):
                evidence = {}
                for evi in range(num_nodes):
                    if evi != target:
                        evidence[node_names[evi]] = X_test[i, evi]
                pred_pos = bp_pos.query(variables=[node_names[target]], evidence=evidence, show_progress=False)
                prob_pos = pred_pos.values[1]
                pred_neg = bp_neg.query(variables=[node_names[target]], evidence=evidence, show_progress=False)
                prob_neg = pred_neg.values[0]
                X_test_pred[i] = prob_pos > prob_neg
            accuracy[rep,target] = np.maximum([1-np.mean(X_test_pred)], [np.mean(X_test_pred)])
            print("Predicting symptom", node_names[target], ":", accuracy[rep, target])
    np.save("data_pgmpy/discr", accuracy)
                          
            
def create_bp_object(X):
    dag = notears_linear(X, lambda1=0.01, loss_type="logistic")
    edges = convert_adj_to_edge_list(dag)
    model = BayesianModel(edges)
    X = pd.DataFrame(X, columns=load_node_names())
    model.fit(data=X, estimator=BayesianEstimator, prior_type="BDeu")
    bp = BeliefPropagation(model)
    return bp