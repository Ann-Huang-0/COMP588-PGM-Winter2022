from cgi import test
from platform import node
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pomegranate.BayesianNetwork import *
from helper import *
from notears import *
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear
from sklearn.model_selection import KFold

# ================================================================================
#               Generative problem -- predict symptom under missing data
# ================================================================================

def predict_prob_varying_num_evidence(method, n_evidence_array):
    X_true =  np.loadtxt("data/X.csv")
    num_nodes = X_true.shape[1]
    if method == "notears":
        BayesNet = symptom_net_parameter_learning(X_true)
    elif method == "chowliu":
        BayesNet = BayesianNetwork.from_samples(X_true, algorithm="chow-liu", pseudocount=5, name="Anxiety Symptom Network")
         
    accuracy = np.zeros((len(n_evidence_array), num_nodes))
    for i, n_evidence in enumerate(n_evidence_array):
        accuracy[i,:] = predict_prob_single_symptom(BayesNet, X_true, n_evidence)
        print("Number of evidence =", str(n_evidence), "finished.")
    np.save("data/predict_prob.npy", accuracy) 
             
    

def predict_prob_single_symptom(BayesNet, X_true, n_evidence):
    num_nodes = X_true.shape[1]
    accuracy = np.zeros(num_nodes)
    for target in range(num_nodes): 
        X_test = copy.deepcopy(X_true)
        X_test = make_masked_matrix_for_prediction(X_test, target, n_evidence)
        #X_test = X_test[:1,:]
        #print(X_test)
        y_test_hat = np.array(BayesNet.predict_proba(X_test))
        accuracy[target] = calculate_thresholded_accuracy(y_test_hat, X_true[:,target], target)
        print("Prediction accuracy for symptom", str(target), ":", accuracy[target])
    return accuracy   


def MAP_varying_num_evidence(method, n_evidence_array):
    X_true =  np.loadtxt("data/X.csv")
    num_nodes = X_true.shape[1]
    if method == "notears":
        BayesNet = symptom_net_parameter_learning(X_true)
        print(BayesNet)
    elif method == "chowliu":
        BayesNet = BayesianNetwork.from_samples(X_true, algorithm="chow-liu", pseudocount=5, name="Anxiety Symptom Network")   
    
    accuracy = np.zeros((len(n_evidence_array), num_nodes))
    for i, n_evidence in enumerate(n_evidence_array):
        accuracy[i,:] = MAP_single_symptom(BayesNet, n_evidence=n_evidence)
        print("Number of evidence =", str(n_evidence), "finished.")
    np.save("data/accuracy_diff_num_evi.npy", accuracy) 
        

def MAP_single_symptom(BayesNet, n_evidence=1, acc_map=False, n_split=5):
    X_true =  np.loadtxt("data/X.csv")
    num_nodes = X_true.shape[1] 
    if n_evidence == 1 and acc_map == True:
        kf = KFold(n_splits=n_split)
        accuracy = np.zeros((num_nodes, num_nodes, kf.get_n_splits()))
        i_fold = 0
        for train_idx, test_idx in kf.split(X_true):     # 10-fold cross validation
            X_train, X_test = X_true[train_idx,:], X_true[test_idx,:]
            # parameter learning of the symptom network
            if method == "notears":
                BayesNet = symptom_net_parameter_learning(X_train)
            elif method == "chowliu":
                BayesNet = BayesianNetwork.from_samples(X_train, algorithm="chow-liu", pseudocount=5, name="Anxiety Symptom Network")  
            # inference
            for evidence in range(num_nodes):
                for mask in range(num_nodes):
                    if evidence != mask:
                        # mask every other variance except evidence
                        groundtruth = copy.deepcopy(X_test[:,mask])
                        var_masked = list(np.arange(num_nodes))
                        var_masked.remove(evidence)
                        X_test_, y_test = mask_samples(X_test, var_masked, del_var_column=False)
                        y_test_hat = np.array(BayesNet.predict(X_test_))
                        accuracy[evidence, mask, i_fold] = np.mean(np.array(y_test_hat[:,mask]) == np.array(groundtruth))
                        print("accuracy: ",  accuracy[evidence, mask, i_fold])
                print("Symptom", str(evidence+1), "finished as the evidence.")
            i_fold += 1
            if i_fold >= 1:
                break
            print("Split", str(i_fold+1), "is completed.")
        np.save("data/121_accuracy", accuracy)           
    
    else:
        accuracy = np.zeros(num_nodes)
        for target in range(num_nodes): 
            X_test = copy.deepcopy(X_true)
            X_test = make_masked_matrix_for_prediction(X_test, target, n_evidence)
            y_test_hat = np.array(BayesNet.predict(X_test))
            accuracy[target] = np.mean(np.array(y_test_hat[:,target]) == np.array(X_true[:,target]))
            print("Prediction accuracy for symptom", str(target), ":", accuracy[target])
        #np.save("data/921_accuracy.npy", accuracy) 
    
    return accuracy          

                
# ================================================================================
#                   Discriminative problem -- binary classification
# ================================================================================

def discriminative(method):
    X_true =  np.loadtxt("data/X.csv")
    num_nodes = X_true.shape[1]
    accuracy = np.zeros(num_nodes)
    if method in ["BayesNet_notears", "BayesNet_chowliu"]:
        for i in range(num_nodes):
            discriminative_notears(X_true, i)
            '''
            var_masked = [i]
            X = copy.deepcopy(X_true)
            X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.2)
            if method == "BayesNet_notears":
                BayesNet = symptom_net_parameter_learning(X_train)
            elif method == "BayesNet_chowliu":
                BayesNet = BayesianNetwork.from_samples(X_train, algorithm="chow-liu", pseudocount=5, name="Anxiety Symptom Network")
            X_test, y_test = mask_samples(X_test, var_masked, del_var_column=False)
            y_test_hat = np.array(BayesNet.predict(X_test))
            accuracy[i] = np.mean(y_test_hat[:,i] == y_test)
            '''

    elif method == "NaiveBayes":   
        for i in range(num_nodes):
            var_masked = [i]
            X, y = mask_samples(X_true, var_masked)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
            clf = BernoulliNB(fit_prior=True)
            clf.fit(X_train, y_train)
            accuracy[i] = clf.score(X_test, y_test)
 
    elif method == "LogisticRegression":
        for i in range(num_nodes):
            var_masked = [i]
            X, y = mask_samples(X_true, var_masked)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
            clf = LogisticRegression(random_state=0).fit(X_train, y_train)
            accuracy[i] = clf.score(X_test, y_test)
    
    elif method == "MLP":
        for i in range(num_nodes):
            var_masked = [i]
            X, y = mask_samples(X_true, var_masked)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 1), max_iter=500, random_state=1)
            clf.fit(X_train, y_train)
            accuracy[i] = clf.score(X_test, y_test)
        
    elif method == "NeuralNet":
        for i in range(num_nodes):
            var_masked = [i]
            X, y = mask_samples(X_true, var_masked)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = tf.keras.Sequential([
                    tf.keras.layers.Dense(11),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(1, activation="softmax")]) 
            model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']) 
            model.fit(X_train, y_train, verbose=0)
            model.evaluate(X_test, y_test)
    
    return accuracy    


def discriminative_notears():
    X_true =  np.loadtxt("data/X.csv")
    node_names = load_node_names()
    num_nodes = len(node_names)
    # seperate the positive and negative cases
    X_pos_pred = np.zeros(X_true.shape)
    X_neg_pred = np.zeros(X_true.shape)
    accuracy = np.zeros(X_true.shape[1])
    for target in range(num_nodes):
        X_pos = copy.deepcopy(X_true[X_true[:,target]==1,:])
        X_neg = copy.deepcopy(X_true[X_true[:,target]==0,:])
        X_pos_pred[:,target] = learn_and_infer(X_true, X_pos, target)
        X_neg_pred[:,target] = learn_and_infer(X_true, X_neg, target)
        print(X_pos_pred[:,target])
        print(X_neg_pred[:,target])
        prediction_target =  X_pos_pred[:,target] > X_neg_pred[:,target]
        print("Accuracy for symptom", str(target), ":", np.mean(prediction_target == X_true[:,target], axis=0))
    # see which network gives a higher probability
    prediction = X_pos_pred > X_neg_pred 
    # calculate prediction accuracy
    accuracy = np.mean(prediction == X_true, axis=0)
    # plot accuracy
    print(accuracy)
    

    
def learn_and_infer(X, X_subset, target):   
    node_names = load_node_names()
    num_nodes = len(node_names) 
    predict_prob = np.zeros(X.shape[0])
        
    dag = notears_linear(X_subset, lambda1=0.01, loss_type="logistic")
    structure = []
    for node in range(num_nodes):
        parents = np.nonzero(dag[:,node])[0]
        structure.append(tuple(parents))
    model = BayesianNetwork.from_structure(X=X_subset, structure=structure, pseudocount=5, state_names=node_names)
        
    X_test = copy.deepcopy(X)
    groundtruth = copy.deepcopy(X[:,target])
    X_test_, y_test = mask_samples(X_test, [target], del_var_column=False)
    y_test_hat = np.array(model.predict_proba(X_test_))
    for i in range(X.shape[0]):
        print(y_test_hat[i, target].values())
        predict_prob[i] = y_test_hat[i, target].values()[0]
    return predict_prob
            
    
    
def discriminative_chowliu():
    pass