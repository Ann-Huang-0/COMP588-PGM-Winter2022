from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pomegranate.BayesianNetwork import *
from helper import *
from sklearn.model_selection import KFold

# ================================================================================
#               Generative problem -- predict symptom under missing data
# ================================================================================

def prediction_varying_num_evidence(method, n_evidence_array):
    X_true =  np.loadtxt("data/X.csv")
    num_nodes = X_true.shape[1]
    if method == "notears":
        BayesNet = symptom_net_parameter_learning(X_true)
    elif method == "chowliu":
        BayesNet = BayesianNetwork.from_samples(X_true, algorithm="chow-liu", pseudocount=5, name="Anxiety Symptom Network")   
    
    accuracy = np.zeros((len(n_evidence_array), num_nodes))
    for i, n_evidence in enumerate(n_evidence_array):
        accuracy[i,:] = predict_single_symptom(BayesNet, n_evidence=n_evidence)
        print("Number of evidence =", str(n_evidence), "finished.")
    np.save("data/accuracy_diff_num_evi.npy", accuracy) 
        

def predict_single_symptom(BayesNet, n_evidence=1, acc_map=False, n_split=5):
    X_true =  np.loadtxt("data/X.csv")
    num_nodes = X_true.shape[1] 
    if n_evidence == 1 and acc_map == True:
        kf = KFold(n_splits=n_split)
        accuracy = np.zeros((num_nodes, num_nodes, kf.get_n_splits()))
        i_fold = 0
        for train_idx, test_idx in kf.split(X_true):     # 10-fold cross validation
            X_train, X_test = X_true[train_idx,:], X_true[test_idx,:]
            # structure and parameter learning of the symptom network
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


def discriminative_totears():
    pass
    
    
def discriminative_chowliu():
    pass