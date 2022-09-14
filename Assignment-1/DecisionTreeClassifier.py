import numpy as np
import pandas as pd
from utils import calc_accuracy

class DecisionTree:
    class Node:
        def __init__(self):
            self.label = None
            self.best_feature_id = None
            self.best_feature_name = None
            self.prev_value = None
            self.parent = None
            self.children = None
            self.present_label = None


    def __init__(self,feature_names,frac):

        self.X = None
        self.labels = None
        self.X_valid = None
        self.labels_valid = None
        self.feature_names = feature_names
        self.frac = frac
        self.labelCategories = None
        self.root_node = None
        self.node_to_prune = None
        self.max_accuracy = None
        self.should_prune = None
        return

    def __get_entropy(self,X_ids):
        """
        Description : Calculates the entropy of given train examples (by index)
        
        Parameters:
        @param X_ids (numpy 1D array of integers) : list containing the index of training examples

        Returns:
        @return entropy (int) : the calculated entropy of the given training examples
        
        """

        labels = np.array([self.labels[i] for i in X_ids])
        labels_unique = np.unique(labels)
        labelCounts = np.array([(labels == val).sum() for val in labels_unique])             
        labelProportion = labelCounts / labels.size           
        entropy = -(labelProportion @ np.log(labelProportion).T) # entropy = sum(-)

        return entropy
    
    def __get_information_gain(self,X_ids,feature_id):
        """
        Description : Calculates the information gain of a given feature on a given set of training examples

        Parameters:
        @param X_ids (numpy 1D array of integers) : list containing the index of training examples
        @param feature_id (int) : ID of feature

        Returns:
        @return info_gain (int) : The information gain of given

        """

        entropy_prev = self.__get_entropy(X_ids)

        feature_vals = np.array([self.X[row][feature_id] for row in X_ids])
        feature_unique = np.unique(feature_vals)
        feature_val_count = np.array([(feature_vals == val).sum() for val in feature_unique]) 
        feature_value_ids = [[row for row in X_ids if self.X[row][feature_id] == feature_val] for feature_val in feature_unique]
        feature_entropies = np.array([self.__get_entropy(val_ids) for val_ids in feature_value_ids])
        entropy_pres = (feature_val_count @ feature_entropies.T) / len(X_ids)
        
        info_gain = entropy_prev - entropy_pres

        return info_gain

    def __get_best_feature(self,X_ids,feature_ids):
        """
        Description : Finds feature with maximum information gain
        
        Parameters:
        @param X_ids (numpy 1D array of int) : list containing the index of training examples
        @param feature_ids (numpy 1D array of int) : ID of features

        Return:
        @param best_feature,best_feature_name (int,string) : ID and Name of feature with maximum information gain

        """

        max_info_gain_array = np.array([self.__get_information_gain(X_ids,feature_id) for feature_id in feature_ids])
        best_feature = feature_ids[np.argmax(max_info_gain_array)]

        return best_feature ,self.feature_names[best_feature]

    
    def fit(self,X,labels):
        """
        Description : Initializes ID3 algorithm to build Decision Tree Classifier

        """
        num_examples = len(labels)
        self.X = X[:int(num_examples*self.frac),:]
        self.X_valid = X[int(num_examples*self.frac):,:]
        self.labels = labels[:int(num_examples*self.frac)]
        self.labels_valid = labels[int(num_examples*self.frac):]        
        self.labelCategories = np.unique(labels)
        self.root_node = None    
        self.max_accuracy = None
        self.node_to_prune = None
        self.should_prune = False    
        X_ids = [i for i in range(len(self.X))]
        feature_ids = [i for i in range(len(self.feature_names))]
        self.root_node = self.__id3_recv(X_ids, feature_ids,self.root_node)
        self.max_accuracy = calc_accuracy(self.labels_valid,self.predict(self.X_valid))
        # print(f"Accuracy before prunning is {self.max_accuracy}")
        # print("Pruning Started")
        # count = 0
        # while True:
        #     print('inside while true')
        #     self.prune(self.root_node)
        #     if self.should_prune:
        #         print('inside if should prune')
        #         count += 1
        #         self.node_to_prune.children = [] 
        #         self.should_prune = False
        #         self.node_to_prune = None
        #     else:
        #         break
        # print(f"Pruning Complete - pruned {count} nodes")
        # print(f"Accuracy after prunning is {self.max_accuracy}")
        return

    def __id3_recv(self,X_ids,feature_ids,node):
        """
        Description : Recursive function to build Decision Tree 
        
        Parameters:
        @param X_ids (numpy 1D array of int) : list containing the index of training examples
        @param feature_ids (numpy 1D array of int) : ID of features
        @param node (an instance of node class) : Present node we are building on

        Returns:
        @return node (an instance of node class) : Present node after building the subtree on this node 
        
        """

        if not node:
            node = self.Node()
        labels = np.array([self.labels[row] for row in X_ids])
        labels_unique = np.unique(labels)
        labelCounts = np.array([(labels == val).sum() for val in labels_unique])
        if len(labels_unique) == 1:
            # print("len(labels_unique) == 1")
            node.label = labels_unique[0]
            return node
        
        if len(feature_ids) == 0:
            # print("len(feature_ids) == 0")
            node.label = labels_unique[np.argmax(labelCounts)]
            return node

        best_feature ,best_feature_name = self.__get_best_feature(X_ids,feature_ids)
        node.best_feature_name = best_feature_name
        node.best_feature_id = best_feature
        node.present_label = labels_unique[np.argmax(labelCounts)]
        node.children = []
        
        feature_ids = np.delete(feature_ids,np.where(feature_ids == best_feature))
        # print(len(feature_ids))
        best_feature_values = np.unique([self.X[row][best_feature] for row in X_ids])

        for val in best_feature_values:
            child = self.Node()
            child.prev_value = val
            child.parent = node
            child_X_ids = np.array([row for row in X_ids if self.X[row][best_feature] == val])
            child.children = []
            # if (not child_X_ids) or len(child_X_ids) == 0:
            if len(child_X_ids) == 0:
                child.label = node.present_label
            else:
                child = self.__id3_recv(child_X_ids,feature_ids,child)
            
            node.children.append(child)
        
        return node

    def predict(self,X_test):
        """
        Description : function to return predictions on given test examples

        Parameters:
        @params X_test (a 2D numpy array, each row is a test example) : the test examples

        Returns:
        @return labels_test (1D numpy array) : the predicted labels by the model
        
        """
        labels_test = []
        a = 0
        b = 0
        for row in X_test:
            # print(row)
            # print(a)
            a = a + 1
            node = self.root_node
            print(node.best_feature_name)
            while node.label == None:
                found = False
                for child in node.children:
                    print(node.best_feature_name)
                    if row[node.best_feature_id] == child.prev_value:
                        node = child
                        found = True
                        break
                if not found:
                    break

            if node.label is None:
                ans = node.present_label
                # b = b + 1
            else:
                ans = node.label
            print("*"*50)
            labels_test.append(ans)
        # print(f"{b} nahi mil paye bhidu")
        return np.array(labels_test)

    def prune(self,node_to_prune):
        
        if node_to_prune.children == [] or node_to_prune.children ==  None:
            print("At leaf")
            return
        print(f'inside prune on node with feature {node_to_prune.best_feature_name}')
        for child_node in node_to_prune.children:
            self.prune(child_node)
        children_buffer = node_to_prune.children
        node_to_prune.children = []
        label_valid_pred = self.predict(self.X_valid)
        accuracy = calc_accuracy(self.labels_valid,label_valid_pred)
        node_to_prune.children = children_buffer
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.node_to_prune = node_to_prune
            self.should_prune = True

        







