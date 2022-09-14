"""
    This file contains main function for Assignment-1-Q1
    Authors: Sidharth Vishwakarma (20CS10082)
             Kulkarni Pranav Suryakant (20CS30029)
"""
from cgi import test
from cmath import nan
from fileinput import close
from unicodedata import name
from helper import calc_entropy, get_accuracy, read_data, train_test_split , calc_entropy, calc_info_gain, best_attribute, fill_missing_values, test_10_random_splits, DecisionTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = 'Dataset_A.csv'
# DATA_FILE = 'test.csv'
OUT_FILE = 'output.txt'


def main():
    """
        This function is the main function for Assignment-1-Q1
    """
    # f = open(OUT_FILE, 'w')
    # data = read_data(DATA_FILE)
    # # print(data)
    # data = fill_missing_values(data)
    # train_data, test_data = train_test_split(data)
    # # print(train_data)
    # # print(test_data)
    # # print(data)
    # # print(data.loc[0])
    # tree = DecisionTree(100, 50)
    # root = tree.build_tree(train_data)
    # # print(data[0])
    # correct = 0
    # for i in range(len(test_data)):
    #     # print(test_data.iloc[i])
    #     # print("HOLA")
    #     # print(test_data.iloc[i]['Segmentation'])
    #     if(tree.predict_segement(test_data.loc[i], root) == test_data.iloc[i]['Segmentation']):
    #         correct += 1
    # # print(root)
    # print(f'Accuracy: {correct/len(test_data)}')
    # tree.print_tree(root)

    # # print(calc_info_gain(data, 'Temp'))
    # print(best_attribute(data))
    # # next_best = best_attribute(data[['Outlook','Temp','Humidity','Wind', 'PlayTennis']])
    # # for i in data[next_best].unique():
    # #     print(i)
    # #     labels = data[next_best] == i
    # #     print(best_attribute(data[labels][['Outlook','Temp','Humidity','Wind', 'PlayTennis']]))
    # #     print()

    # tempd_data = pd.cut(data['Outlook'], bins=3, labels=np.arange(3), right=False)
    # data['Outlook'] = pd.cut(data['Outlook'], 3)
    # print(data['Outlook'])
    # print(tempd_data)

    data = read_data(DATA_FILE)
    data = fill_missing_values(data)
    # accuracy1, best_tree, max_accuracy, validation_data, test_data = test_10_random_splits(data)
    # print(f'Accuracy: {accuracy1}')
    # print("Tree Before Pruning", file=f)
    # best_tree.print_tree(best_tree.root,0,f)
    # best_tree.root.prune(best_tree, validation_data, max_accuracy)
    # print("Tree After Pruning", file=f)
    # best_tree.print_tree(best_tree.root,0,f)
    # print(f'Accuracy after pruning: {get_accuracy(test_data, best_tree)}')

    # f.close()
    
    # depths = np.arange(1, 11)
    # accuracies = []
    # # print(depths)
    # for depth in range(10):
    #     accuracy, _, _, _, _ = test_10_random_splits(data, depth, 10)
    #     accuracies.append(accuracy)
    
    # # save plot accuracies vs depths
    # plt.plot(depths, accuracies)
    # plt.xlabel('Depth')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Depth')
    # plt.savefig('accuracy_vs_depth.png')


        

if __name__ == "__main__":
    main()