import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from treelib import Node, Tree
import math

input_data = pd.read_csv("./data/train.csv")
input_data["Age"] = pd.qcut(input_data["Age"], q=4, labels = [0, 1, 2, 3]) #bins age data into four quartiles
input_data.dropna(subset = ["Age"], inplace=True) #drops any example with incomplete age data
test_data = pd.read_csv("./data/test.csv")
headers = input_data.columns.to_numpy()
learnable_features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
tree = Tree()
def bin_data_by_feature(examples, feature):
    examples_np = examples[feature].tolist()
    unique_values, count = np.unique(examples_np, return_counts = True)
    binned_data = []
    for value in unique_values:
        next_bin = examples.loc[examples[feature] == value]
        binned_data.append(next_bin)
    return binned_data


def calculate_entropy(examples, feature):
    entropy = 0
    survival_np = examples[feature].tolist()
    unique_values, count = np.unique(survival_np, return_counts = True)
    if len(count) < 2:
        return 0
    proportions = [count[0]/len(examples), count[1]/len(examples)]
    for proportion in proportions:
        entropy += -proportion * math.log(proportion,2)
    return entropy

def identify_best_feature(examples, features):
    best_feature = features[0]
    max_information_gain = 0
    whole_set_entropy = calculate_entropy(examples, "Survived")
    for feature in features:
        binned_data = bin_data_by_feature(examples, feature)
        bin_entropy = 0
        for data_bin in binned_data:
            bin_entropy += (len(data_bin)/len(examples)) * calculate_entropy(data_bin, "Survived")
        information_gain = whole_set_entropy - bin_entropy
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = feature
    return best_feature

def ID3(examples, target_feature, features, parent_node = None):
    #gets the distribution of survivors from the whole set
    labels_np = examples[target_feature].tolist()
    labels, count = np.unique(labels_np, return_counts = True)
    #if the data is homogenous:
    if len(labels) == 1:
        tree.create_node(labels[0], str(labels[0]) + str(parent_node), parent = parent_node)
        return None
    #if the there are no more features to train on:
    elif len(features) == 0:
        tree.create_node(examples.mode()[target_feature][0], str(examples.mode()[target_feature][0]) + str(parent_node), parent = parent_node)
        return None
    else:
        best_feature = identify_best_feature(examples, features)
        tree.create_node(best_feature, best_feature, parent = parent_node)
        parent_node = best_feature
        #get different values for best_feature
        examples_np = examples[best_feature].tolist()
        feature_values, count = np.unique(examples_np, return_counts = True)
        binned_data = bin_data_by_feature(examples, best_feature)
        features.remove(best_feature)
        for i in range(0,len(feature_values)):
            tree.create_node(feature_values[i], str(feature_values[i]) + str(parent_node), parent = parent_node)
            ID3(binned_data[i], target_feature, features, str(feature_values[i]) + str(parent_node))
    return None
ID3(input_data, "Survived", learnable_features)
tree.show()
test_np = test_data.to_numpy()
correct = 0
print("PassengerId,Survived")
for example in test_np:
    prediction = None
    if example[3] == "male":
        prediction = 0
    elif example[3] == "female":
        if example[1] == 3:
            prediction = 0
        else:
            prediction = 1

    print(str(example[0]) + "," + str(prediction))
