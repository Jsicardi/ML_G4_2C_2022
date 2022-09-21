from dataclasses import replace
from random import random
import numpy as np
import pandas as pd
import math
from models import Node, Properties, TreeOutput,TreeProperties,Tree
from helper_functions import shannon_entropy,gain
import logging
from collections import Counter

def get_root_node(dataset, target_attribute,parent):
    
    #choose attribute with best gain
    attributes_gain = []
    attributes = dataset.columns.values.tolist()
    attributes.remove(target_attribute)
    for attribute in attributes:
        gain_val = gain(dataset,attribute,target_attribute)
        logging.debug("Gain val for attribute {0} is {1}".format(attribute,gain_val))
        attributes_gain.append(gain(dataset,attribute,target_attribute))

    max_gain = max(attributes_gain)
    attr_index = attributes_gain.index(max_gain)
    attr_name = attributes[attr_index]

    logging.debug("Attribute {0} selected".format(attr_name))

    #create attribute node
    attr_node = Node(parent,[],attr_name,None,len(dataset))

    attr_values = np.unique(dataset[attr_name].values)

    childs = []
    for attr_value in attr_values:
        logging.debug("Using attribute {0} value {1}".format(attr_name,attr_value))

        filtered_dataset = dataset.loc[dataset[attr_name] == attr_value]
        filtered_dataset = filtered_dataset.drop([attr_name], axis=1)
        classifications = np.unique(filtered_dataset[target_attribute].values)
        
        #a leaf is found, so the branch is finished
        if len(classifications) == 1:
            logging.debug("Leaf found with value {0}".format(classifications[0]))
            childs.append(Node(attr_node,[],attr_name,attr_value,len(filtered_dataset)))
            childs[-1].childs.append(Node(childs[-1],[],target_attribute,classifications[0],len(filtered_dataset)))
        else:
            logging.debug("Recursive action needed")
            childs.append(Node(attr_node,[],attr_name,attr_value,len(filtered_dataset)))
            childs[-1].childs.append(get_root_node(filtered_dataset,target_attribute,childs[-1]))
    
    for child in childs:
        attr_node.childs.append(child)
    
    return attr_node


def log_tree(root:Node):
    logging.debug("Node with attribute: {0} Value:{1}".format(root.attribute,root.attribute_value if(root.attribute_value != None) else "-"))
    for root_child in root.childs:
        logging.debug("Child with attribute:{0} Value:{1}".format(root_child.attribute,root_child.attribute_value))
    for root_child in root.childs:
        if(len(root_child.childs) != 0):
            for child in root_child.childs:
                log_tree(child)


def build_tree(treeProperties:TreeProperties):
    training_dataset = treeProperties.traning_dataset
    if not(0 in training_dataset[treeProperties.target_attribute]):
        logging.debug("All positives")
        return Tree(Node(None,[],treeProperties.target_attribute,1,len(training_dataset)))
    if not(1 in training_dataset[treeProperties.target_attribute]):
        logging.debug("All negatives")
        return Tree(Node(None,[],treeProperties.target_attribute,0,len(training_dataset)))
    #TODO case empty attributes

    return Tree(get_root_node(treeProperties.traning_dataset, treeProperties.target_attribute,[]))

def classify_example(treeProperties:TreeProperties,tree:Tree,dataset,row):
    current_node:Node = tree.root

    while(current_node.attribute != treeProperties.target_attribute):
        current_childs = current_node.childs
        selected_child = None
        attr_value = dataset[current_node.attribute].values[row]
        child_examples = []
        for current_child in current_childs:
            child_examples.append(current_child.examples)
            if(current_child.attribute_value == attr_value):
                selected_child = current_child
                break
            
        if(selected_child == None):
            logging.debug("Found case with attribute {0} when classify can't be done".format(current_node.attribute,attr_value))
            selected_idx = child_examples.index(max(child_examples))
            selected_child = current_childs[selected_idx]
            
        current_node = selected_child.childs[0]
    
    return current_node.attribute_value

def classify_with_tree(tree:Tree,treeProperties:TreeProperties,dataset):
    predictions = []
    for row_idx in range(len(dataset.values)):
        predictions.append(classify_example(treeProperties,tree,dataset,row_idx))
    return predictions

def get_training_dataset(datasets,dataset_idx):
    training_datasets = datasets.copy()
    training_datasets.pop(dataset_idx)
    training_dataset = pd.concat(training_datasets)
    
    return training_dataset

def k_cross_classify(datasets,properties:Properties):
    tree:Tree = None
    predictions = []
    test_classifications = []
    treeProperties:TreeProperties = None

    for (dataset_idx,dataset) in enumerate(datasets):
        test_dataset = dataset
        test_classification = test_dataset[properties.target_attribute].values
        test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
        training_dataset = get_training_dataset(datasets,dataset_idx)
        
        treeProperties = TreeProperties(training_dataset,properties.target_attribute,test_dataset,test_classification)
        tree = build_tree(treeProperties)
        
        current_preds = classify_with_tree(tree,treeProperties,treeProperties.test_dataset)
        predictions.append(current_preds)
        test_classifications.append(test_classification)
    
    return TreeOutput(predictions,test_classifications)

def random_forest_classify(training_dataset,test_dataset,properties:Properties):

    trees = []
    predictions = []
    test_classification = test_dataset[properties.target_attribute].values
    test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
    print(len(test_dataset))
    treeProperties = []

    #Create trees from training datasets created from te original
    training_datasets = []
    for i in range(properties.k):
        training_datasets.append(training_dataset.sample(frac=1,replace=True).reset_index(drop=True))
        treeProperties.append(TreeProperties(training_datasets[-1],properties.target_attribute,test_dataset,test_classification))
        trees.append(build_tree(treeProperties[-1]))
    
    predictions = []
    for test_idx in range(len(test_dataset.values)):
        examples_predictions = []
        for (tree_idx,tree) in enumerate(trees):
            examples_predictions.append(classify_example(treeProperties[tree_idx],tree,test_dataset,test_idx))
        occurence_count = Counter(examples_predictions)
        predictions.append(occurence_count.most_common(1)[0][0])
    
    return TreeOutput(predictions,test_classification)
