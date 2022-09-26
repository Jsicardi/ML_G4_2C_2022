from dataclasses import replace
from random import random
import numpy as np
import pandas as pd
import math
from models import Node, NodesTestOutput, Properties, TreeOutput,TreeProperties,Tree
from helper_functions import shannon_entropy,gain
import logging
from collections import Counter

def log_tree(root:Node):
    logging.debug("Node with attribute: {0} Value:{1}".format(root.attribute,root.attribute_value if(root.attribute_value != None) else "-"))
    for root_child in root.childs:
        logging.debug("Child with attribute:{0} Value:{1}".format(root_child.attribute,root_child.attribute_value))
    for root_child in root.childs:
        if(len(root_child.childs) != 0):
            for child in root_child.childs:
                log_tree(child)

def get_root_node(dataset,target_attribute,parent,max_depth,level,max_nodes,total_nodes):
    
    nodes = 0 
    
    #choose attribute with best gain
    attributes_gain = []
    attributes = dataset.columns.values.tolist()
    attributes.remove(target_attribute)
    for attribute in attributes:
        gain_val = gain(dataset,attribute,target_attribute)
        attributes_gain.append(gain(dataset,attribute,target_attribute))

    max_gain = max(attributes_gain)
    attr_index = attributes_gain.index(max_gain)
    attr_name = attributes[attr_index]

    logging.debug("Attribute {0} selected".format(attr_name))

    #create attribute node
    attr_node = Node(parent,[],attr_name,None,len(dataset))
    nodes+=1

    if(nodes+total_nodes >= max_nodes):
        logging.debug("Reached max nodes, deleting attribute node")
        return (Node(parent,[],target_attribute,dataset[target_attribute].mode().values[0],len(dataset)),nodes)

    attr_values = np.unique(dataset[attr_name].values)

    childs = []
    
    for attr_value in attr_values:
        logging.debug("Using attribute {0} value {1}".format(attr_name,attr_value))

        filtered_dataset = dataset.loc[dataset[attr_name] == attr_value]
        filtered_dataset = filtered_dataset.drop([attr_name], axis=1)
        classifications = np.unique(filtered_dataset[target_attribute].values)

        if(nodes+total_nodes >= max_nodes):
            logging.debug("Reached max nodes")
            continue
        
        #a leaf is found, so the branch is finished
        if len(classifications) == 1:
            logging.debug("Leaf found with value {0}".format(classifications[0]))
            childs.append(Node(attr_node,[],attr_name,attr_value,len(filtered_dataset)))
            childs[-1].childs.append(Node(childs[-1],[],target_attribute,classifications[0],len(filtered_dataset)))
            nodes+=1
        else:
            if((max_depth != -1 and level == max_depth-1)):
                logging.debug("Max depth reached")
                logging.debug("Leaf created with value {0}".format(filtered_dataset[target_attribute].mode().values[0]))
                childs.append(Node(attr_node,[],attr_name,attr_value,len(filtered_dataset)))
                childs[-1].childs.append(Node(childs[-1],[],target_attribute,filtered_dataset[target_attribute].mode().values[0],len(filtered_dataset)))
                nodes+=1
            else:
                logging.debug("Recursive action needed")
                childs.append(Node(attr_node,[],attr_name,attr_value,len(filtered_dataset)))
                (new_node,total_new_nodes) = get_root_node(filtered_dataset,target_attribute,childs[-1],max_depth,level+1,max_nodes,total_nodes+nodes)
                nodes+=total_new_nodes
                childs[-1].childs.append(new_node)
    
    for child in childs:
        attr_node.childs.append(child)
    
    return (attr_node,nodes)

def count_tree_nods(root:Node, target_attribute):
    total_nodes = 0
    if(root.attribute == target_attribute):
        return 1
    total_nodes+=1
    for root_child in root.childs:
        for child in root_child.childs:
            total_nodes+=(count_tree_nods(child,target_attribute))

    return total_nodes

def build_tree(treeProperties:TreeProperties):
    training_dataset = treeProperties.traning_dataset
    if not(0 in training_dataset[treeProperties.target_attribute].values):
        logging.debug("All positives")
        return Tree(Node(None,[],treeProperties.target_attribute,1,len(training_dataset)),1)
    if not(1 in training_dataset[treeProperties.target_attribute].values):
        logging.debug("All negatives")
        return Tree(Node(None,[],treeProperties.target_attribute,0,len(training_dataset)),1)

    (root_node,nodes) = get_root_node(treeProperties.traning_dataset, treeProperties.target_attribute,[],treeProperties.max_depth,0,treeProperties.max_nodes,0)
    return Tree(root_node,nodes)

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
    trees = []
    predictions = []
    test_classifications = []
    treeProperties:TreeProperties = None

    for (dataset_idx,dataset) in enumerate(datasets):
        test_dataset = dataset
        test_classification = test_dataset[properties.target_attribute].values
        test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
        training_dataset = get_training_dataset(datasets,dataset_idx)
        
        treeProperties = TreeProperties(training_dataset,properties.target_attribute,test_dataset,test_classification,properties.max_depth,properties.max_nodes)
        trees.append(build_tree(treeProperties))
        trees[-1].nodes = count_tree_nods(trees[-1].root,properties.target_attribute)
        log_tree(trees[-1].root)
        
        current_preds = classify_with_tree(trees[-1],treeProperties,treeProperties.test_dataset)
        predictions.append(current_preds)
        test_classifications.append(test_classification)
    
    return TreeOutput(predictions,test_classifications,trees)

def random_forest_classify(training_dataset,test_dataset,properties:Properties):

    trees = []
    predictions = []
    test_classification = test_dataset[properties.target_attribute].values
    test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
    treeProperties = []

    #Create trees from training datasets created from te original
    training_datasets = []
    for i in range(properties.k):
        training_datasets.append(training_dataset.sample(frac=1,replace=True).reset_index(drop=True))
        treeProperties.append(TreeProperties(training_datasets[-1],properties.target_attribute,test_dataset,test_classification,properties.max_depth,properties.max_nodes))
        trees.append(build_tree(treeProperties[-1]))
        trees[-1].nodes = count_tree_nods(trees[-1].root,properties.target_attribute)
    
    predictions = []
    for test_idx in range(len(test_dataset.values)):
        examples_predictions = []
        for (tree_idx,tree) in enumerate(trees):
            examples_predictions.append(classify_example(treeProperties[tree_idx],tree,test_dataset,test_idx))
        occurence_count = Counter(examples_predictions)
        predictions.append(occurence_count.most_common(1)[0][0])
    
    return TreeOutput(predictions,test_classification,trees)

def tree_nodes_test_depth(training_dataset,test_dataset,properties:Properties):
    
    training_precisions = []
    test_precisions = []
    nodes = []
    depths = []
    trees = []
    training_classification = training_dataset[properties.target_attribute].values
    test_classification = test_dataset[properties.target_attribute].values
    test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
    test_examples = len(test_dataset.values)
    training_examples = len(training_dataset.values)

    treeProperties = []
    
    for depth in range(1,properties.max_depth+1):
        treeProperties.append(TreeProperties(training_dataset,properties.target_attribute,test_dataset,test_classification,depth,properties.max_nodes))
        trees.append(build_tree(treeProperties[-1]))
        trees[-1].nodes = count_tree_nods(trees[-1].root,properties.target_attribute)
        nodes.append(trees[-1].nodes)
        depths.append(depth)

    for (tree_idx,tree) in enumerate(trees):
        correct_examples = 0
        for training_idx in range(training_examples):
            class_predicted = classify_example(treeProperties[tree_idx],tree,training_dataset,training_idx)
            if(class_predicted == training_classification[training_idx]):
                correct_examples+=1
        training_precisions.append(correct_examples/training_examples)

    for (tree_idx,tree) in enumerate(trees):
        correct_examples = 0
        for test_idx in range(test_examples):
            class_predicted = classify_example(treeProperties[tree_idx],tree,test_dataset,test_idx)
            if(class_predicted == test_classification[test_idx]):
                correct_examples+=1
        test_precisions.append(correct_examples/test_examples)

    return NodesTestOutput(training_precisions,test_precisions,nodes,depths)

def single_forest_precision(training_dataset,test_dataset,properties:Properties,depth,max_nodes):
    trees = []
    nodes = []
    test_classification = test_dataset[properties.target_attribute].values
    training_classification = training_dataset[properties.target_attribute].values
    test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
    treeProperties = []

    #Create trees from training datasets created from te original
    training_datasets = []
    for i in range(properties.k):
        training_datasets.append(training_dataset.sample(frac=1,replace=True).reset_index(drop=True))
        treeProperties.append(TreeProperties(training_datasets[-1],properties.target_attribute,test_dataset,test_classification,depth,max_nodes))
        trees.append(build_tree(treeProperties[-1]))
        trees[-1].nodes = count_tree_nods(trees[-1].root,properties.target_attribute)
        nodes.append(trees[-1].nodes)
    
    correct_training_examples = 0
    for training_idx in range(len(training_dataset.values)):
        examples_predictions = []
        for (tree_idx,tree) in enumerate(trees):
            examples_predictions.append(classify_example(treeProperties[tree_idx],tree,training_dataset,training_idx))
        occurence_count = Counter(examples_predictions)
        selected_prediction = occurence_count.most_common(1)[0][0]; 
        if(selected_prediction == training_classification[training_idx]):
            correct_training_examples+=1

    correct_test_examples = 0
    for test_idx in range(len(test_dataset.values)):
        examples_predictions = []
        for (tree_idx,tree) in enumerate(trees):
            examples_predictions.append(classify_example(treeProperties[tree_idx],tree,test_dataset,test_idx))
        occurence_count = Counter(examples_predictions)
        selected_prediction = occurence_count.most_common(1)[0][0]; 
        if(selected_prediction == test_classification[test_idx]):
            correct_test_examples+=1

    return (correct_training_examples / len(training_classification), correct_test_examples / len(test_classification), int(np.average(nodes))) 

def forest_nodes_test_depth(training_dataset,test_dataset,properties:Properties):
    depths = []
    nodes = []
    training_precisions = []
    test_precisions = []

    for depth in range(1,properties.max_depth+1):
        (training_precision,test_precision,tree_nodes) = single_forest_precision(training_dataset,test_dataset,properties,depth)
        nodes.append(tree_nodes)
        training_precisions.append(training_precision)
        test_precisions.append(test_precision)
        depths.append(depth)

    return NodesTestOutput(training_precisions,test_precisions,nodes,depths)

def tree_nodes_test(training_dataset,test_dataset,properties:Properties):
    
    training_precisions = []
    test_precisions = []
    nodes_vec = []
    depths = []
    trees = []
    training_classification = training_dataset[properties.target_attribute].values
    test_classification = test_dataset[properties.target_attribute].values
    test_dataset = test_dataset.drop([properties.target_attribute], axis=1)
    test_examples = len(test_dataset.values)
    training_examples = len(training_dataset.values)

    treeProperties = []
    
    for nodes in range(0,properties.max_nodes+properties.nodes_step,properties.nodes_step):
        treeProperties.append(TreeProperties(training_dataset,properties.target_attribute,test_dataset,test_classification,properties.max_depth,nodes))
        trees.append(build_tree(treeProperties[-1]))
        trees[-1].nodes = count_tree_nods(trees[-1].root,properties.target_attribute)
        nodes_vec.append(trees[-1].nodes)

    for (tree_idx,tree) in enumerate(trees):
        correct_examples = 0
        for training_idx in range(training_examples):
            class_predicted = classify_example(treeProperties[tree_idx],tree,training_dataset,training_idx)
            if(class_predicted == training_classification[training_idx]):
                correct_examples+=1
        training_precisions.append(correct_examples/training_examples)

    for (tree_idx,tree) in enumerate(trees):
        correct_examples = 0
        for test_idx in range(test_examples):
            class_predicted = classify_example(treeProperties[tree_idx],tree,test_dataset,test_idx)
            if(class_predicted == test_classification[test_idx]):
                correct_examples+=1
        test_precisions.append(correct_examples/test_examples)

    return NodesTestOutput(training_precisions,test_precisions,nodes_vec,depths)

def forest_nodes_test(training_dataset,test_dataset,properties:Properties):
    depths = []
    nodes_vec = []
    training_precisions = []
    test_precisions = []

    for nodes in range(0,properties.max_nodes+properties.nodes_step,properties.nodes_step):
        (training_precision,test_precision,tree_nodes) = single_forest_precision(training_dataset,test_dataset,properties,properties.max_depth,nodes)
        nodes_vec.append(tree_nodes)
        training_precisions.append(training_precision)
        test_precisions.append(test_precision)

    return NodesTestOutput(training_precisions,test_precisions,nodes_vec,depths)

