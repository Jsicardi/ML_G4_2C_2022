from itertools import tee
import json
from platform import node
import sys
import pandas as pd
from models import NodesTestOutput, Properties, TreeOutput, TreeProperties

def generate_output(output:TreeOutput,properties:Properties):
    for (curr_idx,current_predictions) in enumerate(output.predictions):
        test_classification = output.test_classifications[curr_idx]
        with open("{0}_{1}_{2}.csv".format(properties.output_file,properties.k,curr_idx), "w") as f:
            f.write("Prediction,Creditability\n")
            for (pred_idx,prediction) in enumerate(current_predictions):
                f.write("{0},{1}\n".format(prediction,test_classification[pred_idx]))
    
    generate_nodes_output(output,properties)

def generate_forest_output(output:TreeOutput,properties:Properties):
    with open("{0}_{1}.csv".format(properties.output_file,properties.k), "w") as f:
        f.write("Prediction,Creditability\n")
        for(pred_idx,prediction) in enumerate(output.predictions):
           f.write("{0},{1}\n".format(prediction,output.test_classifications[pred_idx]))
    
    generate_nodes_output(output,properties)

def generate_nodes_output(output:TreeOutput,properties:Properties):
    with open(properties.nodes_file, "w") as f:
        f.write("Nodes\n")
        for tree in output.trees:
            f.write("{0}\n".format(tree.nodes))

def generate_node_test_depth_output(output:NodesTestOutput,properties:Properties):
    with open("{0}_training.csv".format(properties.output_file),"w") as f:
        f.write("Precision,Depth,Nodes\n")
        for (depth_idx,depth) in enumerate(output.depths):
            f.write("{0},{1},{2}\n".format(output.training_precisions[depth_idx],depth,output.nodes[depth_idx]))
    with open("{0}_test.csv".format(properties.output_file),"w") as f:
        f.write("Precision,Depth,Nodes\n")
        for (depth_idx,depth) in enumerate(output.depths):
            f.write("{0},{1},{2}\n".format(output.test_precisions[depth_idx],depth,output.nodes[depth_idx]))

def generate_node_test_output(output:NodesTestOutput,properties:Properties):
    with open("{0}_training.csv".format(properties.output_file),"w") as f:
        f.write("Precision,Nodes\n")
        for (nodes_idx,nodes) in enumerate(output.nodes):
            f.write("{0},{1}\n".format(output.training_precisions[nodes_idx],nodes))
    with open("{0}_test.csv".format(properties.output_file),"w") as f:
        f.write("Precision,Nodes\n")
        for (nodes_idx,nodes) in enumerate(output.nodes):
            f.write("{0},{1}\n".format(output.test_precisions[nodes_idx],nodes))


# Receive parameters from config.json and encapsulate them into properties object
def parse_properties():

    file = open('config.json')
    json_values = json.load(file)
    file.close()    
    
    type = json_values.get("type")
    
    if type == None:
        print("Type required")
        exit(-1)

    nodes_test = json_values.get("nodes_test")
    if nodes_test == None:
        print("Nodes test value required")
        exit(-1)
    
    dataset = json_values.get("dataset_file")

    if dataset == None:
        print("Dataset file required")
        exit(-1)

    output_file = json_values.get("output_file")
    if output_file == None:
        print("Output file name required")
        exit(-1)

    nodes_file = json_values.get("nodes_file")
    if nodes_file == None:
        print("Nodes file name required")
        exit(-1)

    target_attribute = json_values.get("target_attribute")
    if target_attribute == None:
        print("Target attribute required")
        exit(-1)

    k = json_values.get("k")
    if k == None:
        print("K required")
        exit(-1)
        
    test_percentage = json_values.get("test_percentage")
    if test_percentage == None and (type == "forest" or nodes_test):
        print("Test percentage required")
        exit(-1)
    
    max_depth = json_values.get("max_depth")
    max_nodes = json_values.get("max_nodes")

    if max_depth == None and max_nodes == None and nodes_test:
        print("Max depth or max nodes are required")
        exit(-1)
    if max_depth == None:
        max_depth = sys.maxsize
    if max_nodes == None:
        max_nodes = sys.maxsize

    nodes_step = json_values.get("nodes_step")
    if nodes_step == None and nodes_test:
        print("Nodes step required")
        exit(-1)

    return Properties(type,dataset,output_file,nodes_file,target_attribute,k,test_percentage,nodes_test,max_depth,max_nodes,nodes_step)

def process_dataset_cross_validate(properties:Properties):
    
    dataset = pd.read_csv(properties.dataset_file)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    total_entries = len(dataset)
    k_entries = int(total_entries / properties.k)

    datasets = []
    for iter in range(1,properties.k+1):
        datasets.append(dataset.iloc[(k_entries * (iter-1)):k_entries * (iter)])
    
    return datasets

def process_dataset(properties:Properties):
    
    dataset = pd.read_csv(properties.dataset_file)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    total_entries = len(dataset)
    test_entries = int(total_entries * properties.test_percentage)

    test_dataset = dataset.iloc[:test_entries]
    training_dataset = dataset.iloc[test_entries:]
    
    return (training_dataset,test_dataset)