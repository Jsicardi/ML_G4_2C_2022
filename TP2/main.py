import os
from platform import node
from models import NodesTestOutput, Properties, TreeOutput, TreeProperties,Tree
from parser import generate_forest_output, generate_node_test_output, generate_output, parse_properties, process_dataset, process_dataset_cross_validate
from tree_algorithm import build_tree, classify_with_tree, forest_nodes_test,k_cross_classify, random_forest_classify, tree_nodes_test
import logging

def __main__():

    properties:Properties = parse_properties()
    if(not properties.nodes_test):
        if(properties.type == "tree"):
            os.remove("log.txt")
            logging.basicConfig(filename="log.txt", level=logging.DEBUG)
            datasets = process_dataset_cross_validate(properties)
            output:TreeOutput = k_cross_classify(datasets,properties)
            generate_output(output,properties)
        elif(properties.type == "forest"):
            os.remove("log.txt")
            logging.basicConfig(filename="log.txt", level=logging.DEBUG)
            (training_dataset,test_dataset) = process_dataset(properties)
            output:TreeOutput = random_forest_classify(training_dataset,test_dataset,properties)
            generate_forest_output(output,properties)
    else:
        (training_dataset,test_dataset) = process_dataset(properties)
        os.remove("log.txt")
        logging.basicConfig(filename="log.txt", level=logging.DEBUG)
        output:NodesTestOutput = tree_nodes_test(training_dataset,test_dataset,properties)
        if(properties.type == "tree"):
            output = tree_nodes_test(training_dataset,test_dataset,properties)
        else:
            output = forest_nodes_test(training_dataset,test_dataset,properties)
        generate_node_test_output(output,properties) 



if __name__ == "__main__":
    __main__()