import os
from models import Properties, TreeOutput, TreeProperties,Tree
from parser import generate_forest_output, generate_output, parse_properties, process_dataset, process_forest_dataset
from tree_algorithm import build_tree, classify_with_tree,k_cross_classify, random_forest_classify
import logging

def __main__():

    properties:Properties = parse_properties()
    if(properties.type == "tree"):
        os.remove("log.txt")
        logging.basicConfig(filename="log.txt", level=logging.DEBUG)
        datasets = process_dataset(properties)
        output:TreeOutput = k_cross_classify(datasets,properties)
        generate_output(output,properties)
    elif(properties.type == "forest"):
        os.remove("log.txt")
        logging.basicConfig(filename="log.txt", level=logging.DEBUG)
        (training_dataset,test_dataset) = process_forest_dataset(properties)
        output:TreeOutput = random_forest_classify(training_dataset,test_dataset,properties)
        generate_forest_output(output,properties)



if __name__ == "__main__":
    __main__()