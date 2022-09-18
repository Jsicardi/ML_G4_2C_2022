import os
from models import Properties, TreeOutput, TreeProperties,Tree
from parser import generate_output, parse_properties, process_dataset
from tree_algorithm import build_tree, classify_with_tree,k_cross_classify
import logging

def __main__():

    properties:Properties = parse_properties()
    if(properties.type == "tree"):
        os.remove("log.txt")
        logging.basicConfig(filename="log.txt", level=logging.DEBUG)
        (datasets,attributes_max) = process_dataset(properties)
        output:TreeOutput = k_cross_classify(datasets,attributes_max,properties)
        generate_output(output,properties)


if __name__ == "__main__":
    __main__()