import os
from models import Properties, TreeProperties
from parser import parse_properties, process_dataset
from tree_algorithm import classify_with_tree
import logging

def __main__():

    properties:Properties = parse_properties()
    if(properties.type == "tree"):
        os.remove("log.txt")
        logging.basicConfig(filename="log.txt", level=logging.DEBUG)
        treeProperties:TreeProperties = process_dataset(properties)
        classify_with_tree(treeProperties)
        


if __name__ == "__main__":
    __main__()