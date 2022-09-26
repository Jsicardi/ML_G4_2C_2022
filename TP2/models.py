
from inspect import Attribute


class Properties:
    def __init__(self,type,dataset_file,output_file,nodes_file,target_attribute,k,test_percentage,nodes_test,max_depth):
        self.type = type
        self.dataset_file = dataset_file
        self.output_file = output_file
        self.nodes_file = nodes_file
        self.target_attribute = target_attribute
        self.k = k
        self.test_percentage = test_percentage
        self.nodes_test = nodes_test
        self.max_depth = max_depth
        
class TreeProperties:
    def __init__(self,training_dataset,target_attribute,test_dataset,test_classification,max_depth):
        self.traning_dataset = training_dataset
        self.target_attribute = target_attribute 
        self.test_dataset = test_dataset
        self.test_classification = test_classification
        self.max_depth = max_depth

class TreeOutput:
    def __init__(self,predictions,test_classifications,trees):
        self.predictions = predictions
        self.test_classifications = test_classifications
        self.trees = trees

class NodesTestOutput:
    def __init__(self,training_precisions,test_precisions,nodes,depths):
        self.test_precisions = test_precisions
        self.training_precisions = training_precisions
        self.nodes = nodes
        self.depths = depths

class Tree:
    def __init__(self,root,nodes):
        self.root = root
        self.nodes = nodes

class Node:
    def __init__(self,parent,childs,attribute,attribute_value,examples):
        self.parent = parent
        self.childs = childs
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.examples = examples
