
from inspect import Attribute


class Properties:
    def __init__(self,type,dataset_file,output_file,target_attribute,k):
        self.type = type
        self.dataset_file = dataset_file
        self.output_file = output_file
        self.target_attribute = target_attribute
        self.k = k

class TreeProperties:
    def __init__(self,training_dataset,target_attribute,test_dataset,test_classification,attributes_max):
        self.traning_dataset = training_dataset
        self.target_attribute = target_attribute 
        self.test_dataset = test_dataset
        self.test_classification = test_classification
        self.attributes_max = attributes_max

class Tree:
    def __init__(self,root):
        self.root = root

class Node:
    def __init__(self,parent,childs,attribute,attribute_value):
        self.parent = parent
        self.childs = childs
        self.attribute = attribute
        self.attribute_value = attribute_value
