import json
from locale import normalize
from models import ClassifierOutput, ClassifierProperties, Properties
import numpy as np
import pandas as pd
from probabities_helper import get_probabilities

def generate_classifier_output(properties:ClassifierProperties, output:ClassifierOutput):
    result_file = open("resources/classifier_results.csv", "w")
    result_file.write("Prediction,Probability\n")
    for (pred_idx,prediction) in enumerate(output.predictions):
        result_file.write("{0},{1}\n".format(prediction,max(output.probabilities[pred_idx])))

# Receive parameters from config.json and encapsulate them into properties object
def parse_properties():

    file = open('config.json')
    json_values = json.load(file)
    file.close()    
    
    type = json_values.get("type")
    
    if type == None:
        print("Type required")
        exit(-1)

    examples = json_values.get("examples_file")

    if examples == None:
        print("Examples file required")
        exit(-1)

    queries = json_values.get("queries_file")
    if queries == None:
        print("Queries file required")
        exit(-1)

    return Properties(type,examples,queries)

def parse_file(file):
    xls = pd.ExcelFile(file)
    dataset = xls.parse(0) #first sheet
    return dataset
    
def get_classifier_properties(properties:Properties):
    ex_dataset = parse_file(properties.examples_file)
    query_dataset = parse_file(properties.queries_file)
    (absolute_probs, conditional_probs) = get_probabilities(ex_dataset)
    attributes = ex_dataset.columns[:-1]
    classes = np.unique(ex_dataset[ex_dataset.columns[-1]].values)
    return ClassifierProperties(attributes,classes,absolute_probs,conditional_probs,query_dataset.values)