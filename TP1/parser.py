import json
from locale import normalize
from models import ClassifierOutput, ClassifierProperties, Properties
import numpy as np
import pandas as pd
import pathlib
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

    training = json_values.get("training_file")

    if training == None:
        print("Training file required")
        exit(-1)

    test = None
    if type == "nationality":
        test = json_values.get("test_file")
        if test == None:
            print("Test file required")
            exit(-1)

    categories = json_values.get("categories")
    if categories == None and type == "titles":
        print("Categories required")
        exit(-1)

    max_attributes = json_values.get("max_attributes")

    if max_attributes == None and type == "titles":
        print("Max attributes required")
        exit(-1)
    
    remove_characters = json_values.get("remove_characters")

    test_percentage = json_values.get("test_percentage")

    if test_percentage == None and type == "titles":
        print("Test percentage required")
        exit(-1)

    return Properties(type,training,test,categories,max_attributes,remove_characters,test_percentage)

def parse_xlsx_file(file):
    xls = pd.ExcelFile(file)
    dataset = xls.parse(0) #first sheet
    return dataset

def parse_csv_file(file):
    return pd.read_csv(file)

def get_classifier_properties(properties:Properties):
    training_dataset = None
    if(pathlib.Path(properties.training_file).suffix == ".xlsx"):
        training_dataset = parse_xlsx_file(properties.training_file)
    else:
        training_dataset = parse_csv_file(properties.training_file)

    test_dataset = None
    if(pathlib.Path(properties.test_file).suffix == ".xlsx"):
        test_dataset = parse_xlsx_file(properties.test_file)
    else:
        test_dataset = parse_csv_file(properties.test_file)

    (absolute_probs, conditional_probs) = get_probabilities(training_dataset)
    attributes = training_dataset.columns[:-1]
    classes = np.unique(training_dataset[training_dataset.columns[-1]].values)
    return ClassifierProperties(attributes,classes,absolute_probs,conditional_probs,test_dataset.values)