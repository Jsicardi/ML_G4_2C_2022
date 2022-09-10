import json
from models import ClassifierOutput, ClassifierProperties, NetworkProperties, Properties
import numpy as np
import pandas as pd
import pathlib
from probabities_helper import get_newtwork_probabilities, get_probabilities

def generate_classifier_output(properties:ClassifierProperties, output:ClassifierOutput,og_properties:Properties):
    result_file = open("{0}.csv".format(og_properties.output_file), "w")
    columns_line = "Prediction"
    for class_name in properties.classes:
        columns_line = columns_line + ",prob_{0}".format(class_name)

    result_file.write("{0}\n".format(columns_line))
    
    for (pred_idx,prediction) in enumerate(output.predictions):
        line = "{0}".format(prediction)
        for prob in output.probabilities[pred_idx]:
            line = line + ",{0}".format(prob)
        result_file.write("{0}\n".format(line))

def generate_network_output(properties:NetworkProperties, output:ClassifierOutput,og_properties:Properties):
    result_file = open("{0}.csv".format(og_properties.output_file), "w")
    columns_line = "Prediction"
    for class_name in properties.classes:
        columns_line = columns_line + ",prob_{0}".format(class_name)

    result_file.write("{0}\n".format(columns_line))
    
    for (pred_idx,prediction) in enumerate(output.predictions):
        line = "{0}".format(prediction)
        for prob in output.probabilities[pred_idx]:
            line = line + ",{0}".format(prob)
        result_file.write("{0}\n".format(line))

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

    output_file = json_values.get("output_file")
    if output_file == None:
        print("Output file name required")
        exit(-1)

    test = None
    if type != "titles":
        test = json_values.get("test_file")
        if test == None:
            print("Test file required")
            exit(-1)

    test_categories_file = None
    if type == "titles":
        test_categories_file = json_values.get("test_categories_file")
        if test_categories_file == None:
            print("Test categories file required")
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
    
    network_graph = json_values.get("network_graph")

    if network_graph == None and type == "admission":
        print("Network graph required")
        exit(-1)
    
    discretize_values = json_values.get("discretize_values")

    if discretize_values == None and type == "admission":
        print("Discretize values required")
        exit(-1)

    return Properties(type,training,output_file,test,test_categories_file,categories,max_attributes,remove_characters,test_percentage,network_graph,discretize_values)

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

def get_network_properties(properties:Properties):
    
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

    for (node_idx,discretize_value) in enumerate(properties.discretize_values):
        column_name = properties.network_graph[1][node_idx]
        training_dataset[column_name] = np.where(training_dataset[column_name] < discretize_value, 0, training_dataset[column_name])
        training_dataset[column_name] = np.where(training_dataset[column_name] >= discretize_value, 1, training_dataset[column_name])
        training_dataset[column_name] = training_dataset[column_name].astype(int)
        test_dataset[column_name] = np.where((test_dataset[column_name] < discretize_value) & (test_dataset[column_name] != -1), 0, test_dataset[column_name])
        test_dataset[column_name] = np.where(test_dataset[column_name] >= discretize_value,1 , test_dataset[column_name])
        test_dataset[column_name] = test_dataset[column_name].astype(int)

    (root_probabilities,middle_probabilities,last_probabilities) = get_newtwork_probabilities(training_dataset,properties.network_graph[0],properties.network_graph[1],properties.network_graph[2])

    return NetworkProperties(training_dataset.columns[1:],["no_admit","admit"],root_probabilities,middle_probabilities,last_probabilities,test_dataset.values)