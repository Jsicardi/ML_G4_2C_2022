import json
import pandas as pd
from models import Properties, TreeProperties

# Receive parameters from config.json and encapsulate them into properties object
def parse_properties():

    file = open('config.json')
    json_values = json.load(file)
    file.close()    
    
    type = json_values.get("type")
    
    if type == None:
        print("Type required")
        exit(-1)

    dataset = json_values.get("dataset_file")

    if dataset == None:
        print("Dataset file required")
        exit(-1)

    output_file = json_values.get("output_file")
    if output_file == None:
        print("Output file name required")
        exit(-1)

    target_attribute = json_values.get("target_attribute")
    if target_attribute == None:
        print("Target attribute required")
        exit(-1)

    k = json_values.get("k")
    if k == None:
        print("K required")
        exit(-1)

    return Properties(type,dataset,output_file,target_attribute,k)

def process_dataset(properties:Properties):
    dataset = pd.read_csv(properties.dataset_file)

    attributes_max = []

    attributes = dataset.columns.values.tolist()
    attributes.remove(properties.target_attribute)

    for column_name in attributes:
        if(column_name != properties.target_attribute):
            attributes_max.append(dataset[column_name].max)
    
    total_training = int(len(dataset) * (0.8))
    training_dataset = dataset.iloc[:total_training]
    test_dataset = dataset.iloc[total_training:]
    test_classification = dataset[properties.target_attribute].values
    test_dataset = dataset.drop([properties.target_attribute], axis=1)


    return TreeProperties(training_dataset,properties.target_attribute,test_dataset,test_classification,attributes_max)