import json
import pandas as pd
from models import Properties, TreeOutput, TreeProperties

def generate_output(output:TreeOutput,properties:Properties):
    for (curr_idx,current_predictions) in enumerate(output.predictions):
        test_classification = output.test_classifications[curr_idx]
        with open("{0}_{1}.csv".format(properties.output_file,curr_idx), "w") as f:
            f.write("Prediction,Creditability\n")
            for (pred_idx,prediction) in enumerate(current_predictions):
                f.write("{0},{1}\n".format(prediction,test_classification[pred_idx]))


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
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    attributes_max = []

    attributes = dataset.columns.values.tolist()
    attributes.remove(properties.target_attribute)

    for column_name in attributes:
        if(column_name != properties.target_attribute):
            attributes_max.append(dataset[column_name].max)

    total_entries = len(dataset)
    k_entries = int(total_entries / properties.k)

    datasets = []
    for iter in range(1,properties.k+1):
        datasets.append(dataset.iloc[(k_entries * (iter-1)):k_entries * (iter)])
    
    return (datasets,attributes_max)