import json
import numpy as np
from models import KohonenObservables, KohonenProperties,HierarchicalProperties,HierarchicalObservables
import pandas as pd

def generate_kohonen_results(properties:KohonenProperties, observables:KohonenObservables):
    with open("resources/classifications.csv", "w") as f:
        f.write("Id,Genre,Row,Column\n")
        for (i,id) in enumerate(properties.input_ids):
            f.write("{0},{1},{2},{3}\n".format(id[0],properties.input_genres[i][0], observables.classifications[i].i, observables.classifications[i].j))
    
    with open("resources/u_matrix.csv", "w") as f:
        f.write("Row,Colum,Avg_distance\n")
        for (row_index, col_index) in observables.u_matrix.keys(): 
            f.write("{0},{1},{2}\n".format(row_index,col_index,observables.u_matrix.get((row_index,col_index))))

    with open("resources/weights_matrix.csv", "w") as f:
        f.write("Row,Column,Budget,Popularity,Revenue,Runtime,VoteAverage\n")
        for (row_index, col_index) in observables.weights_matrix.keys(): 
            values = observables.weights_matrix.get((row_index,col_index))
            f.write("{0},{1},{2},{3},{4},{5},{6}\n".format(row_index,col_index,values[0],values[1],values[2],values[3],values[4]))

def generate_hierarchical_results(properties:HierarchicalProperties,observables:HierarchicalObservables):
    with open("resources/h_groups.csv", "w") as f:
        f.write("Genre,Budget,Popularity,Revenue,Runtime,VoteAverage,Group\n")
        for (g_idx, group) in enumerate(observables.groups):
            for (m_idx,member) in enumerate(group.members):
                f.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(group.members_genres[m_idx],member[0],member[1],member[2],member[3],member[4],g_idx))

def get_dataset(path):
    dataset = pd.read_csv(path, delimiter=",")
    return (dataset.loc[:, dataset.columns == "imdb_id"].values,dataset.loc[:, dataset.columns == "genres"].values, dataset.loc[:, dataset.columns.difference(["imdb_id","genres"])].values) 

def parse_kohonen_properties(json_values):
    kohonen_props = json_values.get("kohonen_props")
    if kohonen_props == None:
        print("Kohonen properties are required")
        exit(-1)
    
    dataset_path = kohonen_props.get("dataset_path")
    if dataset_path == None:
        print("Path for dataset is required")
        exit(-1)
    
    (ids,genres,input_set) = get_dataset(dataset_path)

    eta = kohonen_props.get("eta")
    if eta == None or eta <= 0:
        print("Positive eta required")
        exit(-1)
    
    k = kohonen_props.get("k")
    if k == None or k <= 0:
        print("Positive k is required")
        exit(-1)
    
    r = kohonen_props.get("r")
    if r == None or r <= 0:
        print("Positive r is required")
        exit(-1)
    
    epochs = kohonen_props.get("epochs")
    if epochs == None or epochs <= 0:
        print("Positive epochs is required")
        exit(-1)
    
    return KohonenProperties(ids,genres,input_set,eta,k,r,epochs)


def parse_hierarchical_properties(json_values):
    hierarchical_props = json_values.get("hierarchical_props")
    if hierarchical_props == None:
        print("Hierarchical properties are required")
        exit(-1)
    
    dataset_path = hierarchical_props.get("dataset_path")
    if dataset_path == None:
        print("Path for dataset is required")
        exit(-1)
    
    (ids,genres,input_set) = get_dataset(dataset_path)

    k = hierarchical_props.get("k")
    if k == None or k <= 0:
        print("Positive k is required")
        exit(-1)
    
    distance = hierarchical_props.get("distance")
    if distance != 'max' and distance != 'min' and distance != 'avg' and distance != 'centroid'  :
        print("Invalid distance method or distance method missing")
        exit(-1)
    
    return HierarchicalProperties(ids,genres,input_set,distance,k)


def parse_properties():
    file = open('config.json')
    json_values = json.load(file)
    file.close()

    method = json_values.get("method")

    if method == None:
        print("Method is required")
        exit(-1)

    if(method == "kohonen"):
        return parse_kohonen_properties(json_values)

    if(method == "hierarchical"):
        return parse_hierarchical_properties(json_values)
    
    print("Invalid method")
    exit(-1)