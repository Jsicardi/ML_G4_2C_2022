import numpy as np
import pandas as pd

def apply_laplace(dataset,absolute_probs,classes):
    conditional_probs = []
    total_classes = len(classes)
    for classifier in classes:
        class_dataset = dataset.loc[dataset[dataset.columns[-1]].values == classifier]
        conditional_probs.append([])
        for attr_name in class_dataset.columns[:-1]:
           occurences = len(class_dataset.loc[class_dataset[attr_name].values == 1].values)
           total_values = len(class_dataset[attr_name].values)
           conditional_probs[-1].append((occurences + 1) / (total_values + total_classes))
    return conditional_probs

def apply_laplace_network_last(dataset,root_node_column,middle_nodes_columns,last_node_column,root_node_values,middle_node_values,last_node_values):
    last_node_probabilities = []
    for root_value in root_node_values:
        last_node_probabilities.append([])
        for first_middle_value in middle_node_values[0]:
            last_node_probabilities[-1].append([])
            for second_middle_value in middle_node_values[1]:
                last_node_probabilities[-1][-1].append([])
                base_dataset = dataset.loc[(dataset[root_node_column] == root_value) & (dataset[middle_nodes_columns[0]] == first_middle_value) & (dataset[middle_nodes_columns[1]] == second_middle_value)]
                total_values = len(base_dataset)
                for value in last_node_values:
                    ocurrences = len(base_dataset.loc[base_dataset[last_node_column] == value])
                    last_node_probabilities[-1][-1][-1].append((ocurrences + 1) / (total_values + len(last_node_values)))
    return last_node_probabilities

def get_probabilities(dataset):
    classes = np.unique(dataset[dataset.columns[-1]].values)
    absolute_probs = dataset[dataset.columns[-1]].value_counts(normalize=True).values
    conditional_probs = []
    laplace = False
    for classifier in classes:
        class_dataset = dataset.loc[dataset[dataset.columns[-1]].values == classifier]
        conditional_probs.append([])
        for attr_name in class_dataset.columns[:-1]:
            conditional_probs[-1].append(class_dataset[attr_name].value_counts(normalize=True).sort_index(ascending=True)[1])
            if(conditional_probs[-1][-1] >= 1):
               laplace = True 
               break
        if(laplace):
            break

    if(laplace):
        conditional_probs = apply_laplace(dataset,absolute_probs,classes)

    return (absolute_probs,conditional_probs)

def get_newtwork_probabilities(dataset, root_node_column, middle_nodes_columns, last_node_column):
    
    root_probabilities = dataset[root_node_column].value_counts(normalize=True).values
    laplace = False

    root_values = np.unique(dataset[root_node_column].values)
    middle_nodes_probabilities = []
    middle_node_values = []
    for middle_column in middle_nodes_columns:
        middle_nodes_probabilities.append([])
        middle_node_values.append(np.unique(dataset[middle_column].values))
        for root_value in root_values:
            middle_nodes_probabilities[-1].append([])
            for middle_node_value in middle_node_values[-1]:
                root_dataset = dataset.loc[dataset[root_node_column] == root_value]
                middle_nodes_probabilities[-1][-1].append(root_dataset[middle_column].value_counts(normalize=True).values[1-middle_node_value])
    
    last_node_probabilities = []
    last_node_values = np.unique(dataset[last_node_column].values)

    for root_value in root_values:
        last_node_probabilities.append([])
        for first_middle_value in middle_node_values[0]:
            last_node_probabilities[-1].append([])
            for second_middle_value in middle_node_values[1]:
                last_node_probabilities[-1][-1].append([])
                base_dataset = dataset.loc[(dataset[root_node_column] == root_value) & (dataset[middle_nodes_columns[0]] == first_middle_value) & (dataset[middle_nodes_columns[1]] == second_middle_value)]
                probs = base_dataset[last_node_column].value_counts(normalize=True).values
                if(len(probs) == 2):
                    last_node_probabilities[-1][-1][-1].append(probs[1])
                    last_node_probabilities[-1][-1][-1].append(probs[0])
                else:
                    laplace = True
                    break
            if(laplace):
                break
        if(laplace):
            break
    
    if(laplace):
        last_node_probabilities = apply_laplace_network_last(dataset,root_node_column,middle_nodes_columns,last_node_column,root_values,middle_node_values,last_node_values)
                    
    return(root_probabilities,middle_nodes_probabilities,last_node_probabilities)



    
