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