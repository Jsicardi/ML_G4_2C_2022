import math
import numpy as np

def shannon_entropy(dataset,target_attribute):
    entropy = 0
    freqs = dataset[target_attribute].value_counts(normalize=True).values
    for freq in freqs:
        entropy += freq * math.log(freq,2)
    entropy *= -1

    return entropy

def gain(dataset,attribute,target_attribute):
    gain_val = shannon_entropy(dataset,target_attribute)
    attribute_values = np.unique(dataset[attribute].values)
    total_entries = len(dataset.values)

    new_dataset = []
    entropy_sum = 0
    for attribute_value in attribute_values:
        new_dataset = dataset.loc[dataset[attribute] == attribute_value]
        entropy_sum += shannon_entropy(new_dataset,target_attribute) * (len(new_dataset) / total_entries)
    
    gain_val -= entropy_sum
    return gain_val