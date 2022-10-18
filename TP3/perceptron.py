from models import Perceptron, PerceptronObservables, Properties
from helper_functions import d_identity,step_function
import pandas as pd
import numpy as np
import sys
import random

def train_perceptron(perceptron:Perceptron):   
    
    # Add threshold to training set
    BIAS = 1
    training_set = np.insert(perceptron.training_set, 0, BIAS, axis=1)
    w = np.zeros(len(training_set[0]))
    error = sys.maxsize
    min_error = sys.maxsize #2 * len(training_set)
    min_w = np.zeros(len(training_set[0]))

    output_set = perceptron.output_set

    i = len(training_set)
    indexes = []
    previous_inc = None
    epochs = -1
    while error > perceptron.min_error and epochs < perceptron.max_epochs:
        # Always pick at random or random until covered whole training set and then random again?
        #pos = random.randint(0, len(training_set) - 1)
        if(i == len(training_set)):
            epochs+=1
            indexes = random.sample(list(range(len(training_set))),len(list(range(len(training_set)))))
            i = 0
        pos = indexes[i]
        entry = training_set[pos]
        h = np.dot(entry, w)
        O = perceptron.function(h)
        delta_w = perceptron.learning_rate * (output_set[pos] - O) * entry * perceptron.d_function(h)
        w += delta_w
        error = calculate_error(perceptron.function,training_set, output_set,w)
        if error < min_error:
            min_error = error
            min_w = w.copy()
        i+=1
    
    return (min_w,min_error,epochs)

def calculate_error(perceptron_function,training_set, output_set, w):
    error = 0
    for i in range(len(training_set)):
        entry = training_set[i]
        h = np.dot(entry, w)
        O = perceptron_function(h)
        error += (output_set[i] - O)**2
    return error*(1/2)


def classify(perceptron:Perceptron,set,output_set,w):
    results = []
    input_set = np.insert(set, 0, 1, axis=1)
    error = 0
    
    for (i,entry) in enumerate(input_set):
        h = np.dot(entry, w)
        O = perceptron.function(h)
        results.append(O)

    return results

def simple_execute(properties:Properties):
    dataset = pd.read_csv(properties.dataset_path)

    if(properties.dataset_shuffle):
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    total_entries = len(dataset)
    test_entries = int(total_entries * properties.test_proportion)

    test_dataset = dataset.iloc[:test_entries]
    training_dataset = dataset.iloc[test_entries:]

    training_output_set = training_dataset[properties.class_column].values
    test_output_set = test_dataset[properties.class_column].values

    training_dataset = training_dataset.drop([properties.class_column], axis=1)
    test_dataset = test_dataset.drop([properties.class_column], axis=1)

    training_set = training_dataset.values
    test_set = test_dataset.values

    perceptron = Perceptron(training_set,test_set,training_output_set,test_output_set,properties.learning_rate,properties.max_epochs,properties.min_error,step_function,d_identity)

    (min_w,min_error,epochs) = train_perceptron(perceptron)

    training_results = classify(perceptron,perceptron.training_set,perceptron.output_set,min_w)
    test_results = classify(perceptron,perceptron.test_set,perceptron.test_output_set,min_w)

    return (PerceptronObservables([min_w],[epochs],[training_results],[test_results]),perceptron)