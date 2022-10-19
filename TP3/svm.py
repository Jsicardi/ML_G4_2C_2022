import sys
from models import SVM, Properties, SVMObservables
import random
import numpy as np
import pandas as pd
from helper_functions import dw,db,L_error
import math

def train_svm(svm:SVM):

    kw = svm.rate_w
    kb = svm.rate_b
    training_set = svm.training_set
    output = svm.output_set
    
    i = len(training_set)
    w = np.ones(len(training_set[0]))
    b=0

    iters = 0
    error = 0
    min_w = np.ones(len(training_set[0]))
    min_b = sys.maxsize
    min_error = sys.maxsize

    while iters < svm.max_iters:

        kw = svm.rate_w * math.exp(-svm.Aw * iters)
        kb = svm.rate_b * math.exp(-svm.Ab * iters)

        if(i == len(training_set)):
            indexes = random.sample(list(range(len(training_set))),len(list(range(len(training_set)))))
            i = 0
        
        pos = indexes[i]
        entry = training_set[pos]

        t = output[pos] * (np.dot(w,entry) + b)

        w -= kw * svm.dw_function(t,svm.C,w,entry,output[pos])
        b -= kb * svm.db_function(t,svm.C,output[pos])
        
        error = calculate_error(training_set,output,w,b,svm.C)

        if(error < min_error):
            min_error=error
            min_w=w
            min_b=b

        iters+=1
        i+=1

    return (min_w,min_b,iters,min_error)

def calculate_error(training_set,output_set,w,b,C):
    return L_error(w,b,training_set,output_set,C)

def classify(svm:SVM,set,output_set,w,b):
    results = []
    gradient = -(w[0] / w[1])
    intercept =  (b/w[1])

    for (i,entry) in enumerate(set):
        results.append(np.sign(np.dot(entry, w) + b))
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

    svm = SVM(training_set,test_set,training_output_set,test_output_set,properties.rate_w,properties.rate_b,properties.max_iters,properties.min_error,dw,db,properties.C,properties.Aw,properties.Ab)

    (min_w,min_b,iters,min_error) = train_svm(svm)

    training_results =  classify(svm,training_set,training_output_set,min_w,min_b)
    test_results =  classify(svm,test_set,test_output_set,min_w,min_b)


    return (SVMObservables([min_w],[min_b],[iters],[min_error],[training_results],[test_results]),[svm])

def get_training_dataset(datasets,dataset_idx):
    training_datasets = datasets.copy()
    training_datasets.pop(dataset_idx)
    training_dataset = pd.concat(training_datasets)
    
    return training_dataset

def k_cross_execute(properties:Properties):
    dataset = pd.read_csv(properties.dataset_path)

    if(properties.dataset_shuffle):
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    total_entries = len(dataset)
    k_entries = int(total_entries / properties.k)

    datasets = []
    for iter in range(1,properties.k+1):
        datasets.append(dataset.iloc[(k_entries * (iter-1)):k_entries * (iter)])

    svms = []
    weights = []
    intercepts = []
    iterations = []
    errors = []

    training_classifications = []
    test_classifications = []

    for (dataset_idx,dataset) in enumerate(datasets):
        
        test_dataset = dataset
        training_dataset = get_training_dataset(datasets,dataset_idx)

        training_output_set = training_dataset[properties.class_column].values
        test_output_set = test_dataset[properties.class_column].values

        training_dataset = training_dataset.drop([properties.class_column], axis=1)
        test_dataset = test_dataset.drop([properties.class_column], axis=1)

        training_set = training_dataset.values
        test_set = test_dataset.values

        svm = SVM(training_set,test_set,training_output_set,test_output_set,properties.rate_w,properties.rate_b,properties.max_iters,properties.min_error,dw,db,properties.C,properties.Aw,properties.Ab)

        svms.append(svm)

        (min_w,min_b,iters,min_error) = train_svm(svm)

        weights.append(min_w)
        intercepts.append(min_b)
        iterations.append(iters)
        errors.append(min_error)

        training_classifications.append(classify(svm,training_set,training_output_set,min_w,min_b))
        test_classifications.append(classify(svm,test_set,test_output_set,min_w,min_b))
    
    return (SVMObservables(weights,intercepts,iterations,errors,training_classifications,test_classifications),svms)





