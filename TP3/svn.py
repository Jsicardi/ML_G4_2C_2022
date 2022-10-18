import sys
from models import SVN, Properties, SVNObservables
import random
import numpy as np
import pandas as pd
from helper_functions import dw,db
import math

def train_svn(svn:SVN):

    kw = svn.rate_w
    kb = svn.rate_b
    training_set = svn.training_set
    output = svn.output_set
    
    i = len(training_set)
    w = np.zeros(len(training_set[0]))
    b=0
    iters = 0
    error = 0
    
    while iters < svn.max_iters:

        kw = svn.rate_w * math.exp(-svn.A * iters)
        kb = svn.rate_b * math.exp(-svn.A * iters)

        if(i == len(training_set)):
            indexes = random.sample(list(range(len(training_set))),len(list(range(len(training_set)))))
            i = 0
        
        pos = indexes[i]
        entry = training_set[pos]

        t = output[pos] * (np.dot(w,entry) + b)

        print(t)

        w -= kw * svn.dw_function(t,svn.C,w,training_set,output)
        b -= kb * svn.db_function(t,svn.C,output)
        
        iters+=1
        i+=1
    
    return (w,b,iters)

def classify(svn:SVN,set,output_set,w,b):
    results = []

    for (i,entry) in enumerate(set):
        t = output_set[i] * (np.dot(w,entry) + b)
        
        if(t < 1):
            results.append(output_set[i] * (-1))
        else:
            results.append(output_set[i])
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

    svn = SVN(training_set,test_set,training_output_set,test_output_set,properties.rate_w,properties.rate_b,properties.max_iters,properties.min_error,dw,db,properties.C,properties.A)

    (min_w,min_b,iters) = train_svn(svn)

    training_results =  classify(svn,training_set,training_output_set,min_w,min_b)
    test_results =  classify(svn,test_set,test_output_set,min_w,min_b)


    return (SVNObservables(min_w,min_b,iters,training_results,test_results),svn)


