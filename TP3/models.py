from ast import Await
import numpy as np
#The parameters for the running of perceptron
class Perceptron:
    def __init__(self,training_set,test_set,output_set,test_output_set,learning_rate,max_iters,min_error,function,dfunction=None):
        self.training_set = training_set
        self.test_set = test_set
        self.output_set = output_set
        self.test_output_set = test_output_set
        self.max_iters = max_iters
        self.min_error = min_error 
        self.function = function
        self.learning_rate = learning_rate
        self.d_function = dfunction

class SVM:
    def __init__(self,training_set,test_set,output_set,test_output_set,rate_w,rate_b,max_iters,min_error,dw_function,db_function,C,Aw,Ab):
        self.training_set = training_set
        self.test_set = test_set
        self.output_set = output_set
        self.test_output_set = test_output_set
        self.max_iters = max_iters
        self.min_error = min_error 
        self.dw_function = dw_function
        self.db_function = db_function
        self.rate_w = rate_w
        self.rate_b = rate_b
        self.C = C
        self.Aw = Aw
        self.Ab = Ab

class Properties:
    def __init__(self,type,dataset,output_path,weights_path,class_column,rate_w,rate_b,max_iters,min_error,k,test_proportion,dataset_shuffle,C,Aw,Ab):
        self.type = type
        self.dataset_path = dataset
        self.output_path = output_path
        self.weights_path = weights_path
        self.class_column = class_column
        self.rate_w = rate_w
        self.rate_b = rate_b
        self.max_iters = max_iters
        self.min_error = min_error
        self.k = k
        self.test_proportion = test_proportion
        self.dataset_shuffle = dataset_shuffle
        self.C = C
        self.Aw = Aw
        self.Ab = Ab

class PerceptronObservables:
    def __init__(self,w,iters,training_classifications,test_classifications):
        self.w = w
        self.iters = iters
        self.training_classifications = training_classifications
        self.test_classifications = test_classifications

class SVMObservables:
    def __init__(self,weights,intercepts,iters,errors,training_classifications,test_classifications):
        self.weights = weights
        self.intercepts = intercepts
        self.iters = iters
        self.errors = errors
        self.training_classifications = training_classifications
        self.test_classifications = test_classifications