import numpy as np
#The parameters for the running of perceptron
class Perceptron:
    def __init__(self,training_set,test_set,output_set,test_output_set,learning_rate,max_epochs,min_error,function,dfunction=None):
        self.training_set = training_set
        self.test_set = test_set
        self.output_set = output_set
        self.test_output_set = test_output_set
        self.max_epochs = max_epochs
        self.min_error = min_error 
        self.function = function
        self.learning_rate = learning_rate
        self.d_function = dfunction

class Properties:
    def __init__(self,type,dataset,output_path,weights_path,class_column,learning_rate,max_epochs,min_error,k,test_proportion,dataset_shuffle):
        self.type = type
        self.dataset_path = dataset
        self.output_path = output_path
        self.weights_path = weights_path
        self.class_column = class_column
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.min_error = min_error
        self.k = k
        self.test_proportion = test_proportion
        self.dataset_shuffle = dataset_shuffle

class PerceptronObservables:
    def __init__(self,w,epochs,training_classifications,test_classifications):
        self.w = w
        self.epochs = epochs
        self.training_classifications = training_classifications
        self.test_classifications = test_classifications