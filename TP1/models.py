class Properties:
    def __init__(self,type,training_file,output_file,test_file=None,test_categories_file=None,categories=None,max_attributes=None,remove_characters=None,test_percentage=None,network_graph=None,discretize_values=None):
        self.type = type
        self.training_file = training_file
        self.output_file = output_file
        self.test_file = test_file
        self.test_categories_file = test_categories_file
        self.categories = categories
        self.max_attributes = max_attributes
        self.test_percentage = test_percentage
        self.remove_characters = remove_characters
        self.network_graph = network_graph
        self.discretize_values = discretize_values

class ClassifierProperties:
    def __init__(self,attributes,classes,absolute_probs,conditional_probs,test_values):
        self.attributes = attributes
        self.classes = classes
        self.absolute_probs = absolute_probs
        self.conditional_probs = conditional_probs
        self.test_values = test_values

class ClassifierOutput:
    def __init__(self,predictions, probabilities):
        self.predictions = predictions
        self.probabilities = probabilities

class NetworkProperties:
    def __init__(self,attributes,classes,root_probabilities,middle_probabilities,last_probabilities,test_values):
        self.attributes = attributes
        self.classes = classes
        self.root_probabilities = root_probabilities
        self.middle_probabilities = middle_probabilities
        self.last_probabilities = last_probabilities
        self.test_values = test_values