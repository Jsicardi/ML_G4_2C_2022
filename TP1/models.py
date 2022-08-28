from mimetypes import init
from xml.sax.xmlreader import AttributesImpl

class Properties:
    def __init__(self,type,training_file,test_file=None,categories=None,max_attributes=None,remove_characters=None,test_percentage=None):
        self.type = type
        self.training_file = training_file
        self.test_file = test_file
        self.categories = categories
        self.max_attributes = max_attributes
        self.test_percentage = test_percentage
        self.remove_characters = remove_characters

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