from mimetypes import init
from xml.sax.xmlreader import AttributesImpl

class Properties:
    def __init__(self,type,examples_file,queries_file=None,categories=None,max_attributes=None):
        self.type = type
        self.examples_file = examples_file
        self.queries_file = queries_file
        self.categories = categories
        self.max_attributes = max_attributes

class ClassifierProperties:
    def __init__(self,attributes,classes,absolute_probs,conditional_probs,queries):
        self.attributes = attributes
        self.classes = classes
        self.absolute_probs = absolute_probs
        self.conditional_probs = conditional_probs
        self.queries = queries

class ClassifierOutput:
    def __init__(self,predictions, probabilities):
        self.predictions = predictions
        self.probabilities = probabilities