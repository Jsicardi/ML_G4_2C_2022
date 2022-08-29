import numpy as np

from models import ClassifierOutput, NetworkProperties

class BayesianNetwork:

    def get_class_probs(self,properties:NetworkProperties,test_value):
        class_probs = []
        for (class_idx,class_name) in enumerate(properties.classes):
            class_probs.append(properties.last_probabilities[test_value[0]][test_value[1]][test_value[2]-1][class_idx])
        return class_probs
        
    def classify(self,properties:NetworkProperties):
        predictions = []
        probabilities = []
        for test_value in properties.test_values:
            probabilities.append([])
            class_probs = self.get_class_probs(properties,test_value)
            probabilities[-1].extend(class_probs)
            max_prob = max(class_probs)
            predictions.append(properties.classes[class_probs.index(max_prob)])
        return ClassifierOutput(predictions,probabilities)