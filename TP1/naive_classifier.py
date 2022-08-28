from models import ClassifierOutput, ClassifierProperties
import numpy as np

class NaiveClassifier:

    def get_class_probs(self,properties:ClassifierProperties,test_value):
        class_probs = []
        for (class_idx,class_val) in enumerate(properties.classes):
                class_prob = properties.absolute_probs[class_idx]
                for (attr_idx,attr_val) in enumerate(test_value):
                    if attr_val == 1:
                        class_prob *= properties.conditional_probs[class_idx][attr_idx]
                    else:
                        class_prob *= (1 - (properties.conditional_probs[class_idx][attr_idx]))
                class_probs.append(class_prob)
        return class_probs
        
    def classify(self,properties:ClassifierProperties):
        predictions = []
        probabilities = []
        for test_value in properties.test_values:
            probabilities.append([])
            class_probs = self.get_class_probs(properties,test_value)
            total_prob = np.sum(class_probs)
            max_prob = 0
            prediction = ""
            for (class_idx,class_val) in enumerate(properties.classes):
                probabilities[-1].append(class_probs[class_idx] / total_prob)
                if(probabilities[-1][-1] > max_prob):
                    max_prob = probabilities[-1][-1]
                    prediction = class_val
            predictions.append(prediction)
        return ClassifierOutput(predictions,probabilities)

        
    