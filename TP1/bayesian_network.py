import numpy as np

from models import ClassifierOutput, NetworkProperties

class BayesianNetwork:

    def get_class_probs(self,properties:NetworkProperties,test_value):
        class_probs = []
        for (class_idx,class_name) in enumerate(properties.classes):
            
            #check if rank attribute is missing
            root_absolute_probs = []
            root_conditional_probs = []
            if(test_value[2] == -1):
                for (root_idx,probs) in enumerate(properties.last_probabilities):
                    root_conditional_probs.append(probs)
                    root_absolute_probs.append(properties.root_probabilities[root_idx])
            else:
                root_absolute_probs.append(1)
                root_conditional_probs.append(properties.last_probabilities[test_value[2]-1])
            
            #check if gre and gpa attributes are missing
            prob = 0
            if(test_value[0] == -1 and test_value[1] == -1):
                for first_middle_value in [0,1]:
                    for second_middle_value in [0,1]:
                        sum_prob = 0
                        for (root_idx,root_prob) in enumerate(properties.root_probabilities):
                            prod_prob = 1
                            middle_values = [first_middle_value,second_middle_value]
                            for (prob_idx,probs) in enumerate(properties.middle_probabilities):
                                print(probs[root_idx][middle_values[prob_idx]])
                                prod_prob*=(probs[root_idx][middle_values[prob_idx]])
                            
                            prod_prob *= root_prob
                            sum_prob+=prod_prob
                        prob+=(properties.last_probabilities[test_value[2]-1][first_middle_value][second_middle_value][class_idx] * sum_prob)
            
            elif(test_value[0] != -1 and test_value[1] != -1):
                for (cond_idx,conditional_probs) in enumerate(root_conditional_probs):
                    prob+=(conditional_probs[test_value[0]][test_value[1]][class_idx]) * root_absolute_probs[cond_idx]

            class_probs.append(prob)
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