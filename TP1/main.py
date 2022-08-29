from bayesian_network import BayesianNetwork
from models import ClassifierOutput, ClassifierProperties, NetworkProperties, Properties
from naive_classifier import NaiveClassifier
from parser import generate_classifier_output, generate_network_output, get_classifier_properties, get_network_properties, parse_properties
from text_analyzer import analyze_text_file

def __main__():

    properties:Properties = parse_properties()
    if(properties.type == "nationality"):
        classifier_properties:ClassifierProperties = get_classifier_properties(properties)
        classifier:NaiveClassifier = NaiveClassifier()
        output:ClassifierOutput = classifier.classify(classifier_properties)
        generate_classifier_output(classifier_properties,output)
    elif(properties.type == "titles"):
        properties = analyze_text_file(properties)
        classifier_properties:ClassifierProperties = get_classifier_properties(properties)
        classifier:NaiveClassifier = NaiveClassifier()
        output:ClassifierOutput = classifier.classify(classifier_properties)
        generate_classifier_output(classifier_properties,output)
    elif(properties.type == "admission"):
        network_properties:NetworkProperties = get_network_properties(properties)
        network:BayesianNetwork = BayesianNetwork()
        output:ClassifierOutput = network.classify(network_properties)
        generate_network_output(network_properties,output)

if __name__ == "__main__":
    __main__()