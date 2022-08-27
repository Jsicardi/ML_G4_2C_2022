from models import ClassifierOutput, ClassifierProperties, Properties
from naive_classifier import NaiveClassifier
from parser import generate_classifier_output, get_classifier_properties, parse_properties
from text_analyzer import analyze_text_file


def __main__():

    properties:Properties = parse_properties()
    if(properties.type == "nationality"):
        get_classifier_properties(properties)
        classifier_properties:ClassifierProperties = get_classifier_properties(properties)
        classifier:NaiveClassifier = NaiveClassifier()
        output:ClassifierOutput = classifier.classify(classifier_properties)
        generate_classifier_output(classifier_properties,output)
    elif(properties.type == "titles"):
        analyze_text_file(properties)

if __name__ == "__main__":
    __main__()