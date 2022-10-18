from parser import generate_perceptron_output, generate_svm_output, parse_properties
from models import Properties,PerceptronObservables
from perceptron import simple_execute as perceptron_simple_execute
from svm import simple_execute as svm_simple_execute,k_cross_execute

def __main__():

    #Parse parameters
    properties:Properties = parse_properties()

    if(properties.type == "perceptron"):
        (observables,perceptron) = perceptron_simple_execute(properties)
        generate_perceptron_output(observables,properties,perceptron)
    elif(properties.type == "svm"):
        if(properties.k == None):
            (observables,svms) = svm_simple_execute(properties)
            generate_svm_output(observables,properties,svms)
        else:
            (observables,svms) = k_cross_execute(properties)
            generate_svm_output(observables,properties,svms)

if __name__ == "__main__":
    __main__()