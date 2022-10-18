from parser import generate_perceptron_output, generate_svn_output, parse_properties
from models import Properties,PerceptronObservables
from perceptron import simple_execute as perceptron_simple_execute
from svn import simple_execute as svn_simple_execute

def __main__():

    #Parse parameters
    properties:Properties = parse_properties()

    if(properties.type == "perceptron"):
        (observables,perceptron) = perceptron_simple_execute(properties)
        generate_perceptron_output(observables,properties,perceptron)
    elif(properties.type == "svn"):
        (observables,svn) = svn_simple_execute(properties)
        generate_svn_output(observables,properties,svn)
        

if __name__ == "__main__":
    __main__()