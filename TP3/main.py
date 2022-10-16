from parser import generate_perceptron_output, parse_properties
from models import Properties,PerceptronObservables
from perceptron import simple_execute

def __main__():

    #Parse parameters
    properties:Properties = parse_properties()

    if(properties.type == "perceptron"):
        if(properties.k != None):
            print("Hello")
            #observables = simple_cross_validate(properties)
        else:
            (observables,perceptron) = simple_execute(properties)
    
    #Process metrics for data visualization
    generate_perceptron_output(observables,properties,perceptron)
        

if __name__ == "__main__":
    __main__()