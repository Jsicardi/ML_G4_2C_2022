from parser import parse_properties,generate_kohonen_output
from kohonen import execute as kohonen_execute
from models import KohonenObservables
import numpy as np

def __main__():
    properties = parse_properties()
    if(properties.method == "kohonen"):
        observables:KohonenObservables = kohonen_execute(properties)
        generate_kohonen_output(properties,observables)

if __name__ == "__main__":
    __main__()