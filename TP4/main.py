from parser import parse_properties,generate_kohonen_results,generate_hierarchical_results
from kohonen import execute as kohonen_execute
from hierarchical import execute as hierarchical_execute
from models import KohonenObservables,HierarchicalObservables
import numpy as np

def __main__():
    properties = parse_properties()
    if(properties.method == "kohonen"):
        observables:KohonenObservables = kohonen_execute(properties)
        generate_kohonen_results(properties,observables)
    if(properties.method == "hierarchical"):
        observables:HierarchicalObservables = hierarchical_execute(properties)
        generate_hierarchical_results(properties,observables)

if __name__ == "__main__":
    __main__()