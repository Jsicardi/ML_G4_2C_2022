import json

from models import SVM, Perceptron, PerceptronObservables, Properties, SVMObservables

MIN_ITERS = 100
DEFAULT_ERROR = 0

def generate_perceptron_results_output(properties:Properties,iter,training_set,test_set,training_expected,training_predictions,test_expected,test_predictions):
    with open("{0}_training_{1}.csv".format(properties.output_path,iter), "w") as f:
        f.write("X1,X2,Prediction,Expected\n")
        for (entry_index,entry) in enumerate(training_set):
            f.write("{0},{1},{2},{3}\n".format(entry[0],entry[1],int(training_predictions[entry_index]),training_expected[entry_index]))
    with open("{0}_test_{1}.csv".format(properties.output_path,iter), "w") as f:
        f.write("X1,X2,Prediction,Expected\n")
        for (entry_index,entry) in enumerate(test_set):
            f.write("{0},{1},{2},{3}\n".format(entry[0],entry[1],int(test_predictions[entry_index]),test_expected[entry_index]))

def generate_perceptron_weigths_output(properties:Properties,w,iters):
    with open("{0}.csv".format(properties.weights_path), "w") as f:
        f.write("Iters,W0,W1,W2\n")
        for (weigths_idx,weigths) in enumerate(w):
            f.write("{0},{1},{2},{3}\n".format(iters[weigths_idx],weigths[0],weigths[1],weigths[2]))

def generate_perceptron_output(observables:PerceptronObservables,properties:Properties,perceptron:Perceptron):
    generate_perceptron_results_output(properties,0,perceptron.training_set,perceptron.test_set,perceptron.output_set,observables.training_classifications[0],perceptron.test_output_set,observables.test_classifications[0])
    generate_perceptron_weigths_output(properties,observables.w,observables.iters)
    
def generate_svm_weigths_output(properties:Properties,w,b,iters,errors):
    with open("{0}.csv".format(properties.weights_path), "w") as f:
        f.write("Errors,Iters,b,W1,W2\n")
        for (weigths_idx,weigths) in enumerate(w):
            f.write("{0},{1},{2},{3},{4}\n".format(errors[weigths_idx],iters[weigths_idx],b[weigths_idx],w[weigths_idx][0],w[weigths_idx][1]))


def generate_svm_output(observables:SVMObservables,properties:Properties,svms):
    for (svm_idx,svm) in enumerate(svms):
        generate_perceptron_results_output(properties,svm_idx,svm.training_set,svm.test_set,svm.output_set,observables.training_classifications[svm_idx],svm.test_output_set,observables.test_classifications[svm_idx])    
    generate_svm_weigths_output(properties,observables.weights,observables.intercepts,observables.iters,observables.errors)

def parse_properties():
    file = open('config.json')
    json_values = json.load(file)
    file.close()    

    type = json_values.get("type")

    dataset_path = json_values.get("dataset")
    output_path = json_values.get("output_file")
    weights_path = json_values.get("weights_file")
    class_column = json_values.get("class_column")

    if type == None:
        print("Algorithm type required")
        exit(-1)

    k = json_values.get("k")

    test_proportion = json_values.get("test_proportion")

    if k==None and test_proportion==None:
        print("Test proportion or k value is required")
        exit(-1)

    rate_w = json_values.get("rate_w")

    if rate_w == None:
        print("Rate w required")
        exit(-1)

    rate_b = json_values.get("rate_b")

    if rate_b == None and type == "svm":
        print("Rate b required")
        exit(-1)

    iters = json_values.get("max_iters")
    min_error = json_values.get("min_error")

    if iters == None and min_error == None:
        print("Max iterations or min error required")
        exit(-1)

    if iters == None or iters < 0:
        iters = MIN_ITERS
    
    if min_error == None or min_error < 0:
        min_error = DEFAULT_ERROR

    dataset_shuffle = json_values.get("dataset_shuffle")

    if dataset_shuffle == None:
        print("Dataset shuffle required")
        exit(-1)

    C = json_values.get("C")

    if C == None and type=="svm":
        print("C required")
        exit(-1)

    Aw = json_values.get("Aw")

    if Aw == None and type=="svm":
        print("Aw required")
        exit(-1)

    Ab = json_values.get("Aw")

    if Ab == None and type=="svm":
        print("Ab required")
        exit(-1)
    return Properties(type,dataset_path,output_path,weights_path,class_column,rate_w,rate_b,iters,min_error,k,test_proportion,dataset_shuffle,C,Aw,Ab)
    