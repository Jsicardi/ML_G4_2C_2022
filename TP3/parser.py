import json

from models import Perceptron, PerceptronObservables, Properties

MIN_EPOCHS = 10
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

def generate_perceptron_weigths_output(properties:Properties,w,epochs):
    with open("{0}.csv".format(properties.weights_path), "w") as f:
        f.write("Epochs,W0,W1,W2\n")
        for (weigths_idx,weigths) in enumerate(w):
            f.write("{0},{1},{2},{3}\n".format(epochs[weigths_idx],weigths[0],weigths[1],weigths[2]))


def generate_perceptron_output(observables:PerceptronObservables,properties:Properties,perceptron:Perceptron):
    generate_perceptron_results_output(properties,0,perceptron.training_set,perceptron.test_set,perceptron.output_set,observables.training_classifications[0],perceptron.test_output_set,observables.test_classifications[0])
    generate_perceptron_weigths_output(properties,observables.w,observables.epochs)

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

    learning_rate = json_values.get("learning_rate")

    if learning_rate == None:
        print("Learning rate required")
        exit(-1)

    epochs = json_values.get("max_epochs")
    min_error = json_values.get("min_error")

    if epochs == None and min_error == None:
        print("Max iterations or min error required")
        exit(-1)

    if epochs == None or epochs < 0:
        epochs = MIN_EPOCHS
    
    if min_error == None or min_error < 0:
        min_error = DEFAULT_ERROR

    return Properties(type,dataset_path,output_path,weights_path,class_column,learning_rate,epochs,min_error,k,test_proportion)
    