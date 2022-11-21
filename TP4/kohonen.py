from models import KohonenObservables, KohonenProperties, KohonenNeuron
import numpy as np
import random as rn
import sys
import math

def standarize_input(input_set):
    # Calculates standard deviations and means for each field
    field_set = np.array(input_set).transpose()
    field_aggregations = []
    for field in field_set:
        field_aggregations.append([np.mean(field), np.std(field)])
    
    # Build standardized set
    output_set = []
    for entry in input_set:
        aux_row = []
        for index,field in enumerate(entry):
            aux_row.append((field - field_aggregations[index][0]) / field_aggregations[index][1])
            
        output_set.append(aux_row.copy())
    return output_set
    
def execute(properties:KohonenProperties):
    # Initialize input values
    #input_set = standarize_input(properties.input_set)

    input_set = properties.input_set
    # Create lattice of k x k neurons and initialize weights with values of entry chosen at random
    neurons = []
    for i in range(properties.k):
        for j in range(properties.k):
            w = rn.choice(input_set)
            neurons.append(KohonenNeuron(w.copy(), i, j))

    # Initialize epochs, eta and radius
    total_epochs = properties.epochs
    starting_eta = properties.eta
    eta = properties.eta
    starting_r = properties.r
    r = properties.r

    # Loop through inputs
    curr_epochs = 0
    random_ordered_inputs = input_set.copy()
    while(curr_epochs < total_epochs):
        print(curr_epochs)
        rn.shuffle(random_ordered_inputs)
        for entry in random_ordered_inputs:  
            # Find best match among neurons
            winner_neuron = find_winner_neuron(entry, neurons)
            
            # Update weights using Kohonen's rule
            update_neighbours(neurons, winner_neuron, eta, r, entry)
            
        # Update epochs, eta and r
        # r updates 4 times: [0 ; epochs/5] => r, [epochs/5 ; 2*epochs/5] => r/(r * 1/4), ..., [4*epochs/5 ; epochs] => r/r
        if (curr_epochs != 0 and curr_epochs % (total_epochs/5) == 0):
            r = starting_r - (int(curr_epochs / (total_epochs/5)) * (starting_r-1) / 4)
        if (curr_epochs != 0 and curr_epochs % (total_epochs/100) == 0):
            eta = starting_eta - (int(curr_epochs / (total_epochs/100)) * starting_eta / 100)
        curr_epochs += 1
    
    return get_observables(neurons, input_set, properties)

def find_winner_neuron(entry, neurons):
    winner_neuron = None
    winner_diff = sys.maxsize
    
    # Find best match among neurons
    for neuron in neurons:
        difference = np.sum(np.abs(np.subtract(entry, neuron.w)))
        if difference < winner_diff:
            winner_diff = difference
            winner_neuron = neuron
    
    return winner_neuron


def find_neighbours(neurons,central_neuron,r,k):
    neighbourhood = []
    # Find neighbours
    for i in range(math.floor(central_neuron.i - r), math.ceil(central_neuron.i + r + 1)):
        for j in range(math.floor(central_neuron.j - r), math.ceil(central_neuron.j + r + 1)):
            # Check bounds
            if i >= 0 and i < k and j >= 0 and j < k:
                # Check distance
                if (abs(i - central_neuron.i) + abs(j - central_neuron.j)) <= r:
                    neighbourhood.append(neurons[i*k+j])
    return neighbourhood


def update_neighbours(neurons, central_neuron, eta, r, input_value):
    k = int(math.sqrt(len(neurons)))
    neighbourhood = find_neighbours(neurons, central_neuron, r, k)

    # Update neighbour weights
    for i in range(0, len(neighbourhood)):
        neighbourhood[i].update_w(input_value, eta)

# Calculates the U-Matrix and associates an input to each neuron
def get_observables(neurons, standarized_input, properties:KohonenProperties):
    input_classifications = []
    # Find associated neuron for each input
    for i, entry in enumerate(standarized_input):
        input_classifications.append(find_winner_neuron(entry, neurons))
        
    u_matrix = {}
    weights_matrix = {}
    for neuron in neurons:
        neighbourhood = find_neighbours(neurons, neuron, 1, properties.k)
        avg_distance = 0
        for neighbour in neighbourhood:
            avg_distance += np.sum(np.abs(np.subtract(neuron.w, neighbour.w)))
        avg_distance /= len(neighbourhood) - 1
        u_matrix[(neuron.i, neuron.j)] = avg_distance

        for weight in neuron.w:
            if (neuron.i, neuron.j) in weights_matrix:
                weights_matrix[(neuron.i, neuron.j)].append(weight)
            else:
                weights_matrix[(neuron.i, neuron.j)] = [weight]
    
    return KohonenObservables(input_classifications,u_matrix,weights_matrix)