import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import csv
import random
from sklearn.model_selection import train_test_split
from numpy import trapz
from sklearn import metrics
from scipy.integrate import simps
from collections import OrderedDict
from models import Properties
from parser import parse_file

def parse_titles_file(properties:Properties,file_path):
    categories_dic = []
    for category in properties.categories:
        categories_dic.append({})
    words_dic = {}
    length = 0
    titles_dic = []
    title_categories = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            length=length+1
            text=(row[1])
            
            remove_characters = ",;:.\n!\"'?!"
            for character in remove_characters:
                text.replace(character,"")  
            text = text.lower() 
            words = text.split(" ")
            titles_dic.append({})
            cat_idx = properties.categories.index(row[2])
            
            for word in words:
                if (categories_dic[cat_idx].get(word) != None):
                    categories_dic[cat_idx][word]+=1
                else:
                    categories_dic[cat_idx][word]=1
                if words_dic.get(word) != None:
                    words_dic[word] += 1
                else:
                    words_dic[word] = 1 
                if titles_dic[-1].get(word) != None:
                    titles_dic[-1][word] += 1
                else:
                    titles_dic[-1][word] = 1
            
            title_categories.append(row[2])
    
    return(titles_dic,title_categories,categories_dic,words_dic)

def sort_by_ocurrences(properties:Properties,categories_dic):
    sorted_categories_dic = []
    
    for (cat_idx,category) in enumerate(properties.categories):
        sorted_values = sorted(categories_dic[cat_idx].values(),reverse=True)
        sorted_categories_dic.append({})
        for i in sorted_values:
            for k in categories_dic[cat_idx].keys():
                if categories_dic[cat_idx][k] == i:
                    sorted_categories_dic[cat_idx][k] = categories_dic[cat_idx][k]
                    break
    
    return sorted_categories_dic

def get_attributes(properties:Properties,sorted_categories_dic):
    attributes = []
    
    for (cat_idx,category) in enumerate(properties.categories):
        attributes.extend(list(sorted_categories_dic[cat_idx].keys())[:properties.max_attributes]) 
    attributes_dic = {}
    
    for attribute in attributes:
        attributes_dic[attribute] = 0
    
    return attributes_dic

def analyze_text_file(properties:Properties):
    file_path = 'resources/ej2_examples.csv'
    dataset = parse_file(properties.examples_file)
    dataset = dataset[['fecha','titular','categoria']]

    clean_dataset = dataset.loc[dataset[dataset.columns[2]].isin(properties.categories)]
    clean_dataset.to_csv(file_path,index=False)

    (title_words,title_categories,categories_words,words_dic) = parse_titles_file(properties,file_path)
    
    sorted_categories_words = sort_by_ocurrences(properties,categories_words)

    attributes_dic = get_attributes(properties,sorted_categories_words)

    with open("resources/ej2_curated_examples.csv","w") as f:
        f.write("{0},category\n".format(','.join(list(attributes_dic.keys()))))
        for (title_idx,title) in enumerate(title_words):
            title_attrs = []
            for key in list(attributes_dic.keys()):
                if(title.get(key) != None):
                    title_attrs.append(0)
                else:
                    title_attrs.append(1)
            
            title_attrs.append(title_categories[title_idx])
            f.write("{0}\n".format(','.join(map(str,title_attrs))))



