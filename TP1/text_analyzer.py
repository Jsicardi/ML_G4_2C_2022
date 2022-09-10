import os
import pandas as pd
import csv
import pathlib
from models import Properties
from parser import parse_xlsx_file

def parse_training_titles(properties:Properties,file_path):
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
            
            if properties.remove_characters != None:
                for character in properties.remove_characters:
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

def parse_test_titles(properties:Properties,file_path):
    length = 0
    titles_dic = []
    title_categories = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            length=length+1
            text=(row[1])
            
            if properties.remove_characters != None:
                for character in properties.remove_characters:
                    text.replace(character,"")

            text = text.lower() 
            words = text.split(" ")
            titles_dic.append({})
            
            for word in words: 
                if titles_dic[-1].get(word) != None:
                    titles_dic[-1][word] += 1
                else:
                    titles_dic[-1][word] = 1
            
            title_categories.append(row[2])
    
    return(titles_dic,title_categories)

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
        key_list = list(sorted_categories_dic[cat_idx].keys())
        if(len(key_list) >= properties.max_attributes):
            attributes.extend(key_list[:properties.max_attributes])
        else:
            attributes.extend(key_list) 
    attributes_dic = {}
    
    for attribute in attributes:
        attributes_dic[attribute] = 0
    
    return attributes_dic

def generate_training_file(training_title_words,training_title_categories,attributes_dic,training_path):
    if(os.path.exists(training_path)):
        os.remove(training_path)

    with open(training_path,"w") as f:
        f.write("{0},category\n".format(','.join(list(attributes_dic.keys()))))
        for (title_idx,title) in enumerate(training_title_words):
            title_attrs = []
            for key in list(attributes_dic.keys()):
                if(title.get(key) != None):
                    title_attrs.append(0)
                else:
                    title_attrs.append(1)
            
            title_attrs.append(training_title_categories[title_idx])
            f.write("{0}\n".format(','.join(map(str,title_attrs))))

def generate_test_file(test_title_words,attributes_dic,test_path):
    if(os.path.exists(test_path)):
        os.remove(test_path)
        
    with open(test_path,"w") as f:
        f.write("{0}\n".format(','.join(list(attributes_dic.keys()))))
        for (title_idx,title) in enumerate(test_title_words):
            title_attrs = []
            for key in list(attributes_dic.keys()):
                if(title.get(key) != None):
                    title_attrs.append(0)
                else:
                    title_attrs.append(1)
            f.write("{0}\n".format(','.join(map(str,title_attrs))))

def generate_categories_file(test_title_categories,test_categories_path):
    with open(test_categories_path,"w") as f:
        f.write("category\n")
        for (title_idx,title) in enumerate(test_title_categories):
            f.write("{0}\n".format(test_title_categories[title_idx]))

def analyze_text_file(properties:Properties):
    aux_training_path = 'resources/ej2_aux_training.csv'
    aux_test_path = 'resources/ej2_aux_test.csv'

    dataset = parse_xlsx_file(properties.training_file)
    dataset = dataset[['fecha','titular','categoria']]

    dataset = dataset.loc[dataset[dataset.columns[2]].isin(properties.categories)]

    #shuffle dataset to avoid using same order
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    og_dataset = dataset.loc[dataset[dataset.columns[2]] == properties.categories[0]]
    total_training = int(len(og_dataset) * (1 - properties.test_percentage))
    training_dataset = og_dataset.iloc[:total_training]
    test_dataset = og_dataset.iloc[total_training:]

    for (cat_idx,category) in enumerate(properties.categories[1:]):
        og_dataset = dataset.loc[dataset[dataset.columns[2]] == properties.categories[cat_idx+1]]
        total_training = int(len(og_dataset) * (1 - properties.test_percentage))
        training_dataset = pd.concat([training_dataset,og_dataset.iloc[:total_training]])
        test_dataset = pd.concat([test_dataset,og_dataset.iloc[total_training:]])

    training_dataset.to_csv(aux_training_path,index=False)
    test_dataset.to_csv(aux_test_path,index=False)

    (training_title_words,training_title_categories,categories_words,words_dic) = parse_training_titles(properties,aux_training_path)
    (test_title_words,test_title_categories) = parse_test_titles(properties,aux_test_path)

    sorted_categories_words = sort_by_ocurrences(properties,categories_words)

    attributes_dic = get_attributes(properties,sorted_categories_words)

    training_path = "resources/ej2_training.csv"
    test_path = "resources/ej2_test.csv"
    test_categories_path = ("{0}.csv".format(properties.test_categories_file))

    generate_training_file(training_title_words,training_title_categories,attributes_dic,training_path)
    generate_test_file(test_title_words,attributes_dic,test_path)
    generate_categories_file(test_title_categories,test_categories_path)

    aux_file = pathlib.Path(aux_training_path)
    aux_file.unlink()
    aux_file = pathlib.Path(aux_test_path)
    aux_file.unlink()

    return Properties(properties.type,training_path,properties.output_file,test_path,properties.test_categories_file,properties.categories,properties.max_attributes,properties.max_attributes,properties.test_percentage)



