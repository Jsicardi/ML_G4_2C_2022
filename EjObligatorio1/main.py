import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['Grasas_sat','Alcohol','Calorías','Sexo']

def fill_dataset(dataset):
    for column_name in columns[:-1]:
        column_set = np.array(dataset[column_name])
        values_set = column_set[(column_set != 999.99)]
        avg = np.average(values_set)
        for (index,value) in enumerate(column_set):
            if(value == 999.99):
                dataset[column_name][index] = avg
    return dataset 

def boxplot(dataset, x):
    sns.set(rc={'figure.figsize':(7,7)})
    ax=sns.boxplot(x=x, data=dataset)
    plt.savefig("boxplot_{0}.png".format(x))
    plt.close()

def doubleBoxplot(dataset,x,y):
    sns.set(rc={'figure.figsize':(7,7)})
    ax=sns.boxplot(x=x,y=y,data=dataset)
    ax.set_xticklabels(ax.get_xticklabels())    
    plt.savefig("boxplot_{0}_{1}.png".format(x,y))
    plt.close()

def histogram(dataset,x,bins):
    sns.histplot(dataset[x], bins=bins)
    plt.savefig('hist_{0}.png'.format(x))
    plt.close()

def change_sex_format(dataset):
    mod_dataset = dataset
    for (index,value) in enumerate(np.array(dataset['Sexo'])):
        mod_dataset['Sexo'][index] = (0 if value == 'M' else 1)
    return mod_dataset

def scatterplot(dataset,x,y):
    sns.scatterplot(data=dataset,x=x,y=y)
    plt.savefig('scat_{0}_{1}.png'.format(x,y))
    plt.close()

def scatterplot_with_hue(dataset,x,y,hue):
    sns.scatterplot(data=dataset,x=x,y=y,hue=hue)
    plt.savefig('scat_{0}_{1}.png'.format(x,y))
    plt.close()

def __main__():

    xls = pd.ExcelFile("dataset.xls") 

    dataset = xls.parse(0) #first sheet

    #replace empty values with avg
    dataset = fill_dataset(dataset)

    #generate single variable boxplot
    for column_name in columns[:-1]:
        boxplot(dataset,column_name)

    #generate histogram for Sex
    histogram(dataset,'Sexo',2)
    
    #generate scatterplot with Sex
    for column_name in columns[:-1]:
        scatterplot(dataset,column_name,'Sexo')

    #generate boxplots
    for column_name in columns[:-1]:    
        doubleBoxplot(dataset,'Sexo',column_name)
    
    #classify calories by categories
    categories = []
    for cal_value in dataset['Calorías']:
        if cal_value < 1100:
            categories.append('CATE 1')
        elif cal_value < 1700:
            categories.append('CATE 2')
        else:
            categories.append('CATE 3')

    #generate scatterplot by categories
    scatterplot_with_hue(dataset,'Calorías','Alcohol',categories)

    
if __name__ == "__main__":
    __main__()