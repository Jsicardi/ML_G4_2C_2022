import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['Grasas_sat','Alcohol','Calorías','Sexo']

def boxplot(dataset, x):
    sns.set(rc={'figure.figsize':(5,5)})
    ax=sns.boxplot(x=x, data=dataset)
    plt.savefig("boxplot_{0}.png".format(x))
    plt.close()

def histogram(dataset,x,bins):
    sns.histplot(dataset[x], bins=bins)
    plt.savefig('hist_{0}.png'.format(x))
    plt.close()

def scaterplot(dataset,x,y):
    sns.scatterplot(x=x, hue=y, data=dataset)
    plt.savefig('scat_{0}.png'.format(x))
    plt.close()

def pairplot(dataset,Hue):
    sns.pairplot(dataset,hue=Hue,height=2)
    plt.savefig('graf_pares.png')
    plt.close()
       
def joinpoint(dataset,x,y):
    sns.jointplot(x=x, y=y, data=dataset)
    plt.savefig('joinpoint.png')
    plt.close()

def fill_dataset(dataset):
    for column_name in columns[:-1]:
        column_set = np.array(dataset[column_name])
        values_set = column_set[(column_set != 999.99)]
        avg = np.average(values_set)
        for (index,value) in enumerate(column_set):
            if(value == 999.99):
                dataset[column_name][index] = avg
    return dataset    

def __main__():

    xls = pd.ExcelFile("dataset.xls") 

    dataset = xls.parse(0) #first sheet

    dataset = fill_dataset(dataset)

    for column_name in columns[:-1]:
        boxplot(dataset,column_name)

    for column_name in columns:
        histogram(dataset,column_name,7)
    
    for column_name in columns[:-1]:
        scaterplot(dataset,column_name,'Sexo')
    
    pairplot(dataset,'Sexo')
    
    for column_name in columns[:-1]:
        joinpoint(dataset,column_name,'Sexo')
    
if __name__ == "__main__":
    __main__()