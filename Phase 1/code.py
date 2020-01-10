# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:51:07 2019

@author: vishal kumar
ASU ID = 1215200480
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import pearsonr
import csv

task = int(input("\nPlease specify the task number:\n"))
filename = "overdoses.csv"
dataset = pd.read_csv(filename)
dataset["Deaths"] = dataset["Deaths"].str.replace(',','')
dataset["Population"] = dataset["Population"].str.replace(',','')

#Task 1
col_population = np.array(list(map(int, dataset.Population)))
col_deaths = np.array(list(map(int, dataset.Deaths)))
dataset["ODD"] = col_deaths/col_population

if task == 1:
    pearson_cc, p_value = pearsonr(col_population, col_deaths)
    print("\nPearson correlation coefficient is - "+str(pearson_cc)+"\n")


#Task 2
elif task == 2:
    plt.ylabel('Opioid Death Density')
    plt.xlabel('States')
    plt.title('ODD of all 50 states')
    plt.xticks(np.arange(len(dataset["ODD"])), dataset["Abbrev"])
    plt.bar(np.arange(len(dataset["ODD"])), height=dataset["ODD"] , width=0.8, align='center')
    plt.show()


#Task 3
elif task == 3:
    
    max_odd = max(dataset["ODD"])
    min_odd = min(dataset["ODD"])
    max_diff = max_odd - min_odd
    city_odd = dict(zip(dataset["Abbrev"],dataset["ODD"]))
    similarity_dict = {}
    for city in list(city_odd.keys()):
        similarity_dict[city]={}
        city_distance = city_odd[city]
        for other_city in list(city_odd.keys()):
            other_city_dis = city_odd[other_city]
            distance = abs(city_distance - other_city_dis)
            if distance == 0:
                similarity_dict[city][other_city]=1
            elif distance == max_diff:
                similarity_dict[city][other_city]=0
            else:
                similarity_dict[city][other_city]=distance
    
    similarity_matrix_frame = pd.DataFrame.from_dict(similarity_dict)
    similarity_matrix_frame = similarity_matrix_frame.sort_index(axis=1)
    pd.DataFrame(similarity_matrix_frame).to_csv("./task3.csv")
    print("\nFollowing is the similarity matrix: \n"+str(similarity_matrix_frame))
    
    
else:
    print("\nOnly task 1,2 and 3 exit\n")