#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:40:27 2019

@author: ronak
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import random
import pandas
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import matplotlib.pyplot as plt; plt.rcdefaults()

def cos_similarity(x, y):
	dot_product = np.dot(x, y)
	norm_x = np.linalg.norm(x)
	norm_y = np.linalg.norm(y)
	return dot_product / (norm_x * norm_y)

filename = "overdoses.csv"
dataset = pd.read_csv(filename)
dataset["Deaths"] = dataset["Deaths"].str.replace(',','')
dataset["Population"] = dataset["Population"].str.replace(',','')

#Task 1
col_population = np.array(list(map(int, dataset.Population)))
col_deaths = np.array(list(map(int, dataset.Deaths)))

similarity_dict = {}
city_death = dict(zip(dataset["Abbrev"],dataset["Deaths"]))
city_population = dict(zip(dataset["Abbrev"],dataset["Population"]))
city_odd = dataset["Abbrev"]
for city in list(city_odd):
    similarity_dict[city]={}
    city_vec = []
    city_vec.append(int(city_population[city]))
    city_vec.append(int(city_death[city]))
    city_vec = np.array(city_vec)
    
    for other_city in list(city_odd):
        other_city_vec = []
        other_city_vec.append(int(city_population[other_city]))
        other_city_vec.append(int(city_death[other_city]))
        other_city_vec = np.array(other_city_vec)
        
        cos_sim_city = cos_similarity(city_vec, other_city_vec)
        
        similarity_dict[city][other_city]=cos_sim_city
    
similarity_matrix_frame = pd.DataFrame.from_dict(similarity_dict)
similarity_matrix_frame = similarity_matrix_frame.sort_index(axis=1)
pd.DataFrame(similarity_matrix_frame).to_csv("./text2.csv")
print("\nFollowing is the similarity matrix: \n"+str(similarity_matrix_frame))

from copy import deepcopy
'''get the input from the user via command line'''
'''data preocessing adding ODD'''
csv_reader=pd.read_csv("text3.csv")
csv_reader['ODD']=""

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
#print(csv_reader)

'''get the 50*2 table'''
a = {}
Y = []
def gettable():
    
    X = np.array(similarity_matrix_frame.values)
    return X

'''Euclidean Distance Caculator'''
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def getSquaredDistanceforcluster(k):
    X = gettable()
    C_x =random.sample(range(0, 50), k)
    #C_x = np.random.randint(0, 50, size=k)
    randpoints=X[C_x]
    #print(C_x)
    clusters = np.zeros(len(X))
    C = np.array(randpoints, dtype=np.float32)
    #print(C)
    '''To store the value of centroids when it updates'''
    C_old = np.zeros(C.shape)
    '''Error func. - Distance between new centroids and old centroids'''
    error = dist(C, C_old, None)
    iteration=0;
    sumvalue=0
    ''' Loop will run till the error becomes zero'''
    while error != 0 and iteration<500:
        '''Assigning each value to its closest cluster'''
        sumvalue=0
        dis = np.zeros(k)
        for i in range(len(X)):
            distances = dist(X[i], C)
            #print(distances)
            sumvalue=sumvalue+sum(distances)
            cluster = np.argmin(distances)
            clusters[i] = cluster
            dis[cluster]=dis[cluster]+distances[cluster]**2
        '''Storing the old centroid values'''
        C_old = deepcopy(C)
        '''Finding the new centroids by taking the average value'''
        #print(clusters)
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            #print(points)
            #print("-------------------------------------------------")
            if(len(points)>0):
                C[i] = np.mean(points, axis=0)
            #print(C[i])
        error = dist(C, C_old, None)
        iteration=iteration+1;
        sumvalue=sum(dis)
        #print(iteration)
    return sumvalue


def getclusters(k):
    X = gettable()
    C_x =random.sample(range(0, 50), k)
    #C_x = np.random.randint(0, 50, size=k)
    randpoints=X[C_x]
    #print(C_x)
    clusters = np.zeros(len(X))
    C = np.array(randpoints, dtype=np.float32)
    #print(C)
    '''To store the value of centroids when it updates'''
    C_old = np.zeros(C.shape)
    '''Error func. - Distance between new centroids and old centroids'''
    error = dist(C, C_old, None)
    iteration=0;
    sumvalue=0
    ''' Loop will run till the error becomes zero'''
    while error != 0 and iteration<500:
        '''Assigning each value to its closest cluster'''
        sumvalue=0
        for i in range(len(X)):
            distances = dist(X[i], C)
            #print(distances)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        '''Storing the old centroid values'''
        C_old = deepcopy(C)
        '''Finding the new centroids by taking the average value'''
        #print(clusters)
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            #print(points)
            #print("-------------------------------------------------")
            if(len(points)>0):
                C[i] = np.mean(points, axis=0)
            #print(C[i])
        error = dist(C, C_old, None)
        iteration=iteration+1;
        #print(iteration)
    return clusters

def plotObjectivefunction():
    kval=[]
    jval=[]
    for i in range(2,16):
        kval.append(i)
        value=getSquaredDistanceforcluster(i)
        jval.append(value)
    #print(jval)
    #print(kval)
    plt.scatter(kval,jval, c='#050505')
    plt.ylabel('Objective function value',fontdict=font)
    plt.xlabel('Number of clusters(k)(2-15)',fontdict=font)
    plt.title('Objective function value (J) vs. number of clusters (k) plot',fontdict=font)
    plt.show()


X=gettable()
df=pd.DataFrame(X, columns=dataset["Abbrev"])
#df=pd.DataFrame(X, columns=['AL', 'AK', 'AZ'])
print(df)
plotObjectivefunction()
'''clusters=getclusters(5)
df_clusters=pd.DataFrame(clusters, columns=['Cluster number'])
df_clusters.to_csv(index=False)
print(df_clusters)
'''
