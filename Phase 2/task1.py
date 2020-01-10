import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import random
import pandas
import csv
from copy import deepcopy
'''get the input from the user via command line'''
'''data preocessing adding ODD'''
csv_reader=pandas.read_csv("overdoses.csv")
csv_reader['ODD']=""
for row in range(len(csv_reader['Population'])):
    csv_reader['Population'][row]=int(csv_reader['Population'][row].replace(',',''))
    csv_reader['Deaths'][row]=int(csv_reader['Deaths'][row].replace(',',''))
    csv_reader['ODD'][row]=(csv_reader['Deaths'][row]/csv_reader['Population'][row])*100   

headers = csv_reader[:0]
states=csv_reader['Abbrev']


font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
#print(csv_reader)

'''get the 50*2 table'''
def gettable():
    f1 = csv_reader['Population']
    f2 = csv_reader['Deaths']
    X = np.array(list(zip(f1, f2)))
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
    print(jval)
    print(kval)
    plt.plot(kval,jval, c='#050505')
    plt.ylabel('Objective function value',fontdict=font)
    plt.xlabel('Number of clusters(k)(2-15)',fontdict=font)
    plt.title('Objective function value (J) vs. number of clusters (k) plot',fontdict=font)
    plt.show()


X=gettable()
df=pandas.DataFrame(X, columns=['Population', 'Deaths'])
print(df)
plotObjectivefunction()
clusters=getclusters(5)
df_clusters=pandas.DataFrame(clusters, columns=['Cluster number'])
df_clusters.to_csv(index=False)
print(df_clusters)
#value=getSquaredDistanceforcluster(10)
