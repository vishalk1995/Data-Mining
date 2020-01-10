import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import eig
import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
k=5
task = int(input("\nplease enter task number\n"))
c=2
if task == 1:
    df = pd.read_csv('Dataset_1.csv')
if task == 2:
    df = pd.read_csv('Dataset_2.csv')
if task == 3:
    df = pd.read_csv('Dataset_3.csv')
    c=3
df.columns = ['x1', 'x2', 'y']
samples = df.drop('y', axis=1)                              
samples = samples.values

distance_table = euclidean_distances(samples, samples)

smp_smp_graph_dict = {}

for row in range(len(distance_table)):  
    smp_smp_graph_dict[str(row)] = {}                                  
    sample_dis = np.array(distance_table[row])                                 
    k_min_index = sample_dis.argsort()[:k+1]
    for index_index in k_min_index:
        if (index_index!=row):
            smp_smp_graph_dict[str(row)][str(index_index)] = sample_dis[index_index]

sample_number = len(smp_smp_graph_dict)

sample_name = list(smp_smp_graph_dict.keys())
smp_smp_zero_matrix = np.zeros((sample_number, sample_number))
laplacian_df = pd.DataFrame(smp_smp_zero_matrix, index=sample_name, columns=sample_name)    #filling Laplacian with zeroes and samples labels in index
del smp_smp_zero_matrix                                                                                       #of no use now

for index, row in laplacian_df.iterrows():
    k_neighbour_list = list(smp_smp_graph_dict[str(index)].keys())                                                                #filling the laplacian matrix
    for neighbour in k_neighbour_list:
        laplacian_df[str(index)][neighbour] = -1
    row[index] = k

laplacian_df = laplacian_df.transpose()

eigenvalues, eigenvectors = eig(laplacian_df)                                                       #calculating eiganvalues and eiganvectors of laplacian matrix
eiganvaluescopy = list(eigenvalues)

min_eigan = min(eiganvaluescopy)

eiganvaluescopy.remove(min_eigan)

min2_eigan = min_eigan = min(eiganvaluescopy)                                                             #getting second smallest eigan value

min2_index = list(eigenvalues).index(min2_eigan)                                                  #index of that eigan value

vector_min2_array = eigenvectors[:,min2_index]                                                      #getting desired eigan vector
vector_min2 = pd.DataFrame(list(vector_min2_array), index = sample_name)                    #putting sample names as index labels
sorted_vector_min2 = vector_min2.sort_values([0],ascending = False)                                 #Sorting the desired eigan vector

count = []
for i in range(0,c):                                                                                #Creating empty cluster and a list containg number of elements in each cluster
    globals()['group%s' % i] = []
    count.append(0)


def divisionele(array, avg):                                                                        #function to find element with eigan-vector value closest to the mean
    array = np.asarray(array)
    index_val = (np.abs(array - avg)).argmin()
    return array[index_val], index_val

temp_grp1 = []                                                                                      #temporary groups for hadling exchange and division
temp_grp2 = []
intial_pos_len = 0                                                                                  #variables to store length intial groups
intial_neg_len = 0


for element in sorted_vector_min2.index:
    if sorted_vector_min2[0][element] > 0 or sorted_vector_min2[0][element] == 0:
        temp_grp1.append(element)                                                                   #Initial division of elements as per the rules describes in report
    if sorted_vector_min2[0][element] < 0:
        temp_grp2.append(element)

intial_pos_len = len(temp_grp1)                                                                     #storing length of each group
intial_neg_len = len(temp_grp2)

if intial_pos_len > intial_neg_len:                                                                 #
    globals()['group%s' % 0] = temp_grp2                                                            #
    count[0] = intial_neg_len                                                                       #keeping group with smaller length
elif intial_pos_len < intial_neg_len:                                                               #
    globals()['group%s' % 0] = temp_grp1                                                            #
    count[0] = intial_pos_len                                                                       #



sorted_small = sorted_vector_min2
sorted_small = sorted_small.drop(globals()['group%s' % 0], axis = 0)                                #this sorted_small is our 'non-cluster' group, it is sorted and smaller than the original list
count.append(len(sorted_small))                                                                     #count[c] will be used to store the value of sorted_small set

for c_count in range(1,c):
    if not c_count == (c-1):                                                                        #if cluster is not the last cluser to be formed
        largest_grp_size = max(count)                                                               #largest gorup size
        largest_grp_no = count.index(largest_grp_size)                                              #Getting gorup number
        if not largest_grp_no == c:                                                                 #If lagest group is not the 'non-cluster' group then swap it with 'non-cluster' group
            temp_grp = globals()['group%s' % largest_grp_no]
            temp_count = count[largest_grp_no]
            globals()['group%s' % largest_grp_no] = list(sorted_small.index)
            count[largest_grp_no] = len(globals()['group%s' % largest_grp_no])
            sorted_small = sorted_vector_min2
            sorted_small = sorted_small.loc[temp_grp]
            count[c] = temp_count

        average = sorted_small[0].mean()                                                        #getting the mean of 'non-cluster' group
        divi_ele_value, divi_ele_index = divisionele(sorted_small[0], average)
        divi_ele_label = sorted_small.index[sorted_small[0] == divi_ele_value][0]               #getting the label(sample name) of element with value closest to mean
        temp_grp1 = list(sorted_small.loc[:divi_ele_label].index)                               #diving into 2 groups as explained in documentation
        temp_grp2 = sorted_small
        temp_grp2 = list(temp_grp2.drop(temp_grp1, axis = 0).index)
        if len(temp_grp1) > len(temp_grp2):                                                     #
            globals()['group%s' % c_count] = temp_grp2                                          #
            count[c_count] = len(temp_grp2)                                                     #keeping the one with smaller length
        elif len(temp_grp1) < len(temp_grp2):                                                   #
            globals()['group%s' % c_count] = temp_grp1                                          #
            count[c_count] = len(temp_grp1)                                                     #
        sorted_small = sorted_small.drop(globals()['group%s' % c_count], axis = 0)              #sending all other to 'non-cluster' group
        count[c] = len(sorted_small)                                                            #keeping track on count

    elif c_count == (c-1):                                                                          #if the cluster is the las t cluster to be formed than
        globals()['group%s' % c_count] = list(sorted_small.index)                                   #put all remaiing elements in the cluster
        count[c_count] = len(globals()['group%s' % c_count])

count.pop()                                                                                     #removing the count of 'non-cluster' group,  unnecessary now.


''' output is present in groups, named group0 to group(c-1)
    accessed using globals()['group%s'% i]
'''
df["group"] = np.nan
for i in range(0,c):
    for element in (globals()['group%s'% i]):
        df = df.set_value(int(element), 'group', i)
colors = ['r', 'b', 'g', 'y', 'c', 'm']
for i in range(len(df)):
     plt.scatter(df['x1'][i],df['x2'][i], c=colors[int(df['group'][i])], s=10)
plt.ylabel('Value 2',fontdict=font)
plt.xlabel('Value 1',fontdict=font)
if (task==1):
    plt.title('Plot for dataset 1',fontdict=font)
elif task == 2:
    plt.title('Plot for dataset 2',fontdict=font)
elif task == 3:
    plt.title('Plot for dataset 3',fontdict=font)
plt.show()



## Count for each group
for i in range(0,c):
    print("\nNumber of elements in Group "+str(i)+": "+str(count[i]))
