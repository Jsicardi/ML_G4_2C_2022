from models import HierarchicalGroup,HierarchicalProperties,HierarchicalObservables
import numpy as np
import math
import sys

def min_distance(group1:HierarchicalGroup,group2:HierarchicalGroup):
    distances = []
    for member1 in group1.members:
        for member2 in group2.members:
            distances.append(np.linalg.norm(member1 - member2))
    return np.amin(distances)

def max_distance(group1:HierarchicalGroup,group2:HierarchicalGroup):
    distances = []
    for member1 in group1.members:
        for member2 in group2.members:
            distances.append(np.linalg.norm(member1 - member2))
    return np.amax(distances)

def avg_distance(group1:HierarchicalGroup,group2:HierarchicalGroup):
    distances = []
    for member1 in group1.members:
        for member2 in group2.members:
            distances.append(np.linalg.norm(member1 - member2))
    return np.mean(distances)

def centroid_distance(group1:HierarchicalGroup,group2:HierarchicalGroup):
    centroid1 = np.mean(group1.members, axis=0)
    centroid2 = np.mean(group2.members, axis=0)
    return np.linalg.norm(centroid1-centroid2)

def get_distance(group1:HierarchicalGroup,group2:HierarchicalGroup,method):
    if(method == 'min'):
        return min_distance(group1,group2)
    elif(method == 'max'):
        return max_distance(group1,group2)
    elif(method == 'avg'):
        return avg_distance(group1,group2)
    elif(method == 'centroid'):
        return centroid_distance(group1,group2)

def create_groups(input_set,input_genres):
    groups = []
    for (idx,input) in enumerate(input_set):
        groups.append(HierarchicalGroup([input],[input_genres[idx][0]]))
    return groups

def merge_groups(g1:HierarchicalGroup,g2:HierarchicalGroup):
    new_group_values = np.concatenate((g1.members, g2.members), axis=0)
    new_group_genres = np.concatenate((g1.members_genres,g2.members_genres), axis=0)
    return HierarchicalGroup(new_group_values,new_group_genres)

def execute(properties:HierarchicalProperties):
    
    groups = create_groups(properties.input_set,properties.input_genres)
    total_groups = len(groups)
    min_distance = sys.maxsize
    min_g1 = 0
    min_g2 = 0
    
    distances = np.zeros((len(groups), len(groups)))
    np.fill_diagonal(distances, math.inf)

    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            distance = get_distance(groups[i],groups[j],properties.distance_method)
            distances[i][j] = distance
            distances[j][i] = distance

    while(total_groups > properties.k):

        #print(distances)
        
        #calculate distances between groups
        # for i in range(len(groups)):
        #     for j in range(i+1,len(groups)):
        #         groups_distance = distances[i][j]
        #         if(groups_distance <= min_distance): 
        #             min_distance = groups_distance
        #             min_g1=i
        #             min_g2=j
        
        min_distance = np.amin(distances)
        result = np.where(distances == min_distance)
        min_g1 = list(zip(result[0], result[1]))[0][0]
        min_g2 = list(zip(result[0], result[1]))[0][1]
        
        #print('{0},{1}'.format(min_g1,min_g2))
        #print(np.amin(distances))
        #result = np.where(distances == np.amin(distances))
        #print(list(zip(result[0], result[1]))[0][0])

        #merge groups with minor distance
        group2:HierarchicalGroup = groups.pop(min_g2)
        group1:HierarchicalGroup = groups[min_g1]

        new_group:HierarchicalGroup = merge_groups(group1,group2)
        #print(new_group.members)
        groups[min_g1]=new_group

        #erase 
        distances = np.delete(distances,min_g2,0)
        distances = np.delete(distances,min_g2,1)

        for i in range(len(distances)):
            if(i == min_g1):
                distances[i][min_g1] = math.inf
            else:      
                distance = get_distance(groups[i], new_group,properties.distance_method)
                distances[i][min_g1] = distance
                distances[min_g1][i] = distance

        total_groups-=1
        min_distance = sys.maxsize
        print(total_groups)

    
    return HierarchicalObservables(groups)