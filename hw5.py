# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:04:44 2017

@author: Rsoni
"""

import sys
import random
import numpy
import matplotlib.pyplot as plt
import math
from random import randrange
from math import sqrt
from scipy.spatial.distance import cdist
import numpy.ma as ma
from scipy.cluster import hierarchy
from matplotlib import patches

def read_file_raw(file):
    rawData={}
    rawDictLabelAllIds={}
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            #token[0] image id, token[1] class label, token[2] pixels
            imageId=int(token[0])
            classLabel=int(token[1])
            pixels=token[2:]
            pixels=numpy.array(list(map(int,pixels)))
            rawData[imageId]=(classLabel,pixels)
            #rawData.append([classLabel,pixels])   #each image is appended
            if classLabel not in rawDictLabelAllIds:
                #insert label in dict
                rawDictLabelAllIds[classLabel]=[(imageId)]
            else:
                rawDictLabelAllIds[classLabel].append(imageId)
    return rawData,rawDictLabelAllIds

def read_file_raw_matrix(file):
    data_coordinates=[]
    class_labels=[]
    
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            #token[0] image id, token[1] class label, token[2] pixels
            imageId=int(token[0])
            classLabel=int(token[1])
            pixels=token[2:]
            pixels=numpy.array(list(map(int,pixels)))
            data_coordinates.append(pixels)
            class_labels.append(classLabel)
        data_coordinates_matrix=numpy.array(data_coordinates)
        class_labels=numpy.array(class_labels)
        return data_coordinates_matrix,class_labels
    
def read_file_raw_matrix_2467(file):
    data_coordinates=[]
    class_labels=[]
    
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            #token[0] image id, token[1] class label, token[2] pixels
            imageId=int(token[0])
            classLabel=int(token[1])
            pixels=token[2:]
            pixels=numpy.array(list(map(int,pixels)))
            if(classLabel==2 or classLabel==4 or classLabel==6 or classLabel==7):
                data_coordinates.append(pixels) 
                class_labels.append(classLabel)
        data_coordinates_matrix=numpy.array(data_coordinates)
        class_labels=numpy.array(class_labels)
        return data_coordinates_matrix,class_labels
    
def read_file_raw_matrix_67(file):
    data_coordinates=[]
    class_labels=[]
    
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            #token[0] image id, token[1] class label, token[2] pixels
            imageId=int(token[0])
            classLabel=int(token[1])
            pixels=token[2:]
            pixels=numpy.array(list(map(int,pixels)))
            if(classLabel==6 or classLabel==7):
                data_coordinates.append(pixels) 
                class_labels.append(classLabel)
        data_coordinates_matrix=numpy.array(data_coordinates)
        class_labels=numpy.array(class_labels)
        return data_coordinates_matrix,class_labels

def read_file_embedding(file):
    embeddingData={}
    embeddingDataLabelAllIds={}
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            imageId=int(token[0])
            classLabel=int(token[1])
            coordinates=token[2:]
            coordinates=numpy.array(list(map(float,coordinates)))
            embeddingData[imageId]=(classLabel,coordinates)
            if classLabel not in embeddingDataLabelAllIds:
                #insert label in dict
                embeddingDataLabelAllIds[classLabel]=[(imageId)]
            else:
                embeddingDataLabelAllIds[classLabel].append(imageId)
    return embeddingData,embeddingDataLabelAllIds

def read_file_embedding_2467(file):
    embeddingData={}
    embeddingDataLabelAllIds={}
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            imageId=int(token[0])
            classLabel=int(token[1])
            coordinates=token[2:]
            coordinates=numpy.array(list(map(float,coordinates)))
            embeddingData[imageId]=(classLabel,coordinates)
            if(classLabel==2 or classLabel==4 or classLabel==6 or classLabel==7):
                if classLabel not in embeddingDataLabelAllIds:
                    #insert label in dict
                    embeddingDataLabelAllIds[classLabel]=[(imageId)]
                else:
                    embeddingDataLabelAllIds[classLabel].append(imageId)
    return embeddingData,embeddingDataLabelAllIds

def read_file_embedding_67(file):
    embeddingData={}
    embeddingDataLabelAllIds={}
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            imageId=int(token[0])
            classLabel=int(token[1])
            coordinates=token[2:]
            coordinates=numpy.array(list(map(float,coordinates)))
            embeddingData[imageId]=(classLabel,coordinates)
            if(classLabel==6 or classLabel==7):
                if classLabel not in embeddingDataLabelAllIds:
                    #insert label in dict
                    embeddingDataLabelAllIds[classLabel]=[(imageId)]
                else:
                    embeddingDataLabelAllIds[classLabel].append(imageId)
    return embeddingData,embeddingDataLabelAllIds
            
def read_file_embedding_matrix(file):
    data_coordinates=[]
    class_labels=[]
    embeddingDataLabelAllIds={}
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            imageId=int(token[0])
            classLabel=int(token[1])
            coordinates=token[2:]
            coordinates=list(map(float,coordinates))
            data_coordinates.append(coordinates)
            class_labels.append(classLabel)
            if classLabel not in embeddingDataLabelAllIds:
                #insert label in dict
                embeddingDataLabelAllIds[classLabel]=[(imageId)]
            else:
                embeddingDataLabelAllIds[classLabel].append(imageId)
        data_coordinates_matrix=numpy.array(data_coordinates)
        class_labels=numpy.array(class_labels)
        return data_coordinates_matrix,class_labels,embeddingDataLabelAllIds

def read_file_embedding_matrix_2467(file):
    data_coordinates=[]
    class_labels=[]
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            #imageId=int(token[0])
            classLabel=int(token[1])
            coordinates=token[2:]
            coordinates=list(map(float,coordinates))
            if(classLabel==2 or classLabel==4 or classLabel==6 or classLabel==7):
                data_coordinates.append(coordinates) 
                class_labels.append(classLabel)
        data_coordinates_matrix=numpy.array(data_coordinates)
        class_labels=numpy.array(class_labels)
        return data_coordinates_matrix,class_labels

def read_file_embedding_matrix_67(file):
    data_coordinates=[]
    class_labels=[]
    with open(file) as f: 
        for row in f:
            rowStrip=row.strip()
            token=rowStrip.split(",")
            #imageId=int(token[0])
            classLabel=int(token[1])
            coordinates=token[2:]
            coordinates=list(map(float,coordinates))
            if(classLabel==6 or classLabel==7):
                data_coordinates.append(coordinates)
                class_labels.append(classLabel)
        data_coordinates_matrix=numpy.array(data_coordinates)
        class_labels=numpy.array(class_labels)
        return data_coordinates_matrix,class_labels

def convert_grayscale(rawData,rawDictLabelAllIds,classLabel):
    Ids=rawDictLabelAllIds[classLabel]
    random.shuffle(Ids)
    randomId=Ids[0]                 #random image for this class
    #find pixels for this id
    pixelArray=rawData[randomId][1]
    #print(len(pixelArray))
    grayScale=numpy.reshape(pixelArray,(28,28))
    return grayScale
    
colorDict={0:"black",
           1:"blue",
           2:"darkgray",
           3:"darkgreen",
           4:"darkred",
           5:"purple",
           6:"orange",
           7:"yellow",
           8:"chocolate",
           9:"deeppink"}

def calculate_centroids(k,all_points,which_clusters):
    centroid_array=[]
    for cluster_no in range(k):
        new_examples=all_points[which_clusters==cluster_no]
        new_centroid=numpy.mean(new_examples,axis=0)
        centroid_array.append(new_centroid)
    return centroid_array

def kmeans(data_coordinates_matrix,k):
    #k means************
    while(True):
        #centroid_array=data_coordinates_matrix[[1000,2000,3000,4000,5000,6000,7000,8000]]
        centroid_array=initialize_centroids(k,data_coordinates_matrix)
        prev_which_clusters=(-1) * numpy.ones(len(data_coordinates_matrix))
        for i in range(50):
            distance_matrix=cdist(data_coordinates_matrix,centroid_array)
            #print(distance_matrix)  
            which_clusters=numpy.argmin(distance_matrix,axis=1)
            if((prev_which_clusters==which_clusters).all()):
                break;
            for cluster_no in range(k):
                new_examples=data_coordinates_matrix[which_clusters==cluster_no]
                new_centroid=numpy.mean(new_examples,axis=0)
                centroid_array[cluster_no]=new_centroid  
        if(len(set(which_clusters))==k):
            break;        
    #end of k means
    return which_clusters,centroid_array

def initialize_centroids(k,data_coordinates_matrix):
    #randomly select k points
    randomSamplePoints=random.sample(list(data_coordinates_matrix),k)
    return randomSamplePoints

def wcssd(which_clusters,data_coordinates_matrix,centroid_array,k):
    each_cluster_distance_mean_list=[]
    for cluster_no in range(k):
        each_cluster_examples=data_coordinates_matrix[which_clusters==cluster_no]
        each_cluster_distance=cdist(each_cluster_examples,[centroid_array[cluster_no]])
        each_cluster_distance_mean=numpy.sum(each_cluster_distance**2,axis=0)
        each_cluster_distance_mean_list.append(each_cluster_distance_mean)
    overall_distance_mean=numpy.sum(each_cluster_distance_mean_list)
    return overall_distance_mean

def sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist):
    
    a=numpy.zeros(len(data_coordinates_matrix))
    b=numpy.ones(len(data_coordinates_matrix)) * numpy.inf
    
    for cluster_no in range(k):
        this_cluster_dist=pairwise_dist[which_clusters==cluster_no]
        examples_in_cluster=data_coordinates_matrix[which_clusters==cluster_no]
        #examples_not_in_cluster=data_coordinates_matrix[which_clusters!=cluster_no]
        
        len_examples_in_cluster=len(examples_in_cluster)-1
        if(len_examples_in_cluster!=0):        #members present in cluster
            a[which_clusters==cluster_no]=numpy.sum(this_cluster_dist[:,which_clusters==cluster_no],axis=1)/len_examples_in_cluster
            
        for other_cluster in range(k):
            if(other_cluster != cluster_no):
                oc_dist=numpy.mean(this_cluster_dist[:,which_clusters==other_cluster],axis=1)
                b[which_clusters==cluster_no]=numpy.minimum(b[which_clusters==cluster_no],oc_dist)
            
    sc_each=b-a
    sc_each/=numpy.maximum(a,b)
    sc_each[sc_each == numpy.inf]=0
    
    return numpy.mean(sc_each)

def nmi(which_clusters,data_coordinates_matrix,k,class_labels):
    total_points=len(which_clusters) + 0.0
    unique_classes=list(set(class_labels))
    unique_classes.sort()
    prob_classes=numpy.zeros(len(unique_classes))
    prob_clusters=numpy.zeros(k)
    cg_matrix=numpy.zeros((len(unique_classes),k))   
    
    idx=0
    for each_class_label in unique_classes:
        for cluster_no in range(k):
            matched_index_cluster=numpy.arange(total_points)[which_clusters==cluster_no]
            prob_clusters[cluster_no]=len(matched_index_cluster)
            
            matched_index_class=numpy.arange(total_points)[class_labels==each_class_label]
            prob_classes[idx]=len(matched_index_class)
            
            cg_matrix[idx][cluster_no]=len(numpy.intersect1d(matched_index_class,matched_index_cluster))
        idx=idx+1
    #print(prob_clusters)
    #print(prob_classes)
    #print(cg_matrix)
    prob_clusters=prob_clusters/total_points
    prob_classes=prob_classes/total_points
    cg_matrix=cg_matrix/total_points
    
    cg_matrix_copy=numpy.array(cg_matrix)
    cg_matrix_copy[cg_matrix_copy == 0.0]=numpy.finfo(float).eps
    
    
    product_classes_cluster=numpy.array(numpy.transpose(numpy.matrix(prob_classes)))*prob_clusters
    product_classes_cluster[product_classes_cluster == 0.0]=numpy.finfo(float).eps
    
    
    #calculate log
    log_value=numpy.log(cg_matrix_copy/product_classes_cluster)
    log_value_product=cg_matrix*log_value
    
    numerator=numpy.sum(log_value_product)
    
    log_pc=numpy.log(prob_classes)
    log_pc_product=(log_pc*prob_classes)
    sum_pc_product=numpy.sum(log_pc_product)
    
    log_pg=numpy.log(prob_clusters)
    log_pg_product=(log_pg*prob_clusters)
    sum_pg_product=numpy.sum(log_pg_product)
    
    nmi=numerator/((-1)*(sum_pc_product+sum_pg_product))
    return nmi
    
def sample100_data():
    all_points=[]
    new_class_labels=[]
    data_coordinates_matrix,class_labels,embeddingDataLabelAllIds=read_file_embedding_matrix('digits-embedding.csv')
    unique_classes=list(set(class_labels))
    for each_class_label in unique_classes:
        imageIds=embeddingDataLabelAllIds[each_class_label]
        sample_image_ids=random.sample(imageIds,10)
        for i in range(10):
            new_class_labels.append(each_class_label)
        
        #find coordinates for these Ids
        all_points.extend([data_coordinates_matrix[i] for i in sample_image_ids])
    
    #print(all_points)
    all_points=numpy.array(all_points)
    return all_points,new_class_labels

def cluster_membership_acc_dend(all_points,type_dend,k):
    Z = hierarchy.linkage(all_points, type_dend)
#    plt.figure()
#    dn = hierarchy.dendrogram(Z)
    which_clusters=numpy.array(hierarchy.fcluster(Z,k,"maxclust"))
    which_clusters=which_clusters-1
    return which_clusters

def plot_cluster_membership_acc_dend(all_points,type_dend):
    Z = hierarchy.linkage(all_points, type_dend)
    plt.figure()
    dn = hierarchy.dendrogram(Z)

def pca_reduced_data(full_data,n):
    full_data = full_data - numpy.mean(full_data,axis=0)
    cov_matrix = numpy.dot(numpy.transpose(full_data) , full_data)
    s, V = numpy.linalg.eig(cov_matrix)
    idx = s.argsort()[::-1]
    eig_Vec = V[:,idx]
    W = eig_Vec[:,:n]
    X = numpy.matmul(full_data,W) 
    return W.astype("float64"),X.astype("float64")

ques=00
if(ques==11):
    rawData,rawDictLabelAllIds=read_file_raw('digits-raw.csv')
    for key in rawDictLabelAllIds:          #for each class
        grayScale=convert_grayscale(rawData,rawDictLabelAllIds,key)
        plt.figure(key)
        plt.imshow(grayScale,cmap='gray',interpolation='nearest')
        plt.show()

if(ques==12):
    embeddingData,embeddingDataLabelAllIds=read_file_embedding('digits-embedding.csv')
    #randomly pick 1000 ids
    keys=list(embeddingData.keys())
    random.shuffle(keys)
    xCoordList=[]
    yCoordList=[]
    colorSelectedList=[]
    for i in range(1000):
        randomClassLabel=embeddingData[keys[i]][0]
        xCoord=embeddingData[keys[i]][1][0]
        yCoord=embeddingData[keys[i]][1][1]
        colorSelected=colorDict[randomClassLabel]
        xCoordList.append(xCoord)
        yCoordList.append(yCoord)
        colorSelectedList.append(colorSelected)
        
    plt.scatter(xCoordList,yCoordList,c=colorSelectedList)
    plt.show()

mainQues=1
if(mainQues==1):
    if(len(sys.argv) == 3):     
        trainFileName = sys.argv[1]
        k = int(sys.argv[2])     #no of clusters
        data_coordinates_matrix,class_labels,embeddingDataLabelAllIds=read_file_embedding_matrix(trainFileName)
            
        #centroid_array=[[0,0]]
        
        ##k means
        which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
        ##end of k means
        
        #print(centroid_array)
        overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
        print("WC-SSD ",overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        sc=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
        print("SC ",sc)
        nmi=nmi(which_clusters,data_coordinates_matrix,k,class_labels)
        print("NMI ",nmi)
    else:
        print("Number of arguments is not equal to two. Hence invalid input!!")
        exit()

analysisQues=00
if(analysisQues==21):
    #print("analysis of b1")
    full_wcList=[]
    data2467_wcList=[]
    data67_wcList=[]
    full_scList=[]
    data2467_scList=[]
    data67_scList=[]
    trainFileName='digits-embedding.csv'
    full_data_coordinates_matrix,full_class_labels,embeddingDataLabelAllIds=read_file_embedding_matrix(trainFileName)
    data2467_data_coordinates_matrix,data2467_class_labels=read_file_embedding_matrix_2467(trainFileName)
    data67_data_coordinates_matrix,data67_class_labels=read_file_embedding_matrix_67(trainFileName)
    
    variationOfKList=[2, 4, 8, 16, 32]
    for k in variationOfKList:
        #full data
        
        ##k means
        data_coordinates_matrix=full_data_coordinates_matrix
        class_labels=full_class_labels
        which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
        ##end of k means
        
        #print(centroid_array)
        overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
        full_wcList.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
        full_scList.append(scValue)
        
        #2467 data
        ##k means
        data_coordinates_matrix=data2467_data_coordinates_matrix
        class_labels=data2467_class_labels
        which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
        ##end of k means
        
        #print(centroid_array)
        overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
        data2467_wcList.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
        data2467_scList.append(scValue)
        
        #67 data
        ##k means
        data_coordinates_matrix=data67_data_coordinates_matrix
        class_labels=data67_class_labels
        which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
        ##end of k means
        
        #print(centroid_array)
        overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
        data67_wcList.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
        data67_scList.append(scValue)
    
    #end of for loop
    print("variationOfKList=",variationOfKList)
    print("full_wcList=",full_wcList)
    print("data2467_wcList=",data2467_wcList)
    print("data67_wcList=",data67_wcList)
    print("full_scList=",full_scList)
    print("data2467_scList=",data2467_scList)
    print("data67_scList=",data67_scList)
    
    plt.figure(1)
    plt.errorbar(variationOfKList,full_wcList, marker='^',  label = "Full data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    plt.savefig('b1_wc_full.png')
    
    plt.figure(2)
    plt.errorbar(variationOfKList,data2467_wcList, marker='^',  label = "2467 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    plt.savefig('b1_wc_2467.png')
    
    plt.figure(3)
    plt.errorbar(variationOfKList,data67_wcList, marker='^',  label = "67 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    plt.savefig('b1_wc_67.png')
    
    plt.figure(4)
    plt.errorbar(variationOfKList,full_scList, marker='^',  label = "Full data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()
    plt.savefig('b1_sc_full.png')
    
    plt.figure(5)
    plt.errorbar(variationOfKList,data2467_scList, marker='^',  label = "2467 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()
    plt.savefig('b1_sc_2467.png')
    
    plt.figure(6)
    plt.errorbar(variationOfKList,data67_scList, marker='^',  label = "67 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()
    plt.savefig('b1_sc_67.png')

if(analysisQues==23):
    #print("analysis of b3")

    avg_full_wcList=[]
    std_full_wcList=[]
    
    avg_data2467_wcList=[]
    std_data2467_wcList=[]
    
    avg_data67_wcList=[]
    std_data67_wcList=[]
    
    avg_full_scList=[]
    std_full_scList=[]
    
    avg_data2467_scList=[]
    std_data2467_scList=[]
    
    avg_data67_scList=[]
    std_data67_scList=[]
    
    trainFileName='digits-embedding.csv'
    full_data_coordinates_matrix,full_class_labels,embeddingDataLabelAllIds=read_file_embedding_matrix(trainFileName)
    data2467_data_coordinates_matrix,data2467_class_labels=read_file_embedding_matrix_2467(trainFileName)
    data67_data_coordinates_matrix,data67_class_labels=read_file_embedding_matrix_67(trainFileName)
    
    variationOfKList=[2, 4, 8, 16, 32]
    for k in variationOfKList:
        full_wcList=[]
        data2467_wcList=[]
        data67_wcList=[]
        full_scList=[]
        data2467_scList=[]
        data67_scList=[]
        
        for trial in range(10):
            #full data
            
            ##k means
            data_coordinates_matrix=full_data_coordinates_matrix
            class_labels=full_class_labels
            which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
            ##end of k means
            
            #print(centroid_array)
            overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
            full_wcList.append(overall_distance_mean)
            
            pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
            scValue=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
            full_scList.append(scValue)
            
            #2467 data
            ##k means
            data_coordinates_matrix=data2467_data_coordinates_matrix
            class_labels=data2467_class_labels
            which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
            ##end of k means
            
            #print(centroid_array)
            overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
            data2467_wcList.append(overall_distance_mean)
            
            pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
            scValue=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
            data2467_scList.append(scValue)
            
            #67 data
            ##k means
            data_coordinates_matrix=data67_data_coordinates_matrix
            class_labels=data67_class_labels
            which_clusters,centroid_array=kmeans(data_coordinates_matrix,k)
            ##end of k means
            
            #print(centroid_array)
            overall_distance_mean=wcssd(which_clusters,data_coordinates_matrix,centroid_array,k)
            data67_wcList.append(overall_distance_mean)
            
            pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
            scValue=sc(which_clusters,data_coordinates_matrix,centroid_array,k,pairwise_dist)
            data67_scList.append(scValue)
        #after ten trials
        avg_full_wc=numpy.average(full_wcList)
        std_full_wc=numpy.std(full_wcList)/math.sqrt(10)
        avg_full_wcList.append(avg_full_wc)
        std_full_wcList.append(std_full_wc)
        
        avg_data2467_wc=numpy.average(data2467_wcList)
        std_data2467_wc=numpy.std(data2467_wcList)/math.sqrt(10)
        avg_data2467_wcList.append(avg_data2467_wc)
        std_data2467_wcList.append(std_data2467_wc)
        
        avg_data67_wc=numpy.average(data67_wcList)
        std_data67_wc=numpy.std(data67_wcList)/math.sqrt(10)
        avg_data67_wcList.append(avg_data67_wc)
        std_data67_wcList.append(std_data67_wc)
        
        avg_full_sc=numpy.average(full_scList)
        std_full_sc=numpy.std(full_scList)/math.sqrt(10)
        avg_full_scList.append(avg_full_sc)
        std_full_scList.append(std_full_sc)
        
        avg_data2467_sc=numpy.average(data2467_scList)
        std_data2467_sc=numpy.std(data2467_scList)/math.sqrt(10)
        avg_data2467_scList.append(avg_data2467_sc)
        std_data2467_scList.append(std_data2467_sc)
        
        avg_data67_sc=numpy.average(data67_scList)
        std_data67_sc=numpy.std(data67_scList)/math.sqrt(10)
        avg_data67_scList.append(avg_data67_sc)
        std_data67_scList.append(std_data67_sc)
    
    #end of for loop
    print("variationOfKList=",variationOfKList)
    print("avg_full_wcList=",avg_full_wcList)
    print("std_full_wcList=",std_full_wcList)
    
    print("avg_data2467_wcList=",avg_data2467_wcList)
    print("std_data2467_wcList=",std_data2467_wcList)
    
    print("avg_data67_wcList=",avg_data67_wcList)
    print("std_data67_wcList=",std_data67_wcList)
    
    print("avg_full_scList=",avg_full_scList)
    print("std_full_scList=",std_full_scList)
    
    print("avg_data2467_scList=",avg_data2467_scList)
    print("std_data2467_scList=",std_data2467_scList)
    
    print("avg_data67_scList=",avg_data67_scList)
    print("std_data67_scList=",std_data67_scList) 
    
    plt.figure(1)
    plt.errorbar(variationOfKList,avg_full_wcList,std_full_wcList, marker='.',  label = "Full data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.errorbar(variationOfKList,avg_data2467_wcList,std_data2467_wcList, marker='.',  label = "2467 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    
    plt.figure(3)
    plt.errorbar(variationOfKList,avg_data67_wcList,std_data67_wcList, marker='.',  label = "67 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    
    plt.figure(4)
    plt.errorbar(variationOfKList,avg_full_scList,std_full_scList, marker='.',  label = "Full data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()
    
    plt.figure(5)
    plt.errorbar(variationOfKList,avg_data2467_scList,std_data2467_scList, marker='.',  label = "2467 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()
    
    plt.figure(6)
    plt.errorbar(variationOfKList,avg_data67_scList,std_data67_scList, marker='.',  label = "67 data")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()

if(analysisQues==24):
    
    trainFileName='digits-embedding.csv'
    full_data_coordinates_matrix,full_class_labels,embeddingDataLabelAllIds=read_file_embedding_matrix(trainFileName)
    data2467_data_coordinates_matrix,data2467_class_labels=read_file_embedding_matrix_2467(trainFileName)
    data67_data_coordinates_matrix,data67_class_labels=read_file_embedding_matrix_67(trainFileName)
    
    innerQues=00
    if(innerQues==11):
        #full data
        #provide the value of k
        k=8
        which_clusters,centroid_array=kmeans(full_data_coordinates_matrix,k)
        nmi=nmi(which_clusters,full_data_coordinates_matrix,k,full_class_labels)
        print("NMI-full ",nmi)
        #visulalize
        
        random1000 = random.sample(range(len(which_clusters)),1000)
        plt.figure()
        for i in random1000:
            
            plt.scatter(full_data_coordinates_matrix[i][0],full_data_coordinates_matrix[i][1],color = colorDict[which_clusters[i]])
    
    if(innerQues==12):
        #2467 data
        #provide the value of k
        k=4
        which_clusters,centroid_array=kmeans(data2467_data_coordinates_matrix,k)
        nmi=nmi(which_clusters,data2467_data_coordinates_matrix,k,data2467_class_labels)
        print("NMI-2467 ",nmi)
        #visualize
        random1000 = random.sample(range(len(which_clusters)),1000)
        plt.figure()
        for i in random1000:
            
            plt.scatter(data2467_data_coordinates_matrix[i][0],data2467_data_coordinates_matrix[i][1],color = colorDict[which_clusters[i]])

    if(innerQues==13):
        #67 data
        #provide the value of k
        k=2
        which_clusters,centroid_array=kmeans(data67_data_coordinates_matrix,k)
        nmi=nmi(which_clusters,data67_data_coordinates_matrix,k,data67_class_labels)
        print("NMI-67 ",nmi)
        #visualize
        random1000 = random.sample(range(len(which_clusters)),1000)
        plt.figure()
        for i in random1000:
            
            plt.scatter(data67_data_coordinates_matrix[i][0],data67_data_coordinates_matrix[i][1],color = colorDict[which_clusters[i]])

if(analysisQues==31):
    #hierarchical
        
    all_points,new_class_labels=sample100_data()
    
    #plotting the graphs
    plot_cluster_membership_acc_dend(all_points,'single')
    
    plot_cluster_membership_acc_dend(all_points,'average')
    
    plot_cluster_membership_acc_dend(all_points,'complete')
    
    data_coordinates_matrix=all_points
    class_labels=new_class_labels
    wcListK=[]
    scListK=[]
    wcListSingle=[]
    scListSingle=[]
    wcListComplete=[]
    scListComplete=[]
    wcListAverage=[]
    scListAverage=[]
    variationOfKList=[2, 4, 8, 16, 32]
    for k in variationOfKList:
        ##k means
        which_clusters_kMeans,centroid_array=kmeans(data_coordinates_matrix,k)
        ##end of k means
        
        #print(centroid_array)
        overall_distance_mean=wcssd(which_clusters_kMeans,data_coordinates_matrix,centroid_array,k)
        wcListK.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters_kMeans,data_coordinates_matrix,centroid_array,k,pairwise_dist)
        scListK.append(scValue)
        
        #nmi=nmi(which_clusters,data_coordinates_matrix,k,class_labels)
        
        #hierarchical
        which_clusters_single=cluster_membership_acc_dend(all_points,'single',k)
    
        centroid_array=calculate_centroids(k,all_points,which_clusters_single)
        
        overall_distance_mean=wcssd(which_clusters_single,all_points,centroid_array,k)
        wcListSingle.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters_single,all_points,centroid_array,k,pairwise_dist)
        scListSingle.append(scValue)
        
        #hierarchical
        which_clusters_average=cluster_membership_acc_dend(all_points,'average',k)
    
        centroid_array=calculate_centroids(k,all_points,which_clusters_average)
        
        overall_distance_mean=wcssd(which_clusters_average,all_points,centroid_array,k)
        wcListAverage.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters_average,all_points,centroid_array,k,pairwise_dist)
        scListAverage.append(scValue)
        
        #hierarchical
        which_clusters_complete=cluster_membership_acc_dend(all_points,'complete',k)
    
        centroid_array=calculate_centroids(k,all_points,which_clusters_complete)
        
        overall_distance_mean=wcssd(which_clusters_complete,all_points,centroid_array,k)
        wcListComplete.append(overall_distance_mean)
        
        pairwise_dist=cdist(data_coordinates_matrix,data_coordinates_matrix)
        scValue=sc(which_clusters_complete,all_points,centroid_array,k,pairwise_dist)
        scListComplete.append(scValue)
        
        
    #end of for loop
    plt.figure(4)
    plt.errorbar(variationOfKList,wcListK, marker='^',  label = "K means")
    plt.errorbar(variationOfKList,wcListSingle, marker='^',  label = "Single Linkage")
    plt.errorbar(variationOfKList,wcListComplete, marker='^',  label = "Complete Linkage")
    plt.errorbar(variationOfKList,wcListAverage, marker='^',  label = "Average Linkage")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of wc')
    plt.legend()
    plt.show()
    plt.savefig('c3_wc.png')
    
    plt.figure(5)
    plt.errorbar(variationOfKList,scListK, marker='^',  label = "K means")
    plt.errorbar(variationOfKList,scListSingle, marker='^',  label = "Single Linkage")
    plt.errorbar(variationOfKList,scListComplete, marker='^',  label = "Complete Linkage")
    plt.errorbar(variationOfKList,scListAverage, marker='^',  label = "Average Linkage")
    plt.xlabel('Variation of k')
    plt.ylabel('Value of sc')
    plt.legend()
    plt.show()
    plt.savefig('c3_sc.png')
    
    nmiV=nmi(which_clusters_kMeans,all_points,8,new_class_labels)
    print("NMI kmeans ",nmiV)
    
    nmiV=nmi(which_clusters_single,all_points,16,new_class_labels)
    print("NMI Single ",nmiV)
        
    nmiV=nmi(which_clusters_average,all_points,8,new_class_labels)
    print("NMI Average ",nmiV)
    
    nmiV=nmi(which_clusters_complete,all_points,8,new_class_labels)
    print("NMI Complete ",nmiV)
    
if(analysisQues==42):
        
    ##PCA 2
    full_data_coordinates_matrix,full_class_labels=read_file_raw_matrix('digits-raw.csv')
    data_2467_data_coordinates_matrix,data_2467_class_labels=read_file_raw_matrix_2467('digits-raw.csv')
    data_67_data_coordinates_matrix,data_67_class_labels=read_file_raw_matrix_67('digits-raw.csv')
    
    #full data
    full_eigen_vectors,full_reduced_data=pca_reduced_data(full_data_coordinates_matrix,10)
    for i in range(10):
        principal_component=full_eigen_vectors[:,i]
        reshaped_image=numpy.reshape(principal_component,(28,28))
        plt.figure(i)
        plt.imshow(reshaped_image,cmap="gray")
        
    #2467 data
#    full_eigen_vectors,full_reduced_data=pca_reduced_data(data_2467_data_coordinates_matrix,10)
#    for i in range(10):
#        principal_component=full_eigen_vectors[:,i]
#        reshaped_image=numpy.reshape(principal_component,(28,28))
#        plt.figure(i)
#        plt.imshow(reshaped_image,cmap="gray")
        
    #67 data
#    full_eigen_vectors,full_reduced_data=pca_reduced_data(data_67_data_coordinates_matrix,10)
#    for i in range(10):
#        principal_component=full_eigen_vectors[:,i]
#        reshaped_image=numpy.reshape(principal_component,(28,28))
#        plt.figure(i)
#        plt.imshow(reshaped_image,cmap="gray")
        
    
    ##PCA 3
if(analysisQues==43):   
    #full data
    full_data_coordinates_matrix,full_class_labels=read_file_raw_matrix('digits-raw.csv')
    data_2467_data_coordinates_matrix,data_2467_class_labels=read_file_raw_matrix_2467('digits-raw.csv')
    data_67_data_coordinates_matrix,data_67_class_labels=read_file_raw_matrix_67('digits-raw.csv')
    
    sampled_data=[]
    sampled_class_labels=[]
    sample_index=random.sample(range(len(full_data_coordinates_matrix)),1000)
    #sample_index=range(1000)
    for idx in sample_index:
        sampled_data.append(full_data_coordinates_matrix[idx])
        sampled_class_labels.append(full_class_labels[idx])
    sampled_data=numpy.array(sampled_data)
    
    full_eigen_vectors,full_reduced_data=pca_reduced_data(sampled_data,2)
    plt.figure()
    for i in range(1000):
        plt.scatter(full_reduced_data[i][0],full_reduced_data[i][1],color=colorDict[sampled_class_labels[i]])
   
    
    #2467 data
#    sampled_data=[]
#    sampled_class_labels=[]
#    sample_index=random.sample(range(len(data_2467_data_coordinates_matrix)),1000)
#    #sample_index=range(1000)
#    for idx in sample_index:
#        sampled_data.append(data_2467_data_coordinates_matrix[idx])
#        sampled_class_labels.append(data_2467_class_labels[idx])
#    sampled_data=numpy.array(sampled_data)
#    
#    full_eigen_vectors,full_reduced_data=pca_reduced_data(sampled_data,2)
#    plt.figure()
#    for i in range(1000):
#        plt.scatter(full_reduced_data[i][0],full_reduced_data[i][1],color=colorDict[sampled_class_labels[i]])
   
        
    #67 data
#    sampled_data=[]
#    sampled_class_labels=[]
#    sample_index=random.sample(range(len(data_67_data_coordinates_matrix)),1000)
#    #sample_index=range(1000)
#    for idx in sample_index:
#        sampled_data.append(data_67_data_coordinates_matrix[idx])
#        sampled_class_labels.append(data_67_class_labels[idx])
#    sampled_data=numpy.array(sampled_data)
#    
#    full_eigen_vectors,full_reduced_data=pca_reduced_data(sampled_data,2)
#    plt.figure()
#    for i in range(1000):
#        plt.scatter(full_reduced_data[i][0],full_reduced_data[i][1],color=colorDict[sampled_class_labels[i]])
   
    
    
    
            
               
            
        
        
        
        


        
    
       
    

            
                                
                              
            