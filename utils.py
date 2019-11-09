#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:03:19 2018

@author: marin
"""

import numpy as np
from os.path import expanduser
import matplotlib.pyplot as plt
import os

N = 16      # number of strips
bin_Num = 100       # number of bins in histogram
# PATH = expanduser("~/Documents/G4WORKDIR/carbon6/output/")
PATH = os.path.join("data", "raw")
FILE = "run.7.dat"
OUT_PATH = "out/"


"""def loadDict(file = FILE, path = PATH):
    dictionary = {}

    with open(path+file, "r") as f:
        for line in f:
            tmp = line.split(" ")
            if(tmp[2] in dictionary):
                dictionary[tmp[2]].append((int(tmp[0]),float(tmp[-2])))
            else:
                dictionary.update({tmp[2]:[(int(tmp[0]),float(tmp[-2]))]})

    return (file, dictionary)"""


# Load dictionary with data from one file
def loadDict(file, path=PATH):
    dictionary = {}
    with open(os.path.join(path, file), "r") as f:
        for line in f:
            tmp = line.split(" ")
            if(int(tmp[0]) in dictionary):
                dictionary[int(tmp[0])].append((tmp[2], float(tmp[-2])))
            else:
                dictionary.update({int(tmp[0]): [(tmp[2], float(tmp[-2]))]})
    power = len(str(max(dictionary.keys())))
    num_of_events = 10**power
    print("FILE LOADED:  "+os.path.join(path, file))
    return dictionary, num_of_events


#  Load all cosinuses to one array, from one or multiple files
def loadCosFull(file, path=PATH, multiple_files=False):
    array = []
    max_nums = []
    num_of_events = 0
    if(multiple_files is True):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                with open(os.path.join(subdir, file), "r") as f:
                    for line in f:
                        tmp = line.split(" ")
                        array.append(float(tmp[-2]))
                        max_num = int(tmp[0])
                    max_num.append(max_num)
    else:
        with open(os.path.join(path, file), "r") as f:
            for line in f:
                tmp = line.split(" ")
                array.append(float(tmp[-2]))
                max_num = int(tmp[0])
            max_nums.append(max_num)
    for n in max_nums:
        curr_power = len(str(n))
        num_of_events += 10**curr_power
    print("FILE LOADED:  "+os.path.join(path, file))
    return np.asarray(array), num_of_events


#  Load all cosinuses to one array, from multiple files -- UNUSED
#  -->> UNUSED
def loadCosFull_oneFile(path=PATH):
    array = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(subdir, file), "r") as f:
                for line in f:
                    tmp = line.split(" ")
                    array.append(float(tmp[-2]))
    print("FILE LOADED:  "+os.path.join(path, file))
    return np.asarray(array)
#  --<< UNUSED


#  Create numpy histogram   --  UNUSED
def createHist(name_array, bin_num=bin_Num, auto_bins=False):
    # name = name_array[0]; array = name_array[1]
    if(auto_bins is True):
        hist = np.histogram(name_array[0], bins='auto')
    else:
        hist = np.histogram(name_array[0], bins=np.arange(-1, 1, 2/bin_num))
    return (name_array[0], hist)


#  Save Histogram to png image
def plotHist(name_array, bin_num=bin_Num, plot=True, auto_bins=False, toFile=False, normed=False, weights=None, show=False):
    # if not os.path.exists(OUT_PATH+file+"/"):
    #     os.makedirs(OUT_PATH+file+"/")

    if plot is True:
        if auto_bins is True:
            plt.figure()
            n, bins = plt.hist(name_array[1], bins='auto', histtype='step', density=normed, weights=weights)
            if toFile is True:
                if os.path.isfile(OUT_PATH+name_array[0]+".png"):
                    os.remove(OUT_PATH+name_array[0]+".png")   # Opt.: os.system("rm "+OUT_PATH+name_array[0]+".png")
                plt.savefig(OUT_PATH+name_array[0]+".png")
            if show is True:
                plt.show()
            # plt.close()
        else:
            plt.figure()
            n, bins = plt.hist(name_array[1], bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=normed, weights=weights)
            if toFile is True:
                if os.path.isfile(OUT_PATH+name_array[0]+".png"):
                    os.remove(OUT_PATH+name_array[0]+".png")   # Opt.: os.system("rm "+OUT_PATH+name_array[0]+".png")
                plt.savefig(OUT_PATH+name_array[0]+".png")
            if show is True:
                plt.show()
            # plt.close()
    else:
        if auto_bins is True:
            n, bins = np.histogram(name_array[1], bins='auto', density=normed, weights=weights)
        else:
            n, bins = np.histogram(name_array[1], bins=np.arange(-1, 1, 2/bin_num), density=normed, weights=weights)

    return n, bins


def plotToOneHist(name_array, bin_num=bin_Num, auto_bins=False, normed=True, weights=None):
    # if not os.path.exists(OUT_PATH+file+"/"):
    #     os.makedirs(OUT_PATH+file+"/")
    if(auto_bins is True):
        n, bins, patches = plt.hist(name_array[1], bins='auto', histtype='step', density=normed, weights=weights)
    else:
        n, bins, patches = plt.hist(name_array[1], bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=normed, weights=weights)
    return n, bins, patches


#  OLD FUNCTION, UNUSED -- load data for one strip pair
#  -->> UNUSED
def cosForDetectorPair(strip1, strip2, data):
    # print(data[strip1]); print(data[strip2])
    array1 = [];  # array2 = []
    eventNums1 = np.array(data[strip1])[:, 0]
    eventNums2 = np.array(data[strip2])[:, 0]
    cosThetas1 = np.array(data[strip1])[:, 1]
    # cosThetas2 = np.array(data[strip2])[:,1]
    for a, c in zip(eventNums1, cosThetas1):
        for b in eventNums2:
            if(a == b):
                array1.append(c)
                # array2.append(d)
    # return array1, array2
    return (strip1+"-"+strip2, array1)
#  --<< UNUSED


# Load dictonary of histograms for all combos of strips, from multiple files
def histFromDictList(data_list):
    hists = {}
    with open("debug.txt", "w") as f:
        for data in data_list:
            for v in data.values():
                i = 0
                for l in v:
                    i += 1
                    if(l[0][3] == "1"):
                        j = i
                        while(j < len(v)):
                            # if(l[0] != v[j][0]):
                            if((l[0]+"-"+v[j][0]) in hists):
                                hists[l[0]+"-"+v[j][0]].append(float(l[1]))
                                f.write(l[0]+"-"+v[j][0]+":  "+str(l[1])+"\n")
                            else:
                                hists.update({(l[0] + "-" + v[j][0]): [float(l[1])]})
                                f.write(l[0]+"-"+v[j][0]+":  "+str(l[1])+"\n")
                            j += 1
    return hists


# used for debug
def writeHistsToFile(hists):
    with open("hists.txt", "w") as f:
        for h in hists:
            f.write(h+":  "+str(hists[h])+"\n")


# Load dictonary of histograms for all combos of strips
def histFromDict(data):
    hists = {}
    for v in data.values():
        i = 0
        for l in v:
            i += 1
            if(l[0][3] == "1"):
                j = i
                while(j < len(v)):
                    if(l[0] != v[j][0]):
                        if((l[0]+"-"+v[j][0]) in hists):
                            hists[l[0]+"-"+v[j][0]].append(float(l[1]))
                        else:
                            hists.update({(l[0] + "-" + v[j][0]): [float(l[1])]})
                    j += 1
    return hists


# Load dictionary with data from multiple files
def loadAllFiles(path=PATH):
    data_list = []
    tot_num = 0
    for subdir, dirs, files in os.walk(path):
            for file in files:
                data, num_of_events = loadDict(file, subdir)
                data_list.append(data)
                tot_num += num_of_events
    return data_list, tot_num
