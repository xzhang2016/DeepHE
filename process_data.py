##########################################################
#copy right (c) Xue Zhang 2020, all rights reserved.
#use sequence features and network embedding features
#for 2009 essential genes and 8430 nonessential genes
#########################################################

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import math
import os
import pickle

logging.basicConfig(format='%(levelname)s: %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('ProcessDataset')


class ProcessDataset(object):
    def __init__(self, data_dir='data/', trainProp=0.8, ExpName='Balance-10percent', embedF=3, fold=4):
        super(ProcessDataset, self).__init__()
        
        self.data_dir = data_dir
        self.trainProp = trainProp
        self.ExpName = ExpName
        self.embedF = embedF
        self.fold = fold
        
    def getScaledData(self, dataMatrix):
        #return scaled dataset
        scaler = StandardScaler().fit(dataMatrix)
        return scaler.transform(dataMatrix)
    
    def partitionDataset(self):
        """
        embedF: 0 for sequence feature, 1 for embedding feature, and other values for the 
                combination of these two types of features
        """
        if self.embedF == 1:
            fn1 = os.path.join(self.data_dir, 'ess_embedFeature.pickle')
            fn2 = os.path.join(self.data_dir, 'ness_embedFeature.pickle')
        elif self.embedF == 0:
            fn1 = os.path.join(self.data_dir, 'ess_seqFeature.pickle')
            fn2 = os.path.join(self.data_dir, 'ness_seqFeature.pickle')
        else:
            fn1 = os.path.join(self.data_dir, 'ess_seqFeature_embedF.pickle')
            fn2 = os.path.join(self.data_dir, 'ness_seqFeature_embedF.pickle')
        
        if all([os.path.isfile(fn1), os.path.isfile(fn2)]):
            essGeneFeatTable = load_pickle(fn1)
            nessGeneFeatTable = load_pickle(fn2)
        else:
            sys.exit("Feature files {} and {} do not exist, please check!".format(fn1, fn2))
    
        trainData, validationData, testData = splitDataset(essGeneFeatTable, nessGeneFeatTable, self.trainProp, fold=self.fold)
        logger.info('trainData.shape={}*{}.'.format(trainData.shape[0],trainData.shape[1]))
        
        return trainData,validationData,testData

def splitDataset(essFeatTable, nessFeatTable, trainingProp, fold=4):
    """
    The size of nonessential genes is 4 fold of that of essential genes, so
    parameter fold should satisfy 1 <= fold <= 4
    """
    ness_num = math.ceil(essFeatTable.shape[0] * fold)
    nessTotal = nessFeatTable.shape[0]
    if ness_num <= nessTotal:
        nessTable = nessFeatTable[np.random.choice(nessTotal, ness_num, replace=False), :]
    else:
        nessTable = nessFeatTable[np.random.choice(nessTotal, ness_num, replace=True), :]
    
    # calculating training, validation, testing data portion
    validationProp, testingProp = float(1 - trainingProp) / 2, float(1 - trainingProp) / 2

    # shuffling the data to mix the data before splitting the dataset into training, validation and testing data
    np.random.shuffle(essFeatTable)
    np.random.shuffle(nessTable)

    # getting the shape of the reSized dataset to find the training, validation and testing size
    row1, col1 = essFeatTable.shape
    logger.info("essFeatTable has {} rows and {} columns.".format(row1, col1))
    
    row2, col2 = nessTable.shape
    logger.info("nessTable has {} rows and {} columns.".format(row2, col2))
    
    trainingSize = math.ceil(row1 * trainingProp)
    validationSize = int(row1 * validationProp)
    testingSize = row1 - trainingSize - validationSize

    etrainingData = essFeatTable[0:trainingSize, :]
    evalidationData = essFeatTable[trainingSize:(trainingSize + validationSize), :]
    etestingData = essFeatTable[(trainingSize + validationSize):, :]
    
    trainingSize = math.ceil(row2 * trainingProp)
    validationSize = int(row2 * validationProp)
    testingSize = row2 - trainingSize - validationSize
    
    netrainingData = nessTable[0:trainingSize, :]
    nevalidationData = nessTable[trainingSize:(trainingSize + validationSize), :]
    netestingData = nessTable[(trainingSize + validationSize):, :]
    
    trainingData = np.vstack((etrainingData, netrainingData))
    validationData = np.vstack((evalidationData, nevalidationData))
    testingData = np.vstack((etestingData, netestingData))
    
    np.random.shuffle(trainingData)
    np.random.shuffle(validationData)
    np.random.shuffle(testingData)
    
    logger.info("trainingData.shape={}*{}".format(trainingData.shape[0], trainingData.shape[1]))
    logger.info("validationData.shape={}*{}".format(validationData.shape[0], validationData.shape[1]))
    logger.info("testingData.shape={}*{}".format(testingData.shape[0], testingData.shape[1]))

    return trainingData, validationData, testingData
    
def save_pickle(fn, data):
    with open(fn, 'wb') as pickle_out:
        pickle.dump(data, pickle_out)
        
def load_pickle(fn):
    with open(fn, 'rb') as pickle_in:
        data = pickle.load(pickle_in)
    return data
