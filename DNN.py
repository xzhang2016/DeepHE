# Copy right (c) Xue Zhang and Weijia Xiao 2020. All rights reserved.
#
#

import os
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping


logging.basicConfig(format='%(levelname)s: %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('DNN')


# parameter dictionary
paramDict = {
    'epoch': 200,
    'batchSize': 32,
    'dropOut': 0.2,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'activation1': 'relu',
    'activation2': 'sigmoid',
    'monitor': 'val_accuracy',
    'save_best_only': True,
    'mode': 'max'
}

class_weight = {0: 1.0, 1: 4.0}

optimizerDict = {
    'adam': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
}

hl = [128, 256, 512, 1024, 1024, 1024,1024, 1024, 1024, 1024];

def make_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
        logger.info('{} is created.'.format(folder))
    else:
        logger.info('{} is already there.'.format(folder))


class DNN(object):

    def __init__(self, pdata, f_tp, f_fp, f_th, expName, iteration, numHidden=3, result_dir='node2vec/results/'):
        super(DNN, self).__init__()
        self.pdata = pdata
        self.expName = expName
        self.model_dir = os.path.join(result_dir, 'model')
        make_folder(self.model_dir)

        self.evaluationInfo = dict()

        self.trainingData, self.validationData, self.testingData = pdata.partitionDataset()

        X_train, Y_train = separateDataAndClassLabel(self.trainingData)
        X_valid, Y_valid = separateDataAndClassLabel(self.validationData)
        X_test, Y_test = separateDataAndClassLabel(self.testingData)

        #
        X_train = pdata.getScaledData(X_train)
        X_valid = pdata.getScaledData(X_valid)
        X_test = pdata.getScaledData(X_test)

        self.numberOfClasses = encodeClassLabel(Y_train)
        self.numberOfFeature = X_train.shape[1]

        # reshaping class labels
        Y_train_reshaped = np_utils.to_categorical(Y_train, self.numberOfClasses)
        Y_valid_reshaped = np_utils.to_categorical(Y_valid, self.numberOfClasses)
        Y_test_reshaped = np_utils.to_categorical(Y_test, self.numberOfClasses)

        self.dataDict = {
            'train': X_train,
            'trainLabel': Y_train_reshaped,
            'valid': X_valid,
            'validLabel': Y_valid_reshaped,
            'test': X_test,
            'testLabel': Y_test_reshaped
        }

        self.evaluationInfo = buildModel(self.dataDict, self.numberOfFeature, self.numberOfClasses,
                                               f_tp, f_fp, f_th, expName, iteration, self.model_dir, result_dir)
        self.evaluationInfo['numTrain'] = X_train.shape[0]
        self.evaluationInfo['numTest'] = X_test.shape[0]
        self.evaluationInfo['numValidation'] = X_valid.shape[0]
        self.evaluationInfo['numFeature'] = self.numberOfFeature

    def getEvaluationStat(self):
        return self.evaluationInfo

# returns the TP, TN, FP and FN values
def getTPTNValues(test, testPred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(testPred)):
        if test[i] == testPred[i] == 1:
            TP += 1
        elif testPred[i] == 1 and test[i] != testPred[i]:
            FP += 1
        elif test[i] == testPred[i] == 0:
            TN += 1
        elif testPred[i] == 0 and test[i] != testPred[i]:
            FN += 1

    return TP, TN, FP, FN


# separating feature matrix and class label
def separateDataAndClassLabel(dataMatrix):
    featureMatrix = dataMatrix[:, :(dataMatrix.shape[1] - 1)]
    classLabelMatrix = dataMatrix[:, -1]

    return featureMatrix, classLabelMatrix


# returns the number of classes and encode it
def encodeClassLabel(classLabel):
    labelEncoder = LabelEncoder().fit(classLabel)
    labels = labelEncoder.transform(classLabel)
    classes = list(labelEncoder.classes_)
    return len(classes)

# building the DNN model and run with the data, returns a list of metrics
def buildModel(dataDict, numFeat, numberOfClasses, f_tp, f_fp, f_th, expName, iteration, model_dir, result_dir):
    trainData = dataDict['train']
    trainLabel = dataDict['trainLabel']
    validData = dataDict['valid']
    validLabel = dataDict['validLabel']
    testData = dataDict['test']
    testLabel = dataDict['testLabel']

    # building NN model
    model = Sequential()
    model.add(Dense(hl[0], activation = paramDict['activation1'], input_shape = (numFeat, )))
    model.add(Dropout(paramDict['dropOut']))
    for i in range(1, numHidden):
        if i < len(hl):
            model.add(Dense(hl[i], activation = paramDict['activation1']))
            model.add(Dropout(paramDict['dropOut']))
        else:
            model.add(Dense(1024, activation = paramDict['activation1']))
            model.add(Dropout(paramDict['dropOut']))
    
    model.add(Dense(numberOfClasses, activation=paramDict['activation2']))

    model.compile(optimizer=optimizerDict['adam'],
                  loss=paramDict['loss'],
                  metrics=paramDict['metrics'])
    
    # saving best model by validation accuracy
    filePath = os.path.join(model_dir, expName + str(iteration) + '_weights.best.hdf5')
    checkpointer = ModelCheckpoint(filepath=filePath, verbose=0, monitor=paramDict['monitor'], save_best_only=True)
    earlystopper = EarlyStopping(paramDict['monitor'], patience=15, verbose=1)

    # fit the model to the training data and verify with validation data
    model.fit(trainData, trainLabel,
              epochs=paramDict['epoch'],
              callbacks=[checkpointer, earlystopper],
              batch_size=paramDict['batchSize'],
              shuffle=True,
              verbose=1,
              validation_data=(validData, validLabel), class_weight = class_weight)

    # load best model and compile
    model.load_weights(filePath)
    model.compile(optimizer=optimizerDict['adam'],
                  loss=paramDict['loss'],
                  metrics=paramDict['metrics'])
    
    # serialize model to JSON (save the model structure in order to use the saved weights)
    #one time save
    fn = os.path.join(model_dir, 'model3.json')
    if not os.path.isfile(fn):
        model_json = model.to_json()
        with open(fn, 'w') as json_file:
            json_file.write(model_json)
            
    #save model for later use (including model structure and weights)
    model_file = os.path.join(model_dir, expName + str(iteration) + '_model.h5')
    model.save(model_file)
    
    # evaluation scores
    roc_auc = metrics.roc_auc_score(testLabel, model.predict(testData))
    
    #precision here is the auc of precision-recall curve
    precision = metrics.average_precision_score(testLabel, model.predict(testData))

    # get predicted class label
    probs = model.predict_proba(testData)
    testPredLabel = model.predict(testData)
    true_y = list()
    for y_i in range(len(testLabel)):
        true_y.append(testLabel[y_i][1])
    probs = probs[:, 1]

    fpr, tpr, threshold = metrics.roc_curve(true_y, probs)

    for i in range(len(fpr)):
        f_fp.write(str(fpr[i]) + '\t')
    f_fp.write('\n')

    for i in range(len(tpr)):
        f_tp.write(str(tpr[i]) + '\t')
    f_tp.write('\n')

    for i in range(len(threshold)):
        f_th.write(str(threshold[i]) + '\t')
    f_th.write('\n')
    
    #save precision, recall, and thresholds for PR curve plot
    p0, r0, t0 = metrics.precision_recall_curve(true_y, probs)
    fnp0 = os.path.join(result_dir, expName + '_precision.txt')
    fnr0 = os.path.join(result_dir, expName + '_recall.txt')
    fnt0 = os.path.join(result_dir, expName + '_PR_threshold.txt')
    with open(fnp0, 'a') as f0:
        for i in range(len(p0)):
            f0.write(str(p0[i]) + '\t')
        f0.write('\n')
            
    with open(fnr0, 'a') as f0:
        for i in range(len(r0)):
            f0.write(str(r0[i]) + '\t')
        f0.write('\n')
    
    with open(fnt0, 'a') as f0:
        for i in range(len(t0)):
            f0.write(str(t0[i]) + '\t')
        f0.write('\n')
    
    # convert back class label from categorical to integer label
    testLabelRev = np.argmax(testLabel, axis=1)
    testPredLabelRev = np.argmax(testPredLabel, axis=1)

    # get TP, TN, FP, FN to calculate sensitivity, specificity, PPV and accuracy
    TP, TN, FP, FN = getTPTNValues(testLabelRev, testPredLabelRev)

    sensitivity = float(TP) / float(TP + FN)
    specificity = float(TN) / float(TN + FP)
    PPV = float(TP) / float(TP + FP)
    accuracy = float(TP + TN) / float(TP + FP + FN + TN)

    # dictionary to store evaluation stat
    evaluationInfo = {
        'roc_auc': roc_auc,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'PPV': PPV,
        'accuracy': accuracy,
        'batch_size': paramDict['batchSize'],
        'activation': paramDict['activation2'],
        'dropout': paramDict['dropOut']
    }

    return evaluationInfo
    
