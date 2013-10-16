#!/usr/bin/env python

import numpy
import math
import random
import time

#global definitions
kPartitions = 10 #Number of groups or partitions
numOfFeatures = 58 #Number of features
totalMails = 4601
spamMails = 1813
nonspamMails = 2788

def calculateMean(parameterSet):
    mean = [numpy.mean(parameterSet[:, i]) for i in range(parameterSet.shape[1] - 1)]
    return mean

def calculateStandardDeviation(parameterSet):
    sd = [numpy.std(parameterSet[:, i]) for i in range(parameterSet.shape[1] - 1)]
    return sd

def checkSpam(parameterSet):
    ifSpam = parameterSet[:, 57] == 1.0
    spams = parameterSet[ifSpam, :]
    nonSpams = parameterSet[~ifSpam, :]
    totalSpams = len(spams)
    totalNonSpams = len(nonSpams)
    
    return totalSpams, totalNonSpams

def calculateZScores(mails, featureMean, featureSd):
    emailId = 0
    zMap = numpy.empty([4601, 58], dtype=float)
    for emailId in range(len(mails)):
        record = mails[emailId]
        zScoreRecord = [0.0] * numOfFeatures
        for i in range(len(record)):
            if i == 57:
                zScoreRecord[i] = record[i]
            
            else:
                x = record[i]
                mean = featureMean[i]
                sd = featureSd[i]         
                zScore = (float(x) - float(mean)) / float(sd)
                zScoreRecord[i] = zScore
            
        zMap[emailId] = zScoreRecord
    return zMap

def main():
    #read the data
    fp = open("processedData.txt", "r")
    mails = numpy.loadtxt(fp, comments = "numpy loads the dataset into 2D matrix array", delimiter=",")
    totalMails = len(mails) #total no of mails
    mailId = 0
    
    featureMean = calculateMean(mails)
    featureSd = calculateStandardDeviation(mails)
    zScoreMap = calculateZScores(mails, featureMean, featureSd)
   
    groupMap = {} #map to store each mail-record
    
    while mailId < totalMails:
        i = mailId % kPartitions
        if i == 0:
            if mailId == 0:
                groupMap["testingSet"] = [zScoreMap[mailId]]
            else:
                groupMap["testingSet"] = numpy.concatenate((groupMap["testingSet"], [zScoreMap[mailId]]))
        else:
            if mailId == 1:
                groupMap["trainingSet"] = [zScoreMap[mailId]]
            else:
                groupMap["trainingSet"] = numpy.concatenate((groupMap["trainingSet"], [zScoreMap[mailId]]))

        mailId += 1
    
    #print "GroupMap: " + str(groupMap)
    #stochasticTrainingSet = random.shuffle(groupMap["trainingSet"])
    #batchTrainingSet = groupMap["trainingSet"]
    numpy.random.shuffle(groupMap["trainingSet"])
    #print groupMap["trainingSet"]
    linearRegression(groupMap["trainingSet"], groupMap["testingSet"])
    #logisticRegression(groupMap["trainingSet"], groupMap["testingSet"])
    #linearBatRegression(groupMap["trainingSet"], groupMap["testingSet"])
    #logisticBatRegression(groupMap["trainingSet"], groupMap["testingSet"])
    
def linearRegression(trainingSet, testingSet):
    lambdaCnst = 0.0001
    weights = [0.0] * 57
    print trainingSet[0][56]
    recordError = len(trainingSet) * [0.0]
    hwTrain = len(trainingSet) * [0.0]
    fp = open(str(lambdaCnst) + "RMSE.xls", "w")
    converge = False
    prevGradDesc = 0.0
    gradDesc = 0.0
    iteration = 0
    convergenceCriteria = 0.0000001
    print "lambda: " + str(lambdaCnst)
    print "Convergence Criteria: " + str(convergenceCriteria)
     
    while not converge:
        iteration += 1
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            recordError[r] = 0.0
            hw = 0.0
            #print "weights: " + str(weights)
            for i in range(len(record) - 1):
                hw += (float(weights[i]) * float(record[i]))
            
            for i in range(len(record) - 1):
                gradDesc = float(hw - record[57]) * float(record[i])
                weights[i] = weights[i] - (lambdaCnst * gradDesc)
                
            
        if (math.fabs(prevGradDesc - gradDesc)< convergenceCriteria):
            converge = True
        else:
            prevGradDesc = gradDesc
        
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            hwTrain[r] = 0.0
            for i in range(len(record) - 1):
                hwTrain[r] += (float(weights[i]) * float(record[i]))
            recordError[r] = math.pow(float(hwTrain[r] - record[57]), 2)
        
        sumSquareError = float(numpy.sum(recordError)) / 2
        meanSquareError = sumSquareError / len(trainingSet)
        rootMeanSqError = math.sqrt(meanSquareError)
        print "Iteration: " + str(iteration)
        print "sumHw: " + str(hw)
        print "SSE: " + str(sumSquareError) 
        print "MSE: " + " :" + str(meanSquareError)
        print "RMSE :" + str(rootMeanSqError)
        fp.write(str(iteration) + "\t")
        fp.write(str(rootMeanSqError)+ "\n")
        
    
    #testing data
    totalSpamsTest, totalNonSpamsTest = checkSpam(testingSet)
    falsePositive = 0.0
    falseNegative = 0.0
    hwTest = len(testingSet) * [0.0]
    for r in range(len(testingSet)):
        record = testingSet[r]
        hwTest[r] = 0.0
        for i in range(len(record) - 1):
            hwTest[r] += (float(weights[i]) * float(record[i]))
            
        if hwTest[r] > 0.0:
            if record[57] == 0.0:
                falsePositive += 1
        else:
            if record[57] == 1.0:
                falseNegative += 1
    
    errorRate = float(falsePositive + falseNegative) / float(len(testingSet))
    falsePositiveError = float(falsePositive) / float(totalNonSpamsTest)
    falseNegativeError = float(falseNegative) / float(totalSpamsTest)
    
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    
    truePos, falsePos = rocPoints(hwTest, testingSet)
    auc(truePos, falsePos, testingSet)

def linearBatRegression(trainingSet, testingSet):
    lambdaCnst = 0.0001
    weights = [0.0] * 57
    recordError = len(trainingSet) * [0.0]
    hwTrain = len(trainingSet) * [0.0]
    fp = open(str(lambdaCnst) + "RMSE.xls", "w")
    converge = False
    prevGradDesc = 0.0
    iteration = 0

    while not converge:
        iteration += 1
        gradDesc = [0.0] * 57
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            recordError[r] = 0.0
            hw = 0.0
            #print "weights: " + str(weights)
            for i in range(len(record) - 1):
                hw += (float(weights[i]) * float(record[i]))
                gradDesc[i] += float(hw - record[57]) * float(record[i])
                
                
            
        if (math.fabs(prevGradDesc - gradDesc[56])< 0.000001):
            converge = True
        else:
            prevGradDesc = gradDesc[56]
        
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            hwTrain[r] = 0.0
            for i in range(len(record) - 1):
                weights[i] = weights[i] - (lambdaCnst * gradDesc[i]) / len(trainingSet)
                hwTrain[r] += (float(weights[i]) * float(record[i]))
            recordError[r] = math.pow(float(hwTrain[r] - record[57]), 2)
        
        sumSquareError = float(numpy.sum(recordError)) / 2
        meanSquareError = sumSquareError / len(trainingSet)
        rootMeanSqError = math.sqrt(meanSquareError)
        print "Iteration: " + str(iteration)
        print "sumHw: " + str(hw)
        print "SSE: " + str(sumSquareError) 
        print "MSE: " + " :" + str(meanSquareError)
        print "RMSE :" + str(rootMeanSqError)
        fp.write(str(iteration) + "\t")
        fp.write(str(rootMeanSqError)+ "\n")
        
    
    #testing data
    totalSpamsTest, totalNonSpamsTest = checkSpam(testingSet)
    falsePositive = 0.0
    falseNegative = 0.0
    hwTest = len(testingSet) * [0.0]
    for r in range(len(testingSet)):
        record = testingSet[r]
        hwTest[r] = 0.0
        for i in range(len(record) - 1):
            hwTest[r] += (float(weights[i]) * float(record[i]))
            
        if hwTest[r] > 0.0:
            if record[57] == 0.0:
                falsePositive += 1
        else:
            if record[57] == 1.0:
                falseNegative += 1
    
    errorRate = float(falsePositive + falseNegative) / float(len(testingSet))
    falsePositiveError = float(falsePositive) / float(totalNonSpamsTest)
    falseNegativeError = float(falseNegative) / float(totalSpamsTest)
    
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    
    truePos, falsePos = rocPoints(hwTest, testingSet)
    auc(truePos, falsePos, testingSet)


def logisticRegression(trainingSet, testingSet):
    lambdaCnst = 0.1
    weights = [0.0] * 57
    print trainingSet[0][56]
    recordError = len(trainingSet) * [0.0]
    hwTrain = len(trainingSet) * [0.0]
    sumwxTrain = len(trainingSet) * [0.0]
    fp = open(str(lambdaCnst) + "RMSE.xls", "w")
    converge = False
    prevGradDesc = 0.0
    gradDesc = 0.0
    iteration = 0
    
    while not converge:
        iteration += 1
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            recordError[r] = 0.0
            hw = 0.0
            sumwx = 0.0
            #print "weights: " + str(weights)
            for i in range(len(record) - 1):
                sumwx += (float(weights[i]) * float(record[i]))
                hw = 1 / (1 + (1 / math.exp(sumwx)))
                gradDesc = float(hw - record[57]) * hw * (1 - hw) * float(record[i])
                weights[i] = weights[i] - (lambdaCnst * gradDesc)
                
            
        if (math.fabs(prevGradDesc - gradDesc)< 0.00001):
            converge = True
        else:
            prevGradDesc = gradDesc
        
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            hwTrain[r] = 0.0
            sumwxTrain[r] = 0.0
            for i in range(len(record) - 1):
                sumwxTrain[r] += (float(weights[i]) * float(record[i]))
                hwTrain[r] = 1 / (1 + (1 / math.exp(sumwxTrain[r])))
            recordError[r] = math.pow(float(hwTrain[r] - record[57]), 2)
        
        sumSquareError = float(numpy.sum(recordError)) / 2
        meanSquareError = sumSquareError / len(trainingSet)
        rootMeanSqError = math.sqrt(meanSquareError)
        print "Iteration: " + str(iteration)
        print "sumHw: " + str(hw)
        print "SSE: " + str(sumSquareError) 
        print "MSE: " + " :" + str(meanSquareError)
        print "RMSE :" + str(rootMeanSqError)
        fp.write(str(iteration) + "\t")
        fp.write(str(rootMeanSqError)+ "\n")
        
    
    #testing data
    totalSpamsTest, totalNonSpamsTest = checkSpam(testingSet)
    falsePositive = 0.0
    falseNegative = 0.0
    hwTest = len(testingSet) * [0.0]
    for r in range(len(testingSet)):
        record = testingSet[r]
        hwTest[r] = 0.0
        for i in range(len(record) - 1):
            hwTest[r] += (float(weights[i]) * float(record[i]))
            
        if hwTest[r] > 0.0:
            if record[57] == 0.0:
                falsePositive += 1
        else:
            if record[57] == 1.0:
                falseNegative += 1
    
    errorRate = float(falsePositive + falseNegative) / float(len(testingSet))
    falsePositiveError = float(falsePositive) / float(totalNonSpamsTest)
    falseNegativeError = float(falseNegative) / float(totalSpamsTest)
    
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    
    truePos, falsePos = rocPoints(hwTest, testingSet)
    auc(truePos, falsePos, testingSet)


def logisticBatRegression(trainingSet, testingSet):
    lambdaCnst = 0.01
    weights = [0.0] * 57
    recordError = len(trainingSet) * [0.0]
    hwTrain = len(trainingSet) * [0.0]
    swTrain = len(trainingSet) * [0.0]
    fp = open(str(lambdaCnst) + "RMSE.xls", "w")
    converge = False
    prevGradDesc = 0.0
    iteration = 0
     
    while not converge:
        iteration += 1
        gradDesc = len(trainingSet) * [0.0]
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            recordError[r] = 0.0
            hw = 0.0
            sumwx = 0.0
            #print "weights: " + str(weights)
            for i in range(len(record) - 1):
                sumwx += (float(weights[i]) * float(record[i]))
                hw = 1 / (1 + (1 / math.exp(sumwx)))
                gradDesc[i] += float(hw - record[57]) * hw * (1 - hw) * float(record[i])
                
            
        if (math.fabs(prevGradDesc - gradDesc[56])< 0.00001):
            converge = True
        else:
            prevGradDesc = gradDesc[56]
        
        for r in range(len(trainingSet)):
            record = trainingSet[r]
            hwTrain[r] = 0.0
            swTrain[r] = 0.0
            for i in range(len(record) - 1):
                weights[i] = weights[i] - (lambdaCnst * gradDesc[i]) / len(trainingSet)
                swTrain[r] += (float(weights[i]) * float(record[i]))
                hwTrain[r] = 1 / (1 + (1 / math.exp(swTrain[r])))
            recordError[r] = math.pow(float(hwTrain[r] - record[57]), 2)
        
        sumSquareError = float(numpy.sum(recordError))
        meanSquareError = sumSquareError / len(trainingSet)
        rootMeanSqError = math.sqrt(meanSquareError)
        print "Iteration: " + str(iteration)
        print "sumHw: " + str(hw)
        print "SSE: " + str(sumSquareError) 
        print "MSE: " + " :" + str(meanSquareError)
        print "RMSE :" + str(rootMeanSqError)
        fp.write(str(iteration) + "\t")
        fp.write(str(rootMeanSqError)+ "\n")
        
    
    #testing data
    totalSpamsTest, totalNonSpamsTest = checkSpam(testingSet)
    falsePositive = 0.0
    falseNegative = 0.0
    hwTest = len(testingSet) * [0.0]
    for r in range(len(testingSet)):
        record = testingSet[r]
        hwTest[r] = 0.0
        for i in range(len(record) - 1):
            hwTest[r] += (float(weights[i]) * float(record[i]))
            
        if hwTest[r] > 0.0:
            if record[57] == 0.0:
                falsePositive += 1
        else:
            if record[57] == 1.0:
                falseNegative += 1
    
    errorRate = float(falsePositive + falseNegative) / float(len(testingSet))
    falsePositiveError = float(falsePositive) / float(totalNonSpamsTest)
    falseNegativeError = float(falseNegative) / float(totalSpamsTest)
    
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    
    truePos, falsePos = rocPoints(hwTest, testingSet)
    auc(truePos, falsePos, testingSet)
    
    
def rocPoints(hwTest, testingSet):
    rocList = []
    falsePositiveList = []
    truePositiveList = []
    
    filePointer = open("ROCPlot.xls", "w")
    
    for i in hwTest:
        rocList.append(i)
        
    rocList.sort(reverse=True)
    
    for rocDataPoint in rocList:
        recordNum = 0
        falsePositive = 0.0
        falseNegative = 0.0
        truePositive = 0.0
        trueNegative = 0.0
        
        for hw in hwTest:
            if hw > rocDataPoint:
                if testingSet[recordNum][57] == 1.0:
                    truePositive += 1
                else:
                    falsePositive += 1
            else:
                if testingSet[recordNum][57] == 1.0:
                    falseNegative += 1
                else:
                    trueNegative += 1
            recordNum += 1
            
        falsePositiveValue = (float(falsePositive)) / (float( falsePositive + trueNegative))
        truePositiveValue = (float(truePositive)) / (float(truePositive + falseNegative))
        falsePositiveList.append(falsePositiveValue)
        truePositiveList.append(truePositiveValue)
        
    filePointer.write("TP" + "\n")
    for truePosValue in truePositiveList:
        filePointer.write(str(truePosValue)+ "\n")
        
    filePointer.write("FP" + "\n")
    for falsePosValue in falsePositiveList:
        filePointer.write(str(falsePosValue)+ "\n")
        
    return truePositiveList, falsePositiveList
        
def auc(truePositiveList, falsePositiveList, testingSet):
    length = len(testingSet)
    auc = 0.0
    for i in range(1,length):
        truePositiveVal = truePositiveList[i] + truePositiveList[i-1]
        falsePositiveVal = falsePositiveList[i] - falsePositiveList[i-1]
        auc += float(truePositiveVal)* float(falsePositiveVal)
        
    auc = auc / 2.0
    print("AUC: " + str(auc))
    
    
main()
    
