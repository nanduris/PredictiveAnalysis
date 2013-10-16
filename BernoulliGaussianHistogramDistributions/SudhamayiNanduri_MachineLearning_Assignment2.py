#!/usr/bin/env python

import numpy
import math

#global definitions
kPartitions = 10 #Number of groups or partitions
numOfFeatures = 57 #Number of features
totalMails = 4601
spamMails = 1813
nonspamMails = 2788               
        
def main():
    #read the data
    fp = open("processedData.txt", "r")
    mails = numpy.loadtxt(fp, comments = "numpy loads the dataset into 2D matrix array", delimiter=",")
    totalMails = len(mails) #total no of mails
    mailId = 0
    groupMap = {} #map to store each mail-record
    
    while mailId < totalMails:
        i = mailId % kPartitions
        if i not in groupMap:
            groupMap[i] = [mails[mailId]]
        else:
            groupMap[i] = numpy.concatenate((groupMap[i], [mails[mailId]]))
            
        mailId += 1
    
    # run classifiers on this data set
    bernoulliErrorRate = 0.0
    gaussianErrorRate = 0.0
    histogramErrorRate = 0.0
    for k in range(kPartitions):
        trainingGroup, validationGroup = partitionKSets(groupMap, k)
        #calculate error rate for the classifiers
        bernoulliErrorRate += bernoulliClassifier(k, trainingGroup, validationGroup)
        gaussianErrorRate += gaussianClassifier(k, trainingGroup, validationGroup)
        histogramErrorRate += histogram(k, trainingGroup, validationGroup)
        
    #average error rate
    bernoulliAverageError = (float(bernoulliErrorRate)) / (float(kPartitions))
    gaussianAverageError = (float(gaussianErrorRate)) / (float(kPartitions))
    histogramAverageError = (float(histogramErrorRate)) / (float(kPartitions))
    
    print "Error Rate for Bernoulli Classifier: " + str(bernoulliAverageError)
    print "Error Rate for Gaussian Classifier: " + str(gaussianAverageError)
    print "Error Rate for Histogram: " + str(histogramAverageError)


# partition the data into training and validation sets
def partitionKSets(groupMap, k):
    validationGroup = {}
    validationGroup[k] = groupMap[k]
    trainingGroup = {}
    trainingFeatures = numpy.empty([0,58])
    for i, features in groupMap.iteritems():
        if i != k:
            trainingFeatures = numpy.concatenate((trainingFeatures, numpy.array(features)))

    trainingGroup[k] = trainingFeatures
    return trainingGroup, validationGroup

#get spam/non-spam mails
def checkSpam(parameterSet):
    ifSpam = parameterSet[:, 57] == 1.0
    spams = parameterSet[ifSpam, :]
    nonSpams = parameterSet[~ifSpam, :]
    totalSpams = len(spams)
    totalNonSpams = len(nonSpams)
    
    return spams,nonSpams, totalSpams, totalNonSpams

#calculate the mean of the input data set
def calculateMean(parameterSet):
    mean = [numpy.mean(parameterSet[:, i]) for i in range(parameterSet.shape[1] - 1)]
    return mean
    
# Bernoulli Classifier
def bernoulliClassifier(k, trainingGroup, validationGroup):
    
    # Bernoulli classifier on the training set
    trainingSet = trainingGroup[k]
    spamsTrain, nonSpamsTrain, totalSpamsTrain, totalNonSpamsTrain = checkSpam(trainingSet)
    thresholdMean = calculateMean(trainingSet) #average mean
    # the four ranges given in the problem question
    spamBelowThreshold = []
    spamAboveThreshold = []
    nonSpamBelowThreshold = []
    nonSpamAboveThreshold = []
    
    for i in range(trainingSet.shape[1] - 1):
        #prob of feature less than or equal to mean, given spam
        spamBelowThresholdCond = spamsTrain[:, i] <= thresholdMean[i]
        spamBelowThreshold.append(float(len(spamsTrain[spamBelowThresholdCond, :]))/ float(totalSpamsTrain))
        #prob of feature greater than mean, given spam
        spamAboveThresholdCond = spamsTrain[:, i] > thresholdMean[i]
        spamAboveThreshold.append(float(len(spamsTrain[spamAboveThresholdCond, :]))/ float(totalSpamsTrain))
        #prob of feature less than or equal to mean, given non-spam
        nonSpamBelowThresholdCond = nonSpamsTrain[:, i] <= thresholdMean[i]
        nonSpamBelowThreshold.append(float(len(nonSpamsTrain[nonSpamBelowThresholdCond, :]))/ float(totalNonSpamsTrain))
        #prob of feature greater than mean, given non-spam
        nonSpamAboveThresholdCond = nonSpamsTrain[:, i] > thresholdMean[i]
        nonSpamAboveThreshold.append(float(len(nonSpamsTrain[nonSpamAboveThresholdCond, :]))/ float(totalNonSpamsTrain))
        
        
    # Bernoulli classifier on the validating set
    validationSet = validationGroup[k]
    falsePositive = 0.0
    falseNegative = 0.0
    bayesPrediction = []
    spamsValidate, nonSpamsValidate, numOfValidateSpams, numOfValidateNonSpams = checkSpam(validationSet)
    
    #probabilities of spam and non-spam
    prSpam = float(totalSpamsTrain)/ float(totalSpamsTrain + totalNonSpamsTrain)
    prNonSpam = float(totalNonSpamsTrain)/ float(totalSpamsTrain + totalNonSpamsTrain)
    logSpamNSpam = math.log(float(prSpam)) - math.log(float(prNonSpam))
    
    #calculate the Naive Bayes Ratio based on the threshold mean value
    for i in range(len(validationSet)):
        record = validationSet[i]
        num = 0.0
        den = 0.0
        diff = 0.0
        naiveBayesRatio = 0.0
        
        for j in range(len(record) - 1):
            if record[j] > thresholdMean[j]:
                if spamAboveThreshold[j] != 0.0 and nonSpamAboveThreshold[j] != 0.0 and spamAboveThreshold[j] != nonSpamAboveThreshold[j]:
                    num += math.log(float(spamAboveThreshold[j]))
                    den += math.log(float(nonSpamAboveThreshold[j]))
                else:
                    num += math.log((float(spamAboveThreshold[j])) + 1.0)
                    den += math.log((float(nonSpamAboveThreshold[j])) + 2.0)
            else:
                if spamBelowThreshold[j] != 0.0 and spamBelowThreshold[j] != nonSpamBelowThreshold[j]:
                    num += math.log(float(spamBelowThreshold[j]))
                    if nonSpamBelowThreshold[j] != 0.0:
                        den += math.log(float(nonSpamBelowThreshold[j]))
                    else:
                        den += math.log(float(nonSpamBelowThreshold[j]))
                else:
                    num += math.log((float(spamBelowThreshold[j])) + 1.0)
                    den += math.log((float(nonSpamBelowThreshold[j])) + 2.0)
                    
        diff = num - den
        naiveBayesRatio = logSpamNSpam + diff
        bayesPrediction.append(naiveBayesRatio)
        
        if naiveBayesRatio > 0.0:
            if record[57] != 1.0:
                falsePositive += 1
        else:
            if record[57] != 0.0:
                falseNegative += 1
    
    #get ROC datapoints and calculate AUC
    if k == 0:
        truePositiveList, falsePositiveList = rocPoints("Bernoulli", bayesPrediction, validationSet)
        auc("Bernoulli", truePositiveList, falsePositiveList, validationSet)
        
    errorRate = float(falsePositive + falseNegative) / float(len(validationSet))
    falsePositiveError = float(falsePositive) / float(numOfValidateNonSpams)
    falseNegativeError = float(falseNegative) / float(numOfValidateSpams)
    
    #print output
    print "Bernoulli:"
    print "Partition: " + str(k)
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    return errorRate

#calculates the variance for the features
def calculateConditionClassVariance(trainingSet, meanOfFeatures, spamMean, nonSpamMean, totalSpamsTrain, totalNonSpamsTrain):
    
    varianceOfFeature = [0.0] * numOfFeatures
    varianceOfSpam = [0.0] * numOfFeatures
    varianceOfNonSpam = [0.0] * numOfFeatures
    
    for r in range(len(trainingSet)):
        record = trainingSet[r]
        for i in range(len(record) - 1):
            diffVar = record[i] - meanOfFeatures[i]
            varianceOfFeature[i] += float(math.pow((record[i] - meanOfFeatures[i]), 2))
            #conditional variance
            if record[57] != 1.0:
                diffNonSpamVar = record[i] - nonSpamMean[i]
                varianceOfNonSpam[i] += float(math.pow(diffNonSpamVar, 2))
            else:
                diffSpamVar = record[i] - spamMean[i]
                varianceOfSpam[i] += float(math.pow(diffSpamVar, 2))
    
    for j in range(len(varianceOfFeature)):
        varianceOfFeature[j] = float(varianceOfFeature[j]) / float(totalSpamsTrain + totalNonSpamsTrain)
        varianceOfSpam[j] = float(varianceOfSpam[j]) / float(totalSpamsTrain)
        varianceOfNonSpam[j] = float(varianceOfNonSpam[j]) / float(totalNonSpamsTrain)
        
    #Laplace smoothing with lambda
    spamLambda = 0.3
    nonSpamLambda = 0.3
    
    for x in range(len(varianceOfFeature)):
        varianceOfSpam[x] = float(varianceOfSpam[x] * spamLambda) + float((1 - spamLambda) * varianceOfFeature[x])
        varianceOfNonSpam[x] = float(varianceOfNonSpam[x] * nonSpamLambda) + float((1 - nonSpamLambda) * varianceOfFeature[x])
        
    return varianceOfSpam, varianceOfNonSpam

#Gaussian Classifier
def gaussianClassifier(k, trainingGroup, validationGroup):
    #gaussian for trainingSet
    trainingSet = trainingGroup[k]
    spamsTrain, nonSpamsTrain, totalSpamsTrain, totalNonSpamsTrain = checkSpam(trainingSet)
    spamMean = calculateMean(spamsTrain)
    nonSpamMean = calculateMean(nonSpamsTrain)
    meanOfFeatures = calculateMean(trainingSet)
    #conditional variance for the trainingSet features
    varianceOfSpam, varianceOfNonSpam = calculateConditionClassVariance(trainingSet, meanOfFeatures, spamMean, nonSpamMean, totalSpamsTrain, totalNonSpamsTrain)
    
    #guassian for ValidatingSet/testing set
    validationSet = validationGroup[k]
    falsePositive = 0.0
    falseNegative = 0.0
    
    constVal = float(-0.5 * float(math.log(2 * math.pi)))
    prSpam = float(totalSpamsTrain) / float(totalSpamsTrain + totalNonSpamsTrain)
    prNonSpam = float(totalNonSpamsTrain) / float(totalSpamsTrain + totalNonSpamsTrain)
    logSpamNonSpam = math.log(float(prSpam)) - math.log(float(prNonSpam))
    #get spam/non-spam mails from the validating set
    spamsValidate, nonSpamsValidate,totalValidateSpams, totalValidateNonSpams = checkSpam(validationSet)
    naiveBayesPrediction = []
    
    for r in range(len(validationSet)):
        record = validationSet[r]
        num = 0.0
        den = 0.0
        diffVal = 0.0
        naiveBayesRatio = 0.0
        
        for i in range(len(record) - 1):
            diffSpamVar = record[i] - spamMean[i]
            diffNonSpamVar = record[i] - nonSpamMean[i]
            num += constVal - (float(math.pow(diffSpamVar, 2))/ (2.0 * varianceOfSpam[i])) - (float(math.log(math.sqrt(varianceOfSpam[i]))))
            den += constVal - (float(math.pow(diffNonSpamVar, 2))/ (2.0 * varianceOfNonSpam[i])) - (float(math.log(math.sqrt(varianceOfNonSpam[i]))))
            
        diffVal = num - den
        naiveBayesRatio = logSpamNonSpam + diffVal
        
        if naiveBayesRatio >= 0:
            if record[57] != 1.0:
                falsePositive += 1
        else:
            if record[57] != 0.0:
                falseNegative += 1
                
        naiveBayesPrediction.append(naiveBayesRatio)
    
    errorCount = falsePositive + falseNegative
    errorRate = float(errorCount / len(validationSet))
    falsePositiveError = float(falsePositive) / float(totalValidateNonSpams)
    falseNegativeError = float(falseNegative) / float(totalValidateSpams)
    
    #print output
    print "Gaussian:"
    print "Partition: " + str(k)
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    
    
    #get ROC data points and calculate AUC
    if k == 0:
        truePositiveList, falsePositiveList = rocPoints("Gaussian", naiveBayesPrediction, validationSet)
        auc("Gaussian", truePositiveList, falsePositiveList, validationSet)

    return errorRate

# histogram
def histogram(k, trainingGroup, validationGroup):
    #for training data sets
    trainingSet = trainingGroup[k]
    spamsTrain, nonSpamsTrain, totalSpamsTrain, totalNonSpamsTrain = checkSpam(trainingSet)
    thresholdMean = calculateMean(trainingSet)
    spamMean = calculateMean(spamsTrain)
    nonSpamMean = calculateMean(nonSpamsTrain)
    
    spamBinOne = [0.0] * numOfFeatures
    spamBinTwo = [0.0] * numOfFeatures
    spamBinThree = [0.0] * numOfFeatures
    spamBinFour = [0.0] * numOfFeatures
    
    nonSpamBinOne = [0.0] * numOfFeatures
    nonSpamBinTwo = [0.0] * numOfFeatures
    nonSpamBinThree = [0.0] * numOfFeatures
    nonSpamBinFour = [0.0] * numOfFeatures
    
    lowMeanValue = [0.0] * numOfFeatures
    highMeanValue = [0.0] * numOfFeatures
    
    binOneCount = 0
    binTwoCount = 0
    binThreeCount = 0
    binFourCount = 0
    
    #assign lower and higher mean values from spam mean of non-spam mean 
    for i in range(trainingSet.shape[1] - 1):
        if spamMean[i] < nonSpamMean[i]:
            lowMeanValue[i] = spamMean[i]
            highMeanValue[i] = nonSpamMean[i]
        else:
            lowMeanValue[i] = nonSpamMean[i]
            highMeanValue[i] = spamMean[i]
            
    #sort the records into bins based on the conditions
    for r in range(len(trainingSet)):
        record = trainingSet[r]
        for i in range(len(record) - 1):
            #bin 1
            if record[i] <= lowMeanValue[i]:
                binOneCount += 1
                if record[57] == 0.0:
                    nonSpamBinOne[i] += 1
                else:
                    spamBinOne[i] += 1
            #bin 2        
            elif ((record[i] > lowMeanValue[i]) and (record[i] <= thresholdMean[i])):
                binTwoCount += 1
                if record[57] == 0.0:
                    nonSpamBinTwo[i] += 1
                else:
                    spamBinTwo[i] += 1
            #bin 3        
            elif ((record[i] > thresholdMean[i]) and (record[i] <= highMeanValue[i])):
                binThreeCount += 1
                if record[57] == 0.0:
                    nonSpamBinThree[i] += 1
                else:
                    spamBinThree[i] += 1
            #bin 4        
            elif record[i] > highMeanValue[i]:
                binFourCount += 1
                if record[57] == 0.0:
                    nonSpamBinFour[i] += 1
                else:
                    spamBinFour[i] += 1
    
    #print binOneCount
    #print binTwoCount
    #print binThreeCount
    #print binFourCount
    
    spamBinOne = map(lambda count: float(count) / (totalSpamsTrain), spamBinOne)
    spamBinTwo = map(lambda count: float(count) / (totalSpamsTrain), spamBinTwo)
    spamBinThree = map(lambda count: float(count) / (totalSpamsTrain), spamBinThree)
    spamBinFour = map(lambda count: float(count) / (totalSpamsTrain), spamBinFour)
    
    nonSpamBinOne = map(lambda count: float(count) / (totalNonSpamsTrain), nonSpamBinOne)
    nonSpamBinTwo = map(lambda count: float(count) / (totalNonSpamsTrain), nonSpamBinTwo)
    nonSpamBinThree = map(lambda count: float(count) / (totalNonSpamsTrain), nonSpamBinThree)
    nonSpamBinFour = map(lambda count: float(count) / (totalNonSpamsTrain), nonSpamBinFour)
    
    
    
    #histogram on validation set
    validationSet = validationGroup[k]
    spamsValidate, nonSpamsValidate,totalValidateSpams, totalValidateNonSpams = checkSpam(validationSet)
    bayesPredictionList = []
    
    prSpam = (float)(totalSpamsTrain)/(float(totalSpamsTrain + totalNonSpamsTrain))
    prNonSpam = (float)(totalNonSpamsTrain)/(float(totalSpamsTrain + totalNonSpamsTrain))
    
    logSpamNonSpam = math.log(prSpam) - math.log(prNonSpam)
    
    falsePositive = 0.0
    falseNegative = 0.0
    
    for r in range(len(validationSet)):
        record = validationSet[r]
        num = 0.0
        den = 0.0
        diff = 0.0
        naiveBayesRatio = 0.0
        
        # calculation for features in each bin
        for i in range(len(record) - 1):
            if record[i] <= lowMeanValue[i]:
                if spamBinOne[i] != 0.0:
                    num += math.log(spamBinOne[i])
                
                if nonSpamBinOne[i] != 0.0:
                    den += math.log(nonSpamBinOne[i])
            
            elif ((record[i] > lowMeanValue[i]) and (record[i] <= thresholdMean[i])):
                if spamBinTwo[i] != 0.0:
                    num += math.log(spamBinTwo[i])
                else:
                    num += math.log(spamBinTwo[i] + 1.0)
                
                if nonSpamBinTwo[i] != 0.0:
                    den += math.log(nonSpamBinTwo[i])
                else:
                    den += math.log(nonSpamBinTwo[i] + 2.0)
            
            elif ((record[i] > thresholdMean[i]) and (record[i] <= highMeanValue[i])):
                if spamBinThree[i] != 0.0:
                    num += math.log(spamBinThree[i])
                else:
                    num += math.log(spamBinThree[i] + 1.0)
                
                if nonSpamBinThree[i] != 0.0:
                    den += math.log(nonSpamBinThree[i])
                else:
                    den += math.log(nonSpamBinThree[i] + 2.0)
                    
            elif record[i] > highMeanValue[i]:
                if spamBinFour[i] != 0.0:
                    num += math.log(spamBinFour[i])
                else:
                    num += math.log(spamBinFour[i] + 1.0)
                
                if nonSpamBinFour[i] != 0.0:
                    den += math.log(nonSpamBinFour[i])
                else:
                    den += math.log(nonSpamBinFour[i] + 2.0)
                    
        diff = num - den
        naiveBayesRatio = logSpamNonSpam + diff
        bayesPredictionList.append(naiveBayesRatio)
        
        if naiveBayesRatio > 0.0:
            if record[57] == 0.0:
                falsePositive += 1
                
        else:
            if record[57] == 1.0:
                falseNegative += 1
    
    #get ROC data points and calculate AUC
    if k == 0:
        truePositiveList, falsePositiveList = rocPoints("Histogram", bayesPredictionList, validationSet)
        auc("Histogram", truePositiveList, falsePositiveList, validationSet)
        
    errorRate = float(falsePositive + falseNegative) / float(len(validationSet))
    falsePositiveError = float(falsePositive) / float(totalValidateNonSpams)
    falseNegativeError = float(falseNegative) / float(totalValidateSpams)
    
    #print output
    print "Histogram:"
    print "Partition: " + str(k)
    print "Error Rate: " + str(errorRate)
    print "FalsePositiveErrorRate: " + str(falsePositiveError)
    print "FalseNegativeErrorRate: " + str(falseNegativeError)
    return errorRate
        
def rocPoints(classifier, bayesPredictionList, validationSet):
    rocList = []
    falsePositiveList = []
    truePositiveList = []
    
    filePointer = open(classifier + "ROCPlot.xls", "w")
    
    for i in bayesPredictionList:
        rocList.append(i)
        
    rocList.sort(reverse=True)
    
    for rocDataPoint in rocList:
        recordNum = 0
        falsePositive = 0.0
        falseNegative = 0.0
        truePositive = 0.0
        trueNegative = 0.0
        
        for bayesPrediction in bayesPredictionList:
            if bayesPrediction > rocDataPoint:
                if validationSet[recordNum][57] == 1.0:
                    truePositive += 1
                else:
                    falsePositive += 1
            else:
                if validationSet[recordNum][57] == 1.0:
                    falseNegative += 1
                else:
                    trueNegative += 1
            recordNum += 1
            
        falsePositiveValue = (float(falsePositive)) / (float( falsePositive + trueNegative))
        truePositiveValue = (float(truePositive)) / (float(truePositive + falseNegative))
        falsePositiveList.append(falsePositiveValue)
        truePositiveList.append(truePositiveValue)
        
    filePointer.write("TN" + "\n")
    for truePosValue in truePositiveList:
        filePointer.write(str(truePosValue)+ "\n")
        
    filePointer.write("TP" + "\n")
    for falsePosValue in falsePositiveList:
        filePointer.write(str(falsePosValue)+ "\n")
        
    return truePositiveList, falsePositiveList
        

def auc(classifier, truePositiveList, falsePositiveList, validationGroup):
    length = len(validationGroup)
    auc = 0.0
    for i in range(1,length):
        truePositiveVal = truePositiveList[i] + truePositiveList[i-1]
        falsePositiveVal = falsePositiveList[i] - falsePositiveList[i-1]
        auc += float(truePositiveVal)* float(falsePositiveVal)
        
    auc = auc / 2.0
    print(classifier + " AUC: " + str(auc))
   
main()
    
    
    
    
    
    
    
        


