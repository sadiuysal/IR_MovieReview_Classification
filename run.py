import collections
import math
import os
import re,sys
from itertools import combinations
from os import listdir
from os.path import isfile, join
import random

Token = collections.namedtuple('Token', ['type', 'value', 'line', 'column'])
print("Running configuration:")
print(sys.argv)

neg_trainDir=sys.argv[1] # "data/train/neg/"
pos_trainDir=sys.argv[2] # "data/train/pos/"
neg_testDir=sys.argv[3] # "data/test/neg/"
pos_testDir=sys.argv[4] #"data/test/pos/"


def tokenize(text): #tokenize given text with regex
    token_specification = [
        ('WORD',       r'[A-Za-z]([-\']){0,1}[A-Za-z]*'),    # regex for finding words
        ('NEWLINE',  r'\n'),           # regex for Line endings
        ('ABBREVIATION',       r'[A-Za-z]+([.]){1}(?=[ ]{1}[a-z])'), #I added this but found no abbreviation in dataset
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, text):
        type = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if type=='NEWLINE':
            line_start = mo.end()
            line_num += 1
            continue
        else:
            yield Token(type, value, line_num, column)
def createDict(dirName,vocabDict):  #creates dictionaries for direction Name and adds them to vocabDict
    fileList= [dirName+f for f in listdir(dirName) if isfile(join(dirName, f)) and not f.startswith(".")] #take all file paths except the system files
    dict_tokens={} #stores token--frequency pairs
    wordDocfreq={}  #stores word and documentCount pairs
    binaryNofWords=0  #stores sum of number of unique words in documents for binary naive bayes
    for dir in fileList :
        docWordSet=set()
        f=open(dir, "r")
        if f.mode == 'r':
            contents=f.read()
            for token in tokenize(contents):
                word=token.value.casefold()
                docWordSet.add(word)
                if not (dict_tokens.get(word) is None):
                    dict_tokens[word]+=1
                else:
                    dict_tokens[word]=1
                if not (vocabDict.get(word) is None):
                    vocabDict[word]+=1
                else:
                    vocabDict[word]=1
        f.close()
        binaryNofWords+=len(docWordSet)  #update total wordCount for binary Naive Bayes
        for word in docWordSet:   #update wordDocfreq
            if wordDocfreq.get(word) is None:
                wordDocfreq[word]=1
            else:
                wordDocfreq[word]+=1

    return dict_tokens,len(fileList),wordDocfreq,len(fileList),binaryNofWords

def nofWordsCalculator(dictName):  #calculates the number of words according to given dict
    nofWords=0
    for value in dictName.values():
        nofWords+=value
    return nofWords
def condProb(word,className,addkSmoothing,NBType):  #finds conditional probability for given word, className("neg","pos"),Naive Bayes Type
    if className=="neg":  #if negative then here
        if NBType=="bernoulli":
            dictName=neg_wordDocfreq
            docCount=neg_DocCount
        elif NBType=="binary":
            dictName=neg_wordDocfreq
            nofWords=neg_binaryNofWords
        elif NBType=="multinomial":
            dictName=neg_dict
            nofWords=nofWordsNeg
        else:
            print("Unknown NB Type ,please try again.")
    elif className=="pos":   #if positive then here
        if NBType=="bernoulli":
            dictName=pos_wordDocfreq
            docCount=pos_DocCount
        elif NBType=="binary":
            dictName=pos_wordDocfreq
            nofWords=pos_binaryNofWords
        elif NBType=="multinomial":
            dictName=pos_dict
            nofWords=nofWordsPos
        else:
            print("Unknown NB Type ,please try again.")
    else:
        print("Unknown class name while calling condProb(), please choose 'pos' or 'neg'. ")
    count=dictName.get(word)
    if count is None:
        count=0
    if NBType=="bernoulli":
        return (count+1)/(docCount+2)
    else :  #if multinomial or binary I changed count and nofWords values according so that we can return the same statement
        return (count+addkSmoothing)/(nofWords + addkSmoothing * len(vocabDict))

def predictDocs(dirName,NBType,laplaceSmoothing): #predicts documents in direction name, according to NB Type and laplaceSmoothing value
    fileList= [dirName+f for f in listdir(dirName) if isfile(join(dirName, f)) and not f.startswith(".")] #take all file paths except the system files
    result=[]
    negCount,posCount=0,0
    for dir in fileList :
        f=open(dir, "r")
        if f.mode == 'r':
            contents=f.read()
            probNegClass=0
            probPosClass=0
            docWordSet=set()
            for token in tokenize(contents):
                word=token.value.casefold()
                if NBType=="binary" and word in docWordSet:  #if NB Type is binary we should process unique words in our document
                    continue
                probNegClass+=math.log2(condProb(word,"neg",addkSmoothing=laplaceSmoothing,NBType=NBType))
                probPosClass+=math.log2(condProb(word,"pos",addkSmoothing=laplaceSmoothing,NBType=NBType))
                docWordSet.add(word)
            if NBType=="bernoulli":
                for word in vocabDict :
                    if word not in docWordSet:
                        probNegClass+=math.log2(1-condProb(word,"neg",addkSmoothing=laplaceSmoothing,NBType=NBType))
                        probPosClass+=math.log2(1-condProb(word,"pos",addkSmoothing=laplaceSmoothing,NBType=NBType))
            probNegClass+=math.log2(probNegDoc)
            probPosClass+=math.log2(probPosDoc)
            if probNegClass>probPosClass:
                negCount+=1
                result.append(0)  #negative result
            else:
                posCount+=1
                result.append(1)  #positive result
        f.close()
    return posCount,negCount,result

def measureErrors(tp,fp,fn,tn): #helper function for getResults, it calculates recall,precision,Fmeasure
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    fMeasure=2*precision*recall/(precision+recall)
    print("Recall : "+str(recall)+" Precision : "+str(precision)+" fMeasure : "+ str(fMeasure))
def measureMakroMikroAvg(tp1,fp1,fn1,tn1,tp2,fp2,fn2,tn2,Randomization): #helper function for getResults and randomization test
    makroRecall=(tp1/(tp1+fn1)+tp1/(tp2+fn2))/2
    mikroRecall=(tp1+tp2)/(tp1+tp2+fn1+fn2)
    makroPrecision=(tp1/(tp1+fp1)+tp1/(tp1+fp1))/2
    mikroPrecision=(tp1+tp2)/(tp1+tp2+fp1+fp2)
    mikrofMeasure=2*mikroPrecision*mikroRecall/(mikroPrecision+mikroRecall)
    makrofMeasure=2*makroPrecision*makroRecall/(makroPrecision+makroRecall)
    if Randomization:
        return mikrofMeasure
    else:
        print("Makro values: ")
        print("Recall : "+str(makroRecall)+" Precision : "+str(makroPrecision)+" fMeasure : "+ str(makrofMeasure))
        print("Mikro values: ")
        print("Recall : "+str(mikroRecall)+" Precision : "+str(mikroPrecision)+" fMeasure : "+ str(mikrofMeasure))
        return mikrofMeasure
def getResults(pos1,neg1,pos2,neg2):  #gets results
    print("--Positive Review: --")
    measureErrors(tp=pos1,fp=pos2,fn=neg1,tn=neg2)
    print("--Negative Review: --")
    measureErrors(tp=neg2,fp=neg1,fn=pos2,tn=pos1)
    mikroF=measureMakroMikroAvg(pos1,pos2,neg1,neg2,neg2,neg1,pos2,pos1,Randomization=False)
    return mikroF

def randomizationTest(R,F_A,result_A,pos1_A,pos2_A,F_B,result_B,pos1_B,pos2_B): #do randomization test to check system's difference
    count=0
    s=abs(F_A-F_B)
    for itr in range(R):
        newPos1_A=pos1_A   #reassign old values of pos-neg counts at every iteration
        newPos2_A=pos2_A
        newPos1_B=pos1_B
        newPos2_B=pos2_B
        for i in range(len(result_A)):  #for every sample
            p=random.random()
            if p>0.5:
                if result_A[i]!=result_B[i]:   #if results are same no need for action otherwise do
                    if result_A[i]==1:
                        if i<300:
                            newPos1_A+=-1
                            newPos1_B+=1
                        else:
                            newPos2_A+=-1
                            newPos2_B+=1
                    else:
                        if i<300:
                            newPos1_A+=1
                            newPos1_B+=-1
                        else:
                            newPos2_A+=1
                            newPos2_B+=-1
        F_a=measureMakroMikroAvg(newPos1_A,newPos2_A,300-newPos1_A,300-newPos2_A,300-newPos2_A,300-newPos1_A,newPos2_A,newPos1_A,Randomization=True)
        F_b=measureMakroMikroAvg(newPos1_B,newPos2_B,300-newPos1_B,300-newPos2_B,300-newPos2_B,300-newPos1_B,newPos2_B,newPos1_B,Randomization=True)
        sStar=abs(F_a-F_b)
        #print("Sstar: "+str(sStar))
        #print("S: "+str(s))
        if sStar>=s:
            count+=1
    prob=(count+1)/(R+1)
    print("p= "+str(prob))
    if prob<=0.05:
        print("We reject the hypothesis that 'Sytems are not different.'")
    else:
        print("Systems are not significantly different.")





vocabDict={}
neg_dict,nofNegDocs,neg_wordDocfreq,neg_DocCount,neg_binaryNofWords=createDict(neg_trainDir,vocabDict)
pos_dict,nofPosDocs,pos_wordDocfreq,pos_DocCount,pos_binaryNofWords=createDict(pos_trainDir,vocabDict)
probNegDoc=nofNegDocs/(nofNegDocs+nofPosDocs)
probPosDoc=nofPosDocs/(nofNegDocs+nofPosDocs)
nofWordsNeg=nofWordsCalculator(neg_dict)
nofWordsPos=nofWordsCalculator(pos_dict)


#NBTypes: "binary","bernoulli","multinomial"

print(" ------------Bernoulli NB:-------------------              ")
pos1_Ber,neg1,resultPos=predictDocs(pos_testDir,NBType="bernoulli",laplaceSmoothing=1)
pos2_Ber,neg2,resultNeg=predictDocs(neg_testDir,NBType="bernoulli",laplaceSmoothing=1)
result_Ber=resultPos+resultNeg
print("Positive test set Success: "+str(pos1_Ber)+" Fail: "+str(neg1))
print("Negative test set Success: "+str(neg2)+" Fail: "+str(pos2_Ber))
mikroF_Ber=getResults(pos1_Ber,neg1,pos2_Ber,neg2)
print("   -------------Binary NB: ----------------                ")
pos1_Bin,neg1,resultPos=predictDocs(pos_testDir,NBType="binary",laplaceSmoothing=1)
pos2_Bin,neg2,resultNeg=predictDocs(neg_testDir,NBType="binary",laplaceSmoothing=1)
result_Bin=resultPos+resultNeg
print("Positive test set Success: "+str(pos1_Bin)+" Fail: "+str(neg1))
print("Negative test set Success: "+str(neg2)+" Fail: "+str(pos2_Bin))
mikroF_Bin=getResults(pos1_Bin,neg1,pos2_Bin,neg2)
print("   --------------Multinomial NB: ----------------")
pos1_Mul,neg1,resultPos=predictDocs(pos_testDir,NBType="multinomial",laplaceSmoothing=1)
pos2_Mul,neg2,resultNeg=predictDocs(neg_testDir,NBType="multinomial",laplaceSmoothing=1)
result_Mul=resultPos+resultNeg
print("Positive test set Success: "+str(pos1_Mul)+" Fail: "+str(neg1))
print("Negative test set Success: "+str(neg2)+" Fail: "+str(pos2_Mul))
mikroF_Mul=getResults(pos1_Mul,neg1,pos2_Mul,neg2)

print("-----------------Randomization Test---------------")
print("-----Iteration :5000 Rejection : 0.05----")
print("--Bernoulli NB vs Binary NB-- ")
randomizationTest(5000,mikroF_Ber,result_Ber,pos1_Ber,pos2_Ber,mikroF_Bin,result_Bin,pos1_Bin,pos2_Bin)
print("--Bernoulli NB vs Multinomial NB-- ")
randomizationTest(5000,mikroF_Ber,result_Ber,pos1_Ber,pos2_Ber,mikroF_Mul,result_Mul,pos1_Mul,pos2_Mul)
print("--Multinomial NB vs Binary NB-- ")
randomizationTest(5000,mikroF_Mul,result_Mul,pos1_Mul,pos2_Mul,mikroF_Bin,result_Bin,pos1_Bin,pos2_Bin)



