# written by Yonathan Zailer and Assaf Ziv
from random import randint
import csv
import numpy
import math
import sys
import random
from codecs import open
import gc

# kernel function
def ker(E,S,i,t,gama):
    # calculate common_set
    common_set=list(set(S[i]).intersection(S[t]))
    sum=0
    for a_st in common_set:
        a=int(a_st)
        sum=sum+float(E[i][a])*float(E[t][a])
    return sum*(math.pow(len(common_set),gama)-1)/(len(common_set)-1)

# hipotesis
def hipotesis(ALFA, gama, t, E, S, whileLearning):
    sum=0
    # loop passing all alfas
    for i in range(0,t):
        # if learning look at previous alfa
        if whileLearning==True:
            curent_alfa=ALFA[t-1][i]
        else:
            curent_alfa=ALFA[i]
        sum=sum+curent_alfa*ker(E,S,i,t,gama)
    return sum

# training the module
def train(E,S,eta,ro,e_size,T,gama,y_index):
    # create alfa array and init
    alfa=numpy.ndarray(shape=(T,e_size))
    for a in range(e_size):
        alfa[0][a]=0.0
    # passing examples
    for i in range(T):
        if i!=0:
            eta_i = eta/math.sqrt(i)
        else:
             eta_i = eta
        # choosing example
        out_index=i
        if i>e_size-1: # if finished passing examples choose random one
            out_index=randint(0,e_size-1)
        # init alfa_index
        if i==0:
            alfa_index=0
        else:
            alfa_index=i-1
        # updating all current alfas
        for j in range(0,e_size):
            # updating rule
            if out_index!=j:
                alfa[out_index][j]=alfa[alfa_index][j]*(1-eta_i*ro)
            else:
                alfa[out_index][out_index]=alfa[alfa_index][out_index]-2*\
                                eta_i*(hipotesis(alfa,gama,out_index,E,S,True)-
                                float(E[out_index][y_index]))-alfa[alfa_index][out_index]*eta_i*ro
    return alfa



# checking possible prms
def KARMA(E,S,y_index):
    M={}
    e_size=len(E)# examples number
    T=int(e_size) # number of cycles
    # init parameters of algorithm
    for gama in range(1,4):
        for t_eta in range(1,22):
            eta=t_eta*0.1
            c=pow(10,-6)
            while c<pow(10,6):
                    c=c*10
                    ro=1/(c)
                    # training
                    alfa=train(E,S,eta,ro,e_size,T,gama,y_index)
                    # calcolate avg alfa
                    sum=numpy.zeros(shape=(e_size))
                    for z in range(0,T):
                        for x in range(0,e_size):
                            sum[x]=sum[x]+alfa[z][x]
                    # save avg alfa
                    M[gama,eta,c]=sum/(T)
                    gc.collect()
    return M

# finding max value in column
def maxColumn(E,i):
    # print str(i)+" attr"
    flag=False
    for j in range(len(E)):
        if E[j][i]!='':
            if flag==False:
                max=float(E[j][i])
                flag=True
            else:
                if max<float(E[j][i]):
                    max=float(E[j][i])
    return max

# finding min value in column
def minColumn(E,i):
    flag=False
    for j in range(len(E)):
        if E[j][i]!='':
            if flag==False:
                min=float(E[j][i])
                flag=True
            else:
                if min>float(E[j][i]):
                    min=float(E[j][i])
    return min

# data normalization
def normalize(E,y_index):
    #setting the max and min arrays
    max={}
    min={}
    # finding max and min
    for i in range(len(E[0])):
        if i!=y_index:
            max[i]=maxColumn(E,i)
            min[i]=minColumn(E,i)
            if max[i]!=min[i]:
                #a loop to divide the numbers with the max min method for normalization
                for j in range(len(E)):
                    if E[j][i] != '' :
                        E[j][i]=str((float(E[j][i])-min[i]) / (max[i]-min[i]))
    return E



# uploading file
def uploadCSV(file):
     with open(file) as csvinput:
        reader = csv.reader(csvinput)
        lis=list(reader)
        return lis

# preparing data and running algorithm
def articalAlgorithm(file,y_name):
    S=[]
    E=[]
    lis=uploadCSV(file)
    # passing examples-lines
    print len(lis)
    for i in range(len(lis)):
        # removing first column
        row=[]
        for k in range (len(lis[i])):
            if k!=0:
                    row.append(lis[i][k])
        #adding constant
        row.append('1')
        # finding y_name index
        if i==0:
            attr=row
            for j in range(len(attr)):
                if y_name==attr[j]:
                    y_index=j
                    break
            continue
        temp=[]
        # if Y has no value keep going
        if row[y_index]=='':
            continue
        # creating E-examples and S-set of exist values
        for j in range(len(row)):
            if row[j]!='' and j!=y_index:
                temp.append(j)
        S.append(temp)
        E.append(row)
    E=normalize(E,y_index)
    Etrain,Etest,Strain,Stest=randomlyDistributedToTT(E,S)
    M=KARMA(Etrain,Strain,y_index) # alfa per values
    return (M,Etrain,Etest,Strain,Stest,y_index)

# this function divide the data to train and test randomaly in the size 70% 30%
def randomlyDistributedToTT(E,S):
    Etrain=[]
    Etest=[]
    Strain=[]
    Stest=[]
    for i in range(0,len(E)):
    # the percentage
        if random.random() > 0.7:
            Etest.append(E[i])
            Stest.append(S[i])

        else:
            Etrain.append(E[i])
            Strain.append(S[i])
    return (Etrain,Etest,Strain,Stest)

# kernel test function
def kerTest(Etrain,Etest,Strain,Stest,i,t,gama):
    # calculate common_set
    common_set=list(set(Strain[i]).intersection(Stest[t]))
    sum=0
    for a_st in common_set:
        a=int(a_st)
        sum=sum+float(Etrain[i][a])*float(Etest[t][a])
    return sum*(math.pow(len(common_set),gama)-1)/(len(common_set)-1)

# hipotesis test function
def hipotesisTest(ALFA, gama, t, Etrain,Etest,Strain,Stest):
    sum=0
    # loop passing all alfas
    for i in range(0,len(Etrain)):
        curent_alfa=ALFA[i]
        sum=sum+curent_alfa*kerTest(Etrain,Etest,Strain,Stest,i,t,gama)
    return sum




# calculating errors
def checkAlgo(file,y_name):
    min=-1
    min_val=""
    M,Etrain,Etest,Strain,Stest,y_index=articalAlgorithm(file,y_name)
    # calculating error per alfas
    errorM={}
    for gama in range(1,4):
        for t_eta in range(1,22):
            eta=t_eta*0.1
            c=pow(10,-6)
            while c<pow(10,6):
                c=c*10
                ro=1/(c)
                alfa=M[gama,eta,c]
                errorSum=0.0
                # passing examples
                # print len(Etest)
                for i in range (len(Etest)):
                    diff=(hipotesisTest(alfa, gama, i, Etrain,Etest,Strain,Stest)-float(Etest[i][y_index]))
                    if abs(diff)>pow(sys.float_info.max,0.5) or errorSum>pow(sys.float_info.max,0.5):
                        errorSum=sys.float_info.max
                        break
                    else:
                        errorSum=errorSum+math.pow(diff,2)
                errorM[gama,eta,c]=errorSum/len(Etest) # EMS error
                if min==-1 and not math.isnan(errorM[gama,eta,c]):
                    min=errorM[gama,eta,c]
                    min_val=gama,eta,c
                else:
                    if min>errorM[gama,eta,c] and not math.isnan(errorM[gama,eta,c]):
                        min=errorM[gama,eta,c]
                        min_val=gama,eta,c

    print min
    print min_val
    print M[min_val]



################################# scripts that were in use###########################################

# print "Lasik_hyperopic"
# checkAlgo("fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv",'Treatment Param SEQ (intended)')
# 7.2531749543 (1, 0.8, 100000.0)

# print "PRK"
# checkAlgo("fixed 2016-04-12_PRK_care_data_july_16_2015.3.csv",'Treatment Param SEQ (intended)')
# print "Lasik_myopic"
# checkAlgo("fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv",'Treatment Param SEQ (intended)')
#
# print "sara"
# checkAlgo('fixed 2016-04-12_sara_original_unified_add_15.csv',"EDP")
# checkAlgo('check.csv','pos')
# 69.6639915189  (1, 0.1, 10.0)

