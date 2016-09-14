# written by Yonathan Zailer and Assaf Ziv

import csv
import numpy
import graphlab
import math
import sys
import random


# attributes dictionery
att={}




#rearranging the raw data into the format in which the graphlabs
#  recommendation system works with
def rearrangeData(file1,outputFile):
    with open(file1, 'r') as csvinput:
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            # first row in file
            writer.writerow(['user_id','item_id','rating'])
            attributes = []
            data = list(reader)
            row_count = len(data)
            column_count = len(data[0])
            first_row = data[0]
            # saving attributes
            for i in range (1, column_count) :
                attributes = numpy.append(attributes,first_row[i])
            # passing data
            for i in range (1,row_count) :
                current = []
                current = numpy.append(current, [str(i)])
                for j in range (column_count - 1) :
                    row = []
                    row = numpy.append(current, attributes[j])
                    if data[i][j + 1] == '' :
                        continue
                    row = numpy.append(row, data[i][j + 1])
                    writer.writerow(row)


#this method changes the attribute names to numbers so that it could be clearer for the procedure to run without
#any ordering problem(attribute can have alphabetic order which could mess up with the recommendation process).
def changeAtributesNames(file1,file2):
    global att
    with open(file2, 'w') as csvinput2:
        with open(file1, 'r') as csvinput1:
            spamwriter = csv.writer(csvinput2, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            reader = csv.reader(csvinput1)
            rows=list(reader)
            attributes=rows[0]
            l=[]
            letter=1
            #for every attribute exists, put it in the att global variable list
            for attr in attributes:
                att[letter]=attr
                l.append(letter)
                letter=letter+1
            spamwriter.writerow(l)
            #write all other rows as they are
            for i in range(1,len(rows)):
                 spamwriter.writerow(rows[i])

#a function to find the max in a column considering blanks
def maxInColumn(column) :
    max = sys.float_info.max
    max = max * (-1)
    for i in range(0, column.size) :
        if column[i] == '' :
            continue
        if float(column[i]) > max :
            max = float(column[i])
    return max


#a function to find the min in a column considering blanks
def minInColumn(column) :
    min = sys.float_info.max
    for i in range(0, column.size) :
        if column[i] == '' :
            continue
        if float(column[i]) < min :
            min = float(column[i])
    return min


#a function to normalize the columns using the min max method
def normalization(file1,outputFile):
    X = numpy.genfromtxt(file1,delimiter=",",dtype=str)
    with open(file1,'r') as csvinput:
        reader = csv.reader(csvinput)
        data = list(reader)
        #setting the max and min arrays
        max={}
        min={}
        row_count = len(data)
        for i in range(1,X[0].size):
            #finding the max and min for each and every column in the data
            max[i]=maxInColumn(X[1:,i])
            min[i]=minInColumn(X[1:,i])
            #a loop to divide the numbers with the max min method for normalization
            for j in range(1,X[:,i].size):
                if data[j][i] != '' :
                    data[j][i]=(float(X[j][i])-min[i]) / (max[i]-min[i])
        #opening a csv to write the results of the normalization
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            for i in range (row_count) :
                writer.writerow(data[i])
        #opening a csv to write the max and min arrays for later use
        with open('maxMin.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            writer.writerow(['index', 'min', 'max'])
            for i in range(1,len(min.keys()) + 1) :
                writer.writerow([i, min[i], max[i]])

#a function to reverse the normalization procedure
def normalizationReversed(file1,outputFile,y_index):
    X = numpy.genfromtxt(file1,delimiter=",")
    with open('maxMin.csv', 'r') as csvinput1:
        with open(file1,'r') as csvinput2:
            reader1 = csv.reader(csvinput1)
            maxMinData = list(reader1)
            reader2 = csv.reader(csvinput2)
            data = list(reader2)
            row_count = len(data)
            #i stands for column number
            i=y_index
            #j stands for row number
            # print len(X[:,1])
            # print X[:,1].size
            for j in range(1,len(X[:,1])):
                #reversing the values according to this equation: deNormalized = X * (Xmax - Xmin) + Xmin
                data[j][2]= float(data[j][2]) * (float(maxMinData[i][2])- float(maxMinData[i][1])) +\
                            float(maxMinData[i][1])
            #opening a csv to write the results of the normalization
            with open(outputFile, 'w') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                for i in range (row_count) :
                    writer.writerow(data[i])


# this function returns list that holds the file
def uploadCSV(file):
     with open(file,'r') as csvinput:
        reader = csv.reader(csvinput)
        lis=list(reader)
        return lis

# this function remove lines without y value.
def removeRowsWithEmptyY(file1,file2,y_index):
    lis=uploadCSV(file1)
    with open(file2,'w') as csvoutput:
        spamwriter = csv.writer(csvoutput, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in lis:
            if line[y_index]!='':
                spamwriter.writerow(line)

#a function that extract the users column
def extractUsersColumn(file1, file2, columnNumber) :
    X = numpy.genfromtxt(file1,delimiter=",",dtype=str)
    # Y = X[:,-1]
    U = X[1:,0]
    numpy.savetxt(file2, U,fmt='%s', delimiter=",")
    # numpy.savetxt("tempYColumn.csv",Y,fmt='%s', delimiter=",")

# this function updates users names
def changeUsersColumn(file1, file2):
    X = numpy.genfromtxt(file1,delimiter=",",dtype=str)
    for i in range(1,len(X)):
        X[i][0]=str(i)
    numpy.savetxt(file2, X,fmt='%s', delimiter=",")


# this function divide the data to train and test randomaly in the size 70% 30%
def randomlyDistributedToTT(file,test_outout,train_output,y_index):
        X = numpy.genfromtxt(file,delimiter=",",dtype=str)
        Y = []
        for i in range(1,len(X)):
            # the percentage
            if random.random() > 0.7:
                Y.append(X[i])
                X[i][y_index] = ''
        numpy.savetxt(train_output, X,fmt='%s', delimiter=",")
        Y= numpy.array(Y)
        numpy.savetxt(test_outout, Y,fmt='%s', delimiter=",")

# this function calculates RMSE error
def calculateRMSE(file1,file2,y_index):
    test = numpy.genfromtxt(file1,delimiter=",",dtype=str)
    origin = numpy.genfromtxt(file2,delimiter=",",dtype=str)
    sum=0
    # passing test set
    for i in range(1,len(test)):
        tmp1=float(test[i][2])
        tmp2=int(test[i][0])
        tmp3=float(origin[tmp2][y_index])
        sum=sum+math.pow(tmp1-tmp3,2)
    return math.pow((sum/(len(test)-1)),0.5)

# this function for a small data set
def methodOneLessExamples(k,file1,y_index,method):
    removeRowsWithEmptyY(file1,"tmpFile1.csv",y_index)
    changeUsersColumn("tmpFile1.csv","tmpFile2.csv")
    normalization("tmpFile2.csv","tmpFile3.csv")
    changeAtributesNames("tmpFile3.csv","tmpFile4.csv")
    X = numpy.genfromtxt("tmpFile4.csv",delimiter=",",dtype=str)
    size=len(X)-1
    test_size=int(size/k)
    m={}
    RMSE={}
    for i in range(0,k):
        print "step: "+str(i)
        test=numpy.ndarray(shape=(2,(len(X[0]))),dtype=object)
        test[0]=X[0]
        t_index=[x for x in range(1+i*test_size,1+test_size+i*test_size)]
        d_i=1
        train=X
        for d in range(1+i*test_size,1+test_size+i*test_size):
            test[d_i]=X[d]
            d_i=d_i+1
            train[d][y_index]=''
        for j in range(len(test)):
            test[j][y_index]=''
        numpy.savetxt("tmpTest1.csv",test,fmt='%s', delimiter=",")
        rearrangeData("tmpTest1.csv","tmpTest2.csv")
        numpy.savetxt("tmpTrain1.csv",train,fmt='%s', delimiter=",")
        rearrangeData("tmpTrain1.csv","tmpTrain2.csv")
        train = graphlab.SFrame.read_csv("tmpTrain2.csv")
        # choosing method
        if method == 'factorization' :
            trainSettings = graphlab.recommender.factorization_recommender.create(train, target='rating')
        if method == 'item_similarity' :
            trainSettings = graphlab.recommender.item_similarity_recommender.create(train, target='rating')
        if method == 'popularity' :
            trainSettings = graphlab.recommender.popularity_recommender.create(train, target='rating')
        if method == 'ranking_factorization' :
            trainSettings = graphlab.recommender.ranking_factorization_recommender.create(train, target='rating')
        if method == 'auto' :
            trainSettings = graphlab.recommender.create(train, 'user_id', 'item_id')
        # use test
        # test = graphlab.SFrame.read_csv("tmpTest2.csv")
        results = trainSettings.recommend(items=[y_index+1])
        results.save("letsFinish1.csv",format="csv")
        normalizationReversed("letsFinish1.csv","letsFinish2.csv",y_index)
        RMSE[i]=calculateRMSE("letsFinish2.csv","tmpFile2.csv",y_index)
    sum=0
    for i in range(0,k):
        sum=sum+RMSE[i]
    return (sum/k)


# this function for a big data set
def methodOneManyExamples(file1,y_index,method):
    removeRowsWithEmptyY(file1,"tmpFile1.csv",y_index)
    changeUsersColumn("tmpFile1.csv","tmpFile2.csv")
    normalization("tmpFile2.csv","tmpFile3.csv")
    changeAtributesNames("tmpFile3.csv","tmpFile4.csv")
    randomlyDistributedToTT("tmpFile4.csv","tmpTest1.csv","tmpTrain1.csv", y_index)
    rearrangeData("tmpTrain1.csv","tmpTrain2.csv")
    train = graphlab.SFrame.read_csv("tmpTrain2.csv")
    # choosing method
    if method == 'factorization' :
        trainSettings = graphlab.recommender.factorization_recommender.create(train, target='rating')
    if method == 'item_similarity' :
        trainSettings = graphlab.recommender.item_similarity_recommender.create(train, target='rating')
    if method == 'popularity' :
        trainSettings = graphlab.recommender.popularity_recommender.create(train, target='rating')
    if method == 'ranking_factorization' :
        trainSettings = graphlab.recommender.ranking_factorization_recommender.create(train, target='rating')
    if method == 'auto' :
        trainSettings = graphlab.recommender.create(train, 'user_id', 'item_id')

    #calculate the error rate using the Root-mean-square deviation
    results = trainSettings.recommend(items=[y_index+1])
    results.save("letsFinish1.csv",format="csv")
    normalizationReversed("letsFinish1.csv","letsFinish2.csv",y_index)
    RMSE=calculateRMSE("letsFinish2.csv","tmpFile2.csv",y_index)
    print "RMSE ERORR: "+str(RMSE)

################################# scripts that were in use###########################################

#Lasik_hyperopic:
# a=methodOneLessExamples(120,"fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv",10,"item_similarity")
# b=methodOneLessExamples(120,"fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv",10,"popularity")
# c=methodOneLessExamples(120,"fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv",10,"ranking_factorization")
# d=methodOneLessExamples(120,"fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv",10,"factorization")
# print "item_similarity: "+str(a)+",popularity: "+str(b)+",ranking_factorization: "+str(c)+",factorization: "+str(d)

#heart:
# a=methodOneLessExamples(121,"fixed 2016-04-12_sara_original_unified_add_15.csv",35,"item_similarity")
# b=methodOneLessExamples(121,"fixed 2016-04-12_sara_original_unified_add_15.csv",35,"popularity")
# c=methodOneLessExamples(121,"fixed 2016-04-12_sara_original_unified_add_15.csv",35,"ranking_factorization")
# d=methodOneLessExamples(121,"fixed 2016-04-12_sara_original_unified_add_15.csv",35,"factorization")
# print "item_similarity: "+str(a)+",popularity: "+str(b)+",ranking_factorization: "+str(c)+",factorization: "+str(d)

#PRK:
# methodOneManyExamples('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv',10,"item_similarity")
# methodOneManyExamples('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv',10,"popularity")
# methodOneManyExamples('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv',10,"ranking_factorization")
# methodOneManyExamples('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv',10,"factorization")

#Lasik_myopic:
# methodOneManyExamples('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv',10,"item_similarity")
# methodOneManyExamples('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv',10,"popularity")
# methodOneManyExamples('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv',10,"ranking_factorization")
# methodOneManyExamples('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv',10,"factorization")