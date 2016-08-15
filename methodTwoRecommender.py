# written by Yonathan Zailer and Assaf Ziv
from random import randint
import csv
import numpy
import graphlab
import math
import pandas as pd
import sys
import os


predictions_number=0
att={}

#a function to create a synthetic data
def createData(file1):
    with open(file1, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #attr id
        spamwriter.writerow(numpy.array(['m0','m1','m2','m3','m4','m5','m6','m7']))
        # hard coded basis
        p1 =numpy.array([1,2,3,4,5,6,7])
        p2 =numpy.array([1,64,2,3,6,2,1])
        p3 =numpy.array([3,6,8,4,1,9,3])
        p4 =numpy.array([65,2,3,8,1,5,4])
        p5 =numpy.array([69,2,1,5,8,4,7])
        p6 =numpy.array([63,25,14,87,8,2,3])
        p7 =numpy.array([9,9,8,4,1,2,5])
        set_size=1000
        # putInFile=numpy.array([9,9,8,4,1,2,5])
        # create data set
        for index_set in range(set_size):
            # rand coef
            coef={}
            for index in range(7):
                coef[index]=randint(0,9)
            putInFile= coef[0]*p1+coef[1]*p2+coef[2]*p3+coef[3]*p4+coef[4]*p5+coef[5]*p6+coef[6]*p7
            spamwriter.writerow(numpy.append(['U'+str(index_set)],putInFile))

#a function to find the max in a column considering blanks
def maxInColumn(column) :
    max = sys.float_info.max
    max = max * (-1)
    for i in range(0, column.size) :
        if column[i] == '' :
            continue
        if column[i] > max :
            max = column[i]
    return max


#a function to find the min in a column considering blanks
def minInColumn(column) :
    min = sys.float_info.max
    for i in range(0, column.size) :
        if column[i] == '' :
            continue
        if column[i] < min :
            min = column[i]
    return min

#a function to normalize the columns using the min max method
def normalization(file1,outputFile):
    X = numpy.genfromtxt(file1,delimiter=",")
    with open(file1,'r') as csvinput:
        reader = csv.reader(csvinput)
        data = list(reader)
        #setting the max and min arrays
        max={}
        min={}
        row_count = len(data)
        for i in range(1,X[0].size):
            # max[i]=numpy.amax(X[1:,i])
            # min[i]=numpy.amin(X[1:,i])
            #finding the max and min for each and every column in the data
            max[i]=maxInColumn(X[1:,i])
            min[i]=minInColumn(X[1:,i])
            #a loop to divide the numbers with the max min method for normalization
            for j in range(1,X[:,i].size):
                if data[j][i] != '' :
                    data[j][i]=(X[j][i]-min[i]) / (max[i]-min[i])
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

#a function to normalize the columns using the min max method
def normalizationReversed(file1,outputFile):
    X = numpy.genfromtxt(file1,delimiter=",")
    with open('maxMin.csv', 'r') as csvinput1:
        with open(file1,'r') as csvinput2:
            reader1 = csv.reader(csvinput1)
            maxMinData = list(reader1)
            reader2 = csv.reader(csvinput2)
            data = list(reader2)
            row_count = len(data)
            #i stands for column number
            for i in range(1,X[0].size):
                #j stands for row number
                for j in range(1,X[:,i].size):
                    #reversing the values according to this equation: deNormalized = X * (Xmax - Xmin) + Xmin
                    data[j][i]= float(data[j][i]) * (float(maxMinData[i][2])- float(maxMinData[i][1])) +\
                                float(maxMinData[i][1])
            #opening a csv to write the results of the normalization
            with open(outputFile, 'w') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                for i in range (row_count) :
                    writer.writerow(data[i])


#fix function for the laser csv files(deleting empty lines)
def fix(file1,outputFile):
    with open(file1,'r') as csvinput:
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            flag=1
            for row in reader:
                writer.writerow(row)
                # if flag == 1 :
                #     writer.writerow(row)
                #     flag = 0
                # else :
                #     flag = 1

#fix function for the heart csv files(deleting empty lines)
def fix2(file1,outputFile):
    with open(file1,'r') as csvinput:
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            flag=1
            index = 0
            for row in reader:
                if flag == 1 :
                    writer.writerow(numpy.append([index],row))
                    index += 1
                    flag = 0
                else :
                    flag = 1

#a function that delete features randomly to create the sparse
# matrix we need to do recommandation for.
def deleteFeatures(file1,outputFile):
    with open(file1,'r') as csvinput:
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            flag=1
            for row in reader:

                index=randint(0,6)
                if ((randint(0,9)>6)and(flag!=1)):
                    row[index+1]=None
                all.append(row)
                flag=0
            writer.writerows(all)


#a function that removes the y column so that the recommendation
#  can operate according to algorithm
def removeYColumn(file1, file2, columnNumber) :
    #removing the column and saving it to tempYColumn.csv for later addition
    info = pd.read_csv(file1)
    tempYColumn = info[[columnNumber]]
    filtered = info.drop(info.columns[[columnNumber]], axis=1)
    tempYColumn.to_csv('tempYColumn.csv', index = False)
    filtered.to_csv(file2,index = False)

#a function that adds the y colum after the recommendation took
# place on the csv as according to the algorithm
def addYColumn(file1, file2, columnNumber) :
    #taking the tempYColumn.csv and adding it to the after recoomendation csv
    info = pd.read_csv(file1)
    tempYColumn = pd.read_csv('tempYColumn.csv')
    result = pd.concat([info, tempYColumn], axis = columnNumber)
    result.to_csv(file2,index = False)


#rearranging the raw data into the format in which the graphlabs
#  recommendation system works with
def rearrangeData(file1,outputFile):
    with open(file1, 'r') as csvinput:
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            writer.writerow(['user_id','item_id','rating'])
            attributes = []
            data = list(reader)
            row_count = len(data)
            column_count = len(data[0])
            first_row = data[0]
            for i in range (1, column_count) :
                attributes = numpy.append(attributes,first_row[i])
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

#rearrage the data after the recommendation procedure took place so that
# the format would be in the raw data format we got in the first place.
def rearrangeDataReversed(file1,outputFile):
    with open(file1, 'r') as csvinput:
        with open(outputFile, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            attributes = [[1]]
            reader = csv.reader(csvinput)
            data = list(reader)
            index = 1
            #get number of attributes
            while 1 :
                if (data[index][1] == str(index + 1)) :
                    attributes = numpy.append(attributes,[index + 1])
                    index += 1
                else :
                    break
            writer.writerow(attributes)
            #put the number of attributes needed in index
            index -= 1
            #get the number of iterations needed
            k = len(data)
            numberOfIterations = data[len(data) - 1][0]
            for i in range(0, int(numberOfIterations)) :
                row = [[i + 1]]
                #iterate over the right rows
                for j in range(i * index + 1, i * index + 1 + index) :
                    row = numpy.append(row, data[j][2])
                writer.writerow(row)

#this function "fill in the blanks", run the recommendation system of graph lab according
# to the method it gets as variable which could be factorization,
#  item_similarity, popularity, ranking_factorization or auto for graphlab to choose the optimal for you.
def fillInTheBlanks(file1,outputFile,method):
    global  predictions_number
    sf = graphlab.SFrame.read_csv(file1)
    train, test = graphlab.recommender.util.random_split_by_user(sf)
    #make a model according to the method we get and a trainsettings for accuracy calculation
    if method == 'factorization' :
        model = graphlab.recommender.factorization_recommender.create(sf, target='rating')
        trainSettings = graphlab.recommender.factorization_recommender.create(train, target='rating')
    if method == 'item_similarity' :
        model = graphlab.recommender.item_similarity_recommender.create(sf, target='rating')
        trainSettings = graphlab.recommender.item_similarity_recommender.create(train, target='rating')
    if method == 'popularity' :
        model = graphlab.recommender.popularity_recommender.create(sf, target='rating')
        trainSettings = graphlab.recommender.popularity_recommender.create(train, target='rating')
    if method == 'ranking_factorization' :
        model = graphlab.recommender.ranking_factorization_recommender.create(sf, target='rating')
        trainSettings = graphlab.recommender.ranking_factorization_recommender.create(train, target='rating')
    if method == 'auto' :
        model = graphlab.recommender.create(sf, 'user_id', 'item_id')
        trainSettings = graphlab.recommender.create(train, 'user_id', 'item_id')
    #make the recommendation it self for the current data
    results = model.recommend()
    #calculate the error rate using the Root-mean-square deviation
    errorRate = trainSettings.evaluate_rmse(test, target='rating')
    print(errorRate)
    predictions_number=results.num_rows()
    results.save('output_recommander.csv', format='csv')
    #merging the 2 csvs to 1 to create a fully predicted data csv
    test1 = pd.read_csv(file1)
    test2 = pd.read_csv('output_recommander.csv')
    test2 = test2.drop('rank', 1)
    test2.columns = ['user_id', 'item_id','rating']
    test3 = pd.concat([test1, test2])
    test3 = test3.sort(["user_id","item_id"])
    #drop the first column which is row number
    test3.to_csv(outputFile,index = False)

#this function calculate the accuracy of the recommendation output we got on the synthetic data
#  we created, it uses the mse method.
def accuracy(file1,file2):
    global  predictions_number
    rearrangeData(file1,'synthetic_origin_formated.csv')
    with open('synthetic_origin_formated.csv','r') as csvinput1:
        with open(file2, 'r') as csvinput2:
            reader1 = csv.reader(csvinput1)
            reader2 = csv.reader(csvinput2)
            sum=0
            origin = list(reader1)
            result= list(reader2)
            size=len(result)
            #sum up the errors to sum variable
            for i in range(1,size):
                sum=sum+math.pow((float(origin[i][2])-float(result[i][2])),2)
            #finally calculate the accuracy by dividing sum by the number of predictions.
            acc=sum/( predictions_number)
            print acc

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

#the reversed function for the one above, it restores the original attribute names to the file
#  we got as output from the recommendation process.
def changeAtributesNamesReversed(file1,file2):
    global att
    with open(file2, 'w') as csvinput2:
        with open(file1, 'r') as csvinput1:
            spamwriter = csv.writer(csvinput2, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            reader = csv.reader(csvinput1)
            rows=list(reader)
            attributes=rows[0]
            l=[]
            #take all the attribute from the att global variable list
            for index in range (1, len(attributes) + 1):
                l.append(att[index])
            #write the attribute row to the csv file
            spamwriter.writerow(l)
            #write all other rows as they are
            for i in range(1,len(rows)):
                 spamwriter.writerow(rows[i])


#this function runs the recommending scenario for the given file and uses the recommenderMethod it's given,
#the toDelete variable indicates wheather to delete csv's that were created during the recommending process.
def runRecommenderScenario(fileName, label, recommenderMethod,toDelete,YcolumnNumber) :

    removeYColumn(fileName, 'WithoutY_fixed 2016-04-12_sara_original_unified_add_15.csv', YcolumnNumber)
    changeAtributesNames('WithoutY_fixed 2016-04-12_sara_original_unified_add_15.csv','changed_attributes.csv')
    # normalization('changed_attributes.csv',"real_normalized.csv")
    # rearrangeData("real_normalized.csv",'real_recommenderFormated.csv')
    rearrangeData('changed_attributes.csv','real_recommenderFormated.csv')
    fillInTheBlanks('real_recommenderFormated.csv','recommender_result_' + recommenderMethod + ".csv", recommenderMethod)
    rearrangeDataReversed('recommender_result_' + recommenderMethod + ".csv",'originalFormatedNoAttributes_results.csv')
    changeAtributesNamesReversed('originalFormatedNoAttributes_results.csv','originalFormated_results.csv')
    # normalizationReversed('originalFormated_results.csv','originalFormatedDeNormalized_results.csv')
    # addYColumn('originalFormatedDeNormalized_results.csv','originalFormatedDeNormalizedWithY_results_' +
    #            label + "_" + recommenderMethod + ".csv", YcolumnNumber)
    addYColumn('originalFormated_results.csv','originalFormatedDeNormalizedWithY_results_' +
               label + "_" + recommenderMethod + ".csv", YcolumnNumber)
    if (toDelete == 1) :
        os.remove('WithoutY_fixed 2016-04-12_sara_original_unified_add_15.csv')
        os.remove('changed_attributes.csv')
        # os.remove("real_normalized.csv")
        os.remove('real_recommenderFormated.csv')
        os.remove('originalFormated_results.csv')
        os.remove('recommender_result_' + recommenderMethod + ".csv")
        os.remove('originalFormatedNoAttributes_results.csv')
        # os.remove('originalFormatedDeNormalized_results.csv')
        os.remove('output_recommander.csv')
        os.remove('tempYColumn.csv')
        # os.remove('maxMin.csv')
        print "deleted inprocess csv's"



################################# scripts that were in use ###########################################

# #hyperopic
#
# #Lasik hyperopic with factorization
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv', "hyperopic",
#                        "factorization" , 0, 10)
#
# #Lasik hyperopic with popularity
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv', "hyperopic",
#                        "popularity" ,0, 10)
#
# #Lasik hyperopic with ranking_factorization
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv', "hyperopic",
#                        "ranking_factorization" ,0, 10)
#
# #Lasik hyperopic with item_similarity
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv', "hyperopic",
#                        "item_similarity" ,0, 10)
# #myopic
#
# #Lasik myopic with factorization
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv', "myopic","factorization" , 0, 10)
#
# #Lasik myopic with popularity
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv', "myopic", "popularity" ,0, 10)
#
# #Lasik myopic with ranking_factorization
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv', "myopic", "ranking_factorization" ,0, 10)
#
# #Lasik myopic with item_similarity
# runRecommenderScenario('fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv', "myopic", "item_similarity" ,0, 10)

#PRK

# #PRK with factorization
# runRecommenderScenario('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv', "prk","factorization" , 0, 10)
#
# #PRK with popularity
# runRecommenderScenario('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv', "prk", "popularity" ,0, 10)
#
# #PRK with ranking_factorization
# runRecommenderScenario('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv', "prk", "ranking_factorization" ,0, 10)

#PRK with item_similarity
runRecommenderScenario('fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv', "prk", "item_similarity" ,1, 10)

#Heart

#heart with factorization
runRecommenderScenario('fixed FULL 2016-04-12_sara_original_unified_add_15_.csv', "heart",  "factorization" , 1, 35)

#heart with popularity
runRecommenderScenario('fixed FULL 2016-04-12_sara_original_unified_add_15_.csv', "heart","popularity" ,1, 35)

#heart with ranking_factorization
runRecommenderScenario('fixed FULL 2016-04-12_sara_original_unified_add_15_.csv', "heart","ranking_factorization" ,1, 35)

#heart with item_similarity
runRecommenderScenario('fixed FULL 2016-04-12_sara_original_unified_add_15_.csv', "heart","item_similarity" ,1, 35)


################################# synthetic creating and using ###########################################

# # creating the synthetic data and doing the recommendation and accuracy on it.
# createData('synthetic_origin.csv')
# normalization('synthetic_origin.csv',"normalized.csv")
# deleteFeatures("normalized.csv",'synthetic_with_holes.csv')
# rearrangeData('synthetic_with_holes.csv','synthetic_formated.csv')
#
# fillInTheBlanks('synthetic_formated.csv','synthetic_result_factorization.csv', "factorization")
# print "result for factorization"
# accuracy("normalized.csv",'synthetic_result_factorization.csv')
#
# fillInTheBlanks('synthetic_formated.csv','synthetic_result_item_similarity.csv',"item_similarity")
# print "result for item similarity"
# accuracy("normalized.csv",'synthetic_result_item_similarity.csv')
#
# fillInTheBlanks('synthetic_formated.csv','synthetic_result_popularity.csv',"popularity")
# print "result for popularity"
# accuracy("normalized.csv",'synthetic_result_popularity.csv')
#
# fillInTheBlanks('synthetic_formated.csv','synthetic_result_ranking_factorization.csv',"ranking_factorization")
# print "result for ranking factorization"
# accuracy("normalized.csv",'synthetic_result_ranking_factorization.csv')

################################# fixes that were used ###########################################

# #fixs for the csv's that we got (empty lines and empty columns removing, adding users column in heart)
# fix('2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv','fixed 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv')
# fix('2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv','fixed 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv')
# fix('2016-04-12_PRK_care_data_july_16_2015.3.csv','fixed 2016-04-12_PRK_care_data_july_16_2015.3.csv')
# fix2('2016-04-12_sara_original_unified_add_15.csv','fixed 2016-04-12_sara_original_unified_add_15.csv')


# # removing the rows in the csv's in which the y value is emtpy
# removeRowsWithEmptyY('fixed 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv',
#                      'fixed FULL 2016-04-12_Lasik_hyperopic_care_data_july_16_2015.3.csv',10)
# removeRowsWithEmptyY('fixed 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv',
#                      'fixed FULL 2016-04-12_Lasik_myopic_care_data_july_16_2015.3.csv',10)
# removeRowsWithEmptyY('fixed 2016-04-12_PRK_care_data_july_16_2015.3.csv',
#                      'fixed FULL 2016-04-12_PRK_care_data_july_16_2015.3.csv',10)