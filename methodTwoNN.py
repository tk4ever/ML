# written by Yonathan Zailer and Assaf Ziv
__author__ = 'assaf'
import tensorflow as tf
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Flatten, Dense, Dropout
import numpy
import csv
import pandas as pd
import random
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
import h5py

attSize = 0

#this function operates to make a randomly distributed train and test sets,
# the percentage of the data distributes is 70 percent for the train data
#and 30 percent for the test data.
#the return values are nd arrays of X_train , Y_train and X_test, Y_test
def randomlyDistributed(input_file, columnNumber) :
    with open(input_file) as data:
        with open("test.csv", 'w') as test:
            with open("train.csv", 'w') as train:
                header = next(data)
                test.write(header)
                train.write(header)
                for line in data:
                    #if we got a number that is smaller than 0.7 put it in the train
                    if random.random() < 0.7:
                        train.write(line)
                    else:
                        test.write(line)
    X_train , Y_train= loadXsAndYs("train.csv", columnNumber)
    X_test , Y_test = loadXsAndYs("test.csv", columnNumber)
    return X_train, Y_train, X_test, Y_test

#a function that removes the y column so that the recommendation
#  can operate according to algorithm
def removeYColumn(file1, file2, columnNumber) :
    #removing the column and saving it to tempYColumn.csv for later addition
    info = pd.read_csv(file1)
    tempYColumn = info[[columnNumber]]
    filtered = info.drop(info.columns[[columnNumber]], axis=1)
    tempYColumn.to_csv('Ys.csv', index = False)
    filtered.to_csv(file2,index = False)


#this function load the X's set(attributes set) and Y's set (to be predicted column)
# from the given file, by using the removeYcolumn function, and removes the first row
# that has the attribute name string, furthermore it casts the data to float from string
def loadXsAndYs (filename, columnNumber) :
    #divid the data into Xs and Ys csvs using the removeYcolumn function
    removeYColumn(filename, 'Xs.csv', columnNumber)
    X = numpy.genfromtxt('Xs.csv',delimiter=",",dtype=str)
    X = numpy.delete(X, (0), axis=0)
    X = X.astype(float)
    X = X.reshape(X.shape[0], X.shape[1])
    Y = numpy.genfromtxt('Ys.csv',delimiter=",",dtype=str)
    Y = numpy.delete(Y, (0), axis=0)
    Y = Y.astype(float)
    Y = Y.reshape(Y.shape[0], 1)
    return X, Y

#a function in which generates data to test the accuracy of the regression model
#we chose the plus data to test the accuracy, in which the last column has the sum of
#all the other columns
def generatePlusData(outputFile):
    with open(outputFile, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        for i in range(0, 30000) :
            a = random.random() * 10
            b = random.random() * 10
            c = random.random() * 10
            d = random.random() * 10
            e = random.random() * 10
            f = random.random() * 10
            g = random.random() * 10
            h = random.random() * 10
            i = random.random() * 10
            total = a + b + c + d + e + f + g + h + i
            writer.writerow([a, b, c, d, e, f, g, h, i, total])

#the baseline model is a sort of routine function that is passed to the KerasRegressor
#it sets the architecture of the model and returns it
def baseline_model() :
    global attSize
    # create model
    print attSize
    model = Sequential()
    model.add(Dense(200, input_dim=attSize,activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam')
    return model

#this function operates the kfold procedure we are required to use in order to calculate
#the accuracy of our sparse data set.
def kfold(estimator, X, Y):
    global attSize
    #set a seed variable that will change randomly
    seed = 7
    numpy.random.seed(seed)
    #set the kfold settings n_folds=attSize mean leave one out
    kfold = KFold(n=len(X), n_folds=attSize, random_state=seed)
    #perform the cross validation with score
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    return results.mean()

#this is the function in which operates the regression and uses all the other function
#to complete it, it gets the filename which holds the data, a boolean variable that indicates
#whether the data is sparse, and yColumnNumber so that we know what column should we predict
def operateRegression(filename, isSparse, yColumnNumber) :
    global attSize
    with open('scores' + filename + '.csv','w') as csvoutput:
        spamwriter = csv.writer(csvoutput, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # generatePlusData('plusData.csv')
        # X_train, Y_train, X_test, Y_test = randomlyDistributed(
        #     'plusData.csv', 9)
        # X_train, Y_train, X_test, Y_test = randomlyDistributed(
        #     'originalFormatedDeNormalizedWithY_results_myopic_factorization.csv', 16)
        if isSparse == True :
            #if we use the leave one out kfold validation method
            X, Y = loadXsAndYs(filename, yColumnNumber)
            attSize = X[0].size
            estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=100, verbose=0)
            spamwriter.writerow([kfold(estimator, X, Y)])
        else:
            for i in range (0, 5) :
                print i
                # #if we use the 70% 30% validation method of train and test
                X_train, Y_train, X_test, Y_test = randomlyDistributed(filename, yColumnNumber)
                attSize = X_train[0].size
                #define the model of the DNN
                model = Sequential()
                model.add(Dense(200, input_dim=attSize,activation='relu'))
                model.add(Dense(200, activation='relu'))
                model.add(Dense(200, activation='relu'))
                model.add(Dense(200, activation='relu'))

                # model.add(Dense(200, activation='relu'))
                # model.add(Dense(200, activation='relu'))
                # model.add(Dense(200, activation='relu'))
                # model.add(Dense(200, activation='relu'))
                # model.add(Dense(200, activation='relu'))

                model.add(Dense(1))
                model.compile(loss='mse', optimizer='adam')
                #start training
                model.fit(X_train, Y_train, nb_epoch=25, batch_size=attSize, verbose=1)
                #evaluate the score of the test set.
                score = model.evaluate(X_test, Y_test, batch_size=attSize / 2)
                predicted = model.predict(X_test)
                #save the weights to weights file for later use
                model.save_weights('model_weights' + filename + '.hdf5', overwrite = 1)
                print score
                spamwriter.writerow([score])

################################ scripts that were in use ###########################################

# #heart
# operateRegression('originalFormatedDeNormalizedWithY_results_heart_factorization.csv', True, 35)
#
# operateRegression('originalFormatedDeNormalizedWithY_results_heart_item_similarity.csv', True, 35)
#
# operateRegression('originalFormatedDeNormalizedWithY_results_heart_popularity.csv', True, 35)

# operateRegression('originalFormatedDeNormalizedWithY_results_heart_ranking_factorization.csv', True, 35)
#
# #hyperopic
# operateRegression('originalFormatedDeNormalizedWithY_results_hyperopic_factorization.csv', True, 10)
#
# operateRegression('originalFormatedDeNormalizedWithY_results_hyperopic_item_similarity.csv', True, 10)
#
# operateRegression('originalFormatedDeNormalizedWithY_results_hyperopic_popularity.csv', True, 10)
#
# operateRegression('originalFormatedDeNormalizedWithY_results_hyperopic_ranking_factorization.csv', True, 10)

#myopic
operateRegression('originalFormatedDeNormalizedWithY_results_myopic_factorization.csv', False, 10)

operateRegression('originalFormatedDeNormalizedWithY_results_myopic_item_similarity.csv', False, 10)

operateRegression('originalFormatedDeNormalizedWithY_results_myopic_popularity.csv', False, 10)

operateRegression('originalFormatedDeNormalizedWithY_results_myopic_ranking_factorization.csv', False, 10)

#prk
operateRegression('originalFormatedDeNormalizedWithY_results_prk_factorization.csv', False, 10)

operateRegression('originalFormatedDeNormalizedWithY_results_prk_item_similarity.csv', False, 10)

operateRegression('originalFormatedDeNormalizedWithY_results_prk_popularity.csv', False, 10)

operateRegression('originalFormatedDeNormalizedWithY_results_prk_ranking_factorization.csv', False, 10)
