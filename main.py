# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 07:56:45 2019
@author: soohl254

This is the main program for the analysis of gamma, neutrona and cherenkov data 
using machine learning algorithms.
"""


import os
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib
import matplotlib.pyplot as plt
import random
import math
#import pydot

from matplotlib.legend_handler import HandlerLine2D
from sklearn import preprocessing
from sklearn import metrics
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.tree import export_graphviz 
from mpl_toolkits.mplot3d import axes3d
   

def normalize_activities(pndfrm,feat):
    #Normalize selected gamma activities to 1 (for each feature).
    newFrame=pndfrm.copy()
    columnNames = list(feat)
    newFrame['gamma_sum'] = newFrame[columnNames].sum(axis=1)
    newFrame[columnNames] = newFrame[columnNames].div(newFrame.gamma_sum, axis='index')
    newFrame=newFrame.drop(columns = ['gamma_sum'])
    return newFrame

def calculate_activity(dtaFrm):
    #Half-lives of gamma isotopes is in days
    halflife={'Y91':58.5, 'Zr95':64, 'Nb95':35, 'Ru106':372, 'Cs134':2.065*365, 'Cs137':30.1*365,
          'Eu154':8.6*365, 'Ce141':32.5, 'Ce144':285, 'Sr90':28.8*365}  

    #Convert days to seconds
    d2s=86400

    #Convert Atomic density to Activity using A=N*lambda=N*ln(2)/t½. Denote isotope+A.
    for iso in halflife:
        dtaFrm[iso+'A']=dtaFrm[iso]*1e24*(np.log(2)/(halflife[iso]*d2s))

    return dtaFrm
 
        
def pca_with_scoresPlot(score, coeff, dataframe, feat, labels=None):
    #Plots the PC-score (value) from a PCA analysis for all data points. Shows the contribution by  diffrent features to the PCs
    #xs is PC1; ys is PC2 
    pc1s = score[:,0]
    pc2s = score[:,1]
    n = coeff.shape[0]
    #Rescale the pc-values so they span a range of 1 in each dimension
    scalepc1 = 1.0/(pc1s.max() - pc1s.min())
    scalepc2 = 1.0/(pc2s.max() - pc2s.min())
    
    plt.figure()   
    plt.scatter(pc1s*scalepc1, pc2s*scalepc2, c=dataframe.CT, cmap='Greens')#if more classes, c = y)

    #Display the importance of the features projected onto PC1 and PC2
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, feat[i], color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

def applynoise(x, columns):
    #Generates noise from a random Gaussian distribution with mean at 1 and a sigma of 0.05.
    for col in columns:
        x[col]*random.normalvariate(1,0.05)
    return x


def RndForReg(X_train, y_train, X_test, Y_test, par, fn):
    plot_components=False  
     
    boot=par['bootstrap']
    max_dep=par['max_depth']
    max_feat=['max_features']
    min_samp_leaf=par['min_samples_leaf']
    min_samp_split=par['min_samples_split']
    n_est=par['n_estimators']  
    max_f=len(fn)
    
    mse_rF = []   
    regressor = RandomForestRegressor(bootstrap=boot, max_depth=max_dep, max_features=max_f, min_samples_leaf=min_samp_leaf, min_samples_split=min_samp_split, n_estimators= n_est)
    regressor.fit(X_train, y_train.flatten())
    y_pred = regressor.predict(X_test)
    mse_p = mean_squared_error(Y_test, y_pred)
    mse_rF.append(mse_p)
    msemin = np.argmin(mse_rF)

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse_rF), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse_rF)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('RandomForest')
            plt.xlim(xmin=-1)
        plt.show()            
    
    return y_pred, regressor


def PLSReg(X_train, y_train, X_test, Y_test):
    # Define PLS object
    pls = PLSRegression(n_components=4)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)
    return y_pred 


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    return accuracy


def calc_learning_curves(estimator, X_train, y_train, X_test, y_test):
    #A learning curve shows the validation and training score of an estimator for varying numbers 
    #of training samples. It is a tool to find out how much we benefit from adding more training 
    #data and whether the estimator suffers more from a variance error or a bias error.
    
    train_score = []
    test_score = []
    training_set_size = np.linspace(5, len(X_train), 20, dtype='int')

    for i in training_set_size:
        # fit the model only using limited training examples
        estimator.fit(X_train[0:i, :], y_train[0:i].flatten())
        train_rmse = np.sqrt(metrics.mean_squared_error(y_train[0:i, :], estimator.predict(X_train[0:i, :])))
        test_rmse = np.sqrt (metrics.mean_squared_error(y_test, estimator.predict(X_test)))

        train_score.append(train_rmse)
        test_score.append(test_rmse)
    
    return training_set_size, train_score, test_score


def under_or_overfitting(X_train, y_train, X_test, y_test, fn):
    rmse_estimator=[]
    estimator = np.arange(5, 50)
   
    for i in estimator:
        regressor_estimator = RandomForestRegressor(n_estimators=i, max_depth=10, max_features=len(fn), min_samples_leaf=2, min_samples_split=3)
        regressor_estimator.fit(X_train, y_train.flatten())
        y_pred_estimator = regressor_estimator.predict(X_test)
     
        rmse_p_estimator = np.sqrt(metrics.mean_squared_error(y_test, y_pred_estimator))
        rmse_estimator.append(rmse_p_estimator)
    rmse_min_estimator = np.argmin(rmse_estimator) 
    

    rmse_depth=[]
    depth = np.arange(1, 20)
    for i in depth:
        regressor_depth = RandomForestRegressor(n_estimators=10, max_depth=i, max_features=len(fn), min_samples_leaf=2, min_samples_split=3)
        #regressor_depth = RandomForestRegressor(n_estimators=10, max_depth=i, max_features=5, min_samples_leaf=2, min_samples_split=3)

        regressor_depth.fit(X_train, y_train.flatten())
        y_pred_depth = regressor_depth.predict(X_test)
     
        rmse_p_depth = np.sqrt(metrics.mean_squared_error(y_test, y_pred_depth))
        rmse_depth.append(rmse_p_depth)
    rmse_min_depth = np.argmin(rmse_depth) 
    
    rmse_feat=[]

    feat = np.arange(1, len(fn)+1)
    for i in feat:
        regressor_feat = RandomForestRegressor(n_estimators=10, max_depth=10, max_features=i, min_samples_leaf=2, min_samples_split=3)
        regressor_feat.fit(X_train, y_train.flatten())
        y_pred_feat = regressor_feat.predict(X_test)
        #feat_imp = y_pred_feat.feature_importances_
     
        rmse_p_feat = np.sqrt(metrics.mean_squared_error(y_test, y_pred_feat))
        rmse_feat.append(rmse_p_feat)
    rmse_min_feat = np.argmin(rmse_feat) 

    return rmse_min_estimator, rmse_estimator, estimator, rmse_min_depth, rmse_depth, depth, rmse_min_feat, rmse_feat, feat


def Grid_Search_CV_RFR(X_train, y_train, X_test, y_test):
    #Default RandomForestRegression
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train.flatten())
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    r2_default_train = metrics.r2_score(y_train, y_pred_train)
    r2_default_test = metrics.r2_score(y_test, y_pred_test)
    #base_accuracy = evaluate(rf, X_test, y_test.flatten())
    params = rf.get_params()
    #print('Default parameters in the fit are ', params)
    #print('R2 Default Train={:0.5f}'.format(r2_default_train))
    #print('R2 Default Test={:0.5f}'.format(r2_default_test))

    #Instantiate Grid search model to optimize hyperparameters for Random Forest Regression
    param_grid = {
    'bootstrap': [True, False],   #Replacement or not
    'max_depth': [5, 7, 10],    #Depth of each tree in the forest (1-32). 4, 8, 12, 16, 20
    'max_features': ['auto'],     #Number of features to consider when looking for the best split
    'min_samples_leaf': [5, 7],   #Minimum number of samples required to be at a leaf node
    'min_samples_split': [6, 8],  #Minimum samples required to split an internal node (10-100%). 8, 10, 12
    'n_estimators': [10, 20, 30]   #Number of trees in the forest (<200). 10, 20, 30, 40
    }
 
    grid_search =     GridSearchCV(rf, param_grid, cv = 5, refit=True, verbose=0)
    grid_results =    grid_search.fit(X_train, y_train.flatten())
    best_parameters = grid_search.best_params_  
    best_result =     grid_search.best_score_
    best_estimator =  grid_search.best_estimator_   #Optimized randomForestRegressor    
    
    y_pred_grid_train = best_estimator.predict(X_train)
    y_pred_grid_test =  best_estimator.predict(X_test)
    r2_grid_train =     metrics.r2_score(y_train, y_pred_grid_train)
    r2_grid_test =      metrics.r2_score(y_test, y_pred_grid_test)
    
    #print('Optimised parameters in the fit are ', best_parameters)
    #grid_accuracy =   evaluate(best_estimator, X_test, y_test.flatten())  
    #print('base_accuracy={:0.5f}'.format(base_accuracy), 'grid_accuracy={:0.5f}'.format(grid_accuracy), '. Relative improvement of {:0.2f}%'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy)) 
    
    return best_parameters, best_result, best_estimator


  
def main():
    #Read in modelled PWR 17x17 data and put in a DataFrame
    fueldata = pd.read_csv('dataforMVA_strategicPWR_fin_New_Erik_v2.csv', header = 0, index_col = 0)
    fueldata = fueldata.drop(columns = ['cherenkovnobeta'])
    print("Hello!")
    
    #Split the full data set according to fuel parameter properties. 
    #subfueldata = fueldata[fueldata['CT']>0*365].sample(n= 1000, random_state=1)
    subfueldata = fueldata[(fueldata['CT']>0*365) & (fueldata['CT']<20*365)].sample(n=1000, random_state=1)

    #Define which features to use. "All" are all features, "X" are selected features, "Iso" are all gamma isotopes.
    selectionAll   =['Cs137A','Cs134A','Eu154A','Zr95A','Nb95A','Ru106A','Ce141A','Ce144A','Sr90A','tau','cherenkov']
    selectionAllIso=['Cs137A','Cs134A','Eu154A','Zr95A','Nb95A','Ru106A','Ce141A','Ce144A','Sr90A']

    selectionIso=['Cs134','Eu154','Cs137'] 
    #selectionIso=['Eu154','Cs137'] 
    #selectionIso=['Cs137'] 

    #selectionX  =['Cs137','tau','cherenkov']
    #selectionX  =['Eu154','Cs137','tau','cherenkov']
    #selectionX  =['Cs134','Eu154','Cs137','cherenkov']
    selectionX  =['Cs134','Eu154','Cs137','tau','cherenkov']

    features=list(selectionX)

    #*******Explore the dataset using PCA*******         
    #Mean-center the data: Subtract the mean and then divide the result by the standard deviation
    #Reduce dimensionality of the data using decomposition into n_components
    X = subfueldata.loc[:,features].values
    X_std = StandardScaler().fit_transform(X)
    pca = decomposition.PCA(n_components =len(features))
    pca.fit(X_std)
    Xt = pca.transform(X_std)
    print("Explained variance ratio for the components using PCA is:", pca.explained_variance_ratio_) 

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(Xt[:,0], Xt[:,1], Xt[:,2], c=subfueldata.CT, cmap='Greens')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA using {0}'.format(features))

    #Visualize scores plot with two PC-scores using pca_with_scoresPlot
    #pca.components_ is the set of all eigenvectors (or loadings) for your projection space (one eigenvector for each PC).
    #pca.components_ outputs an array of [n_components, n_features]
    pca_with_scoresPlot(Xt[:,0:2], np.transpose(pca.components_[0:2, :]), subfueldata, features)
    plt.xlim(-0.5,1)
    plt.ylim(-1.5,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()
    
    #*******Start of REGRESSION part to predict IE, BU and CT values.*******
    #Use the gamma activities calculated from the atomic densities given in Serpent in units of 1e24/cm3.
    calculate_activity(subfueldata)

#    for row in subfueldata.itertuples(index = True):   
#        subdata['TotGamAct'] = subfueldata.apply(lambda row: row.Y91A+row.Zr95A+row.Nb95A+row.Ru106A+row.Cs134A+row.Cs137A+row.Eu154A+row.Ce141A+row.Ce144A+row.Sr90A, axis=1)
#    for i in features:
#        for row in subfueldata.itertuples(index=True): 
#            if row[i]< 1.0e-10:
#                print(i, row[i])
##            if subfueldata[row,i]< 1.0e-10:
##                subfueldata.iat[row, i]=0.0
    
#    for row in subfueldata.itertuples(index=True): 
#        print(row.Cs134A, row.Cs137A)
#        if row.Nb95A< 1.0e-10: #Or if row.Nb95A/row.TotGamAct<0.01
#            subfueldata.at[row.Index, 'Nb95A'] = 0.0
          
    #subfueldata = subfueldata.drop(columns=['fuelType','reactorType','Ce141','Nb95','Y91','Zr95','Ce144','Ru106','Sr90', 'Ce141A','Nb95A','Y91A','Zr95A','Ce144A','Ru106A','Sr90A'])
   
    #Split the data into a training DataFrame and test DataFrame. Training size is 80% of original data.
    traindf, testdf = train_test_split(subfueldata, random_state=0, test_size=0.2)
 
    testdf = applynoise(testdf, selectionAll)

    norm_traindf= normalize_activities(traindf,selectionIso)
    norm_testdf = normalize_activities(testdf,selectionIso)

    #Create training and test data with gamma isotopes, neutron signature and cherenkov light.
    #StandardScaler standardizes features by removing the mean and scaling to unit variance for each column in the data set.
    ss = StandardScaler()
    Xtrain=norm_traindf.loc[:,selectionX]
    Xtrainf=ss.fit_transform(Xtrain)
    
    Xtest=norm_testdf.loc[:,selectionX]
    Xtestf=ss.transform(Xtest)

    #Create training and test data to predict IE, BU and CT
    YIEtrain=norm_traindf.loc[:,['IE']]
    YBUtrain=norm_traindf.loc[:,['BU']]
    YCTtrain=norm_traindf.loc[:,['CT']]
    YIEtest=norm_testdf.loc[:,['IE']]
    YBUtest=norm_testdf.loc[:,['BU']]
    YCTtest=norm_testdf.loc[:,['CT']]
    
    #Make nparrays out of dataframes
    YIEtrain=YIEtrain.values
    YBUtrain=YBUtrain.values
    YCTtrain=YCTtrain.values
    YIEtest=YIEtest.values
    YBUtest=YBUtest.values
    YCTtest=YCTtest.values
    
    #*********************************************
    #Check learning curves to determine if and when generated data is enough. Study RMSE as fcn of dataset size.
    print('Time for learning curves')
    RndFor = RandomForestRegressor()
    setSizeCT, trainScoreCT, testScoreCT = calc_learning_curves(RndFor, Xtrainf, YCTtrain, Xtestf, YCTtest)
    setSizeBU, trainScoreBU, testScoreBU = calc_learning_curves(RndFor, Xtrainf, YBUtrain, Xtestf, YBUtest)
    setSizeIE, trainScoreIE, testScoreIE = calc_learning_curves(RndFor, Xtrainf, YIEtrain, Xtestf, YIEtest)
    
    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.semilogy(setSizeCT, trainScoreCT, c='gold')
    plt.semilogy(setSizeCT, testScoreCT, c='steelblue')
    plt.legend(['Training set', 'Validation set'])
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('Learning Curve CT')
    
    plt.subplot(132)
    plt.semilogy(setSizeBU, trainScoreBU, c='gold')
    plt.semilogy(setSizeBU, testScoreBU, c='steelblue')
    plt.legend(['Training set', 'Validation set'])
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('Learning Curve BU')
    
    plt.subplot(133)
    plt.semilogy(setSizeIE, trainScoreIE, c='gold')
    plt.semilogy(setSizeIE, testScoreIE, c='steelblue')
    plt.legend(['Training set', 'Validation set'])
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('Learning Curve IE')
    plt.show()

    #*********************************************  
    #Investigare under- and overfitting
    print('Study overfitting...')
    RMSE_min_est_CT, RMSE_est_CT, est_CT, RMSE_min_dep_CT, RMSE_dep_CT, dep_CT, RMSE_min_feat_CT, RMSE_feat_CT, feat_CT  = under_or_overfitting(Xtrainf, YCTtrain, Xtestf, YCTtest, features)
    RMSE_min_est_BU, RMSE_est_BU, est_BU, RMSE_min_dep_BU, RMSE_dep_BU, dep_BU, RMSE_min_feat_BU, RMSE_feat_BU, feat_BU  = under_or_overfitting(Xtrainf, YBUtrain, Xtestf, YBUtest, features)
    RMSE_min_est_IE, RMSE_est_IE, est_IE, RMSE_min_dep_IE, RMSE_dep_IE, dep_IE, RMSE_min_feat_IE, RMSE_feat_IE, feat_IE  = under_or_overfitting(Xtrainf, YIEtrain, Xtestf, YIEtest, features)
    
    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)   
    plt.plot(est_CT, np.array(RMSE_est_CT), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(est_CT[RMSE_min_est_CT], np.array(RMSE_est_CT)[RMSE_min_est_CT], 'P', ms=7, mfc='red')
    plt.xlabel('n_estimators for CT')
    plt.ylabel('RMSE')
    plt.title('RandomForest')   
    plt.subplot(132)   
    plt.plot(est_BU, np.array(RMSE_est_BU), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(est_BU[RMSE_min_est_BU], np.array(RMSE_est_BU)[RMSE_min_est_BU], 'P', ms=7, mfc='red')
    plt.xlabel('n_estimators for BU')
    plt.ylabel('RMSE')
    plt.title('RandomForest')   
    plt.subplot(133)   
    plt.plot(est_IE, np.array(RMSE_est_IE), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(est_IE[RMSE_min_est_IE], np.array(RMSE_est_IE)[RMSE_min_est_IE], 'P', ms=7, mfc='red')
    plt.xlabel('n_estimators for IE')
    plt.ylabel('RMSE')
    plt.title('RandomForest')
    plt.show() 
    
    #Plot max_depth
    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)   
    plt.plot(dep_CT, np.array(RMSE_dep_CT), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(dep_CT[RMSE_min_dep_CT], np.array(RMSE_dep_CT)[RMSE_min_dep_CT], 'P', ms=7, mfc='red')
    plt.xlabel('max_depth for CT')
    plt.ylabel('RMSE')
    plt.title('RandomForest')    
    plt.subplot(132)   
    plt.plot(dep_BU, np.array(RMSE_dep_BU), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(dep_BU[RMSE_min_dep_BU], np.array(RMSE_dep_BU)[RMSE_min_dep_BU], 'P', ms=7, mfc='red')
    plt.xlabel('max_depth for BU')
    plt.ylabel('RMSE')
    plt.title('RandomForest')   
    plt.subplot(133)   
    plt.plot(dep_IE, np.array(RMSE_dep_IE), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(dep_IE[RMSE_min_dep_IE], np.array(RMSE_dep_IE)[RMSE_min_dep_IE], 'P', ms=7, mfc='red')
    plt.xlabel('max_depth for IE')
    plt.ylabel('RMSE')
    plt.title('RandomForest')
    plt.show() 
    
    #Plot max_features
    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)   
    plt.plot(feat_CT, np.array(RMSE_feat_CT), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(feat_CT[RMSE_min_feat_CT], np.array(RMSE_feat_CT)[RMSE_min_feat_CT], 'P', ms=7, mfc='red')
    plt.xlabel('max_features for CT')
    plt.ylabel('RMSE')
    plt.title('RandomForest')   
    plt.subplot(132)   
    plt.plot(feat_BU, np.array(RMSE_feat_BU), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(feat_BU[RMSE_min_feat_BU], np.array(RMSE_feat_BU)[RMSE_min_feat_BU], 'P', ms=7, mfc='red')
    plt.xlabel('max_features for BU')
    plt.ylabel('RMSE')
    plt.title('RandomForest')    
    plt.subplot(133)   
    plt.plot(feat_IE, np.array(RMSE_feat_IE), '-v', color = 'blue', mfc='blue', ms=1)
    plt.plot(feat_IE[RMSE_min_feat_IE], np.array(RMSE_feat_IE)[RMSE_min_feat_IE], 'P', ms=7, mfc='red')
    plt.xlabel('max_features for IE')
    plt.ylabel('RMSE')
    plt.title('RandomForest')
    plt.show() 
    
    #*********************************************
    #Optimise Random Forest regression   
    print('Hyperparameter optimisation...')
    #print('Grid optimization for CT...')
    bestParamsCT, bestResultsCT, bestEstimatorCT = Grid_Search_CV_RFR(Xtrainf, YCTtrain, Xtestf, YCTtest)   
    bestParamsBU, bestResultsBU, bestEstimatorBU = Grid_Search_CV_RFR(Xtrainf, YBUtrain, Xtestf, YBUtest)  
    bestParamsIE, bestResultsIE, bestEstimatorIE = Grid_Search_CV_RFR(Xtrainf, YIEtrain, Xtestf, YIEtest) 
    #print ('bestParam for CTs=', bestParamsCT)
    #print ('bestParam for BUs=', bestParamsBU)
    #print ('bestParam for IEs=', bestParamsIE)
    #print('bestResult for CT=', bestResults)  

    df = subfueldata[selectionX]
    i_CT=bestEstimatorCT.feature_importances_  #Numerical values of feature importances starting with the most important one
    i_BU=bestEstimatorBU.feature_importances_  #The name for each feature is the same as in the order in "features"
    i_IE=bestEstimatorIE.feature_importances_
    pos = np.arange(len(i_CT))
    print('pos=', pos)
    print(i_CT)
    print(features)

    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.bar(pos, i_CT)
    plt.xticks(pos, features, rotation=45)
    plt.title('Feature importance CT')
    plt.subplot(132)
    plt.bar(pos, i_BU)
    plt.xticks(pos, features, rotation=45)
    plt.title('Feature importance BU')
    plt.subplot(133)
    plt.bar(pos, i_IE)
    plt.xticks(pos, features, rotation=45)
    plt.title('Feature importance IE')
    plt.show()
    
    #*********************************************
    #Do random forest regression using optimized hyperparameters from Grid search
    yCT_predicted, rCT = RndForReg(Xtrainf, YCTtrain, Xtestf, YCTtest, bestParamsCT, features)
    yBU_predicted, rBU = RndForReg(Xtrainf, YBUtrain, Xtestf, YBUtest, bestParamsBU, features)
    yIE_predicted, rIE = RndForReg(Xtrainf, YIEtrain, Xtestf, YIEtest, bestParamsIE, features)
    print('Test CT RMSE RndFor:', np.sqrt(metrics.mean_squared_error(YCTtest, yCT_predicted)), "days")      
    print('Test BU RMSE RndFor:', np.sqrt(metrics.mean_squared_error(YBUtest, yBU_predicted)), "MWd/kgU")       
    print('Test IE RMSE RndFor:', np.sqrt(metrics.mean_squared_error(YIEtest, yIE_predicted)), "%")   

    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.scatter(yCT_predicted/365, YCTtest/365)
    plt.xlabel('Predicted CT')
    plt.ylabel('True CT')
    plt.title('True vs predicted CT RndFor')

    plt.subplot(132)
    plt.scatter(yBU_predicted, YBUtest)
    plt.xlabel('Predicted BU')
    plt.ylabel('True BU')
    plt.title('True vs predicted BU RndFor')

    plt.subplot(133)
    plt.scatter(yIE_predicted, YIEtest)
    plt.xlabel('Predicted IE')
    plt.ylabel('True IE')
    plt.title('True vs predicted IE RndFor')
    plt.show()

    #Do PLS regression. 
    #yCT_predicted_pls = PLSReg(Xtrainf, YCTtrain, Xtestf, YCTtest)
    #yBU_predicted_pls = PLSReg(Xtrainf, YBUtrain, Xtestf, YBUtest)
    #yIE_predicted_pls = PLSReg(Xtrainf, YIEtrain, Xtestf, YIEtest)
    #print('Test CT RMSE PLS:', np.sqrt(metrics.mean_squared_error(YCTtest, yCT_predicted_pls)), "days")   
    #print('Test BU RMSE PLS:', np.sqrt(metrics.mean_squared_error(YBUtest, yBU_predicted_pls)), "MWd/kgU")   
    #print('Test IE RMSE PLS:', np.sqrt(metrics.mean_squared_error(YIEtest, yIE_predicted_pls)), "%")  
  
    #fig = plt.figure(figsize=(10,3))
    #plt.subplot(131)
    #plt.scatter(yCT_predicted_pls/365, YCTtest/365, c='g')
    #plt.xlabel('Predicted CT')
    #plt.ylabel('True CT')
    #plt.title('True vs predicted CT RndFor')
    
    #plt.subplot(132)
    #plt.scatter(yBU_predicted_pls, YBUtest, c='g')
    #plt.xlabel('Predicted BU')
    #plt.ylabel('True BU')
    #plt.title('True vs predicted BU RndFor')

    #plt.subplot(133)
    #plt.scatter(yIE_predicted_pls, YIEtest, c='g')
    #plt.xlabel('Predicted IE')
    #plt.ylabel('True IE')
    #plt.title('True vs predicted IE RndFor')
    #plt.show()

    #score = r2_score(Y_test, y_pred)
    
    #return rBU
    return subfueldata
   
    
if __name__ == '__main__':
    main()
    

