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
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import decomposition
from mpl_toolkits.mplot3d import axes3d

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

from matplotlib.legend_handler import HandlerLine2D
    

import random
import math

from sklearn.tree import export_graphviz
#import pydot
   
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
    #Plots the PC-score (or value) plot from a PCA analysis for all data points. 
    #This plot also slows the contribution by the diffrent features to the PCs
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

def applynoise(x):
    #Generates noise from a random Gaussian distribution with mean at 1 and a sigma of 0.05.
    return x*random.normalvariate(1,0.05)



def RndForReg(X_train, y_train, X_test, Y_test, par):
    plot_components=False
     
    bootstrap=par['bootstrap']
    max_depth=par['max_depth']
    max_features=par['max_features']
    min_samples_leaf=par['min_samples_leaf']
    min_samples_split=par['min_samples_split']
    n_estimators=par['n_estimators']
   
    
    mse_rF = []   
    regressor = RandomForestRegressor(bootstrap=True, max_depth=25, max_features=5, min_samples_leaf=3, min_samples_split=6, n_estimators= 30)
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
        
    #regressor = RandomForestRegressor(n_estimators=10, max_depth=16,random_state=0,min_samples_leaf=5)  
    #regressor.fit(X_train, y_train.values.ravel())  
    #y_pred = regressor.predict(X_test) 
    #print(metrics.accuracy_score(Y_test, y_pred)) #Is there something similar for continous values?
    
    #VISUALIZATION
    # Pull out one tree from the forest
    #tree = regressor.estimators_[5]

    # Export the image to a dot file
    #export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

    # Use dot file to create a graph
    #(graph, ) = pydot.graph_from_dot_file('tree.dot')

    # Write graph to a png file
    #graph.write_png('tree.png')

    return y_pred, regressor#, tree.png

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


def Grid_Search_CV_RFR(X_train, y_train, X_test, y_test, sel_features):
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
  

#    feats = {} # a dict to hold feature_name: feature_importance
#    for feature, importance in zip(sel_features, best_estimator.feature_importances_):
#        feats[feature] = importance #add the name/value pair 
    
    return best_parameters, best_result, best_estimator


def under_or_overfitting(X_train, y_train, X_test, y_test):

    rmse_estimator=[]
    estimator = np.arange(5, 50)
    for i in estimator:
        regressor_estimator = RandomForestRegressor(n_estimators=i, max_depth=10, max_features=5, min_samples_leaf=2, min_samples_split=3)
        regressor_estimator.fit(X_train, y_train.flatten())
        y_pred_estimator = regressor_estimator.predict(X_test)
     
        rmse_p_estimator = np.sqrt(metrics.mean_squared_error(y_test, y_pred_estimator))
        rmse_estimator.append(rmse_p_estimator)
    rmse_min_estimator = np.argmin(rmse_estimator) 
    

    rmse_depth=[]
    depth = np.arange(1, 20)
    for i in depth:
        regressor_depth = RandomForestRegressor(n_estimators=10, max_depth=i, max_features=5, min_samples_leaf=2, min_samples_split=3)
        regressor_depth.fit(X_train, y_train.flatten())
        y_pred_depth = regressor_depth.predict(X_test)
     
        rmse_p_depth = np.sqrt(metrics.mean_squared_error(y_test, y_pred_depth))
        rmse_depth.append(rmse_p_depth)
    rmse_min_depth = np.argmin(rmse_depth) 
    
    rmse_feat=[]
    feat = np.arange(1, 6)
    for i in feat:
        regressor_feat = RandomForestRegressor(n_estimators=10, max_depth=10, max_features=i, min_samples_leaf=2, min_samples_split=3)
        regressor_feat.fit(X_train, y_train.flatten())
        y_pred_feat = regressor_feat.predict(X_test)
        #feat_imp = y_pred_feat.feature_importances_
     
        rmse_p_feat = np.sqrt(metrics.mean_squared_error(y_test, y_pred_feat))
        rmse_feat.append(rmse_p_feat)
    rmse_min_feat = np.argmin(rmse_feat) 

    return rmse_min_estimator, rmse_estimator, estimator, rmse_min_depth, rmse_depth, depth, rmse_min_feat, rmse_feat, feat



def calc_learning_curves(estimator, X_train, y_train, X_test, y_test):
    #A learning curve shows the validation and training score of an estimator for varying numbers 
    #of training samples. It is a tool to find out how much we benefit from adding more training 
    #data and whether the estimator suffers more from a variance error or a bias error.
    
    train_score = []
    test_score = []
    training_set_sizes = np.linspace(5, len(X_train), 20, dtype='int')

    for i in training_set_sizes:
        # fit the model only using limited training examples
        estimator.fit(X_train[0:i, :], y_train[0:i].flatten())
        train_rmse = np.sqrt(metrics.mean_squared_error(y_train[0:i, :], estimator.predict(X_train[0:i, :])))
        test_rmse = np.sqrt (metrics.mean_squared_error(y_test, estimator.predict(X_test)))

        train_score.append(train_rmse)
        test_score.append(test_rmse)
    
    return training_set_sizes, train_score, test_score

  
def main():
    #Read in modelled PWR 17x17 data and put in a DataFrame
    fueldata = pd.read_csv('dataforMVA_strategicPWR_fin_New_Erik_v2.csv', header = 0, index_col = 0)
    fueldata = fueldata.drop(columns = ['cherenkovnobeta'])
    print("Hello!")
    
    #Split the full data set for furhter analysis. 
    subfueldata = fueldata[fueldata['CT']>0*365].sample(n= 1000, random_state=1)
    #subdata = subfueldata.loc[:,sel_cols]

    
    #*******Explore the dataset using PCA*******  
    #features = ('Cs137','Sr90','tau','cherenkov')
    features  = ('Cs137','Cs134','Eu154','tau','cherenkov')
    
    #Mean-center the data: Subtract the mean and then divide the result by the standard deviation
    #Reduce dimensionality of the data using decomposition into n_components
    X = subfueldata.loc[:,features].values
    X_std = StandardScaler().fit_transform(X)
    pca = decomposition.PCA(n_components = 5)
    pca.fit(X_std)
    Xt = pca.transform(X_std)
    print("Explained variance ratio for the 5 components using PCA is:", pca.explained_variance_ratio_) 

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    #Do a scatter plot pf PC1-3 and color it with CT
    ax.scatter3D(Xt[:,0], Xt[:,1], Xt[:,2], c=subfueldata.CT, cmap='Greens')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    #ax.set_title('PCA using Cs137, Sr90, tau and cherenkov')
    ax.set_title('PCA using Cs137, Cs134, Eu154, tau and cherenkov')

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
    #Define features to use in analysis. All describes all features, X describes selected features, Iso describes gamma isotopes.
    selectionAll=['Cs137A','Cs134A','Eu154A','Zr95A','Nb95A','Ru106A','Ce141A','Ce144A','Sr90A','tau','cherenkov']
    selectionAllIso=['Cs137A','Cs134A','Eu154A','Zr95A','Nb95A','Ru106A','Ce141A','Ce144A','Sr90A']
    selectionIso=['Cs137A','Cs134A','Eu154A'] 
    selectionX  =['Cs137A','Cs134A','Eu154A','tau','cherenkov']
    sel_cols= ('BU','CT','IE','Cs137A','Cs134A','Eu154A','tau','cherenkov')
    
    #Use the gamma activities calculated from the atomic densities given in Serpent in units of 1e24/cm3.
    calculate_activity(subfueldata)
    subdata = subfueldata.copy()
    subdata = subdata.drop(columns=['fuelType','reactorType','Ce141','Nb95','Y91','Zr95','Ce144','Ru106','Sr90', 'Ce141A','Nb95A','Y91A','Zr95A','Ce144A','Ru106A','Sr90A'])

#    for row in subfueldata.itertuples(index = True):   
#        subfueldata['TotGamAct'] = subfueldata.apply(lambda row: row.Y91A+row.Zr95A+row.Nb95A+row.Ru106A+row.Cs134A+row.Cs137A+row.Eu154A+row.Ce141A+row.Ce144A+row.Sr90A, axis=1)
#    #Set minimum activity level for gamma emitting isotopes to 0.1% of total activity
#    gamma_emitting_isotopes = ('Y91A','Zr95A','Nb95A','Ru106A','Cs134A','Cs137A','Eu154A','Ce141A','Ce144A','Sr90A')
#    for i in gamma_emitting_isotopes:
#        for row in subdata.itertuples(index = True): 
#            #if getattr(row, i)/getattr(row, 'TotGamAct')<0.001:
#            num = subdata._get_numeric_data()
#            num[num < 1.0e-20] = 0.0

    
    #Split the data into a training DataFrame and test DataFrame. Training size is 80% of original data.
    traindf, testdf = train_test_split(subdata, random_state=0, test_size=0.2)
    norm_traindf = normalize_activities(traindf,selectionIso)
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
    plt.legend(['Training set', 'Validation set'], fontsize=12)
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('Learning Curve CT')
    
    plt.subplot(132)
    plt.semilogy(setSizeBU, trainScoreBU, c='gold')
    plt.semilogy(setSizeBU, testScoreBU, c='steelblue')
    plt.legend(['Training set', 'Validation set'])#, fontsize=12)
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('Learning Curve BU')
    
    plt.subplot(133)
    plt.semilogy(setSizeIE, trainScoreIE, c='gold')
    plt.semilogy(setSizeIE, testScoreIE, c='steelblue')
    plt.legend(['Training set', 'Validation set'], fontsize=12)
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('Learning Curve IE')
    plt.show()

    #*********************************************   
    #Investigare under- and overfitting
    print('Study overfitting...')
    RMSE_min_est_CT, RMSE_est_CT, est_CT, RMSE_min_dep_CT, RMSE_dep_CT, dep_CT, RMSE_min_feat_CT, RMSE_feat_CT, feat_CT  = under_or_overfitting(Xtrainf, YCTtrain, Xtestf, YCTtest)
    RMSE_min_est_BU, RMSE_est_BU, est_BU, RMSE_min_dep_BU, RMSE_dep_BU, dep_BU, RMSE_min_feat_BU, RMSE_feat_BU, feat_BU  = under_or_overfitting(Xtrainf, YBUtrain, Xtestf, YBUtest)
    RMSE_min_est_IE, RMSE_est_IE, est_IE, RMSE_min_dep_IE, RMSE_dep_IE, dep_IE, RMSE_min_feat_IE, RMSE_feat_IE, feat_IE  = under_or_overfitting(Xtrainf, YIEtrain, Xtestf, YIEtest)
    
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
    plt.plot(dep_CT[RMSE_min_dep_CT], np.array(RMSE_dep_CT)[RMSE_min_dep_CT], 'P', ms=5, mfc='red')
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
    plt.plot(feat_CT[RMSE_min_feat_CT], np.array(RMSE_feat_CT)[RMSE_min_feat_CT], 'P', ms=5, mfc='red')
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
    bestParamsCT, bestResultsCT, bestEstimatorCT = Grid_Search_CV_RFR(Xtrainf, YCTtrain, Xtestf, YCTtest, selectionX)   
    bestParamsBU, bestResultsBU, bestEstimatorBU = Grid_Search_CV_RFR(Xtrainf, YBUtrain, Xtestf, YBUtest, selectionX)  
    bestParamsIE, bestResultsIE, bestEstimatorIE = Grid_Search_CV_RFR(Xtrainf, YIEtrain, Xtestf, YIEtest, selectionX)  
    print ('bestParam for CTs=', bestParamsCT)
    print ('bestParam for BUs=', bestParamsBU)
    print ('bestParam for IEs=', bestParamsIE)
    #print('bestResult for CT=', bestResults)  

    feature_name=('Cs137A','Cs134A','Eu154A','tau','cherenkov') 
    df = subdata[['Cs137A','Cs134A','Eu154A','tau','cherenkov']]
    f_CT=df.columns
    i_CT=bestEstimatorCT.feature_importances_
    i_BU=bestEstimatorBU.feature_importances_
    i_IE=bestEstimatorCT.feature_importances_
    pos = np.arange(len(i_CT))

    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.bar(pos, i_CT)
    plt.xticks(pos, feature_name, rotation=45)
    plt.title('Feature importance CT')
    plt.subplot(132)
    plt.bar(pos, i_BU)
    plt.xticks(pos, feature_name, rotation=45)
    plt.title('Feature importance BU')
    plt.subplot(133)
    plt.bar(pos, i_IE)
    plt.xticks(pos, feature_name, rotation=45)
    plt.title('Feature importance IE')
    
    #*********************************************
    #Do random forest regression using optimized hyperparameters from Grid search
    #In order to use best parameters, import these manuallt to the method "RndForReg"
    yCT_predicted, rCT = RndForReg(Xtrainf, YCTtrain, Xtestf, YCTtest, bestParamsCT)
    yBU_predicted, rBU = RndForReg(Xtrainf, YBUtrain, Xtestf, YBUtest, bestParamsBU)
    yIE_predicted, rIE = RndForReg(Xtrainf, YIEtrain, Xtestf, YIEtest, bestParamsIE)
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
    

