# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 07:56:45 2019
@author: soohl254

This is the main program for the analysis of gamma, neutrona and cherenkov data 
using machine learning algorithms.
"""

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import decomposition
from mpl_toolkits.mplot3d import axes3d

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import random
import math
   
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
    
def pca_with_scoresPlot(score,coeff,dataframe, feat, labels=None):
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

def rndForReg():
    rfrbu = RandomForestRegressor(n_estimators=10,max_depth=16,random_state=0,min_samples_leaf=5)
    rfrct = RandomForestRegressor(n_estimators=10,max_depth=16,random_state=0,min_samples_leaf=5)
    rfrie = RandomForestRegressor(n_estimators=10,max_depth=16,random_state=0,min_samples_leaf=5)
    
    rfrbu.fit(Xtrainf, YBUtrain)
    rfrct.fit(Xtrainf, YCTtrain)
    rfrie.fit(Xtrainf, YIEtrain)
    
    YBUpred = rfrbu.predict(Xtest)
    YBUpredtrain = rfrbu.predict(Xtrainf)
    
    YCTpred = rfrct.predict(Xtest)
    YCTpredtrain = rfrct.predict(Xtrainf)
    
    YIEpred = rfrie.predict(Xtest)
    YIEpredtrain = rfrie.predict(Xtrainf)
    
    plt.figure()
    plt.plot(YBUtest,YBUpred,'o',YBUtrain,YBUpredtrain,'ro')
    print('BU test and train mse')
    print(np.sqrt(mean_squared_error(YBUtest,YBUpred)))
    print(np.sqrt(mean_squared_error(YBUtrain,YBUpredtrain)))
    
    plt.figure()
    plt.plot(YCTtest,YCTpred,'o',YCTtrain,YCTpredtrain,'ro')
    print('CT test and train mse')
    print(np.sqrt(mean_squared_error(YCTtest,YCTpred)))
    print(np.sqrt(mean_squared_error(YCTtrain,YCTpredtrain)))
    
    plt.figure()
    plt.plot(YIEtest,YIEpred,'o',YIEtrain,YIEpredtrain,'ro')
    print('IE test and train mse')
    print(np.sqrt(mean_squared_error(YIEtest,YIEpred)))
    print(np.sqrt(mean_squared_error(YIEtrain,YIEpredtrain)))


def main():
    #Read in modelled PWR 17x17 data and put in a DataFrame
    fueldata = pd.read_csv('dataforMVA_strategicPWR_fin_New_Erik_v2.csv', header = 0, index_col = 0)
    fueldata = fueldata.drop(columns = ['cherenkovnobeta'])
    print("Hello!")
    
    #Split the full data set for furhter analysis. 
    fueldata_shortCT = fueldata[fueldata['CT']<=20*365].sample(n= 1000, random_state=1)
    fueldata_longCT  = fueldata[fueldata['CT']>20*365].sample(n= 1000, random_state=1)
    
    #*******Explore the full dataset using PCA*******  
    features = ('Cs137','Sr90','tau','cherenkov')
    
    #Mean-center the data: Subtract the mean and then divide the result by the standard deviation
    #Reduce dimensionality of the data using decomposition into n_components
    X = fueldata_shortCT.loc[:,features].values
    X_std = StandardScaler().fit_transform(X)
    pca = decomposition.PCA(n_components = 4)
    pca.fit(X_std)
    Xt = pca.transform(X_std)
    print("Explained variance ratio for the 4 components is ", pca.explained_variance_ratio_) 

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    #Do a scatter plot pf PC1-3 and color it with CT
    ax.scatter3D(Xt[:,0], Xt[:,1], Xt[:,2], c=fueldata_shortCT.CT, cmap='Greens')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA on fueldata_shortCT using Cs137, Sr90, tau and cherenkov')

    #Visualize the PC-scores using pca_with_scoresPlot
    #pca.components_ is the set of all eigenvectors (or loadings) for your projection space (one eigenvector for each PC).
    #pca.components_ outputs an array of [n_components, n_features]
    #Scores plot with two PCs
    pca_with_scoresPlot(Xt[:,0:2], np.transpose(pca.components_[0:2, :]), fueldata_shortCT, features)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()
    
    #*******Start of machine learning to predict IE, BU and CT values.*******
    #Use the gamma activities calculated from the atomic densities given in Serpent in units of 1e24/cm3.
    calculate_activity(fueldata_shortCT)
       
    #Split the data into a training DataFrame and test DataFrame. Training size is 80% of original data.
    traindf, testdf = train_test_split(fueldata_shortCT, random_state=0, test_size=0.2)
    
    #Define features to use in analysis. All describes all features, X describes selected features, Iso describes gamma isotopes.
    selectionAll=['Cs137A','Cs134A','Eu154A','Zr95A','Nb95A','Ru106A','Ce141A','Ce144A','Sr90A','tau','cherenkov']
    selectionAllIso=['Cs137A','Cs134A','Eu154A','Zr95A','Nb95A','Ru106A','Ce141A','Ce144A','Sr90A']
    selectionX  =['Cs137A','Cs134A','Eu154A','tau','cherenkov']
    selectionIso=['Cs137A','Cs134A','Eu154A']   
    
    norm_traindf = normalize_activities(traindf,selectionIso)
    norm_testdf = normalize_activities(testdf,selectionIso)
    
    #Create training and test data with gamma isotopes, neutron signature and cherenkov light.
    #StandardScaler standardizes features by removing the mean and scaling to unit variance for each column in the data set.
    ss = StandardScaler()
    Xtrain=norm_traindf.loc[:,selectionX]
    Xtrainf=ss.fit_transform(Xtrain)
    
    Xtest=norm_testdf.loc[:,selectionX]
    Xtest=ss.transform(Xtest)

    #Create training and test data to predict IE, BU and CT
    YIEtrain=norm_traindf.loc[:,['IE']]
    YBUtrain=norm_traindf.loc[:,['BU']]
    YCTtrain=norm_traindf.loc[:,['CT']]
    YIEtest=norm_testdf.loc[:,['IE']]
    YBUtest=norm_testdf.loc[:,['BU']]
    YCTtest=norm_testdf.loc[:,['CT']]


    #Do random forest regression...
    #Use rndForReg



if __name__ == '__main__':
    main()
    

