#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:34:39 2020
#
@author: pczaf
"""
import numpy as np
import pandas as pd
#
# Import rdkit stuff
#
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
#
# Import the machine learning tools 
#
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn import metrics
from sklearn.metrics import explained_variance_score
#
#Import matplot for our figures
#
import matplotlib.pyplot as plt
#
# Load the dataframe from a .csv file, stores the smiles strings for our compound and their PIC50a ctivities 
#
training_data = pd.read_csv('Data_new.csv')
training_smiles = np.array(training_data['Parent_SMILES'])
training_activity = np.array(training_data['Value'])

#
test_data = pd.read_csv('liter.csv')
test_smiles = np.array(test_data['Parent_SMILES'])
test_activity = np.array(test_data['PIC50'])

smiles = np.hstack((training_smiles, test_smiles))

activity = np.hstack((training_activity, test_activity))

#
# Generate Morgan`s circular fingerprint. This family of fingerprints, 
#better known as circular fingerprints , is built by applying 
#the Morgan algorithm to a set of user-supplied atom invariants. When 
#generating Morgan fingerprints, the radius of the fingerprint can be changed to increase precision :
#
def generate_FP_matrix(smiles):
    morgan_matrix = np.zeros((1,1024))
    l=len(smiles)
#    
    for i in range(l):
        try:
            compound = Chem.MolFromSmiles(smiles[i])
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(compound, 2, nBits = 1024)
            fp = fp.ToBitString()
            matrix_row = np.array ([int(x) for x in list(fp)])
            morgan_matrix = np.row_stack((morgan_matrix, matrix_row))
#            
            if i%500==0:
                percentage = np.round(0.1*(i/1),1)
                print ('Calculating fingerprint', percentage,  '% done')
#       
        except:
            print ('problem with index', i)
    morgan_matrix = np.delete(morgan_matrix, 0, axis = 0)
#    
    print('\n')
    print('Morgan Matrix Dimension is', morgan_matrix.shape)
#    
    return morgan_matrix
#
morgan_matrix_feature_training = generate_FP_matrix(training_smiles)
feature_training =  np.array(morgan_matrix_feature_training)
target_training = np.array(training_activity)

morgan_matrix_feature_test = generate_FP_matrix(test_smiles)
features_test = np.array(morgan_matrix_feature_test)
targets_test = np.array(test_activity)

morgan_matrix_feature_cross = generate_FP_matrix(smiles)

features_cross = np.array(morgan_matrix_feature_cross)
targets_cross = np.array(activity)

# Define the regressor to be used in our model. Here a Supporting Vectot Machine
#is chosen. Hyperparameters were optimized ina different step througha grid search.
#
regressor=SVR(kernel='rbf', C = 1, gamma='scale', epsilon = 0.2, max_iter=-1)
#
# Spit thre data into training and test set.
#
train_feature=feature_training
test_feature=features_test

train_target=target_training
test_target=targets_test

#
regressor.fit(train_feature,train_target)
prediction=regressor.predict(test_feature)
errors = abs(prediction - test_target)
#
for i in range(len(prediction)):
#    
    print (test_target[i])
#    
print ("stop")
#
for i in range(len(prediction)):
#    
#    print (prediction[i])
#
print('errors are', errors) 

def largest(arr,n): 
    # Initialize maximum element 
    max = arr[0]
    # Traverse array elements from second 
    # and compare every element with  
    # current max 
    for i in range(1, n): 
        if arr[i] > max: 
            max = arr[i] 
    return max
  
# Driver Code 

#---------------DONE----------------------------------------------------------------------------------------------------------
# Plotting a a  graph now of our predictions vs actual values. 
#-----------------------------------------------------------------------------------------------------------------------------
x = np.arange(len(prediction))
#
plt.plot( x, test_target,    marker='', color='blue', linewidth=1.0, linestyle='dashed', label="actual")
plt.plot( x, prediction, marker='', color='red', linewidth=1.0, label='predicted')
#
plt.savefig('predict_morga_SVR.png', dpi=600)
#
plt.legend()
#
plt.show()
#
print('------------------------------------------------------------------------------------------')
print('CALCULATING SIMPLE MODEL')
print('------------------------------------------------------------------------------------------')
#-----------------DONE---------------------------------------------------------------------------------------------------------
# Calculating statitcs now for the one of
#------------------------------------------------------------------------------------------------------------------------------
print ('Calculating Mean Absolute Error...')
print('Mean Absolute Error:', round(np.mean(errors), 2), 'units.',  'Standard error:',  round(np.std(errors), 2) ) 
#
print ('Calculating Largest error...')
#
arr= errors
n = len(arr) 
Ans = largest(arr,n) 
#
print ("Largest prediction error is",Ans) 
#
#-----------------DONE---------------------------------------------------------------------------------------------------------
# Performing the 10-fold cross validation and leav-one-out validation 
#------------------------------------------------------------------------------------------------------------------------------
print('------------------------------------------------------------------------------------------')
print('CALCULATING CROSS-VALIDATION MODEL')
print('------------------------------------------------------------------------------------------')
#------------------------------------------------------------------------------------------------------------------------------
print ('Calculating 10-fold cross validation...')
#
train_features, test_features, train_targets, test_targets = train_test_split(features_cross, targets_cross, test_size = 0.10, random_state = 7)
#
scores = cross_validate(regressor, train_features, train_targets, cv=10, scoring=('r2', 'explained_variance'), return_train_score=True)
#
print('The cross-validated variance is:', scores['test_explained_variance'].mean())
print('The cross-validated sigma variance is:', scores['test_explained_variance'].std() * 2)
#      
print('The cross-validated q2 is:', scores['train_r2'].mean())
print('The cross-validated sigma q2 is:',scores['train_r2'].std() * 2)
#
predictions_cv=cross_val_predict(regressor,train_features, train_targets, cv=10)
#
with open('predictions.csv', 'w') as the_file:
    l=len(predictions_cv)      
    for i in range(l):
        the_file.write(str(predictions_cv[i]) + "," + str(train_targets[i]))
        the_file.write(' \n')
        
errors_cv = abs(predictions_cv - train_targets)
#
print ('Calculating Cross-Validated Mean Absolute Error...')
print('Mean CV-Absolute Error:', round(np.mean(errors_cv), 2), 'units.',  'Standard error:',  round(np.std(errors_cv), 2) ) 
#
arr_cv=errors_cv
n_cv=len(arr_cv)
Ans_cv = largest(arr_cv,n_cv) 

print ("Largest cross-validated prediction error is",Ans_cv) 

#---------------DONE----------------------------------------------------------------------------------------------------------
# Plotting a a  graph now of our cross-validated predictions vs actual values. 
#-----------------------------------------------------------------------------------------------------------------------------
x = np.arange(len(predictions_cv))
#
plt.plot( x, train_targets,    marker='', color='blue', linewidth=1.0, linestyle='dashed', label="actual")
plt.plot( x, predictions_cv, marker='', color='red', linewidth=1.0, label='predicted')
#
plt.savefig('predict.png', dpi=600)
#
plt.legend()
#
plt.show()
#
print('Calculating AUC (Area Under The Curve) of the ROC (Receiver Operating Characteristics...)')   
#---------------DONE----------------------------------------------------------------------------------------------------------
# ROC and AUC are reponses of binary hit based on a threshold (6 in our case), we need to convert out data to ones and zeros.
#-----------------------------------------------------------------------------------------------------------------------------
for i, item in enumerate(train_targets):
   if item > 6:
       train_targets[i] = 1.0
   else:
       train_targets[i] = 0.0       
#
for i, item in enumerate(predictions_cv):
   if item > 6:
       predictions_cv[i] = 1.0
   else:
       predictions_cv[i] = 0.0
#              
fpr, tpr, thresholds = metrics.roc_curve(train_targets, predictions_cv, pos_label=1)
print ('The AUC score is:', metrics.auc(fpr, tpr))
#
#-----------------DONE-------------------------------------------------------------------------------------------------------------
