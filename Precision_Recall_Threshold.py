#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:49:03 2019

@author: arthurmendes
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

########################
# Fundamental Dataset Exploration
########################
GoT_df = pd.read_excel('GOT_character_predictions.xlsx')

GoT_df['out_house'] = 0

GoT_df['out_culture'] = 0

# Cultures with high probabylity of dying (Sum of isAlive/Count isAlive > 0.5)

cultures = ['Vale',
            'Meereen',
            'Lhazareen',
            'Astapor',
            'Valyrian',
            'Astapori',
            'Westerman',
            'Pentoshi']



for val in enumerate(GoT_df.loc[ : , 'culture']):
    '''
    Creating a flag if the culture appear in the list of cultures, which means
    that they have a high chance of dying.
    '''
    if val[1] in cultures:
        GoT_df.loc[val[0], 'out_culture'] = 1
        
        

# Houses with high probabylity of dying (Sum of isAlive/Count isAlive > 0.5)

house = ['Thenn',
         'Khal',
         'House Toyne',
         'House Rosby',
         'House Roote',
         'House Rambton',
         'House Pemford',
         'House Moore',
         'House Mallery',
         'House Grandison',
         'House Farrow',
         'House Egen',
         'House Dustin',
         'House Cole',
         'House Cockshaw',
         'House Chelsted',
         'House Cerwyn',
         'House Cafferen',
         'House Bywater',
         'House Byrch',
         'House Bushy',
         'House Ball',
         'Good Masters',
         'Band of Nine',
         'House Blackwood',
         'House Blackfyre',
         'House Strong',
         'House Cassel',
         'House Targaryen',
         'Stormcrows',
         'House Darry',
         'Brave Companions',
         'House Velaryon',
         'House Tully',
         'House Clegane']



for val in enumerate(GoT_df.loc[ : , 'house']):
    '''
    Creating a flag if the house appear in the list of houses, which means
    that they have a high probability of dying.
    '''
    if val[1] in house:
        GoT_df.loc[val[0], 'out_house'] = 1
        
        
        
        
# Creating flags for titles
        
GoT_df['title'] = GoT_df['title'].fillna('Unknown')     
          
GoT_df['out_title'] = 0  


for val in enumerate(GoT_df.loc[ : , 'title']):
    '''
    Lords seem to die a lot
    '''
    if val[1].startswith('Lord') == True:
        GoT_df.loc[val[0], 'out_title'] = 1
        



for val in enumerate(GoT_df.loc[ : , 'title']):
    '''
    Prince and princess seem to die a lot
    '''
    if 'Prince' in val[1]:
        GoT_df.loc[val[0], 'out_title'] = 1



         
###############################################################################
# Data Preparation
###############################################################################

print(
      GoT_df
      .isnull()
      .sum()
      )

GoT_Chosen   = GoT_df.loc[:,['S.No',
                                'male',
                                'dateOfBirth',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book3_A_Storm_Of_Swords',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isAliveMother',
                                'isAliveFather',
                                'isAliveHeir',
                                'isAliveSpouse',
                                'isMarried',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'isAlive',
                                'out_house',
                                'out_culture',
                                'out_title']]


# Flagging missing values

for col in GoT_Chosen:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if GoT_Chosen[col].isnull().any():
        GoT_Chosen['m_'+col] = GoT_df[col].isnull().astype(int)
        
        


# Date of birth being filled with the median
fill = GoT_Chosen['dateOfBirth'].median()

GoT_Chosen['dateOfBirth'] = GoT_Chosen['dateOfBirth'].fillna(fill)


# Outlier flags
    
low_dateOfBirth = 200


GoT_Chosen['out_dateOfBirth'] = 0


for val in enumerate(GoT_Chosen.loc[ : , 'dateOfBirth']):
    
    if val[1] <= low_dateOfBirth:
        GoT_Chosen.loc[val[0], 'out_dateOfBirth'] = 1



# Checking again
print(
      GoT_Chosen
      .isnull()
      .sum()
      )



GoT_data   = GoT_Chosen.loc[:,[ 'male',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book3_A_Storm_Of_Swords',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isMarried',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'm_isAliveSpouse',
                                'out_house',
                                'out_culture',
                                'out_title',
                                'out_dateOfBirth']]


GoT_target   = GoT_Chosen.loc[:,['isAlive']]


X_train, X_test, y_train, y_test = train_test_split(
            GoT_data,
            GoT_target.values.ravel(),
            test_size = 0.25,
            random_state = 508,
            stratify = GoT_target)


###############################################################################
# Random Forest in scikit-learn
###############################################################################

# Following the same procedure as other scikit-learn modeling techniques

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 11,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))



########################
# Parameter tuning with GridSearchCV
########################

# Creating a hyperparameter grid
estimator_space = pd.np.arange(50, 500, 50)
leaf_space = pd.np.arange(1, 200, 10)
criterion_space = ['gini', 'entropy']
bootstrap_space = [False]
warm_start_space = [True]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)

'''
Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion': 'gini', 
'min_samples_leaf': 21, 'n_estimators': 200, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8019

Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion': 'entropy',
'min_samples_leaf': 11, 'n_estimators': 350, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8129
'''

# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))



# Using the optimal Random Forrest 

# Full forest using entropy
optimal_forrest = RandomForestClassifier(n_estimators = 350,
                                         criterion = 'entropy',
                                         max_depth = None,
                                         min_samples_leaf = 11,
                                         bootstrap = False,
                                         warm_start = True,
                                         random_state = 508)


# Fitting the models
optimal_forrest_fit = optimal_forrest.fit(X_train, y_train)

full_forest_predict = optimal_forrest.predict(X_test)

y_scores = optimal_forrest.predict_proba(X_test)[:, 1]

# Scoring the gini model
print('Training Score', optimal_forrest_fit.score(X_train, y_train).round(4))
print('Testing Score:', optimal_forrest_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = optimal_forrest_fit.score(X_train, y_train)
gini_full_test  = optimal_forrest_fit.score(X_test, y_test)


# Generate the precision-recall curve for the classifier:
p, r, thresholds = precision_recall_curve(y_test, y_scores)


################################

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['Predicted Dead', 'Predicted ALive'], 
                       index=['Dead', 'Alive']))
    
    print ('\nClassification report:\n', classification_report(y_test, y_pred_adj))
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)


precision_recall_threshold(p, r, thresholds, 0.47)



###############################################################################
# ROC curve for Random Forest
###############################################################################


# Compute predicted probabilities: y_pred_prob
y_pred_prob = optimal_forrest_fit.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
#plt.savefig('ROC Random Forest')
plt.show()



metrics.auc(fpr, tpr)


###############################################################################
# Variable importance
###############################################################################


import pandas as pd
feature_importances = pd.DataFrame(optimal_forrest_fit.feature_importances_,
                                   index = X_train.columns,
                                   columns=
                                   ['importance']).sort_values('importance',   
                                   ascending=False)

print(feature_importances)



print ('\nClassification report:\n', classification_report(y_test, full_forest_predict))
print ('\nConfusion matrix:\n',confusion_matrix(y_test, full_forest_predict))





###############################################################################
# Gradient Boosted Machines
###############################################################################


GoT_data   = GoT_Chosen.loc[:,[ 'male',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book3_A_Storm_Of_Swords',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isMarried',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'out_house',
                                'out_culture',
                                'out_dateOfBirth']]


GoT_target   = GoT_Chosen.loc[:,['isAlive']]


X_train, X_test, y_train, y_test = train_test_split(
            GoT_data,
            GoT_target.values.ravel(),
            test_size = 0.25,
            random_state = 508)



from sklearn.ensemble import GradientBoostingClassifier

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.3,
                                  n_estimators = 80,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = True,
                                  random_state = 508,)


gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)


########################
# Applying GridSearchCV
########################

# Creating a hyperparameter grid'

learn_space = pd.np.arange(0.1, 2.5, 0.1)
estimator_space = pd.np.arange(50, 250, 10)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}



# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)



# Creating a GridSearchCV object
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv = 3)


# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)

gbm_grid_pred = gbm_grid_cv.predict(X_test)

'''
Tuned GBM Parameter: {'criterion': 'friedman_mse', 'learning_rate': 0.1, 
'max_depth': 2, 'n_estimators': 50}
Tuned GBM Accuracy: 0.8136

Tuned GBM Parameter: {'criterion': 'friedman_mse', 'learning_rate': 0.1, 
'max_depth': 3, 'n_estimators': 220}
Tuned GBM Accuracy: 0.8252

Tuned GBM Parameter: {'criterion': 'friedman_mse', 'learning_rate': 1.1, 
'max_depth': 2, 'n_estimators': 140}
'''

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))


# Building a optimal GBM

gbm_optimal = GradientBoostingClassifier(loss = 'deviance',
                                         learning_rate = 0.1,
                                         n_estimators = 220,
                                         max_depth = 3,
                                         criterion = 'friedman_mse',
                                         warm_start = True,
                                         random_state = 508,)


gbm_optimal_fit = gbm_optimal.fit(X_train, y_train)


gbm_optimal_predict = gbm_optimal_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_optimal_fit.score(X_train, y_train)
gmb_basic_test  = gbm_optimal_fit.score(X_test, y_test)

y_scores = gbm_optimal_fit.predict_proba(X_test)[:,1]

# Precision Recall


precision_recall_threshold(p, r, thresholds, 0.55)


###############################################################################
# ROC curve for GBM
###############################################################################

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = gbm_optimal_fit.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
#plt.savefig('ROC for GBM')
plt.show()

metrics.auc(fpr, tpr)


###############################################################################
# Classification Report and Confussion Matrix
###############################################################################


print ('\nClassification report:\n', classification_report(y_test, gbm_optimal_predict))
print ('\nConfusion matrix:\n',confusion_matrix(y_test, gbm_optimal_predict))

