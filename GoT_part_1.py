#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:27:37 2019

@author: arthurmendes
"""


# Loading new libraries
from sklearn.ensemble import RandomForestClassifier

# Loading other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



########################
# Data Preparation
########################
GoT_df = pd.read_excel('GOT_character_predictions.xlsx')

########################
# Fundamental Dataset Exploration
########################

# Column names
GoT_df.columns


# Displaying the first rows of the DataFrame
print(GoT_df.head())


# Dimensions of the DataFrame
GoT_df.shape


# Information about each variable
GoT_df.info()


# Descriptive statistics
GoT_df.describe().round(2)



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
                                'isAlive']]




for col in GoT_Chosen:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if GoT_Chosen[col].isnull().any():
        GoT_Chosen['m_'+col] = GoT_df[col].isnull().astype(int)
        
        
        



"""
########################
# Working with Categorical Variables
########################

# One-Hot Encoding Qualitative Variables
house_dummies = pd.get_dummies(list(GoT_Chosen['house']), drop_first = True)


# Concatenating One-Hot Encoded Values with the Larger DataFrame
GoT_df_2 = pd.concat(
        [GoT_Chosen.loc[:,:],
         house_dummies],
         axis = 1)
"""


# Checking again
print(
      GoT_Chosen
      .isnull()
      .sum()
      )



GoT_data   = GoT_df.loc[:,['S.No',
                                'male',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book3_A_Storm_Of_Swords',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isMarried',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'm_isAliveMother',
                                'm_isAliveFather',
                                'm_isAliveHeir',
                                'm_isAliveSpouse']]


GoT_target   = GoT_df.loc[:,['isAlive']]


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
                                     min_samples_leaf = 15,
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

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1000, 100)
leaf_space = pd.np.arange(1, 200, 10)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



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



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

'''
Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion': 'entropy',
 'min_samples_leaf': 11, 'n_estimators': 100, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.793
'''

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))




########################
# Parameter tuning with GridSearchCV
########################

# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 200, 10)
leaf_space = pd.np.arange(1, 50, 3)
criterion_space = ['entropy']
bootstrap_space = [True]
warm_start_space = [True]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_hyped = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_hyped = GridSearchCV(full_forest_grid, param_grid, cv = 3)



# Fit it to the training data
full_hyped.fit(X_train, y_train)

'''
Tuned Logistic Regression Parameter: {'bootstrap': True, 'criterion': 'entropy',
 'min_samples_leaf': 4, 'n_estimators': 130, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.7971
'''


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_hyped.best_params_)
print("Tuned Logistic Regression Accuracy:", full_hyped.best_score_.round(4))
