#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:27:37 2019

@author: arthurmendes

"""


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics


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
                                'isAlive',
                                'out_house',
                                'out_culture']]


# Flagging missing values

for col in GoT_Chosen:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if GoT_Chosen[col].isnull().any():
        GoT_Chosen['m_'+col] = GoT_df[col].isnull().astype(int)
        
        


# Checking again
print(
      GoT_Chosen
      .isnull()
      .sum()
      )



GoT_data   = GoT_Chosen.loc[:,['male',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book3_A_Storm_Of_Swords',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isMarried',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'm_isAliveFather',
                                'm_isAliveHeir',
                                'm_isAliveSpouse',
                                'out_house',
                                'out_culture']]


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


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))

'''
Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion': 'gini', 
'min_samples_leaf': 11, 'n_estimators': 100, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8108
'''

########################
# Parameter tuning with GridSearchCV
########################

# Creating a hyperparameter grid
estimator_space = pd.np.arange(50, 130, 10)
leaf_space = pd.np.arange(1, 50, 3)
criterion_space = ['gini']
bootstrap_space = [False]
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
hyped_model = GridSearchCV(full_hyped, param_grid, cv = 3)



# Fit it to the training data
hyped_model.fit(X_train, y_train)

'''
Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion': 'gini', 
'min_samples_leaf': 4, 'n_estimators': 90, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8122
'''


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", hyped_model.best_params_)
print("Tuned Logistic Regression Accuracy:", hyped_model.best_score_.round(4))




###############################################################################
# Random Forest in scikit-learn
###############################################################################

# Following the same procedure as other scikit-learn modeling techniques

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 90,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = False,
                                     warm_start = True,
                                     random_state = 508)


# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)

full_forest_predict = full_forest_gini.predict(X_test)

# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

###############################################################################
# ROC curve
###############################################################################

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = full_gini_fit.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

metrics.auc(fpr, tpr)


###############################################################################
# Variable importance
###############################################################################


import pandas as pd
feature_importances = pd.DataFrame(full_gini_fit.feature_importances_,
                                   index = X_train.columns,
                                   columns=
                                   ['importance']).sort_values('importance',   
                                   ascending=False)

print(feature_importances)



print ('\nClasification report:\n', classification_report(y_test, full_forest_predict))
print ('\nConfussion matrix:\n',confusion_matrix(y_test, full_forest_predict))


###############################################################################
# Gradient Boosted Machines
###############################################################################

from sklearn.ensemble import GradientBoostingClassifier

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
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

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


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



# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))



