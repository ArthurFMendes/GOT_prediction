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
# Fundamental Dataset Exploration
########################
GoT_df = pd.read_excel('GOT_character_predictions.xlsx')

GoT_df['out_house'] = 0

for house in GoT_df['house']:
    percentage = GoT_df['isAlive'].sum() / GoT_df['isAlive'].count()
    if percentage >= 0.9:
        GoT_df.loc[house, 'out_house'] = 1


'''
        
# Housing outlier
GoT_df.house.nunique()

GoT_df['out_house'] = 0

percent = GoT_df['isAlive'].sum() / GoT_df['isAlive'].count()



for house in Got_df['house']:
    percentage = GoT_df['isAlive'].sum() / GoT_df['isAlive'].count()
        
    
    
housing['out_Lot_Area'] = 0


for val in enumerate(housing.loc[ : , 'Lot Area']):
    
    if val[1] >= lot_area_hi:
        housing.loc[val[0], 'out_Lot_Area'] = 1

'''
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



GoT_data   = GoT_Chosen.loc[:,['S.No',
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
hyped_model = GridSearchCV(full_hyped, param_grid, cv = 3)



# Fit it to the training data
hyped_model.fit(X_train, y_train)

'''
Tuned Logistic Regression Parameter: {'bootstrap': True, 'criterion': 'entropy',
 'min_samples_leaf': 4, 'n_estimators': 130, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.7971
'''


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_hyped.best_params_)
print("Tuned Logistic Regression Accuracy:", full_hyped.best_score_.round(4))



########################
# Best param
########################

# Creating a hyperparameter grid
estimator_space = pd.np.arange(80, 90, 1)
leaf_space = pd.np.arange(1, 5, 1)
criterion_space = ['entropy']
bootstrap_space = [False]
warm_start_space = [True]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
best_pm = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
Best_rf = GridSearchCV(best_pm, param_grid, cv = 3)



# Fit it to the training data
Best_rf.fit(X_train, y_train)

'''
Tuned Logistic Regression Parameter: {'bootstrap': True, 'criterion': 'entropy',
 'min_samples_leaf': 4, 'n_estimators': 130, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.7971
'''


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", Best_rf.best_params_)
print("Tuned Logistic Regression Accuracy:", Best_rf.best_score_.round(4))




########################
# Building Random Forest Model Based on Best Parameters
########################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 3,
                                    n_estimators = 86,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)

###############################################################################
# Gradient Boosted Machines
###############################################################################

"""
Prof. Chase:
    Gradient boosted machines (gbms) are like decision trees, but instead of
    starting fresh with each iteration, they learn from mistakes made in
    previous iterations.
"""

from sklearn.ensemble import GradientBoostingClassifier

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )


"""
Prof. Chase:
    Notice above that we are using friedman_mse as the criterion. Friedman
    proposed that instead of focusing on one MSE value for the entire tree,
    the algoirthm should localize its optimal MSE for each region of the tree.
"""


gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)


"""
Prof. Chase:
    It appears the model is not generalizing well. Let's try to work on that
    using GridSearhCV.
"""


########################
# Creating a confusion matrix
########################

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true = y_test,
                       y_pred = gbm_basic_predict))


# Visualizing a confusion matrix
import seaborn as sns

labels = ['Is Alive', 'Not Alive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = gbm_basic_predict)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Greys')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


########################
# Creating a classification report
########################

from sklearn.metrics import classification_report

print(classification_report(y_true = y_test,
                            y_pred = gbm_basic_predict))



# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = gbm_basic_predict,
                            target_names = labels))




########################
# Applying Gradiebt Boosting
########################


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

'''
Tuned GBM Parameter: {'criterion': 'friedman_mse', 'learning_rate': 0.1,
 'max_depth': 2, 'n_estimators': 50}
Tuned GBM Accuracy: 0.7951
'''

########################
# Applying Neural Network
########################

from sklearn.neural_network import MLPClassifier


clf = MLPClassifier(solver='lbfgs',
                     hidden_layer_sizes=(100,), random_state= 508)

clf.fit(X_train, y_train)


# Print the optimal parameters and best score

print("Tuned GBM Accuracy:", clf.score(X_train, y_train).round(3))




'''                 
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
'''