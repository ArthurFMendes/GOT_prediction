#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:02:58 2019

@author: arthurmendes
"""


import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Adjusting the Threshold
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t)
    """
    return [1 if y >= t else 0 for y in y_scores]


# Generate new class predictions based on the adjusted_classes function above
# and view the resulting confusion matrix and classification report

def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """


    y_pred_adj = adjusted_classes(y_scores, t)
    print('\nConfusion Matrix:\n',
          pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
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


# Plotting precision and recall

def plot_precision_recall(precisions, recalls, thresholds):
    """
    Creates a plot that analyze precision and recall for each different
    threshold
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')





########################
# Fundamental Dataset Exploration
########################
GoT_df = pd.read_excel('GOT_character_predictions.xlsx')


# Creating a dropped dataset to graph

GoT_toDrop   = GoT_df.loc[:,['S.No',
                             'male',
                             'dateOfBirth',
                             'book1_A_Game_Of_Thrones',
                             'book2_A_Clash_Of_Kings',
                             'book3_A_Storm_Of_Swords',
                             'book4_A_Feast_For_Crows',
                             'book5_A_Dance_with_Dragons',
                             'isMarried',
                             'age',
                             'isNoble',
                             'numDeadRelations',
                             'popularity',
                             'isAlive',]]

df_dropped = GoT_toDrop.dropna()


# Plotting numerical variables

# Date of Birth
sns.distplot(df_dropped['dateOfBirth'],
             bins = 35,
             color = 'g')

plt.xlabel('Birth Year')
plt.show()

# Age
sns.distplot(df_dropped['age'],
             bins = 30,
             color = 'b')

plt.xlabel('Age')
plt.show()

# Number of dead relatives
sns.distplot(df_dropped['numDeadRelations'],
             color = 'y')

plt.xlabel('Number of Dead Relatives')
plt.show()


# Pair plot
p = sns.pairplot(df_dropped[(df_dropped.age >= 0)]
[["popularity", "numDeadRelations", "age", "isAlive"]],
hue = "isAlive", vars = ["popularity", "numDeadRelations", "age"],
 kind = "reg", size = 4.)
plt.savefig('Pairplot.png')
plt.show()



# Create A vizualization for the cultures

cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()


GoT_df.loc[:, "culture"] = [get_cult(x) for x in
                         GoT_df.culture.fillna("")]

data = GoT_df.groupby(
        ["culture", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
data.loc[:, "total"]= data.sum(axis = 1)

p = data[data.index != ""].sort_values("total")[[0, 1]].plot.barh(
        stacked = True, rot = 0, figsize = (14, 12),)
_ = p.set(xlabel = "No. of Characters", ylabel = "Culture"), p.legend(
        ["Dead", "Alive"], loc = "lower right")
plt.savefig('Culture Analysis.png')
plt.show()

# Cultures with high probabylity of dying (Sum of isAlive/Count isAlive > 0.5)

GoT_df['out_culture'] = 0

cultures = ['Vale',
            'Meereen',
            'Lhazareen',
            'Astapor',
            'Valyrian',
            'Astapori',
            'Pentoshi',
            'Westerman',
            'Westermen',
            'Westeros',
            'Sistermen',
            'Riverlands',
            'Qohor']


for val in enumerate(GoT_df.loc[ : , 'culture']):
    '''
    Creating a flag if the culture appear in the list of cultures, which means
    that they have a high chance of dying.
    '''
    if val[1] in cultures:
        GoT_df.loc[val[0], 'out_culture'] = 1




# Good cultures
GoT_df['good_culture'] = 0

good_culture = ['Andal',
                'Andals',
                'Asshai',
                'Asshai\'i',
                'Crannogmen',
                'Dorne',
                'First Men',
                'Ibbenese',
                'Lhazarene',
                'Lyseni',
                'Naathi',
                'Norvos',
                'Qarth',
                'Reachmen',
                'Rhoynar',
                'Stormlander',
                'Summer Islander',
                'Summer Island',
                'Summer Isles',
                'The Reach',
                'Westerlands']



for val in enumerate(GoT_df.loc[ : , 'culture']):
    '''
    Creating a flag if the culture appear in the list of cultures, which means
    that they have a high chance of dying.
    '''
    if val[1] in cultures:
        GoT_df.loc[val[0], 'good_culture'] = 1


# Houses with high probabylity of dying (Sum of isAlive/Count isAlive > 0.5)



GoT_df['out_house'] = 0

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

# Good Houses
        
GoT_df['good_house'] = 0

house = ['Antler Men',
         'Black Ears',
         'Burned Men',
         'Chataya\'s brothel',
         'Citadel',
         'Company of the Cat',
         'Drowned men',
         'Faceless Men',
         'Golden Company',
         'Graces',
         'Happy Port',
         'House Allyrion',
         'House Ambrose',
         'House Ashford',
         'House Baelish',
         'House Banefort',
         'House Baratheon of Dragonstone',
         'House Baratheon of King\'s Landing',
         'House Belmore',
         'House Bettley',
         'House Blackbar',
         'House Blackberry',
         'House Blackmont',
         'House Blanetree',
         'House Blount',
         'House Boggs',
         'House Bolling',
         'House Bolton of the Dreadfort',
         'House Borrell',
         'House Broom',
         'House Brune of Brownhollow',
         'House Brune of the Dyre Den',
         'House Buckler',
         'House Bulwer',
         'House Celtigar',
         'House Chester',
         'House Chyttering',
         'House Clifton',
         'House Codd',
         'House Coldwater',
         'House Tyrell',
         'House Condon',
         'House Coklyn',
         'House Connington',
         'House Corbray',
         'House Costayne',
         'House Cox',
         'House Crane',
         'House Cupps',
         'House Dalt',
         'House Deddings',
         'House Dondarrion',
         'House Drinkwater',
         'House Drumm',
         'House Durrandon',
         'House Erenford',
         'House Errol',
         'House Estren',
         'House Farman',
         'House Farwynd',
         'House Flint',
         'House Foote',
         'House Fossoway',
         'House Fowler']


for val in enumerate(GoT_df.loc[ : , 'house']):
    '''
    Creating a flag if the house appear in the list of houses, which means
    that they have a high probability of dying.
    '''
    if val[1] in house:
        GoT_df.loc[val[0], 'good_house'] = 1

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
                             'culture',
                             'house',
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
                             'age',
                             'isAlive',
                             'out_house',
                             'out_culture',
                             'out_title',
                             'good_culture',
                             'good_house']]


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



# Age being filled with the median
fill = GoT_Chosen['age'].median()

GoT_Chosen['age'] = GoT_Chosen['age'].fillna(fill)


# Outlier flags

HighAge = 60


GoT_Chosen['out_age'] = 0


for val in enumerate(GoT_Chosen.loc[ : , 'dateOfBirth']):

    if val[1] >= low_dateOfBirth:
        GoT_Chosen.loc[val[0], 'out_age'] = 1


# Outlier flag for number of dead relatives


HighDeadRela = 3


GoT_Chosen['out_numrelatives'] = 0


for val in enumerate(GoT_Chosen.loc[ : , 'numDeadRelations']):

    if val[1] > HighDeadRela:
        GoT_Chosen.loc[val[0], 'out_numrelatives'] = 1


# Age being filled with the median
fill = 2 #Means that their heir is no shown in the series

GoT_Chosen['isAliveHeir'] = GoT_Chosen['isAliveHeir'].fillna(fill)


# house being filled with empty spaces
fill = " " #Means that their heir is no shown in the series

GoT_Chosen['house'] = GoT_Chosen['house'].fillna(fill)

GoT_Chosen.loc[:, "house"] = pd.factorize(GoT_Chosen.house)[0]



# culture being filled with empty spaces
fill = " " #Means that their heir is no shown in the series

GoT_Chosen['culture'] = GoT_Chosen['culture'].fillna(fill)

GoT_Chosen.loc[:, "culture"] = pd.factorize(GoT_Chosen.culture)[0]

###############################################################################
# Correlation Analysis
###############################################################################

GoT_Chosen.head()


df_corr = GoT_Chosen.corr().round(2)


print(df_corr)


df_corr.loc['isAlive'].sort_values(ascending = False)



########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Game of Thrones Correlarion Heatmap.png')
plt.show()


# Checking again
print(
      GoT_Chosen
      .isnull()
      .sum()
      )

# Dividing the data into train and test

GoT_data   = GoT_Chosen.loc[:,[ 'male',
                                'culture',
                                'house',
                                'dateOfBirth',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'm_isAliveSpouse',
                                'out_house',
                                'out_culture',
                                'out_dateOfBirth',
                                'good_culture',
                                'out_age',
                                'good_house']]

GoT_target   = GoT_Chosen.loc[:,['isAlive']]


X_train, X_test, y_train, y_test = train_test_split(
            GoT_data,
            GoT_target.values.ravel(),
            test_size = 0.1,
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
estimator_space = pd.np.arange(10, 100, 5)
leaf_space = pd.np.arange(1, 30, 1)
criterion_space = ['entropy']
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

Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion':
'entropy', 'min_samples_leaf': 11, 'n_estimators': 460, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8108

Tuned Logistic Regression Parameter: {'bootstrap': False, 'criterion': 'entropy',
'min_samples_leaf': 11, 'n_estimators': 50, 'warm_start': True}
Tuned Logistic Regression Accuracy: 0.8167
'''

# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))


# Using the optimal Random Forrest

# Full forest using entropy
optimal_forrest = RandomForestClassifier(n_estimators = 50,
                                         criterion = 'entropy',
                                         max_depth = None,
                                         min_samples_leaf = 6,
                                         bootstrap = False,
                                         warm_start = True,
                                         random_state = 508)


# Fitting the models
optimal_forrest_fit = optimal_forrest.fit(X_train, y_train)

full_forest_predict = optimal_forrest.predict(X_test)



# Scoring the gini model
print('Training Score', optimal_forrest_fit.score(X_train, y_train).round(4))
print('Testing Score:', optimal_forrest_fit.score(X_test, y_test).round(4))

# Cross Validation
RF_Cross_Val = cross_val_score(optimal_forrest, X_test, y_test, cv = 3)

print(pd.np.mean(RF_Cross_Val).round(4))


# Saving score objects
gini_full_train = optimal_forrest_fit.score(X_train, y_train)
gini_full_test  = optimal_forrest_fit.score(X_test, y_test)


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
plt.savefig('ROC Random Forest')
plt.show()


# AUC Score
metrics.auc(fpr, tpr)


###############################################################################
# Variable importance
###############################################################################

feature_importances = pd.DataFrame(optimal_forrest_fit.feature_importances_,
                                   index = X_train.columns,
                                   columns=
                                   ['Importance']).sort_values('Importance',
                                   ascending=False)

print(feature_importances)


## If we dont change the threshold
print ('\nClassification report:\n', classification_report(y_test, full_forest_predict))
print ('\nConfusion matrix:\n',confusion_matrix(y_test, full_forest_predict))


##################################
# Precision recall, changing the threshhold
##################################

# Probability for the threshold test
y_scores = optimal_forrest.predict_proba(X_test)[:, 1]

# Generate the precision-recall curve for the classifier:
p, r, thresholds = precision_recall_curve(y_test, y_scores)


# Adjusting the Threshold to t = 0.51
precision_recall_threshold(p, r, thresholds, 0.51)


# Plotting precision and recall
plot_precision_recall(p, r, thresholds)


# Final submission
my_submission = pd.DataFrame({ 'Test': y_test, 
                              'Is Alive': full_forest_predict})
my_submission.to_excel('GoTfinal.xlsx', index=False)


###############################################################################
# Gradient Boosted Machines
###############################################################################


GoT_data   = GoT_Chosen.loc[:,[ 'male',
                                'book1_A_Game_Of_Thrones',
                                'book2_A_Clash_Of_Kings',
                                'book4_A_Feast_For_Crows',
                                'book5_A_Dance_with_Dragons',
                                'isNoble',
                                'numDeadRelations',
                                'popularity',
                                'out_house',
                                'out_culture',
                                'out_dateOfBirth',
                                'good_culture']]

GoT_target   = GoT_Chosen.loc[:,['isAlive']]


X_train, X_test, y_train, y_test = train_test_split(
            GoT_data,
            GoT_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = GoT_target)



# Building a learner gbm
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

from sklearn.model_selection import GridSearchCV


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

Tuned GBM Parameter: {'criterion': 'friedman_mse', 'learning_rate': 0.1,
'max_depth': 5, 'n_estimators': 90}
Tuned GBM Accuracy: 0.8191
'''

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))


# Building a optimal GBM

gbm_optimal = GradientBoostingClassifier(loss = 'deviance',
                                         learning_rate = 0.1,
                                         n_estimators = 90,
                                         max_depth = 3,
                                         criterion = 'friedman_mse',
                                         warm_start = False,
                                         random_state = 508,)


gbm_optimal_fit = gbm_optimal.fit(X_train, y_train)


gbm_optimal_predict = gbm_optimal_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_optimal_fit.score(X_train, y_train)
gmb_basic_test  = gbm_optimal_fit.score(X_test, y_test)


###################################
# ROC curve for GBM
###################################

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
plt.savefig('ROC for GBM')
plt.show()

# AUC Score
metrics.auc(fpr, tpr)


#####################################
# Classification Report and Confussion Matrix
#####################################


print ('\nClassification report:\n', classification_report(y_test, gbm_optimal_predict))
print ('\nConfusion matrix:\n',confusion_matrix(y_test, gbm_optimal_predict))


##################################
# Precision recall, changing the threshhold
##################################


# Creating a threshold recall
y_scores = gbm_optimal.predict_proba(X_test)[:, 1]


# Generate the precision-recall curve for the classifier:
p, r, thresholds = precision_recall_curve(y_test, y_scores)


# Adjusting the Threshold
precision_recall_threshold(p, r, thresholds, 0.51)


# Plotting precision and recall
plot_precision_recall(p, r, thresholds)


##################################
# Cross Validation Score
##################################

GBM_cross_val = cross_val_score(gbm_optimal, GoT_data, GoT_target, cv = 3)

print(pd.np.mean(GBM_cross_val))
