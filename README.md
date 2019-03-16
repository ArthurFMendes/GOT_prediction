# Game of Thrones: life or death

  This is data analysis on who will live or die on the fictional series "Game of Thrones". We will be walking through the data analysis process and making ajustments as needed.

  We will start this analysis by importing the data and doing a quick exploration, to understand what are the variables and how we shuld proceed. The lastest version of the code can be found [here](https://github.com/ArthurFMendes/GOT_prediction/blob/master/LordFlag.py).
  
```python

########################
# Fundamental Dataset Exploration
########################
GoT_df = pd.read_excel('GOT_character_predictions.xlsx')

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
```
## Binary variables

Most variables that we are going to be utilizing on our predictions are *binary*, including the dependent variable (y variable). 

For feature engineering, we created a few new variables, including missing values on a few different variables, which in the case of this dataset indicated that the relative is unknown, which can some times tell the importance of the caracter. Other features (as outlier for houses, cultures and titles) were also engineered.

By the end of the exploration, this were the variables chosen for the development of the machine leaning models:

```python
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
                                'm_isAliveFather',
                                'm_isAliveHeir',
                                'm_isAliveSpouse',
                                'out_house',
                                'out_culture',
                                'out_title']]


GoT_target   = GoT_Chosen.loc[:,['isAlive']]

```

## Machine Learning Algorithms

We chose to go ahead with two different machine learning models (so far).

1. **Random Forest**
2. **Gradient Boosting Machine**

********

## Random Forest

This machine learning algorithm is an ensemble learning method for classification or regression that work by constructing a variety of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. 

By ensembling different trees together in different regions of your data, you end up with a much higher predictive accuracy than otherwise.

### How did we measure performance?

In the process of creating a model, performance measurement is an essential task. Because of that, we chose not to rely on only one measurement, we use **score, AUC and ROC, Classification report and Confusion matrix**. The score is a simple way to compare the results with other linear models (similar to R-squared). 

The Score was as follows:
```
Training Score 0.8136
Testing Score: 0.8214
```
AUC - ROC curve is a performance measurement for classification problems. ROC is a probability curve and AUC represents the degree of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.

The ROC curve is mapped with TPR(True Positive Rate) against the FPR(False Positive Rate) where TPR is on y-axis and FPR is on the x-axis.

The ROC was as follows:

![alt text](
        https://github.com/ArthurFMendes/GOT_prediction/blob/master/ROC%20Random%20Forrest.png
      )

With a AUC score of
```
metrics.auc(fpr, tpr)
0.8646138807429129
```
#### Classification Report

The classification report shows the main classification metrics precision, recall and f1-score on a per-class basis. The metrics are defined in terms of true and false positives, and true and false negatives. Positive and negative in this case are generic names for the classes of a binary classification problem. There are four ways to check if the predictions are right or wrong:
```
TN / True Negative: case was negative and predicted negative
TP / True Positive: case was positive and predicted positive
FN / False Negative: case was positive but predicted negative
FP / False Positive: case was negative but predicted positive
```
#### Precision – What percent of your predictions were correct?

Precision is the ability of a classifier not to label an instance positive that is actually negative. For each class it is defined as the ratio of true positives to the sum of true and false positives.
```
TP – True Positives
FP – False Positives
```
Precision – Accuracy of positive predictions.
```
Precision = TP/(TP + FP)
```
#### Recall – What percent of the positive cases did you catch?

Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.
```
FN – False Negatives
```
Recall: Fraction of positives that were correctly identified.
```
Recall = TP/(TP+FN)
```
#### F1 score – What percent of positive predictions were correct? 

The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.
```
F1 Score = 2*(Recall * Precision) / (Recall + Precision)
```
The classification Report:

|              | Precision  | Recall  |f1-score |
| :----------- |:-----------| :-------|:--------|
| 0            | 0.85       | 0.42    |0.56     |
| 1            | 0.83       |   0.98  |0.90     |
| avg / total  | 0.84       |    0.83 |0.81     |

#### Confusion Matrix

A confusion matrix is a table that is used to describe the performance of a classification model on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.

The terms of a confusion matrix are the same as the classification report, only give in the actual number of observations that are true posite, true negative, false positive and false negative.

|              | Predicted 0   | Predicted 1  |
| :----------- |:--------------| :------------|
| **Actual 0**     | 52           |   72         |
| **Actual 1**     | 9            |   354        |


#### Variable Importance
Another benefit that Random Forest brings is the **variable importance**. The model automatically create a list of variables and its importance for the results. This are the most important features once we call the function:

|                            |Importance|
| :------------- |:-------------|
|popularity                    |0.228659|
|out_dateOfBirth               |0.166500|
|book4_A_Feast_For_Crows       |0.159622|
|out_house                     |0.156254|
|male                          |0.060581|
|book1_A_Game_Of_Thrones       |0.050642|
|numDeadRelations              |0.038546|
|out_culture                   |0.027060|
|book2_A_Clash_Of_Kings        |0.026905|
|book5_A_Dance_with_Dragons    |0.025907|
|isNoble                       |0.024055|
|book3_A_Storm_Of_Swords       |0.019651|
|isMarried                     |0.005941|
|m_isAliveSpouse               |0.005469|
|out_title                     |0.004208|

#### References

[AUC - ROC curce](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

[Classification Report](https://muthu.co/understanding-the-classification-report-in-sklearn/)

[Simple guide to confusion matrix terminology](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
