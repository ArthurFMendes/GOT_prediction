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

1. **Random Forrest**
2. **Gradiant Boosting Machine**
