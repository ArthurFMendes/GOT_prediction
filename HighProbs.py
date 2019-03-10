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
        
        
        
