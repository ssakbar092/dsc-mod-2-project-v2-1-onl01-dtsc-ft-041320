## Module 2 Project
This is the second project which consists of Data Analysis on the Kings County Reality Dataset creating a meaning insightful and predicting the prices of the houses in King County  by deriving Linear Regression and creating Models to analyse which feature of the house could substantially create a great increase in the prices of the houses. This project is aimed to assist to the Real Estate companies and aiding the other Real Estate Companies in deciding what factors could determining and setting up the prices of the houses in King County.


## -------------------------------------OUTLINE BEGINS------------------------------------- 

This repository consists of the following list of files
√ A Complete Linear Regression Analysis along with One of Four Exploratory Data Analysis(EDA) questions answered(Since it was group project)
√ A complete ReadMe.md file that gives the outline of the project
√ A Blog Post based on the Project that gives an idea to my peers about the project


## DATASET

For this project we strictly used a credible dataset to keep it concise and to prevent our data from any other outliers by not deriving any more data through Web Scraping to complete our Linear Regression and predict our Model.

√ "kc_house_data.csv"

## Exploratory Data Analysis Questions

1.What are the most/least expensive homes based on zipcodes (cheapest area to live in vs most expensive)?

2.What's the lowest grade/condition with the highest profit and vice versa?

3.Correlation/relationship between yr_built vs grade? (what age is considered vintage/"desirable old"?).

4.Whats the best month/time of year for buying and selling? (build model to predict year)


## Project Content

In this project we have combined all essential libraries that we learned so far to create a dynamic project that could yield some results and gain a great experience in learning and understanding how each aspect of the libraries and more important in deriving the results. Overall the process we have acquired the OSEMIN method

√ Obtain and load the dataset

√ Scrub and clean the data

√ Explore and understand the data value

√ Model the data through testing processes

√ Interpret the results and conclude



## Obtain and load the dataset -- https://bit.ly/3gZRRbo

√. Imported Some Essential Libraries to start of with 

 ```import numpy as np
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   %matplotlib inline

from scipy import stats
from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import RFE 
```


√. Loading and Verifying the Dataset


``` data = pd.read_csv('kc_house_data.csv') ```

```print(df.isnull().sum(),'\n\n')
 print(df.info(),'\n\n')
 print(df.nunique())
 print(df[df.duplicated()])
 ```


## Scrubbing and Cleaning Data
√.```df['waterfront'] = df['waterfront'].fillna(0.0) - Replaced all the null values --- Replaced all Null Values

√. df = df.drop(['lat', 'long', 'sqft_living15', 'sqft_lot15', 'id','view'], axis=1) -- Dropped all unncessary columns

df['date'] = df['date'].astype('datetime64[ns]') 

#Converting the 'sqft_basement' from object to a float datatype
df['sqft_basement'] = df['sqft_basement'].astype('float64')
