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


## Scrubbing and Cleaning Data -- https://bit.ly/2UhsOHo
√. Replaced all the null values in (waterfront,yr_renovated,sqft_basement)

√. Converted the date column in date time format

√.Converted the 'sqft_basement' from int to float64

√. Converted the 'yr_renovated' and 'sqft_basement' into binary to save it from creating Label Encoder

√. Feature Engineering ::

 Created a new column in the Dataframe that gives the Age of the House and the Percentage of the House Area
 
 ```df['age'] = df['yr_sold'] - df['yr_built']```
 ``` df['perc'] = (df['sqft_lot'] - (df['sqft_above'] / df['floors'])) / df['sqft_lot'] ```
 
√. Created a function to Remove Outliers

``` 
    def remove_outliers(df, column_name, threshold=4):
    z_scores = stats.zscore(df[column_name])
    indices = np.abs(np.where(z_scores > threshold))
    return indices[0]
    
columns_to_check = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']

#['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']
all_indices = []
for column in columns_to_check:
    indices = remove_outliers(df, column, threshold=3)
    all_indices.extend(indices)
all_indices = np.unique(all_indices)

# Remove outliers 4 standard deviations from mean in all columns
df = df.drop(index=all_indices)
df.shape
```

## Exploratory Data Analysis

#### 1.What are the most/least expensive homes based on zipcodes (cheapest area to live in vs most expensive)?

![image](https://user-images.githubusercontent.com/47164862/83890602-95ef6c00-a711-11ea-9d03-08c141846537.png)

Analysis of EDA::
~1910 & ~1940's and 2000's are among the highest sold homes.
Surprisingly yr_build and yr_renovation had correlations of 0.053965 & 0.117858 with the house price
homes in the 1910's were inspired by the Victotian style architecture. These houses can make a profit by being sold to historical societies or used in movie sets. Renovation is likely negligible because buyers who purchase lived in home probably plan to do renovations of their own, making the current renovations irrelevant




#### 2.What's the lowest grade/condition with the highest profit and vice versa?

![image](https://user-images.githubusercontent.com/47164862/83888356-56278500-a70f-11ea-95ee-b29bb70fbc24.png)

![image](https://user-images.githubusercontent.com/47164862/83888417-68a1be80-a70f-11ea-891d-215b4c3e4bdf.png)


Analysis of EDA::


The violin plot and the joint plot plotted above derives clearly that there is clearly a great relationship between the "Grade" of the house and "Price". The houses that yield to the most of the least grade i.e grade 3 is about 260,000, where as when we compare with the Grade 12 house, they yield about 1,500,000 .This clearly depicts that as the grade of the houses goes higher the rate of price goes higher which inturns more profit through the sale of King County residential department.




#### 3.Correlation/relationship between yr_built vs grade? (what age is considered vintage/"desirable old"?).

![image](https://user-images.githubusercontent.com/47164862/83890276-2c6f5d80-a711-11ea-872a-c5cbed155e3a.png)

Analysis of EDA::

As grade increases, the price of the home increases. Grade and Price had a (positive) correlation coefficient of 0.667964
Yr_built and grade had a (positive) correlation of 0.447854. This is low. Homes built in the years 1920-1911 had the lowest grades, and lowest average price sold.Homes built in the year 1984-1975 had the highest grades and average home prices.Homes in the 70s are considered vintage






####  4.Whats the best month/time of year for buying and selling? (build model to predict year).

![image](https://user-images.githubusercontent.com/47164862/83889014-3cd30880-a710-11ea-907c-442b10f555fb.png)

Analysis of EDA::


With the help of the plot graph it is clear that the months of the year does effect on the sale of the house and its prices. This plot graph depicts clearly that the peak month for the Selling houses is 'April' on an average of about $560,000 followed by least month for the sale is February with the average sale of $510,000. As we see there is quite a significant amount of drop in the February but then spikes again March and stays consistent till September or October. 
Based on the data, this graph shows the average sell prices of homes based on the month of the year. The highs and lows range from a price difference of around 10%.  The peak months to sell are March, April, May, and June - we could assuming this is due to tax return season, and end of the public school year.  The low months, being better for buyers, are December, January, and February - which could be explained by the holidays and low average temperatures


## Initializing for Modeling
√. Created a function to highlight the Multicolinearity with the threshold of 0.75
``` df_corr=df.corr().abs()

# Creating a function to highlight the correlation above the threshold value of 0.75
def Colli_thresh(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val > 0.75 else 'black'
    return 'color: %s' % color

highlight_thresh = df_corr.style.applymap(Colli_thresh)
highlight_thresh
```
√.Created a seaborn heatmap to visualize the multicolinearity among the other columns
 ``` 
 corr_cont = df[continuous].corr().abs()
(corr_cont)

# Mapping the correlation among the continous variable
sns.heatmap(corr_cont, annot=True)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()
```
![image](https://user-images.githubusercontent.com/47164862/83902276-b45d6380-a721-11ea-8664-25ef85ea6c7c.png)
![image](https://user-images.githubusercontent.com/47164862/83902316-c939f700-a721-11ea-8450-d8ff0081f5a5.png)

√. Conducted the Shapiro and KS-Test to check for Normality -- https://bit.ly/37bm8Qk
√. Converted all the numerical columns into Normal by Log Transformation, Mean Normalization , Standardization and Creating Function
```
def norm_feat(series):
    return (series - series.mean())/series.std()
for feat in norm_check.columns:
    df[feat] = norm_feat(df[feat])
# print(df.describe())

norm_df = norm_feat(norm_check)
norm_df.head()
norm_df.hist(figsize  = [8, 8]);
```

√. Created a dummy variables using Label Encoder and through a simple way of using Pandas ``` pd.get_dummies(x,drop_first=True)```


## Modeling for Data -- https://bit.ly/3cAvupU

√. Used ```sns.jointplot() ``` to see the linear regression of all the features in individual graphs
√. Created a complete Function to Get the Full Regression Analysis

## Model 

![image](https://user-images.githubusercontent.com/47164862/83892405-3b0b4400-a714-11ea-996a-5457eccecd69.png)
![image](https://user-images.githubusercontent.com/47164862/83892650-92111900-a714-11ea-8222-f8cc9e344c97.png)

## Final Model

√. Used a train test split method model to check linear regression model
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))
```

√. Used the same Linear Regression to run the same model in Scikit Learn
```
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Train-Test Split using Sklearn
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)
```

√.  Build the function to manually complete the K-Fold Validation
√.  Used the Cross the Validation Score from Scikit learn model
```
cv_5_results  = np.mean(cross_val_score(linreg, X, y, cv=5,  scoring='neg_mean_squared_error'))
cv_10_results = np.mean(cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error'))
cv_20_results = np.mean(cross_val_score(linreg, X, y, cv=20, scoring='neg_mean_squared_error'))

print('5-Fold Cross-Validation: ' ,cv_5_results)
print('10-Fold Cross-Validation: ' ,cv_10_results)
print('20-Fold Cross-Validation: ' ,cv_10_results)
```

## Predictions with the Essential Features
#### GRADE
![image](https://user-images.githubusercontent.com/47164862/83893933-3fd0f780-a716-11ea-8388-a7c9b5d66105.png)

#### Square Foot Living
![image](https://user-images.githubusercontent.com/47164862/83894408-e3220c80-a716-11ea-853f-d4106d5fcae8.png)

#### Count of Bathrooms
![image](https://user-images.githubusercontent.com/47164862/83894503-fe8d1780-a716-11ea-8375-a7a45abc35e1.png)


#### Interpretation

The final linear regression model predicts that the features that could yield the better prices. * The selected features predicts 61% and 65% of the variance of the target price.The P-value indicates that all relationships aren't random and all feature coefficients reveal positive correlation. With the nature of the data irrespective numerous of modeling it ranges the the R2 value between 55% to 65% . The data results can range low in R2 due to some additional factors like outliers,normalization, multicolinearity.
Train Mean Squared Error yielded 0.00229 where as Test Mean Squared Error yielded 0.002234 which was 0.00005 difference.

#### Y= -7.60*(price) + 1.07*(bedrooms) + 2.94*(bathrooms)+ 4.82*(sqft_living)+ 4.02*(grade)+ 5.54*(yr_built)+ 1.22*(floors)+6.18* (waterfront)


## Recommendations

√. Increase the size and the number of the bedrooms::
 Based on the demographic area of the house, buyers are more attracted towards the style of the bedrooms and the size of the bedroom. In most cases the bedrooms could be large and just few of them or make a decent sized bedrooms and give more features like the ceiling and lighting fixtures and more autonamious tech that could ease them for opening lights, close the blinds. Designs on the ceiling in the bedroom 

√. Increase in the size and the number of bathrooms::
With the high demand of rooms comes a high demand of the bathrooms. If the house is predicted to be built on large scale. It should be an ease of convenience in placing bathrooms in each floor and not necessarily a full bath but could add just half bath. Add little more designed quality bath tubs and faucets.

√. Size of the House::
As we discussed earlier in situations compared to size of the bedrooms, there is high possibilty that buyers would love to maintain square foot living of the house in a ratio to scale manner where all the bedrooms are in one proportion where as the living and dining would make a great difference. Not just living area but also the Patio and Lawn should be given a good amount of area. After all the with the size of house there should be some leisure in the big lawn area.

√. Grade and Condition ::-
Based on the regression these are considered to be the most important features that determines the house prices. There are list of grades that determine whether these are graded with the new stylish models or old school model based on the ranking. So does comes with the condition. Condition of the house determines if the house is in the state of living or not. As we know that the older the homes the conditions gets degraded. Buyers in the market for King County are looking for better quality conditioned houses that not neccessarily are built brand new but would keep it maintained to check for the houses worn and tear condition and update it from time to time.


√.  Classify Houses With Vintage:
It is recommended to suggest the buyers to offer them a competitative better rate when it comes to Vintage Houses that has been taken care for a long period of time.

## Future Work
As discussed earlier there is great possibilty of outliers, linearity and non-normalized and with the restriction of given time which consices us to give limited result but guide you as one of the best innovative methods to improve the selling prices. But as we continue to build models we can predict best suited houses for the buyers that could potentially be more benefit the firm by analyzing the needs by narrowing the buyers and pin pointing the questions that could yield best houses for the buyers. Also build a model for the sellers and other realty states so that they can predict what prices to be set and what type of houses to build in the upcoming venture based on the demographic area.

## Conclusion
This concludes the dataset with linear regression model and predictions.

