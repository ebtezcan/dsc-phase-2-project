# King County Renovation: Most Effective Way to Increase the Value of Your Home 

**Author:** E. Berke Tezcan

***

## TABLE OF CONTENTS 

*Click to jump to matching Markdown Header.*<br><br>
 
- **[Introduction](#INTRODUCTION)<br>**
- **[OBTAIN](#OBTAIN)**<br>
- **[SCRUB/EXPLORE](#SCRUB/EXPLORE)**<br>
- **[MODEL](#MODEL)**<br>
- **[iNTERPRET](#iNTERPRET)**<br>
- **[Conclusions/Recommendations](#CONCLUSIONS-&-RECOMMENDATIONS)<br>**
___

# INTRODUCTION

For this project, we were hired by a home owner in King County, Washington who wants to renovate their home. They would like us to analyze the real estate data of the county and give them insights as to what to focus their renovation efforts on in order to increase their property's value. 

Real estate prices are affected by a myriad of -what we can define as- internal parameters like the square footage, floor count, bedroom count, finishes, how many cars can fit into the garage etc. There are also parameters that we can define as external and can not (easily) be changed. These include attributes like zipcode, latitude, longitude, view from the house, and school districts. In order to accurately model and pinpoint the most important parameters that affect the sale price of a home, we need to incorporate both internal and external parameters.

We are given a dataset that includes information about the real estate in King County and will be using this dataset to create a multiple linear regression (MLR) model. We defined our goal to be to find the top 3 internal parameters that affect a home's sale price the most in King County specifically. This will ensure that the home owner can actually keep these parameters in mind while renovating rather than getting insights about external parameters that they can't necessarily do anything to change.

# OBTAIN

## Data Understanding/EDA



```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import numpy as np
from scipy import stats
```


```python
df = pd.read_csv('data/kc_house_data.csv')
```


```python
pd.set_option('display.max_columns',0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  float64
     9   view           21534 non-null  float64
     10  condition      21597 non-null  int64  
     11  grade          21597 non-null  int64  
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB
    


```python
df['bathrooms'].unique()
```




    array([1.  , 2.25, 3.  , 2.  , 4.5 , 1.5 , 2.5 , 1.75, 2.75, 3.25, 4.  ,
           3.5 , 0.75, 4.75, 5.  , 4.25, 3.75, 1.25, 5.25, 6.  , 0.5 , 5.5 ,
           6.75, 5.75, 8.  , 7.5 , 7.75, 6.25, 6.5 ])




```python
df['condition'].unique()
```




    array([3, 5, 4, 1, 2], dtype=int64)




```python
df['grade'].unique()
```




    array([ 7,  6,  8, 11,  9,  5, 10, 12,  4,  3, 13], dtype=int64)




```python
df['view'].unique()
```




    array([ 0., nan,  3.,  4.,  2.,  1.])



# SCRUB/EXPLORE

## Addressing Null Values


```python
df.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3842
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64



To address the missing values in the view column we can take the more conservative approach and say that these 63 houses/apartments did not have a view with minimal impact to the overall dataset since we have 21597 data points overall.


```python
df['view'].fillna(0, inplace=True)
df.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view                0
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3842
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64




```python
df[df['yr_built']==df['yr_renovated']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



There are no oddities with the year built and year renovated columns so far it seems like.


```python
df[df['yr_renovated'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>12</th>
      <td>114101516</td>
      <td>5/28/2014</td>
      <td>310000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1430</td>
      <td>19901</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1430</td>
      <td>0.0</td>
      <td>1927</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7558</td>
      <td>-122.229</td>
      <td>1780</td>
      <td>12697</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8091400200</td>
      <td>5/16/2014</td>
      <td>252700.0</td>
      <td>2</td>
      <td>1.50</td>
      <td>1070</td>
      <td>9643</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1070</td>
      <td>0.0</td>
      <td>1985</td>
      <td>NaN</td>
      <td>98030</td>
      <td>47.3533</td>
      <td>-122.166</td>
      <td>1220</td>
      <td>8386</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1794500383</td>
      <td>6/26/2014</td>
      <td>937000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>2450</td>
      <td>2691</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1750</td>
      <td>700.0</td>
      <td>1915</td>
      <td>NaN</td>
      <td>98119</td>
      <td>47.6386</td>
      <td>-122.360</td>
      <td>1760</td>
      <td>3573</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5101402488</td>
      <td>6/24/2014</td>
      <td>438000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1520</td>
      <td>6380</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>790</td>
      <td>730.0</td>
      <td>1948</td>
      <td>NaN</td>
      <td>98115</td>
      <td>47.6950</td>
      <td>-122.304</td>
      <td>1520</td>
      <td>6235</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21576</th>
      <td>1931300412</td>
      <td>4/16/2015</td>
      <td>475000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1190</td>
      <td>1200</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1190</td>
      <td>0.0</td>
      <td>2008</td>
      <td>NaN</td>
      <td>98103</td>
      <td>47.6542</td>
      <td>-122.346</td>
      <td>1180</td>
      <td>1224</td>
    </tr>
    <tr>
      <th>21577</th>
      <td>8672200110</td>
      <td>3/17/2015</td>
      <td>1090000.0</td>
      <td>5</td>
      <td>3.75</td>
      <td>4170</td>
      <td>8142</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>10</td>
      <td>4170</td>
      <td>0.0</td>
      <td>2006</td>
      <td>NaN</td>
      <td>98056</td>
      <td>47.5354</td>
      <td>-122.181</td>
      <td>3030</td>
      <td>7980</td>
    </tr>
    <tr>
      <th>21579</th>
      <td>1972201967</td>
      <td>10/31/2014</td>
      <td>520000.0</td>
      <td>2</td>
      <td>2.25</td>
      <td>1530</td>
      <td>981</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1480</td>
      <td>50.0</td>
      <td>2006</td>
      <td>NaN</td>
      <td>98103</td>
      <td>47.6533</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1282</td>
    </tr>
    <tr>
      <th>21581</th>
      <td>191100405</td>
      <td>4/21/2015</td>
      <td>1580000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3410</td>
      <td>10125</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>3410</td>
      <td>?</td>
      <td>2007</td>
      <td>NaN</td>
      <td>98040</td>
      <td>47.5653</td>
      <td>-122.223</td>
      <td>2290</td>
      <td>10125</td>
    </tr>
    <tr>
      <th>21583</th>
      <td>7202300110</td>
      <td>9/15/2014</td>
      <td>810000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>3990</td>
      <td>7838</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>3990</td>
      <td>0.0</td>
      <td>2003</td>
      <td>NaN</td>
      <td>98053</td>
      <td>47.6857</td>
      <td>-122.046</td>
      <td>3370</td>
      <td>6814</td>
    </tr>
  </tbody>
</table>
<p>3842 rows × 21 columns</p>
</div>




```python
df['waterfront'].unique()
```




    array([nan,  0.,  1.])




```python
df['yr_renovated'].unique()
```




    array([   0., 1991.,   nan, 2002., 2010., 1992., 2013., 1994., 1978.,
           2005., 2003., 1984., 1954., 2014., 2011., 1983., 1945., 1990.,
           1988., 1977., 1981., 1995., 2000., 1999., 1998., 1970., 1989.,
           2004., 1986., 2007., 1987., 2006., 1985., 2001., 1980., 1971.,
           1979., 1997., 1950., 1969., 1948., 2009., 2015., 1974., 2008.,
           1968., 2012., 1963., 1951., 1962., 1953., 1993., 1996., 1955.,
           1982., 1956., 1940., 1976., 1946., 1975., 1964., 1973., 1957.,
           1959., 1960., 1967., 1965., 1934., 1972., 1944., 1958.])



I assumed here that the NA values meant that the place was not renovated. This is also the more conservative approach so I'm comfortable with replacing these values with 0.


```python
df['yr_renovated'].fillna(0, inplace=True)
```


```python
df.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view                0
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated        0
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64



Similar to other columns, putting 0's for the NaN values in the waterfront column is the more conservative approach and makes sense.


```python
df['waterfront'].fillna(0, inplace=True)
```


```python
df.isna().sum()
```




    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64



## Addressing Placeholder Values


```python
df['sqft_basement'].unique()
```




    array(['0.0', '400.0', '910.0', '1530.0', '?', '730.0', '1700.0', '300.0',
           '970.0', '760.0', '720.0', '700.0', '820.0', '780.0', '790.0',
           '330.0', '1620.0', '360.0', '588.0', '1510.0', '410.0', '990.0',
           '600.0', '560.0', '550.0', '1000.0', '1600.0', '500.0', '1040.0',
           '880.0', '1010.0', '240.0', '265.0', '290.0', '800.0', '540.0',
           '710.0', '840.0', '380.0', '770.0', '480.0', '570.0', '1490.0',
           '620.0', '1250.0', '1270.0', '120.0', '650.0', '180.0', '1130.0',
           '450.0', '1640.0', '1460.0', '1020.0', '1030.0', '750.0', '640.0',
           '1070.0', '490.0', '1310.0', '630.0', '2000.0', '390.0', '430.0',
           '850.0', '210.0', '1430.0', '1950.0', '440.0', '220.0', '1160.0',
           '860.0', '580.0', '2060.0', '1820.0', '1180.0', '200.0', '1150.0',
           '1200.0', '680.0', '530.0', '1450.0', '1170.0', '1080.0', '960.0',
           '280.0', '870.0', '1100.0', '460.0', '1400.0', '660.0', '1220.0',
           '900.0', '420.0', '1580.0', '1380.0', '475.0', '690.0', '270.0',
           '350.0', '935.0', '1370.0', '980.0', '1470.0', '160.0', '950.0',
           '50.0', '740.0', '1780.0', '1900.0', '340.0', '470.0', '370.0',
           '140.0', '1760.0', '130.0', '520.0', '890.0', '1110.0', '150.0',
           '1720.0', '810.0', '190.0', '1290.0', '670.0', '1800.0', '1120.0',
           '1810.0', '60.0', '1050.0', '940.0', '310.0', '930.0', '1390.0',
           '610.0', '1830.0', '1300.0', '510.0', '1330.0', '1590.0', '920.0',
           '1320.0', '1420.0', '1240.0', '1960.0', '1560.0', '2020.0',
           '1190.0', '2110.0', '1280.0', '250.0', '2390.0', '1230.0', '170.0',
           '830.0', '1260.0', '1410.0', '1340.0', '590.0', '1500.0', '1140.0',
           '260.0', '100.0', '320.0', '1480.0', '1060.0', '1284.0', '1670.0',
           '1350.0', '2570.0', '1090.0', '110.0', '2500.0', '90.0', '1940.0',
           '1550.0', '2350.0', '2490.0', '1481.0', '1360.0', '1135.0',
           '1520.0', '1850.0', '1660.0', '2130.0', '2600.0', '1690.0',
           '243.0', '1210.0', '1024.0', '1798.0', '1610.0', '1440.0',
           '1570.0', '1650.0', '704.0', '1910.0', '1630.0', '2360.0',
           '1852.0', '2090.0', '2400.0', '1790.0', '2150.0', '230.0', '70.0',
           '1680.0', '2100.0', '3000.0', '1870.0', '1710.0', '2030.0',
           '875.0', '1540.0', '2850.0', '2170.0', '506.0', '906.0', '145.0',
           '2040.0', '784.0', '1750.0', '374.0', '518.0', '2720.0', '2730.0',
           '1840.0', '3480.0', '2160.0', '1920.0', '2330.0', '1860.0',
           '2050.0', '4820.0', '1913.0', '80.0', '2010.0', '3260.0', '2200.0',
           '415.0', '1730.0', '652.0', '2196.0', '1930.0', '515.0', '40.0',
           '2080.0', '2580.0', '1548.0', '1740.0', '235.0', '861.0', '1890.0',
           '2220.0', '792.0', '2070.0', '4130.0', '2250.0', '2240.0',
           '1990.0', '768.0', '2550.0', '435.0', '1008.0', '2300.0', '2610.0',
           '666.0', '3500.0', '172.0', '1816.0', '2190.0', '1245.0', '1525.0',
           '1880.0', '862.0', '946.0', '1281.0', '414.0', '2180.0', '276.0',
           '1248.0', '602.0', '516.0', '176.0', '225.0', '1275.0', '266.0',
           '283.0', '65.0', '2310.0', '10.0', '1770.0', '2120.0', '295.0',
           '207.0', '915.0', '556.0', '417.0', '143.0', '508.0', '2810.0',
           '20.0', '274.0', '248.0'], dtype=object)



The sqft_basement column's dtype needs to be int like other sqft columns but it is dtype object due to the "?" values.


```python
df[df['sqft_basement']=='?']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1321400060</td>
      <td>6/27/2014</td>
      <td>257500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1715</td>
      <td>?</td>
      <td>1995</td>
      <td>0.0</td>
      <td>98003</td>
      <td>47.3097</td>
      <td>-122.327</td>
      <td>2238</td>
      <td>6819</td>
    </tr>
    <tr>
      <th>18</th>
      <td>16000397</td>
      <td>12/5/2014</td>
      <td>189000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1200</td>
      <td>9850</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1200</td>
      <td>?</td>
      <td>1921</td>
      <td>0.0</td>
      <td>98002</td>
      <td>47.3089</td>
      <td>-122.210</td>
      <td>1060</td>
      <td>5095</td>
    </tr>
    <tr>
      <th>42</th>
      <td>7203220400</td>
      <td>7/7/2014</td>
      <td>861990.0</td>
      <td>5</td>
      <td>2.75</td>
      <td>3595</td>
      <td>5639</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>3595</td>
      <td>?</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98053</td>
      <td>47.6848</td>
      <td>-122.016</td>
      <td>3625</td>
      <td>5639</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1531000030</td>
      <td>3/23/2015</td>
      <td>720000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3450</td>
      <td>39683</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>3450</td>
      <td>?</td>
      <td>2002</td>
      <td>0.0</td>
      <td>98010</td>
      <td>47.3420</td>
      <td>-122.025</td>
      <td>3350</td>
      <td>39750</td>
    </tr>
    <tr>
      <th>112</th>
      <td>2525310310</td>
      <td>9/16/2014</td>
      <td>272500.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1540</td>
      <td>12600</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1160</td>
      <td>?</td>
      <td>1980</td>
      <td>0.0</td>
      <td>98038</td>
      <td>47.3624</td>
      <td>-122.031</td>
      <td>1540</td>
      <td>11656</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21442</th>
      <td>3226049565</td>
      <td>7/11/2014</td>
      <td>504600.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>2360</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1390</td>
      <td>?</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6931</td>
      <td>-122.330</td>
      <td>2180</td>
      <td>5009</td>
    </tr>
    <tr>
      <th>21447</th>
      <td>1760650900</td>
      <td>7/21/2014</td>
      <td>337500.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2330</td>
      <td>4907</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2330</td>
      <td>?</td>
      <td>2013</td>
      <td>0.0</td>
      <td>98042</td>
      <td>47.3590</td>
      <td>-122.081</td>
      <td>2300</td>
      <td>3836</td>
    </tr>
    <tr>
      <th>21473</th>
      <td>6021503707</td>
      <td>1/20/2015</td>
      <td>352500.0</td>
      <td>2</td>
      <td>2.50</td>
      <td>980</td>
      <td>1010</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>980</td>
      <td>?</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.6844</td>
      <td>-122.387</td>
      <td>980</td>
      <td>1023</td>
    </tr>
    <tr>
      <th>21519</th>
      <td>2909310100</td>
      <td>10/15/2014</td>
      <td>332000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2380</td>
      <td>5737</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2380</td>
      <td>?</td>
      <td>2010</td>
      <td>0.0</td>
      <td>98023</td>
      <td>47.2815</td>
      <td>-122.356</td>
      <td>2380</td>
      <td>5396</td>
    </tr>
    <tr>
      <th>21581</th>
      <td>191100405</td>
      <td>4/21/2015</td>
      <td>1580000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3410</td>
      <td>10125</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>3410</td>
      <td>?</td>
      <td>2007</td>
      <td>0.0</td>
      <td>98040</td>
      <td>47.5653</td>
      <td>-122.223</td>
      <td>2290</td>
      <td>10125</td>
    </tr>
  </tbody>
</table>
<p>454 rows × 21 columns</p>
</div>



It seems like only 454 rows out of 21,597 have ? as their value. Once again being conservative I'm going to assume that these houses/apartments don't have a basement and replace the ? with 0's and change the dtype to int.


```python
df['sqft_basement'] = df['sqft_basement'].map(lambda x: x.replace('?', '0') if x=='?' else x)
# df['sqft_basement'] = df['sqft_basement'].map(lambda x: int(float(x)))
df['sqft_basement'] = df['sqft_basement'].astype('float').astype('int64')
```


```python
df['sqft_basement'].dtype
```




    dtype('int64')




```python
df['renovated'] = df['yr_renovated']!=0
df['renovated'] = df['renovated'].astype('int')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>renovated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['has_basement'] = df['sqft_basement']!=0
df['has_basement'] = df['has_basement'].astype('int')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>renovated</th>
      <th>has_basement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['id'].duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>renovated</th>
      <th>has_basement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>94</th>
      <td>6021501535</td>
      <td>12/23/2014</td>
      <td>700000.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1580</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1290</td>
      <td>290</td>
      <td>1939</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.6870</td>
      <td>-122.386</td>
      <td>1570</td>
      <td>4500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>314</th>
      <td>4139480200</td>
      <td>12/9/2014</td>
      <td>1400000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>4290</td>
      <td>12103</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>2690</td>
      <td>1600</td>
      <td>1997</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5503</td>
      <td>-122.102</td>
      <td>3860</td>
      <td>11244</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>325</th>
      <td>7520000520</td>
      <td>3/11/2015</td>
      <td>240500.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1240</td>
      <td>12092</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>960</td>
      <td>280</td>
      <td>1922</td>
      <td>1984.0</td>
      <td>98146</td>
      <td>47.4957</td>
      <td>-122.352</td>
      <td>1820</td>
      <td>7460</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>346</th>
      <td>3969300030</td>
      <td>12/29/2014</td>
      <td>239900.0</td>
      <td>4</td>
      <td>1.00</td>
      <td>1000</td>
      <td>7134</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1000</td>
      <td>0</td>
      <td>1943</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.4897</td>
      <td>-122.240</td>
      <td>1020</td>
      <td>7138</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>372</th>
      <td>2231500030</td>
      <td>3/24/2015</td>
      <td>530000.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>2180</td>
      <td>10754</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1100</td>
      <td>1080</td>
      <td>1954</td>
      <td>0.0</td>
      <td>98133</td>
      <td>47.7711</td>
      <td>-122.341</td>
      <td>1810</td>
      <td>6929</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20165</th>
      <td>7853400250</td>
      <td>2/19/2015</td>
      <td>645000.0</td>
      <td>4</td>
      <td>3.50</td>
      <td>2910</td>
      <td>5260</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>2910</td>
      <td>0</td>
      <td>2012</td>
      <td>0.0</td>
      <td>98065</td>
      <td>47.5168</td>
      <td>-121.883</td>
      <td>2910</td>
      <td>5260</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20597</th>
      <td>2724049222</td>
      <td>12/1/2014</td>
      <td>220000.0</td>
      <td>2</td>
      <td>2.50</td>
      <td>1000</td>
      <td>1092</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>990</td>
      <td>10</td>
      <td>2004</td>
      <td>0.0</td>
      <td>98118</td>
      <td>47.5419</td>
      <td>-122.271</td>
      <td>1330</td>
      <td>1466</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20654</th>
      <td>8564860270</td>
      <td>3/30/2015</td>
      <td>502000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2680</td>
      <td>5539</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2680</td>
      <td>0</td>
      <td>2013</td>
      <td>0.0</td>
      <td>98045</td>
      <td>47.4759</td>
      <td>-121.734</td>
      <td>2680</td>
      <td>5992</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20764</th>
      <td>6300000226</td>
      <td>5/4/2015</td>
      <td>380000.0</td>
      <td>4</td>
      <td>1.00</td>
      <td>1200</td>
      <td>2171</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1200</td>
      <td>0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98133</td>
      <td>47.7076</td>
      <td>-122.342</td>
      <td>1130</td>
      <td>1598</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21565</th>
      <td>7853420110</td>
      <td>5/4/2015</td>
      <td>625000.0</td>
      <td>3</td>
      <td>3.00</td>
      <td>2780</td>
      <td>6000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>9</td>
      <td>2780</td>
      <td>0</td>
      <td>2013</td>
      <td>0.0</td>
      <td>98065</td>
      <td>47.5184</td>
      <td>-121.886</td>
      <td>2850</td>
      <td>6000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 23 columns</p>
</div>



There are duplicated entries for the same houses but since these are additional sales of the same house, they are valid data points and there is no reason to drop them.

## Linearity Check

### Checking for Linearity of Parameters


```python
import seaborn as sns

def plot(df, target='price'):
    fig, ax = plt.subplots(nrows = len(df.columns), figsize=(10,200))
    
    for i, col in enumerate(df.columns):
#         sns.lmplot(x=col, y=target, data=df)
        ax[i].scatter(df[col], df[target])
        ax[i].set_xlabel(col)
        ax[i].set_ylabel(target)
        ax[i].set_title(f"{col} vs. {target}")
        
```


```python
plot(df=df, target='price')
```


    
![png](output_46_0.png)
    



```python
sns.scatterplot(x=df['zipcode'], y=df['price']);
```


    
![png](output_47_0.png)
    


Zipcode is clearly a categorical column with no linear relationship with our target: 'price'. In order to keep this information in our model, we need to one hot encode this column.


```python
encoder = OneHotEncoder(sparse=False, drop='first')
cat_cols=['zipcode']
data_ohe = encoder.fit_transform(df[cat_cols])
df_ohe = pd.DataFrame(data_ohe, columns=encoder.get_feature_names(cat_cols), index=df.index)
df_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 69 columns</p>
</div>




```python
df_ohe = pd.concat([df.drop('zipcode', axis=1), df_ohe], axis=1)
df_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>...</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0.0</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>263000018</td>
      <td>5/21/2014</td>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1509</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>6600060120</td>
      <td>2/23/2015</td>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>1830</td>
      <td>7200</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>1523300141</td>
      <td>6/23/2014</td>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>47.5944</td>
      <td>-122.299</td>
      <td>1020</td>
      <td>2007</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>291310100</td>
      <td>1/16/2015</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>0</td>
      <td>2004</td>
      <td>0.0</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410</td>
      <td>1287</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>1523300157</td>
      <td>10/15/2014</td>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>1020</td>
      <td>1357</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 91 columns</p>
</div>



We are ready to fit the data with a linear regression model now. Since the building and improving on the model will be an iterative process, we can write a function that will not only create the model but also show us our QQ Plot as well as the residual information so we can check for normality and homoscedasticity.

## Initial Model Prior to Addressing Multicollinearity


```python
def model_lin_reg(df=df_clean, target='price'):
    
    features = ' + '.join(df.drop(target, axis=1).columns)
    f  = f"{target}~"+features
    model = smf.ols(f, df).fit()
    display(model.summary())
    fig, ax = plt.subplots(ncols=2, figsize=(15,5))
    sm.graphics.qqplot(model.resid,line='45',fit=True, ax=ax[0])
    sns.scatterplot(x=model.predict(df, transform=True), y=model.resid, ax=ax[1])
    ax[1].set_ylabel('Residuals')
    ax[1].set_xlabel('Predicted')
    plt.axhline();
    return model
```


```python
model_lin_reg(df=df_ohe)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.813</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.809</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   201.2</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>16:09:16</td>     <th>  Log-Likelihood:    </th> <td>-2.8926e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th>  <td>5.794e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21138</td>      <th>  BIC:               </th>  <td>5.831e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>   458</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>          <td>-2.545e+07</td> <td> 6.19e+06</td> <td>   -4.112</td> <td> 0.000</td> <td>-3.76e+07</td> <td>-1.33e+07</td>
</tr>
<tr>
  <th>date[T.1/12/2015]</th>  <td> 6.803e+04</td> <td> 1.62e+05</td> <td>    0.419</td> <td> 0.675</td> <td> -2.5e+05</td> <td> 3.87e+05</td>
</tr>
<tr>
  <th>date[T.1/13/2015]</th>  <td> 7.407e+04</td> <td> 1.62e+05</td> <td>    0.456</td> <td> 0.648</td> <td>-2.44e+05</td> <td> 3.92e+05</td>
</tr>
<tr>
  <th>date[T.1/14/2015]</th>  <td> 3.766e+04</td> <td> 1.62e+05</td> <td>    0.232</td> <td> 0.816</td> <td> -2.8e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>date[T.1/15/2015]</th>  <td> 5.788e+04</td> <td> 1.62e+05</td> <td>    0.357</td> <td> 0.721</td> <td> -2.6e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.1/16/2015]</th>  <td> 1.319e+04</td> <td> 1.62e+05</td> <td>    0.081</td> <td> 0.935</td> <td>-3.04e+05</td> <td> 3.31e+05</td>
</tr>
<tr>
  <th>date[T.1/17/2015]</th>  <td> 1.532e+05</td> <td> 2.27e+05</td> <td>    0.675</td> <td> 0.500</td> <td>-2.92e+05</td> <td> 5.98e+05</td>
</tr>
<tr>
  <th>date[T.1/19/2015]</th>  <td>-1.373e+04</td> <td>  1.7e+05</td> <td>   -0.081</td> <td> 0.936</td> <td>-3.48e+05</td> <td>  3.2e+05</td>
</tr>
<tr>
  <th>date[T.1/2/2015]</th>   <td> 7.584e+04</td> <td> 1.62e+05</td> <td>    0.467</td> <td> 0.640</td> <td>-2.42e+05</td> <td> 3.94e+05</td>
</tr>
<tr>
  <th>date[T.1/20/2015]</th>  <td> 7.568e+04</td> <td> 1.62e+05</td> <td>    0.466</td> <td> 0.641</td> <td>-2.42e+05</td> <td> 3.94e+05</td>
</tr>
<tr>
  <th>date[T.1/21/2015]</th>  <td>  7.04e+04</td> <td> 1.62e+05</td> <td>    0.435</td> <td> 0.664</td> <td>-2.47e+05</td> <td> 3.88e+05</td>
</tr>
<tr>
  <th>date[T.1/22/2015]</th>  <td> 5.225e+04</td> <td> 1.62e+05</td> <td>    0.322</td> <td> 0.747</td> <td>-2.66e+05</td> <td>  3.7e+05</td>
</tr>
<tr>
  <th>date[T.1/23/2015]</th>  <td> 3.699e+04</td> <td> 1.62e+05</td> <td>    0.228</td> <td> 0.820</td> <td>-2.81e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>date[T.1/24/2015]</th>  <td>-1.045e+05</td> <td>  1.8e+05</td> <td>   -0.582</td> <td> 0.560</td> <td>-4.56e+05</td> <td> 2.47e+05</td>
</tr>
<tr>
  <th>date[T.1/25/2015]</th>  <td> 3.207e+04</td> <td> 1.85e+05</td> <td>    0.173</td> <td> 0.863</td> <td>-3.31e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.1/26/2015]</th>  <td> 1.453e+04</td> <td> 1.62e+05</td> <td>    0.090</td> <td> 0.929</td> <td>-3.03e+05</td> <td> 3.33e+05</td>
</tr>
<tr>
  <th>date[T.1/27/2015]</th>  <td> 7.388e+04</td> <td> 1.62e+05</td> <td>    0.456</td> <td> 0.648</td> <td>-2.44e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.1/28/2015]</th>  <td> 4.985e+04</td> <td> 1.62e+05</td> <td>    0.308</td> <td> 0.758</td> <td>-2.67e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>date[T.1/29/2015]</th>  <td> 7.743e+04</td> <td> 1.62e+05</td> <td>    0.477</td> <td> 0.634</td> <td>-2.41e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>date[T.1/30/2015]</th>  <td>  1.13e+05</td> <td> 1.63e+05</td> <td>    0.694</td> <td> 0.488</td> <td>-2.06e+05</td> <td> 4.32e+05</td>
</tr>
<tr>
  <th>date[T.1/31/2015]</th>  <td>-6.142e+04</td> <td> 2.27e+05</td> <td>   -0.270</td> <td> 0.787</td> <td>-5.07e+05</td> <td> 3.84e+05</td>
</tr>
<tr>
  <th>date[T.1/5/2015]</th>   <td> 6.731e+04</td> <td> 1.62e+05</td> <td>    0.416</td> <td> 0.678</td> <td> -2.5e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.1/6/2015]</th>   <td> 7.594e+04</td> <td> 1.62e+05</td> <td>    0.467</td> <td> 0.640</td> <td>-2.42e+05</td> <td> 3.94e+05</td>
</tr>
<tr>
  <th>date[T.1/7/2015]</th>   <td> 7.751e+04</td> <td> 1.62e+05</td> <td>    0.478</td> <td> 0.632</td> <td> -2.4e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.1/8/2015]</th>   <td> 3.653e+04</td> <td> 1.62e+05</td> <td>    0.225</td> <td> 0.822</td> <td>-2.81e+05</td> <td> 3.54e+05</td>
</tr>
<tr>
  <th>date[T.1/9/2015]</th>   <td> 1.119e+05</td> <td> 1.63e+05</td> <td>    0.687</td> <td> 0.492</td> <td>-2.07e+05</td> <td> 4.31e+05</td>
</tr>
<tr>
  <th>date[T.10/1/2014]</th>  <td> 6.697e+04</td> <td> 1.62e+05</td> <td>    0.415</td> <td> 0.678</td> <td> -2.5e+05</td> <td> 3.84e+05</td>
</tr>
<tr>
  <th>date[T.10/10/2014]</th> <td> 5.141e+04</td> <td> 1.62e+05</td> <td>    0.318</td> <td> 0.750</td> <td>-2.65e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.10/11/2014]</th> <td> 6.385e+05</td> <td> 1.97e+05</td> <td>    3.240</td> <td> 0.001</td> <td> 2.52e+05</td> <td> 1.02e+06</td>
</tr>
<tr>
  <th>date[T.10/12/2014]</th> <td> 7.394e+04</td> <td> 1.86e+05</td> <td>    0.399</td> <td> 0.690</td> <td> -2.9e+05</td> <td> 4.38e+05</td>
</tr>
<tr>
  <th>date[T.10/13/2014]</th> <td> 1.023e+05</td> <td> 1.62e+05</td> <td>    0.632</td> <td> 0.527</td> <td>-2.15e+05</td> <td> 4.19e+05</td>
</tr>
<tr>
  <th>date[T.10/14/2014]</th> <td> 5.965e+04</td> <td> 1.61e+05</td> <td>    0.369</td> <td> 0.712</td> <td>-2.57e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.10/15/2014]</th> <td>  3.97e+04</td> <td> 1.61e+05</td> <td>    0.246</td> <td> 0.806</td> <td>-2.77e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.10/16/2014]</th> <td> 4.875e+04</td> <td> 1.61e+05</td> <td>    0.302</td> <td> 0.763</td> <td>-2.68e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.10/17/2014]</th> <td> 5.381e+04</td> <td> 1.62e+05</td> <td>    0.333</td> <td> 0.739</td> <td>-2.63e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>date[T.10/18/2014]</th> <td> 1.243e+05</td> <td> 1.73e+05</td> <td>    0.716</td> <td> 0.474</td> <td>-2.16e+05</td> <td> 4.64e+05</td>
</tr>
<tr>
  <th>date[T.10/19/2014]</th> <td> 5.763e+04</td> <td>  1.8e+05</td> <td>    0.321</td> <td> 0.748</td> <td>-2.94e+05</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>date[T.10/2/2014]</th>  <td> 4.299e+04</td> <td> 1.62e+05</td> <td>    0.266</td> <td> 0.790</td> <td>-2.74e+05</td> <td>  3.6e+05</td>
</tr>
<tr>
  <th>date[T.10/20/2014]</th> <td> 8.606e+04</td> <td> 1.62e+05</td> <td>    0.533</td> <td> 0.594</td> <td>-2.31e+05</td> <td> 4.03e+05</td>
</tr>
<tr>
  <th>date[T.10/21/2014]</th> <td> 3.387e+04</td> <td> 1.61e+05</td> <td>    0.210</td> <td> 0.834</td> <td>-2.83e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>date[T.10/22/2014]</th> <td> 4.807e+04</td> <td> 1.62e+05</td> <td>    0.298</td> <td> 0.766</td> <td>-2.69e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.10/23/2014]</th> <td> 3.916e+04</td> <td> 1.62e+05</td> <td>    0.242</td> <td> 0.809</td> <td>-2.78e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.10/24/2014]</th> <td> 2.043e+04</td> <td> 1.62e+05</td> <td>    0.126</td> <td> 0.900</td> <td>-2.97e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>date[T.10/25/2014]</th> <td> 1.032e+05</td> <td> 1.76e+05</td> <td>    0.587</td> <td> 0.557</td> <td>-2.42e+05</td> <td> 4.48e+05</td>
</tr>
<tr>
  <th>date[T.10/26/2014]</th> <td>-5294.3178</td> <td>  1.8e+05</td> <td>   -0.029</td> <td> 0.976</td> <td>-3.57e+05</td> <td> 3.47e+05</td>
</tr>
<tr>
  <th>date[T.10/27/2014]</th> <td> 2.828e+04</td> <td> 1.61e+05</td> <td>    0.175</td> <td> 0.861</td> <td>-2.88e+05</td> <td> 3.45e+05</td>
</tr>
<tr>
  <th>date[T.10/28/2014]</th> <td> 4.598e+04</td> <td> 1.61e+05</td> <td>    0.285</td> <td> 0.776</td> <td> -2.7e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.10/29/2014]</th> <td> 7.268e+04</td> <td> 1.61e+05</td> <td>    0.450</td> <td> 0.653</td> <td>-2.44e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.10/3/2014]</th>  <td> 4.948e+04</td> <td> 1.62e+05</td> <td>    0.306</td> <td> 0.760</td> <td>-2.68e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>date[T.10/30/2014]</th> <td>   8.9e+04</td> <td> 1.62e+05</td> <td>    0.551</td> <td> 0.582</td> <td>-2.28e+05</td> <td> 4.06e+05</td>
</tr>
<tr>
  <th>date[T.10/31/2014]</th> <td> 2.422e+04</td> <td> 1.63e+05</td> <td>    0.149</td> <td> 0.882</td> <td>-2.94e+05</td> <td> 3.43e+05</td>
</tr>
<tr>
  <th>date[T.10/4/2014]</th>  <td> 4334.6301</td> <td>  1.8e+05</td> <td>    0.024</td> <td> 0.981</td> <td>-3.48e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.10/5/2014]</th>  <td>-2239.1184</td> <td> 1.85e+05</td> <td>   -0.012</td> <td> 0.990</td> <td>-3.66e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>date[T.10/6/2014]</th>  <td> 3.892e+04</td> <td> 1.62e+05</td> <td>    0.241</td> <td> 0.810</td> <td>-2.78e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.10/7/2014]</th>  <td> 5.128e+04</td> <td> 1.61e+05</td> <td>    0.318</td> <td> 0.751</td> <td>-2.65e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.10/8/2014]</th>  <td> 3.032e+04</td> <td> 1.62e+05</td> <td>    0.188</td> <td> 0.851</td> <td>-2.87e+05</td> <td> 3.47e+05</td>
</tr>
<tr>
  <th>date[T.10/9/2014]</th>  <td> 4.504e+04</td> <td> 1.62e+05</td> <td>    0.279</td> <td> 0.780</td> <td>-2.72e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.11/1/2014]</th>  <td>  7.81e+04</td> <td> 1.76e+05</td> <td>    0.444</td> <td> 0.657</td> <td>-2.67e+05</td> <td> 4.23e+05</td>
</tr>
<tr>
  <th>date[T.11/10/2014]</th> <td> 8.014e+04</td> <td> 1.62e+05</td> <td>    0.496</td> <td> 0.620</td> <td>-2.37e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>date[T.11/11/2014]</th> <td> 2.938e+04</td> <td> 1.62e+05</td> <td>    0.181</td> <td> 0.856</td> <td>-2.89e+05</td> <td> 3.47e+05</td>
</tr>
<tr>
  <th>date[T.11/12/2014]</th> <td> 2.512e+04</td> <td> 1.62e+05</td> <td>    0.155</td> <td> 0.876</td> <td>-2.92e+05</td> <td> 3.42e+05</td>
</tr>
<tr>
  <th>date[T.11/13/2014]</th> <td> 5.822e+04</td> <td> 1.61e+05</td> <td>    0.361</td> <td> 0.718</td> <td>-2.58e+05</td> <td> 3.75e+05</td>
</tr>
<tr>
  <th>date[T.11/14/2014]</th> <td> 6.781e+04</td> <td> 1.62e+05</td> <td>    0.419</td> <td> 0.675</td> <td>-2.49e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.11/15/2014]</th> <td> 1.537e+05</td> <td> 1.97e+05</td> <td>    0.782</td> <td> 0.435</td> <td>-2.32e+05</td> <td> 5.39e+05</td>
</tr>
<tr>
  <th>date[T.11/16/2014]</th> <td>-6662.0780</td> <td> 1.85e+05</td> <td>   -0.036</td> <td> 0.971</td> <td> -3.7e+05</td> <td> 3.57e+05</td>
</tr>
<tr>
  <th>date[T.11/17/2014]</th> <td> 3.116e+04</td> <td> 1.62e+05</td> <td>    0.193</td> <td> 0.847</td> <td>-2.85e+05</td> <td> 3.48e+05</td>
</tr>
<tr>
  <th>date[T.11/18/2014]</th> <td> 6.164e+04</td> <td> 1.61e+05</td> <td>    0.382</td> <td> 0.703</td> <td>-2.55e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>date[T.11/19/2014]</th> <td> 8.054e+04</td> <td> 1.62e+05</td> <td>    0.499</td> <td> 0.618</td> <td>-2.36e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>date[T.11/2/2014]</th>  <td> 1.222e+05</td> <td> 2.27e+05</td> <td>    0.538</td> <td> 0.591</td> <td>-3.23e+05</td> <td> 5.67e+05</td>
</tr>
<tr>
  <th>date[T.11/20/2014]</th> <td> 5.771e+04</td> <td> 1.62e+05</td> <td>    0.357</td> <td> 0.721</td> <td>-2.59e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>date[T.11/21/2014]</th> <td> 2.835e+04</td> <td> 1.62e+05</td> <td>    0.176</td> <td> 0.861</td> <td>-2.88e+05</td> <td> 3.45e+05</td>
</tr>
<tr>
  <th>date[T.11/22/2014]</th> <td> 7.583e+04</td> <td> 1.72e+05</td> <td>    0.442</td> <td> 0.659</td> <td>-2.61e+05</td> <td> 4.12e+05</td>
</tr>
<tr>
  <th>date[T.11/23/2014]</th> <td> 1.032e+05</td> <td>  1.8e+05</td> <td>    0.575</td> <td> 0.565</td> <td>-2.49e+05</td> <td> 4.55e+05</td>
</tr>
<tr>
  <th>date[T.11/24/2014]</th> <td> 7.264e+04</td> <td> 1.62e+05</td> <td>    0.450</td> <td> 0.653</td> <td>-2.44e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.11/25/2014]</th> <td> 5.312e+04</td> <td> 1.62e+05</td> <td>    0.329</td> <td> 0.743</td> <td>-2.64e+05</td> <td>  3.7e+05</td>
</tr>
<tr>
  <th>date[T.11/26/2014]</th> <td> 5.529e+04</td> <td> 1.62e+05</td> <td>    0.341</td> <td> 0.733</td> <td>-2.63e+05</td> <td> 3.73e+05</td>
</tr>
<tr>
  <th>date[T.11/28/2014]</th> <td> 6.328e+04</td> <td> 1.97e+05</td> <td>    0.322</td> <td> 0.748</td> <td>-3.22e+05</td> <td> 4.49e+05</td>
</tr>
<tr>
  <th>date[T.11/29/2014]</th> <td> 7.361e+04</td> <td> 1.85e+05</td> <td>    0.397</td> <td> 0.691</td> <td> -2.9e+05</td> <td> 4.37e+05</td>
</tr>
<tr>
  <th>date[T.11/3/2014]</th>  <td> 4.359e+04</td> <td> 1.62e+05</td> <td>    0.270</td> <td> 0.787</td> <td>-2.73e+05</td> <td>  3.6e+05</td>
</tr>
<tr>
  <th>date[T.11/30/2014]</th> <td> 2.687e+05</td> <td> 2.27e+05</td> <td>    1.183</td> <td> 0.237</td> <td>-1.77e+05</td> <td> 7.14e+05</td>
</tr>
<tr>
  <th>date[T.11/4/2014]</th>  <td>  4.77e+04</td> <td> 1.62e+05</td> <td>    0.295</td> <td> 0.768</td> <td>-2.69e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.11/5/2014]</th>  <td> 4.512e+04</td> <td> 1.62e+05</td> <td>    0.279</td> <td> 0.780</td> <td>-2.72e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.11/6/2014]</th>  <td>  6.82e+04</td> <td> 1.62e+05</td> <td>    0.422</td> <td> 0.673</td> <td>-2.49e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.11/7/2014]</th>  <td> 3.051e+04</td> <td> 1.62e+05</td> <td>    0.189</td> <td> 0.850</td> <td>-2.86e+05</td> <td> 3.47e+05</td>
</tr>
<tr>
  <th>date[T.11/8/2014]</th>  <td> 1.371e+05</td> <td> 1.76e+05</td> <td>    0.780</td> <td> 0.436</td> <td>-2.08e+05</td> <td> 4.82e+05</td>
</tr>
<tr>
  <th>date[T.11/9/2014]</th>  <td>-1.242e+04</td> <td> 1.85e+05</td> <td>   -0.067</td> <td> 0.947</td> <td>-3.76e+05</td> <td> 3.51e+05</td>
</tr>
<tr>
  <th>date[T.12/1/2014]</th>  <td> 2.633e+04</td> <td> 1.61e+05</td> <td>    0.163</td> <td> 0.870</td> <td> -2.9e+05</td> <td> 3.43e+05</td>
</tr>
<tr>
  <th>date[T.12/10/2014]</th> <td> 5.943e+04</td> <td> 1.62e+05</td> <td>    0.368</td> <td> 0.713</td> <td>-2.57e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.12/11/2014]</th> <td> 4.551e+04</td> <td> 1.62e+05</td> <td>    0.282</td> <td> 0.778</td> <td>-2.71e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.12/12/2014]</th> <td> 6.022e+04</td> <td> 1.62e+05</td> <td>    0.372</td> <td> 0.710</td> <td>-2.57e+05</td> <td> 3.77e+05</td>
</tr>
<tr>
  <th>date[T.12/13/2014]</th> <td> 3.757e+04</td> <td>  1.8e+05</td> <td>    0.209</td> <td> 0.834</td> <td>-3.14e+05</td> <td>  3.9e+05</td>
</tr>
<tr>
  <th>date[T.12/14/2014]</th> <td> 8.817e+04</td> <td>  1.8e+05</td> <td>    0.491</td> <td> 0.623</td> <td>-2.64e+05</td> <td>  4.4e+05</td>
</tr>
<tr>
  <th>date[T.12/15/2014]</th> <td> 3.594e+04</td> <td> 1.62e+05</td> <td>    0.222</td> <td> 0.824</td> <td>-2.81e+05</td> <td> 3.53e+05</td>
</tr>
<tr>
  <th>date[T.12/16/2014]</th> <td> 5.408e+04</td> <td> 1.62e+05</td> <td>    0.334</td> <td> 0.738</td> <td>-2.63e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>date[T.12/17/2014]</th> <td> 5.331e+04</td> <td> 1.62e+05</td> <td>    0.329</td> <td> 0.742</td> <td>-2.64e+05</td> <td>  3.7e+05</td>
</tr>
<tr>
  <th>date[T.12/18/2014]</th> <td> 4.639e+04</td> <td> 1.62e+05</td> <td>    0.287</td> <td> 0.774</td> <td>-2.71e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>date[T.12/19/2014]</th> <td> 1.028e+05</td> <td> 1.62e+05</td> <td>    0.633</td> <td> 0.527</td> <td>-2.16e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>date[T.12/2/2014]</th>  <td>  2.78e+04</td> <td> 1.61e+05</td> <td>    0.172</td> <td> 0.863</td> <td>-2.89e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>date[T.12/20/2014]</th> <td> 1.079e+04</td> <td>  1.7e+05</td> <td>    0.063</td> <td> 0.950</td> <td>-3.23e+05</td> <td> 3.45e+05</td>
</tr>
<tr>
  <th>date[T.12/21/2014]</th> <td> 5.639e+04</td> <td> 1.97e+05</td> <td>    0.287</td> <td> 0.774</td> <td>-3.29e+05</td> <td> 4.42e+05</td>
</tr>
<tr>
  <th>date[T.12/22/2014]</th> <td> 4.044e+04</td> <td> 1.62e+05</td> <td>    0.250</td> <td> 0.803</td> <td>-2.77e+05</td> <td> 3.58e+05</td>
</tr>
<tr>
  <th>date[T.12/23/2014]</th> <td> 5.606e+04</td> <td> 1.62e+05</td> <td>    0.346</td> <td> 0.729</td> <td>-2.61e+05</td> <td> 3.73e+05</td>
</tr>
<tr>
  <th>date[T.12/24/2014]</th> <td> 7.062e+04</td> <td> 1.63e+05</td> <td>    0.433</td> <td> 0.665</td> <td>-2.49e+05</td> <td>  3.9e+05</td>
</tr>
<tr>
  <th>date[T.12/26/2014]</th> <td> 4.028e+04</td> <td> 1.63e+05</td> <td>    0.248</td> <td> 0.804</td> <td>-2.79e+05</td> <td> 3.59e+05</td>
</tr>
<tr>
  <th>date[T.12/27/2014]</th> <td> 2.754e+04</td> <td> 1.97e+05</td> <td>    0.140</td> <td> 0.889</td> <td>-3.58e+05</td> <td> 4.13e+05</td>
</tr>
<tr>
  <th>date[T.12/29/2014]</th> <td> 5.676e+04</td> <td> 1.62e+05</td> <td>    0.351</td> <td> 0.726</td> <td> -2.6e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>date[T.12/3/2014]</th>  <td> 6.593e+04</td> <td> 1.62e+05</td> <td>    0.408</td> <td> 0.683</td> <td>-2.51e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>date[T.12/30/2014]</th> <td> 1.047e+05</td> <td> 1.62e+05</td> <td>    0.645</td> <td> 0.519</td> <td>-2.14e+05</td> <td> 4.23e+05</td>
</tr>
<tr>
  <th>date[T.12/31/2014]</th> <td> 7.635e+04</td> <td> 1.62e+05</td> <td>    0.470</td> <td> 0.638</td> <td>-2.42e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.12/4/2014]</th>  <td> 4.659e+04</td> <td> 1.62e+05</td> <td>    0.288</td> <td> 0.773</td> <td> -2.7e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>date[T.12/5/2014]</th>  <td> 8.073e+04</td> <td> 1.62e+05</td> <td>    0.499</td> <td> 0.618</td> <td>-2.37e+05</td> <td> 3.98e+05</td>
</tr>
<tr>
  <th>date[T.12/6/2014]</th>  <td> 4.309e+04</td> <td> 1.73e+05</td> <td>    0.248</td> <td> 0.804</td> <td>-2.97e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>date[T.12/7/2014]</th>  <td>-2.062e+04</td> <td> 1.97e+05</td> <td>   -0.105</td> <td> 0.916</td> <td>-4.06e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.12/8/2014]</th>  <td>   6.5e+04</td> <td> 1.62e+05</td> <td>    0.402</td> <td> 0.688</td> <td>-2.52e+05</td> <td> 3.82e+05</td>
</tr>
<tr>
  <th>date[T.12/9/2014]</th>  <td>  6.11e+04</td> <td> 1.62e+05</td> <td>    0.378</td> <td> 0.705</td> <td>-2.56e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>date[T.2/1/2015]</th>   <td> 1.146e+05</td> <td> 1.97e+05</td> <td>    0.583</td> <td> 0.560</td> <td>-2.71e+05</td> <td>    5e+05</td>
</tr>
<tr>
  <th>date[T.2/10/2015]</th>  <td> 4.414e+04</td> <td> 1.62e+05</td> <td>    0.272</td> <td> 0.785</td> <td>-2.73e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.2/11/2015]</th>  <td> 7.121e+04</td> <td> 1.62e+05</td> <td>    0.440</td> <td> 0.660</td> <td>-2.46e+05</td> <td> 3.88e+05</td>
</tr>
<tr>
  <th>date[T.2/12/2015]</th>  <td> 7.664e+04</td> <td> 1.62e+05</td> <td>    0.472</td> <td> 0.637</td> <td>-2.41e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.2/13/2015]</th>  <td> 8.436e+04</td> <td> 1.62e+05</td> <td>    0.522</td> <td> 0.602</td> <td>-2.33e+05</td> <td> 4.01e+05</td>
</tr>
<tr>
  <th>date[T.2/14/2015]</th>  <td> 2.728e+04</td> <td> 1.85e+05</td> <td>    0.147</td> <td> 0.883</td> <td>-3.36e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.2/15/2015]</th>  <td> 9.796e+04</td> <td> 2.27e+05</td> <td>    0.431</td> <td> 0.666</td> <td>-3.47e+05</td> <td> 5.43e+05</td>
</tr>
<tr>
  <th>date[T.2/16/2015]</th>  <td> 5.582e+04</td> <td> 1.73e+05</td> <td>    0.322</td> <td> 0.748</td> <td>-2.84e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>date[T.2/17/2015]</th>  <td> 5.242e+04</td> <td> 1.62e+05</td> <td>    0.324</td> <td> 0.746</td> <td>-2.65e+05</td> <td> 3.69e+05</td>
</tr>
<tr>
  <th>date[T.2/18/2015]</th>  <td> 5.559e+04</td> <td> 1.61e+05</td> <td>    0.344</td> <td> 0.731</td> <td>-2.61e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>date[T.2/19/2015]</th>  <td> 7.238e+04</td> <td> 1.62e+05</td> <td>    0.448</td> <td> 0.654</td> <td>-2.44e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.2/2/2015]</th>   <td> 7.324e+04</td> <td> 1.62e+05</td> <td>    0.451</td> <td> 0.652</td> <td>-2.45e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.2/20/2015]</th>  <td> 7.888e+04</td> <td> 1.62e+05</td> <td>    0.488</td> <td> 0.626</td> <td>-2.38e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>date[T.2/21/2015]</th>  <td> 1.024e+05</td> <td> 1.85e+05</td> <td>    0.552</td> <td> 0.581</td> <td>-2.61e+05</td> <td> 4.66e+05</td>
</tr>
<tr>
  <th>date[T.2/22/2015]</th>  <td> 2.317e+04</td> <td> 1.67e+05</td> <td>    0.139</td> <td> 0.889</td> <td>-3.03e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>date[T.2/23/2015]</th>  <td> 5.887e+04</td> <td> 1.62e+05</td> <td>    0.364</td> <td> 0.716</td> <td>-2.58e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.2/24/2015]</th>  <td> 5.066e+04</td> <td> 1.61e+05</td> <td>    0.314</td> <td> 0.754</td> <td>-2.66e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>date[T.2/25/2015]</th>  <td> 9.108e+04</td> <td> 1.61e+05</td> <td>    0.564</td> <td> 0.573</td> <td>-2.25e+05</td> <td> 4.07e+05</td>
</tr>
<tr>
  <th>date[T.2/26/2015]</th>  <td> 5.819e+04</td> <td> 1.62e+05</td> <td>    0.359</td> <td> 0.719</td> <td>-2.59e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.2/27/2015]</th>  <td> 8.307e+04</td> <td> 1.62e+05</td> <td>    0.512</td> <td> 0.608</td> <td>-2.35e+05</td> <td> 4.01e+05</td>
</tr>
<tr>
  <th>date[T.2/28/2015]</th>  <td> 1.327e+05</td> <td> 1.76e+05</td> <td>    0.755</td> <td> 0.451</td> <td>-2.12e+05</td> <td> 4.78e+05</td>
</tr>
<tr>
  <th>date[T.2/3/2015]</th>   <td> 3.639e+04</td> <td> 1.62e+05</td> <td>    0.224</td> <td> 0.823</td> <td>-2.82e+05</td> <td> 3.54e+05</td>
</tr>
<tr>
  <th>date[T.2/4/2015]</th>   <td>  7.95e+04</td> <td> 1.62e+05</td> <td>    0.491</td> <td> 0.624</td> <td>-2.38e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>date[T.2/5/2015]</th>   <td> 4.569e+04</td> <td> 1.62e+05</td> <td>    0.282</td> <td> 0.778</td> <td>-2.72e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>date[T.2/6/2015]</th>   <td> 7.031e+04</td> <td> 1.62e+05</td> <td>    0.434</td> <td> 0.665</td> <td>-2.47e+05</td> <td> 3.88e+05</td>
</tr>
<tr>
  <th>date[T.2/7/2015]</th>   <td> 4.109e+04</td> <td> 1.85e+05</td> <td>    0.222</td> <td> 0.825</td> <td>-3.22e+05</td> <td> 4.05e+05</td>
</tr>
<tr>
  <th>date[T.2/9/2015]</th>   <td> 7.098e+04</td> <td> 1.62e+05</td> <td>    0.438</td> <td> 0.661</td> <td>-2.47e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.3/1/2015]</th>   <td> 2.909e+04</td> <td> 1.73e+05</td> <td>    0.168</td> <td> 0.867</td> <td>-3.11e+05</td> <td> 3.69e+05</td>
</tr>
<tr>
  <th>date[T.3/10/2015]</th>  <td> 6.837e+04</td> <td> 1.62e+05</td> <td>    0.423</td> <td> 0.672</td> <td>-2.48e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.3/11/2015]</th>  <td> 7.258e+04</td> <td> 1.61e+05</td> <td>    0.449</td> <td> 0.653</td> <td>-2.44e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.3/12/2015]</th>  <td> 7.413e+04</td> <td> 1.62e+05</td> <td>    0.459</td> <td> 0.647</td> <td>-2.43e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.3/13/2015]</th>  <td> 1.209e+05</td> <td> 1.62e+05</td> <td>    0.748</td> <td> 0.455</td> <td>-1.96e+05</td> <td> 4.38e+05</td>
</tr>
<tr>
  <th>date[T.3/14/2015]</th>  <td> 1.021e+05</td> <td> 1.73e+05</td> <td>    0.588</td> <td> 0.556</td> <td>-2.38e+05</td> <td> 4.42e+05</td>
</tr>
<tr>
  <th>date[T.3/15/2015]</th>  <td> 9.327e+04</td> <td> 1.85e+05</td> <td>    0.503</td> <td> 0.615</td> <td> -2.7e+05</td> <td> 4.57e+05</td>
</tr>
<tr>
  <th>date[T.3/16/2015]</th>  <td> 1.093e+05</td> <td> 1.62e+05</td> <td>    0.677</td> <td> 0.498</td> <td>-2.07e+05</td> <td> 4.26e+05</td>
</tr>
<tr>
  <th>date[T.3/17/2015]</th>  <td> 8.078e+04</td> <td> 1.61e+05</td> <td>    0.500</td> <td> 0.617</td> <td>-2.36e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>date[T.3/18/2015]</th>  <td> 7.683e+04</td> <td> 1.61e+05</td> <td>    0.476</td> <td> 0.634</td> <td> -2.4e+05</td> <td> 3.93e+05</td>
</tr>
<tr>
  <th>date[T.3/19/2015]</th>  <td> 9.148e+04</td> <td> 1.62e+05</td> <td>    0.566</td> <td> 0.572</td> <td>-2.25e+05</td> <td> 4.08e+05</td>
</tr>
<tr>
  <th>date[T.3/2/2015]</th>   <td> 1.286e+05</td> <td> 1.62e+05</td> <td>    0.791</td> <td> 0.429</td> <td> -1.9e+05</td> <td> 4.47e+05</td>
</tr>
<tr>
  <th>date[T.3/20/2015]</th>  <td> 6.824e+04</td> <td> 1.62e+05</td> <td>    0.422</td> <td> 0.673</td> <td>-2.49e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.3/21/2015]</th>  <td> 6.505e+04</td> <td> 1.66e+05</td> <td>    0.392</td> <td> 0.695</td> <td> -2.6e+05</td> <td>  3.9e+05</td>
</tr>
<tr>
  <th>date[T.3/22/2015]</th>  <td> 2.261e+05</td> <td>  1.8e+05</td> <td>    1.259</td> <td> 0.208</td> <td>-1.26e+05</td> <td> 5.78e+05</td>
</tr>
<tr>
  <th>date[T.3/23/2015]</th>  <td> 8.544e+04</td> <td> 1.61e+05</td> <td>    0.529</td> <td> 0.597</td> <td>-2.31e+05</td> <td> 4.02e+05</td>
</tr>
<tr>
  <th>date[T.3/24/2015]</th>  <td> 7.513e+04</td> <td> 1.61e+05</td> <td>    0.466</td> <td> 0.641</td> <td>-2.41e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.3/25/2015]</th>  <td> 8.621e+04</td> <td> 1.61e+05</td> <td>    0.535</td> <td> 0.593</td> <td> -2.3e+05</td> <td> 4.02e+05</td>
</tr>
<tr>
  <th>date[T.3/26/2015]</th>  <td> 1.082e+05</td> <td> 1.61e+05</td> <td>    0.671</td> <td> 0.502</td> <td>-2.08e+05</td> <td> 4.24e+05</td>
</tr>
<tr>
  <th>date[T.3/27/2015]</th>  <td> 7.932e+04</td> <td> 1.61e+05</td> <td>    0.492</td> <td> 0.623</td> <td>-2.37e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.3/28/2015]</th>  <td> 7.601e+04</td> <td>  1.7e+05</td> <td>    0.446</td> <td> 0.655</td> <td>-2.58e+05</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>date[T.3/29/2015]</th>  <td> -1.98e+04</td> <td>  1.7e+05</td> <td>   -0.116</td> <td> 0.907</td> <td>-3.54e+05</td> <td> 3.14e+05</td>
</tr>
<tr>
  <th>date[T.3/3/2015]</th>   <td> 1.295e+05</td> <td> 1.62e+05</td> <td>    0.800</td> <td> 0.424</td> <td>-1.88e+05</td> <td> 4.47e+05</td>
</tr>
<tr>
  <th>date[T.3/30/2015]</th>  <td> 7.789e+04</td> <td> 1.61e+05</td> <td>    0.482</td> <td> 0.630</td> <td>-2.39e+05</td> <td> 3.94e+05</td>
</tr>
<tr>
  <th>date[T.3/31/2015]</th>  <td> 9.818e+04</td> <td> 1.62e+05</td> <td>    0.607</td> <td> 0.544</td> <td>-2.19e+05</td> <td> 4.15e+05</td>
</tr>
<tr>
  <th>date[T.3/4/2015]</th>   <td> 9.591e+04</td> <td> 1.61e+05</td> <td>    0.594</td> <td> 0.552</td> <td> -2.2e+05</td> <td> 4.12e+05</td>
</tr>
<tr>
  <th>date[T.3/5/2015]</th>   <td> 7.057e+04</td> <td> 1.62e+05</td> <td>    0.437</td> <td> 0.662</td> <td>-2.46e+05</td> <td> 3.87e+05</td>
</tr>
<tr>
  <th>date[T.3/6/2015]</th>   <td> 8.419e+04</td> <td> 1.62e+05</td> <td>    0.520</td> <td> 0.603</td> <td>-2.33e+05</td> <td> 4.02e+05</td>
</tr>
<tr>
  <th>date[T.3/7/2015]</th>   <td>  8.48e+04</td> <td> 1.79e+05</td> <td>    0.473</td> <td> 0.637</td> <td>-2.67e+05</td> <td> 4.37e+05</td>
</tr>
<tr>
  <th>date[T.3/8/2015]</th>   <td> 1.941e+05</td> <td> 2.27e+05</td> <td>    0.854</td> <td> 0.393</td> <td>-2.51e+05</td> <td> 6.39e+05</td>
</tr>
<tr>
  <th>date[T.3/9/2015]</th>   <td> 7.361e+04</td> <td> 1.62e+05</td> <td>    0.455</td> <td> 0.649</td> <td>-2.43e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.4/1/2015]</th>   <td> 1.051e+05</td> <td> 1.61e+05</td> <td>    0.651</td> <td> 0.515</td> <td>-2.11e+05</td> <td> 4.22e+05</td>
</tr>
<tr>
  <th>date[T.4/10/2015]</th>  <td> 9.357e+04</td> <td> 1.62e+05</td> <td>    0.579</td> <td> 0.563</td> <td>-2.23e+05</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>date[T.4/11/2015]</th>  <td> 7.553e+04</td> <td> 1.67e+05</td> <td>    0.453</td> <td> 0.650</td> <td>-2.51e+05</td> <td> 4.02e+05</td>
</tr>
<tr>
  <th>date[T.4/12/2015]</th>  <td> 5.507e+04</td> <td> 1.67e+05</td> <td>    0.330</td> <td> 0.741</td> <td>-2.72e+05</td> <td> 3.82e+05</td>
</tr>
<tr>
  <th>date[T.4/13/2015]</th>  <td> 1.319e+05</td> <td> 1.61e+05</td> <td>    0.817</td> <td> 0.414</td> <td>-1.85e+05</td> <td> 4.48e+05</td>
</tr>
<tr>
  <th>date[T.4/14/2015]</th>  <td> 7.981e+04</td> <td> 1.61e+05</td> <td>    0.495</td> <td> 0.621</td> <td>-2.36e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>date[T.4/15/2015]</th>  <td> 1.128e+05</td> <td> 1.62e+05</td> <td>    0.698</td> <td> 0.485</td> <td>-2.04e+05</td> <td>  4.3e+05</td>
</tr>
<tr>
  <th>date[T.4/16/2015]</th>  <td> 7.432e+04</td> <td> 1.62e+05</td> <td>    0.460</td> <td> 0.646</td> <td>-2.42e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.4/17/2015]</th>  <td> 7.595e+04</td> <td> 1.62e+05</td> <td>    0.470</td> <td> 0.638</td> <td>-2.41e+05</td> <td> 3.93e+05</td>
</tr>
<tr>
  <th>date[T.4/18/2015]</th>  <td>  1.07e+05</td> <td> 1.76e+05</td> <td>    0.608</td> <td> 0.543</td> <td>-2.38e+05</td> <td> 4.52e+05</td>
</tr>
<tr>
  <th>date[T.4/19/2015]</th>  <td> 9.229e+04</td> <td> 1.73e+05</td> <td>    0.532</td> <td> 0.595</td> <td>-2.48e+05</td> <td> 4.32e+05</td>
</tr>
<tr>
  <th>date[T.4/2/2015]</th>   <td> 9.088e+04</td> <td> 1.61e+05</td> <td>    0.563</td> <td> 0.573</td> <td>-2.25e+05</td> <td> 4.07e+05</td>
</tr>
<tr>
  <th>date[T.4/20/2015]</th>  <td> 5.602e+04</td> <td> 1.62e+05</td> <td>    0.347</td> <td> 0.729</td> <td>-2.61e+05</td> <td> 3.73e+05</td>
</tr>
<tr>
  <th>date[T.4/21/2015]</th>  <td> 1.008e+05</td> <td> 1.61e+05</td> <td>    0.625</td> <td> 0.532</td> <td>-2.15e+05</td> <td> 4.17e+05</td>
</tr>
<tr>
  <th>date[T.4/22/2015]</th>  <td>  8.13e+04</td> <td> 1.61e+05</td> <td>    0.504</td> <td> 0.614</td> <td>-2.35e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>date[T.4/23/2015]</th>  <td> 8.358e+04</td> <td> 1.61e+05</td> <td>    0.518</td> <td> 0.604</td> <td>-2.33e+05</td> <td>    4e+05</td>
</tr>
<tr>
  <th>date[T.4/24/2015]</th>  <td> 1.005e+05</td> <td> 1.61e+05</td> <td>    0.623</td> <td> 0.533</td> <td>-2.16e+05</td> <td> 4.17e+05</td>
</tr>
<tr>
  <th>date[T.4/25/2015]</th>  <td> 1.168e+05</td> <td> 1.66e+05</td> <td>    0.706</td> <td> 0.480</td> <td>-2.08e+05</td> <td> 4.41e+05</td>
</tr>
<tr>
  <th>date[T.4/26/2015]</th>  <td> 7.091e+04</td> <td> 1.67e+05</td> <td>    0.426</td> <td> 0.670</td> <td>-2.56e+05</td> <td> 3.98e+05</td>
</tr>
<tr>
  <th>date[T.4/27/2015]</th>  <td> 7.666e+04</td> <td> 1.61e+05</td> <td>    0.476</td> <td> 0.634</td> <td>-2.39e+05</td> <td> 3.93e+05</td>
</tr>
<tr>
  <th>date[T.4/28/2015]</th>  <td> 9.376e+04</td> <td> 1.61e+05</td> <td>    0.582</td> <td> 0.561</td> <td>-2.22e+05</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>date[T.4/29/2015]</th>  <td> 1.136e+05</td> <td> 1.61e+05</td> <td>    0.704</td> <td> 0.481</td> <td>-2.02e+05</td> <td>  4.3e+05</td>
</tr>
<tr>
  <th>date[T.4/3/2015]</th>   <td> 1.089e+05</td> <td> 1.62e+05</td> <td>    0.674</td> <td> 0.500</td> <td>-2.08e+05</td> <td> 4.26e+05</td>
</tr>
<tr>
  <th>date[T.4/30/2015]</th>  <td> 9.605e+04</td> <td> 1.62e+05</td> <td>    0.595</td> <td> 0.552</td> <td>-2.21e+05</td> <td> 4.13e+05</td>
</tr>
<tr>
  <th>date[T.4/4/2015]</th>   <td> 1.215e+05</td> <td>  1.8e+05</td> <td>    0.677</td> <td> 0.498</td> <td> -2.3e+05</td> <td> 4.73e+05</td>
</tr>
<tr>
  <th>date[T.4/5/2015]</th>   <td> 8.902e+04</td> <td> 1.73e+05</td> <td>    0.513</td> <td> 0.608</td> <td>-2.51e+05</td> <td> 4.29e+05</td>
</tr>
<tr>
  <th>date[T.4/6/2015]</th>   <td> 7.828e+04</td> <td> 1.62e+05</td> <td>    0.485</td> <td> 0.628</td> <td>-2.38e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.4/7/2015]</th>   <td> 1.232e+05</td> <td> 1.61e+05</td> <td>    0.764</td> <td> 0.445</td> <td>-1.93e+05</td> <td> 4.39e+05</td>
</tr>
<tr>
  <th>date[T.4/8/2015]</th>   <td> 8.854e+04</td> <td> 1.61e+05</td> <td>    0.549</td> <td> 0.583</td> <td>-2.28e+05</td> <td> 4.05e+05</td>
</tr>
<tr>
  <th>date[T.4/9/2015]</th>   <td> 7.992e+04</td> <td> 1.61e+05</td> <td>    0.495</td> <td> 0.621</td> <td>-2.37e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>date[T.5/1/2015]</th>   <td> 1.236e+05</td> <td> 1.62e+05</td> <td>    0.765</td> <td> 0.444</td> <td>-1.93e+05</td> <td>  4.4e+05</td>
</tr>
<tr>
  <th>date[T.5/10/2014]</th>  <td> 7.679e+04</td> <td> 1.76e+05</td> <td>    0.437</td> <td> 0.662</td> <td>-2.68e+05</td> <td> 4.22e+05</td>
</tr>
<tr>
  <th>date[T.5/10/2015]</th>  <td> 5.498e+04</td> <td> 1.97e+05</td> <td>    0.280</td> <td> 0.780</td> <td>-3.31e+05</td> <td> 4.41e+05</td>
</tr>
<tr>
  <th>date[T.5/11/2014]</th>  <td> 9.365e+04</td> <td> 1.97e+05</td> <td>    0.476</td> <td> 0.634</td> <td>-2.92e+05</td> <td> 4.79e+05</td>
</tr>
<tr>
  <th>date[T.5/11/2015]</th>  <td> 1.335e+05</td> <td> 1.63e+05</td> <td>    0.821</td> <td> 0.412</td> <td>-1.85e+05</td> <td> 4.52e+05</td>
</tr>
<tr>
  <th>date[T.5/12/2014]</th>  <td> 1.383e+04</td> <td> 1.62e+05</td> <td>    0.086</td> <td> 0.932</td> <td>-3.03e+05</td> <td> 3.31e+05</td>
</tr>
<tr>
  <th>date[T.5/12/2015]</th>  <td> 1.037e+05</td> <td> 1.62e+05</td> <td>    0.639</td> <td> 0.523</td> <td>-2.14e+05</td> <td> 4.22e+05</td>
</tr>
<tr>
  <th>date[T.5/13/2014]</th>  <td> 6.619e+04</td> <td> 1.61e+05</td> <td>    0.410</td> <td> 0.682</td> <td> -2.5e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>date[T.5/13/2015]</th>  <td> 1.146e+05</td> <td> 1.63e+05</td> <td>    0.703</td> <td> 0.482</td> <td>-2.05e+05</td> <td> 4.34e+05</td>
</tr>
<tr>
  <th>date[T.5/14/2014]</th>  <td> 4.359e+04</td> <td> 1.62e+05</td> <td>    0.270</td> <td> 0.787</td> <td>-2.73e+05</td> <td>  3.6e+05</td>
</tr>
<tr>
  <th>date[T.5/14/2015]</th>  <td> 1.908e+05</td> <td> 1.68e+05</td> <td>    1.138</td> <td> 0.255</td> <td>-1.38e+05</td> <td> 5.19e+05</td>
</tr>
<tr>
  <th>date[T.5/15/2014]</th>  <td> 3.854e+04</td> <td> 1.62e+05</td> <td>    0.239</td> <td> 0.811</td> <td>-2.78e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>date[T.5/15/2015]</th>  <td> 5.578e+04</td> <td> 2.27e+05</td> <td>    0.245</td> <td> 0.806</td> <td> -3.9e+05</td> <td> 5.01e+05</td>
</tr>
<tr>
  <th>date[T.5/16/2014]</th>  <td> 4.014e+04</td> <td> 1.62e+05</td> <td>    0.248</td> <td> 0.804</td> <td>-2.77e+05</td> <td> 3.57e+05</td>
</tr>
<tr>
  <th>date[T.5/17/2014]</th>  <td>-6.369e+04</td> <td> 2.27e+05</td> <td>   -0.280</td> <td> 0.779</td> <td>-5.09e+05</td> <td> 3.82e+05</td>
</tr>
<tr>
  <th>date[T.5/18/2014]</th>  <td> 7.263e+04</td> <td> 1.72e+05</td> <td>    0.423</td> <td> 0.672</td> <td>-2.64e+05</td> <td> 4.09e+05</td>
</tr>
<tr>
  <th>date[T.5/19/2014]</th>  <td> 5.975e+04</td> <td> 1.62e+05</td> <td>    0.370</td> <td> 0.711</td> <td>-2.57e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.5/2/2014]</th>   <td> 4.136e+04</td> <td> 1.62e+05</td> <td>    0.256</td> <td> 0.798</td> <td>-2.76e+05</td> <td> 3.58e+05</td>
</tr>
<tr>
  <th>date[T.5/2/2015]</th>   <td> 1.002e+05</td> <td> 1.73e+05</td> <td>    0.578</td> <td> 0.564</td> <td> -2.4e+05</td> <td>  4.4e+05</td>
</tr>
<tr>
  <th>date[T.5/20/2014]</th>  <td> 4.395e+04</td> <td> 1.61e+05</td> <td>    0.273</td> <td> 0.785</td> <td>-2.72e+05</td> <td>  3.6e+05</td>
</tr>
<tr>
  <th>date[T.5/21/2014]</th>  <td> 4.668e+04</td> <td> 1.61e+05</td> <td>    0.289</td> <td> 0.772</td> <td> -2.7e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>date[T.5/22/2014]</th>  <td> 3.296e+04</td> <td> 1.61e+05</td> <td>    0.204</td> <td> 0.838</td> <td>-2.83e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>date[T.5/23/2014]</th>  <td> 2.925e+04</td> <td> 1.62e+05</td> <td>    0.181</td> <td> 0.856</td> <td>-2.87e+05</td> <td> 3.46e+05</td>
</tr>
<tr>
  <th>date[T.5/24/2014]</th>  <td> 8.123e+04</td> <td> 1.68e+05</td> <td>    0.484</td> <td> 0.628</td> <td>-2.48e+05</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>date[T.5/24/2015]</th>  <td>   6.2e+04</td> <td> 2.27e+05</td> <td>    0.273</td> <td> 0.785</td> <td>-3.83e+05</td> <td> 5.07e+05</td>
</tr>
<tr>
  <th>date[T.5/25/2014]</th>  <td> -3.29e+04</td> <td> 1.76e+05</td> <td>   -0.187</td> <td> 0.852</td> <td>-3.78e+05</td> <td> 3.12e+05</td>
</tr>
<tr>
  <th>date[T.5/26/2014]</th>  <td> 9.481e+04</td> <td>  1.7e+05</td> <td>    0.557</td> <td> 0.578</td> <td>-2.39e+05</td> <td> 4.29e+05</td>
</tr>
<tr>
  <th>date[T.5/27/2014]</th>  <td> 4.602e+04</td> <td> 1.61e+05</td> <td>    0.285</td> <td> 0.775</td> <td> -2.7e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.5/27/2015]</th>  <td> 3.727e+05</td> <td> 2.27e+05</td> <td>    1.641</td> <td> 0.101</td> <td>-7.24e+04</td> <td> 8.18e+05</td>
</tr>
<tr>
  <th>date[T.5/28/2014]</th>  <td>  4.93e+04</td> <td> 1.61e+05</td> <td>    0.306</td> <td> 0.760</td> <td>-2.67e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.5/29/2014]</th>  <td> 7.221e+04</td> <td> 1.62e+05</td> <td>    0.447</td> <td> 0.655</td> <td>-2.45e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.5/3/2014]</th>   <td> 1.532e+05</td> <td>  1.8e+05</td> <td>    0.853</td> <td> 0.394</td> <td>-1.99e+05</td> <td> 5.05e+05</td>
</tr>
<tr>
  <th>date[T.5/3/2015]</th>   <td> 1.184e+05</td> <td> 1.68e+05</td> <td>    0.703</td> <td> 0.482</td> <td>-2.12e+05</td> <td> 4.49e+05</td>
</tr>
<tr>
  <th>date[T.5/30/2014]</th>  <td> 5.897e+04</td> <td> 1.62e+05</td> <td>    0.364</td> <td> 0.716</td> <td>-2.58e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.5/31/2014]</th>  <td> 5.279e+04</td> <td> 1.73e+05</td> <td>    0.304</td> <td> 0.761</td> <td>-2.87e+05</td> <td> 3.93e+05</td>
</tr>
<tr>
  <th>date[T.5/4/2014]</th>   <td>-9806.1064</td> <td> 1.76e+05</td> <td>   -0.056</td> <td> 0.956</td> <td>-3.55e+05</td> <td> 3.35e+05</td>
</tr>
<tr>
  <th>date[T.5/4/2015]</th>   <td>  8.65e+04</td> <td> 1.61e+05</td> <td>    0.536</td> <td> 0.592</td> <td> -2.3e+05</td> <td> 4.03e+05</td>
</tr>
<tr>
  <th>date[T.5/5/2014]</th>   <td> 3.339e+04</td> <td> 1.62e+05</td> <td>    0.207</td> <td> 0.836</td> <td>-2.83e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>date[T.5/5/2015]</th>   <td>   1.1e+05</td> <td> 1.61e+05</td> <td>    0.681</td> <td> 0.496</td> <td>-2.06e+05</td> <td> 4.26e+05</td>
</tr>
<tr>
  <th>date[T.5/6/2014]</th>   <td> 3.796e+04</td> <td> 1.62e+05</td> <td>    0.235</td> <td> 0.814</td> <td>-2.79e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>date[T.5/6/2015]</th>   <td> 1.098e+05</td> <td> 1.61e+05</td> <td>    0.680</td> <td> 0.497</td> <td>-2.07e+05</td> <td> 4.26e+05</td>
</tr>
<tr>
  <th>date[T.5/7/2014]</th>   <td> 5.086e+04</td> <td> 1.61e+05</td> <td>    0.315</td> <td> 0.753</td> <td>-2.66e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>date[T.5/7/2015]</th>   <td> 9.735e+04</td> <td> 1.62e+05</td> <td>    0.602</td> <td> 0.547</td> <td>-2.19e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>date[T.5/8/2014]</th>   <td>  5.61e+04</td> <td> 1.62e+05</td> <td>    0.347</td> <td> 0.728</td> <td>-2.61e+05</td> <td> 3.73e+05</td>
</tr>
<tr>
  <th>date[T.5/8/2015]</th>   <td> 1.121e+05</td> <td> 1.62e+05</td> <td>    0.692</td> <td> 0.489</td> <td>-2.05e+05</td> <td>  4.3e+05</td>
</tr>
<tr>
  <th>date[T.5/9/2014]</th>   <td> 4.127e+04</td> <td> 1.62e+05</td> <td>    0.255</td> <td> 0.798</td> <td>-2.75e+05</td> <td> 3.58e+05</td>
</tr>
<tr>
  <th>date[T.5/9/2015]</th>   <td> 5.752e+04</td> <td> 1.85e+05</td> <td>    0.310</td> <td> 0.756</td> <td>-3.06e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>date[T.6/1/2014]</th>   <td> 1.317e+05</td> <td> 1.72e+05</td> <td>    0.767</td> <td> 0.443</td> <td>-2.05e+05</td> <td> 4.68e+05</td>
</tr>
<tr>
  <th>date[T.6/10/2014]</th>  <td> 5.979e+04</td> <td> 1.61e+05</td> <td>    0.370</td> <td> 0.711</td> <td>-2.57e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.6/11/2014]</th>  <td> 8.144e+04</td> <td> 1.61e+05</td> <td>    0.504</td> <td> 0.614</td> <td>-2.35e+05</td> <td> 3.98e+05</td>
</tr>
<tr>
  <th>date[T.6/12/2014]</th>  <td> 5.128e+04</td> <td> 1.61e+05</td> <td>    0.318</td> <td> 0.751</td> <td>-2.65e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.6/13/2014]</th>  <td> 7.397e+04</td> <td> 1.62e+05</td> <td>    0.458</td> <td> 0.647</td> <td>-2.43e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.6/14/2014]</th>  <td> 8.621e+04</td> <td>  1.7e+05</td> <td>    0.506</td> <td> 0.613</td> <td>-2.48e+05</td> <td>  4.2e+05</td>
</tr>
<tr>
  <th>date[T.6/15/2014]</th>  <td>-2.909e+04</td> <td> 1.72e+05</td> <td>   -0.169</td> <td> 0.865</td> <td>-3.66e+05</td> <td> 3.07e+05</td>
</tr>
<tr>
  <th>date[T.6/16/2014]</th>  <td> 5.303e+04</td> <td> 1.61e+05</td> <td>    0.329</td> <td> 0.742</td> <td>-2.63e+05</td> <td> 3.69e+05</td>
</tr>
<tr>
  <th>date[T.6/17/2014]</th>  <td> 8.244e+04</td> <td> 1.61e+05</td> <td>    0.511</td> <td> 0.609</td> <td>-2.34e+05</td> <td> 3.99e+05</td>
</tr>
<tr>
  <th>date[T.6/18/2014]</th>  <td> 8.762e+04</td> <td> 1.61e+05</td> <td>    0.543</td> <td> 0.587</td> <td>-2.29e+05</td> <td> 4.04e+05</td>
</tr>
<tr>
  <th>date[T.6/19/2014]</th>  <td> 4.913e+04</td> <td> 1.61e+05</td> <td>    0.304</td> <td> 0.761</td> <td>-2.67e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.6/2/2014]</th>   <td> 4.202e+04</td> <td> 1.62e+05</td> <td>    0.260</td> <td> 0.795</td> <td>-2.75e+05</td> <td> 3.59e+05</td>
</tr>
<tr>
  <th>date[T.6/20/2014]</th>  <td> 6.359e+04</td> <td> 1.61e+05</td> <td>    0.394</td> <td> 0.693</td> <td>-2.53e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>date[T.6/21/2014]</th>  <td> 4.655e+04</td> <td>  1.7e+05</td> <td>    0.273</td> <td> 0.785</td> <td>-2.87e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>date[T.6/22/2014]</th>  <td> 3.956e+04</td> <td> 1.67e+05</td> <td>    0.237</td> <td> 0.813</td> <td>-2.88e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>date[T.6/23/2014]</th>  <td> 5.032e+04</td> <td> 1.61e+05</td> <td>    0.312</td> <td> 0.755</td> <td>-2.65e+05</td> <td> 3.66e+05</td>
</tr>
<tr>
  <th>date[T.6/24/2014]</th>  <td> 6.552e+04</td> <td> 1.61e+05</td> <td>    0.406</td> <td> 0.684</td> <td>-2.51e+05</td> <td> 3.82e+05</td>
</tr>
<tr>
  <th>date[T.6/25/2014]</th>  <td> 6.899e+04</td> <td> 1.61e+05</td> <td>    0.428</td> <td> 0.669</td> <td>-2.47e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.6/26/2014]</th>  <td> 4.393e+04</td> <td> 1.61e+05</td> <td>    0.273</td> <td> 0.785</td> <td>-2.72e+05</td> <td>  3.6e+05</td>
</tr>
<tr>
  <th>date[T.6/27/2014]</th>  <td> 2.525e+04</td> <td> 1.61e+05</td> <td>    0.156</td> <td> 0.876</td> <td>-2.91e+05</td> <td> 3.42e+05</td>
</tr>
<tr>
  <th>date[T.6/28/2014]</th>  <td>  9.41e+04</td> <td> 1.67e+05</td> <td>    0.565</td> <td> 0.572</td> <td>-2.33e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>date[T.6/29/2014]</th>  <td>-2.383e+04</td> <td> 1.72e+05</td> <td>   -0.139</td> <td> 0.890</td> <td> -3.6e+05</td> <td> 3.13e+05</td>
</tr>
<tr>
  <th>date[T.6/3/2014]</th>   <td> 5.187e+04</td> <td> 1.61e+05</td> <td>    0.322</td> <td> 0.748</td> <td>-2.64e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.6/30/2014]</th>  <td> 4.908e+04</td> <td> 1.62e+05</td> <td>    0.304</td> <td> 0.761</td> <td>-2.68e+05</td> <td> 3.66e+05</td>
</tr>
<tr>
  <th>date[T.6/4/2014]</th>   <td> 8.561e+04</td> <td> 1.61e+05</td> <td>    0.531</td> <td> 0.596</td> <td>-2.31e+05</td> <td> 4.02e+05</td>
</tr>
<tr>
  <th>date[T.6/5/2014]</th>   <td> 3.913e+04</td> <td> 1.61e+05</td> <td>    0.242</td> <td> 0.808</td> <td>-2.77e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.6/6/2014]</th>   <td> 5.061e+04</td> <td> 1.62e+05</td> <td>    0.313</td> <td> 0.754</td> <td>-2.66e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.6/7/2014]</th>   <td>-3.891e+04</td> <td>  1.8e+05</td> <td>   -0.217</td> <td> 0.828</td> <td>-3.91e+05</td> <td> 3.13e+05</td>
</tr>
<tr>
  <th>date[T.6/8/2014]</th>   <td> 9.254e+04</td> <td> 1.68e+05</td> <td>    0.549</td> <td> 0.583</td> <td>-2.38e+05</td> <td> 4.23e+05</td>
</tr>
<tr>
  <th>date[T.6/9/2014]</th>   <td> 4.664e+04</td> <td> 1.61e+05</td> <td>    0.289</td> <td> 0.773</td> <td> -2.7e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>date[T.7/1/2014]</th>   <td> 7.264e+04</td> <td> 1.61e+05</td> <td>    0.450</td> <td> 0.652</td> <td>-2.43e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.7/10/2014]</th>  <td> 4.922e+04</td> <td> 1.61e+05</td> <td>    0.305</td> <td> 0.760</td> <td>-2.67e+05</td> <td> 3.66e+05</td>
</tr>
<tr>
  <th>date[T.7/11/2014]</th>  <td> 6.856e+04</td> <td> 1.62e+05</td> <td>    0.424</td> <td> 0.671</td> <td>-2.48e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.7/12/2014]</th>  <td> 6.368e+04</td> <td> 1.69e+05</td> <td>    0.376</td> <td> 0.707</td> <td>-2.68e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>date[T.7/13/2014]</th>  <td> 5.458e+04</td> <td> 1.85e+05</td> <td>    0.294</td> <td> 0.768</td> <td>-3.09e+05</td> <td> 4.18e+05</td>
</tr>
<tr>
  <th>date[T.7/14/2014]</th>  <td> 7.302e+04</td> <td> 1.61e+05</td> <td>    0.453</td> <td> 0.651</td> <td>-2.43e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.7/15/2014]</th>  <td> 4.392e+04</td> <td> 1.61e+05</td> <td>    0.272</td> <td> 0.785</td> <td>-2.72e+05</td> <td>  3.6e+05</td>
</tr>
<tr>
  <th>date[T.7/16/2014]</th>  <td>  5.91e+04</td> <td> 1.61e+05</td> <td>    0.366</td> <td> 0.714</td> <td>-2.57e+05</td> <td> 3.75e+05</td>
</tr>
<tr>
  <th>date[T.7/17/2014]</th>  <td> 4.475e+04</td> <td> 1.62e+05</td> <td>    0.277</td> <td> 0.782</td> <td>-2.72e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>date[T.7/18/2014]</th>  <td> 5.829e+04</td> <td> 1.61e+05</td> <td>    0.361</td> <td> 0.718</td> <td>-2.58e+05</td> <td> 3.75e+05</td>
</tr>
<tr>
  <th>date[T.7/19/2014]</th>  <td> 9.856e+04</td> <td> 1.76e+05</td> <td>    0.560</td> <td> 0.575</td> <td>-2.46e+05</td> <td> 4.43e+05</td>
</tr>
<tr>
  <th>date[T.7/2/2014]</th>   <td> 7.063e+04</td> <td> 1.61e+05</td> <td>    0.437</td> <td> 0.662</td> <td>-2.46e+05</td> <td> 3.87e+05</td>
</tr>
<tr>
  <th>date[T.7/20/2014]</th>  <td>-5623.3992</td> <td>  1.7e+05</td> <td>   -0.033</td> <td> 0.974</td> <td> -3.4e+05</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>date[T.7/21/2014]</th>  <td> 4.591e+04</td> <td> 1.61e+05</td> <td>    0.284</td> <td> 0.776</td> <td> -2.7e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.7/22/2014]</th>  <td> 3.277e+04</td> <td> 1.61e+05</td> <td>    0.203</td> <td> 0.839</td> <td>-2.83e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>date[T.7/23/2014]</th>  <td> 4.054e+04</td> <td> 1.61e+05</td> <td>    0.251</td> <td> 0.802</td> <td>-2.76e+05</td> <td> 3.57e+05</td>
</tr>
<tr>
  <th>date[T.7/24/2014]</th>  <td> 4.783e+04</td> <td> 1.61e+05</td> <td>    0.296</td> <td> 0.767</td> <td>-2.69e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>date[T.7/25/2014]</th>  <td>  6.36e+04</td> <td> 1.61e+05</td> <td>    0.394</td> <td> 0.693</td> <td>-2.53e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>date[T.7/26/2014]</th>  <td> 9.555e+04</td> <td>  1.7e+05</td> <td>    0.561</td> <td> 0.575</td> <td>-2.38e+05</td> <td> 4.29e+05</td>
</tr>
<tr>
  <th>date[T.7/27/2014]</th>  <td> 1.243e+05</td> <td> 2.27e+05</td> <td>    0.547</td> <td> 0.584</td> <td>-3.21e+05</td> <td> 5.69e+05</td>
</tr>
<tr>
  <th>date[T.7/28/2014]</th>  <td> 3.098e+04</td> <td> 1.61e+05</td> <td>    0.192</td> <td> 0.848</td> <td>-2.85e+05</td> <td> 3.47e+05</td>
</tr>
<tr>
  <th>date[T.7/29/2014]</th>  <td>  3.81e+04</td> <td> 1.61e+05</td> <td>    0.236</td> <td> 0.813</td> <td>-2.78e+05</td> <td> 3.54e+05</td>
</tr>
<tr>
  <th>date[T.7/3/2014]</th>   <td> 3.972e+04</td> <td> 1.62e+05</td> <td>    0.246</td> <td> 0.806</td> <td>-2.77e+05</td> <td> 3.57e+05</td>
</tr>
<tr>
  <th>date[T.7/30/2014]</th>  <td> 4.537e+04</td> <td> 1.62e+05</td> <td>    0.281</td> <td> 0.779</td> <td>-2.71e+05</td> <td> 3.62e+05</td>
</tr>
<tr>
  <th>date[T.7/31/2014]</th>  <td> 5.867e+04</td> <td> 1.62e+05</td> <td>    0.363</td> <td> 0.716</td> <td>-2.58e+05</td> <td> 3.75e+05</td>
</tr>
<tr>
  <th>date[T.7/4/2014]</th>   <td> 9.084e+04</td> <td> 1.97e+05</td> <td>    0.462</td> <td> 0.644</td> <td>-2.95e+05</td> <td> 4.76e+05</td>
</tr>
<tr>
  <th>date[T.7/5/2014]</th>   <td> 1.891e+05</td> <td> 1.73e+05</td> <td>    1.090</td> <td> 0.276</td> <td>-1.51e+05</td> <td> 5.29e+05</td>
</tr>
<tr>
  <th>date[T.7/6/2014]</th>   <td>  7.43e+04</td> <td> 1.85e+05</td> <td>    0.401</td> <td> 0.689</td> <td>-2.89e+05</td> <td> 4.38e+05</td>
</tr>
<tr>
  <th>date[T.7/7/2014]</th>   <td> 5.491e+04</td> <td> 1.62e+05</td> <td>    0.340</td> <td> 0.734</td> <td>-2.62e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>date[T.7/8/2014]</th>   <td> 3.754e+04</td> <td> 1.61e+05</td> <td>    0.233</td> <td> 0.816</td> <td>-2.78e+05</td> <td> 3.54e+05</td>
</tr>
<tr>
  <th>date[T.7/9/2014]</th>   <td> 5.091e+04</td> <td> 1.61e+05</td> <td>    0.316</td> <td> 0.752</td> <td>-2.65e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>date[T.8/1/2014]</th>   <td> 5.964e+04</td> <td> 1.62e+05</td> <td>    0.369</td> <td> 0.712</td> <td>-2.57e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.8/10/2014]</th>  <td> 4.276e+04</td> <td> 1.85e+05</td> <td>    0.231</td> <td> 0.818</td> <td>-3.21e+05</td> <td> 4.06e+05</td>
</tr>
<tr>
  <th>date[T.8/11/2014]</th>  <td> 6.691e+04</td> <td> 1.61e+05</td> <td>    0.415</td> <td> 0.678</td> <td>-2.49e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>date[T.8/12/2014]</th>  <td> 6.879e+04</td> <td> 1.61e+05</td> <td>    0.426</td> <td> 0.670</td> <td>-2.47e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.8/13/2014]</th>  <td> 4.686e+04</td> <td> 1.61e+05</td> <td>    0.290</td> <td> 0.772</td> <td>-2.69e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>date[T.8/14/2014]</th>  <td> 4.269e+04</td> <td> 1.61e+05</td> <td>    0.264</td> <td> 0.791</td> <td>-2.74e+05</td> <td> 3.59e+05</td>
</tr>
<tr>
  <th>date[T.8/15/2014]</th>  <td> 8.373e+04</td> <td> 1.62e+05</td> <td>    0.517</td> <td> 0.605</td> <td>-2.34e+05</td> <td> 4.01e+05</td>
</tr>
<tr>
  <th>date[T.8/16/2014]</th>  <td> 7.211e+04</td> <td>  1.8e+05</td> <td>    0.402</td> <td> 0.688</td> <td> -2.8e+05</td> <td> 4.24e+05</td>
</tr>
<tr>
  <th>date[T.8/17/2014]</th>  <td> 2.482e+04</td> <td>  1.8e+05</td> <td>    0.138</td> <td> 0.890</td> <td>-3.27e+05</td> <td> 3.77e+05</td>
</tr>
<tr>
  <th>date[T.8/18/2014]</th>  <td> 6.429e+04</td> <td> 1.62e+05</td> <td>    0.398</td> <td> 0.691</td> <td>-2.52e+05</td> <td> 3.81e+05</td>
</tr>
<tr>
  <th>date[T.8/19/2014]</th>  <td> 7.431e+04</td> <td> 1.61e+05</td> <td>    0.460</td> <td> 0.645</td> <td>-2.42e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>date[T.8/2/2014]</th>   <td> 8.373e+04</td> <td> 1.76e+05</td> <td>    0.476</td> <td> 0.634</td> <td>-2.61e+05</td> <td> 4.29e+05</td>
</tr>
<tr>
  <th>date[T.8/20/2014]</th>  <td> 5.201e+04</td> <td> 1.61e+05</td> <td>    0.323</td> <td> 0.747</td> <td>-2.64e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.8/21/2014]</th>  <td>  3.99e+04</td> <td> 1.61e+05</td> <td>    0.247</td> <td> 0.805</td> <td>-2.77e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.8/22/2014]</th>  <td> 3.269e+04</td> <td> 1.61e+05</td> <td>    0.203</td> <td> 0.839</td> <td>-2.84e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>date[T.8/23/2014]</th>  <td> 7.772e+04</td> <td> 1.72e+05</td> <td>    0.453</td> <td> 0.651</td> <td>-2.59e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>date[T.8/24/2014]</th>  <td> 5.157e+04</td> <td> 1.97e+05</td> <td>    0.262</td> <td> 0.793</td> <td>-3.34e+05</td> <td> 4.37e+05</td>
</tr>
<tr>
  <th>date[T.8/25/2014]</th>  <td> 5.985e+04</td> <td> 1.61e+05</td> <td>    0.371</td> <td> 0.711</td> <td>-2.56e+05</td> <td> 3.76e+05</td>
</tr>
<tr>
  <th>date[T.8/26/2014]</th>  <td> 5.291e+04</td> <td> 1.61e+05</td> <td>    0.328</td> <td> 0.743</td> <td>-2.63e+05</td> <td> 3.69e+05</td>
</tr>
<tr>
  <th>date[T.8/27/2014]</th>  <td> 3.934e+04</td> <td> 1.61e+05</td> <td>    0.244</td> <td> 0.807</td> <td>-2.77e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.8/28/2014]</th>  <td> 6.143e+04</td> <td> 1.62e+05</td> <td>    0.380</td> <td> 0.704</td> <td>-2.55e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>date[T.8/29/2014]</th>  <td> 4.822e+04</td> <td> 1.62e+05</td> <td>    0.298</td> <td> 0.766</td> <td>-2.69e+05</td> <td> 3.66e+05</td>
</tr>
<tr>
  <th>date[T.8/3/2014]</th>   <td> 8.831e+04</td> <td> 2.27e+05</td> <td>    0.389</td> <td> 0.697</td> <td>-3.57e+05</td> <td> 5.33e+05</td>
</tr>
<tr>
  <th>date[T.8/30/2014]</th>  <td> 1.118e+05</td> <td> 2.28e+05</td> <td>    0.491</td> <td> 0.623</td> <td>-3.34e+05</td> <td> 5.58e+05</td>
</tr>
<tr>
  <th>date[T.8/31/2014]</th>  <td> 7.317e+04</td> <td> 1.85e+05</td> <td>    0.395</td> <td> 0.693</td> <td> -2.9e+05</td> <td> 4.37e+05</td>
</tr>
<tr>
  <th>date[T.8/4/2014]</th>   <td> 6.574e+04</td> <td> 1.61e+05</td> <td>    0.407</td> <td> 0.684</td> <td>-2.51e+05</td> <td> 3.82e+05</td>
</tr>
<tr>
  <th>date[T.8/5/2014]</th>   <td> 5.734e+04</td> <td> 1.61e+05</td> <td>    0.355</td> <td> 0.722</td> <td>-2.59e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>date[T.8/6/2014]</th>   <td> 4.191e+04</td> <td> 1.62e+05</td> <td>    0.260</td> <td> 0.795</td> <td>-2.75e+05</td> <td> 3.58e+05</td>
</tr>
<tr>
  <th>date[T.8/7/2014]</th>   <td> 6.086e+04</td> <td> 1.62e+05</td> <td>    0.376</td> <td> 0.707</td> <td>-2.56e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>date[T.8/8/2014]</th>   <td>  6.74e+04</td> <td> 1.62e+05</td> <td>    0.417</td> <td> 0.677</td> <td>-2.49e+05</td> <td> 3.84e+05</td>
</tr>
<tr>
  <th>date[T.8/9/2014]</th>   <td> 1.033e+05</td> <td> 1.97e+05</td> <td>    0.525</td> <td> 0.600</td> <td>-2.82e+05</td> <td> 4.89e+05</td>
</tr>
<tr>
  <th>date[T.9/1/2014]</th>   <td> 9.074e+04</td> <td> 1.73e+05</td> <td>    0.523</td> <td> 0.601</td> <td>-2.49e+05</td> <td> 4.31e+05</td>
</tr>
<tr>
  <th>date[T.9/10/2014]</th>  <td> 4.157e+04</td> <td> 1.61e+05</td> <td>    0.257</td> <td> 0.797</td> <td>-2.75e+05</td> <td> 3.58e+05</td>
</tr>
<tr>
  <th>date[T.9/11/2014]</th>  <td> 7.055e+04</td> <td> 1.61e+05</td> <td>    0.437</td> <td> 0.662</td> <td>-2.46e+05</td> <td> 3.87e+05</td>
</tr>
<tr>
  <th>date[T.9/12/2014]</th>  <td> 1.498e+04</td> <td> 1.62e+05</td> <td>    0.093</td> <td> 0.926</td> <td>-3.02e+05</td> <td> 3.32e+05</td>
</tr>
<tr>
  <th>date[T.9/13/2014]</th>  <td> 8.144e+04</td> <td> 1.76e+05</td> <td>    0.463</td> <td> 0.643</td> <td>-2.63e+05</td> <td> 4.26e+05</td>
</tr>
<tr>
  <th>date[T.9/14/2014]</th>  <td> 6.964e+04</td> <td>  1.8e+05</td> <td>    0.388</td> <td> 0.698</td> <td>-2.82e+05</td> <td> 4.22e+05</td>
</tr>
<tr>
  <th>date[T.9/15/2014]</th>  <td> 2.052e+04</td> <td> 1.62e+05</td> <td>    0.127</td> <td> 0.899</td> <td>-2.96e+05</td> <td> 3.37e+05</td>
</tr>
<tr>
  <th>date[T.9/16/2014]</th>  <td> 3.815e+04</td> <td> 1.61e+05</td> <td>    0.236</td> <td> 0.813</td> <td>-2.78e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>date[T.9/17/2014]</th>  <td> 2.567e+04</td> <td> 1.62e+05</td> <td>    0.159</td> <td> 0.874</td> <td>-2.91e+05</td> <td> 3.43e+05</td>
</tr>
<tr>
  <th>date[T.9/18/2014]</th>  <td>  4.86e+04</td> <td> 1.62e+05</td> <td>    0.301</td> <td> 0.764</td> <td>-2.68e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>date[T.9/19/2014]</th>  <td> 7.247e+04</td> <td> 1.62e+05</td> <td>    0.448</td> <td> 0.654</td> <td>-2.44e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.9/2/2014]</th>   <td> 6.291e+04</td> <td> 1.62e+05</td> <td>    0.389</td> <td> 0.697</td> <td>-2.54e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>date[T.9/20/2014]</th>  <td> 9.853e+04</td> <td> 1.76e+05</td> <td>    0.560</td> <td> 0.575</td> <td>-2.46e+05</td> <td> 4.43e+05</td>
</tr>
<tr>
  <th>date[T.9/21/2014]</th>  <td>-2.162e+04</td> <td> 1.76e+05</td> <td>   -0.123</td> <td> 0.902</td> <td>-3.66e+05</td> <td> 3.23e+05</td>
</tr>
<tr>
  <th>date[T.9/22/2014]</th>  <td> 5.286e+04</td> <td> 1.61e+05</td> <td>    0.327</td> <td> 0.743</td> <td>-2.64e+05</td> <td> 3.69e+05</td>
</tr>
<tr>
  <th>date[T.9/23/2014]</th>  <td> 7.304e+04</td> <td> 1.61e+05</td> <td>    0.453</td> <td> 0.651</td> <td>-2.43e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>date[T.9/24/2014]</th>  <td> 2.913e+04</td> <td> 1.61e+05</td> <td>    0.180</td> <td> 0.857</td> <td>-2.87e+05</td> <td> 3.45e+05</td>
</tr>
<tr>
  <th>date[T.9/25/2014]</th>  <td> 4.459e+04</td> <td> 1.62e+05</td> <td>    0.276</td> <td> 0.783</td> <td>-2.72e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>date[T.9/26/2014]</th>  <td> 4.775e+04</td> <td> 1.61e+05</td> <td>    0.296</td> <td> 0.767</td> <td>-2.69e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>date[T.9/27/2014]</th>  <td>-1.319e+05</td> <td> 1.76e+05</td> <td>   -0.750</td> <td> 0.454</td> <td>-4.77e+05</td> <td> 2.13e+05</td>
</tr>
<tr>
  <th>date[T.9/28/2014]</th>  <td> 8.464e+04</td> <td> 1.85e+05</td> <td>    0.456</td> <td> 0.648</td> <td>-2.79e+05</td> <td> 4.48e+05</td>
</tr>
<tr>
  <th>date[T.9/29/2014]</th>  <td> 5.723e+04</td> <td> 1.62e+05</td> <td>    0.354</td> <td> 0.723</td> <td>-2.59e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>date[T.9/3/2014]</th>   <td> 6.344e+04</td> <td> 1.62e+05</td> <td>    0.393</td> <td> 0.694</td> <td>-2.53e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>date[T.9/30/2014]</th>  <td> 3.846e+04</td> <td> 1.62e+05</td> <td>    0.237</td> <td> 0.812</td> <td>-2.79e+05</td> <td> 3.56e+05</td>
</tr>
<tr>
  <th>date[T.9/4/2014]</th>   <td> 5.176e+04</td> <td> 1.62e+05</td> <td>    0.320</td> <td> 0.749</td> <td>-2.65e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>date[T.9/5/2014]</th>   <td> 5.859e+04</td> <td> 1.61e+05</td> <td>    0.363</td> <td> 0.717</td> <td>-2.58e+05</td> <td> 3.75e+05</td>
</tr>
<tr>
  <th>date[T.9/6/2014]</th>   <td> 3.968e+04</td> <td> 1.76e+05</td> <td>    0.226</td> <td> 0.822</td> <td>-3.05e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>date[T.9/7/2014]</th>   <td> 2.516e+04</td> <td> 1.97e+05</td> <td>    0.128</td> <td> 0.898</td> <td> -3.6e+05</td> <td> 4.11e+05</td>
</tr>
<tr>
  <th>date[T.9/8/2014]</th>   <td> 6.088e+04</td> <td> 1.62e+05</td> <td>    0.377</td> <td> 0.706</td> <td>-2.56e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>date[T.9/9/2014]</th>   <td> 5.418e+04</td> <td> 1.61e+05</td> <td>    0.336</td> <td> 0.737</td> <td>-2.62e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>id</th>                 <td>-8.566e-07</td> <td> 3.99e-07</td> <td>   -2.148</td> <td> 0.032</td> <td>-1.64e-06</td> <td>-7.49e-08</td>
</tr>
<tr>
  <th>bedrooms</th>           <td>-2.758e+04</td> <td> 1548.715</td> <td>  -17.811</td> <td> 0.000</td> <td>-3.06e+04</td> <td>-2.45e+04</td>
</tr>
<tr>
  <th>bathrooms</th>          <td> 2.604e+04</td> <td> 2671.682</td> <td>    9.747</td> <td> 0.000</td> <td> 2.08e+04</td> <td> 3.13e+04</td>
</tr>
<tr>
  <th>sqft_living</th>        <td>   96.8500</td> <td>   14.574</td> <td>    6.646</td> <td> 0.000</td> <td>   68.284</td> <td>  125.416</td>
</tr>
<tr>
  <th>sqft_lot</th>           <td>    0.2472</td> <td>    0.039</td> <td>    6.364</td> <td> 0.000</td> <td>    0.171</td> <td>    0.323</td>
</tr>
<tr>
  <th>floors</th>             <td>-4.503e+04</td> <td> 3194.878</td> <td>  -14.095</td> <td> 0.000</td> <td>-5.13e+04</td> <td>-3.88e+04</td>
</tr>
<tr>
  <th>waterfront</th>         <td> 6.891e+05</td> <td> 1.48e+04</td> <td>   46.561</td> <td> 0.000</td> <td>  6.6e+05</td> <td> 7.18e+05</td>
</tr>
<tr>
  <th>view</th>               <td> 5.624e+04</td> <td> 1754.239</td> <td>   32.059</td> <td> 0.000</td> <td> 5.28e+04</td> <td> 5.97e+04</td>
</tr>
<tr>
  <th>condition</th>          <td> 2.728e+04</td> <td> 1941.072</td> <td>   14.055</td> <td> 0.000</td> <td> 2.35e+04</td> <td> 3.11e+04</td>
</tr>
<tr>
  <th>grade</th>              <td> 5.842e+04</td> <td> 1841.547</td> <td>   31.726</td> <td> 0.000</td> <td> 5.48e+04</td> <td>  6.2e+04</td>
</tr>
<tr>
  <th>sqft_above</th>         <td>  108.0880</td> <td>   14.578</td> <td>    7.414</td> <td> 0.000</td> <td>   79.514</td> <td>  136.662</td>
</tr>
<tr>
  <th>sqft_basement</th>      <td>   56.8494</td> <td>   15.084</td> <td>    3.769</td> <td> 0.000</td> <td>   27.284</td> <td>   86.415</td>
</tr>
<tr>
  <th>yr_built</th>           <td> -733.8554</td> <td>   64.750</td> <td>  -11.334</td> <td> 0.000</td> <td> -860.770</td> <td> -606.940</td>
</tr>
<tr>
  <th>yr_renovated</th>       <td> 2913.6565</td> <td>  384.006</td> <td>    7.588</td> <td> 0.000</td> <td> 2160.976</td> <td> 3666.337</td>
</tr>
<tr>
  <th>lat</th>                <td> 1.904e+05</td> <td> 6.39e+04</td> <td>    2.981</td> <td> 0.003</td> <td> 6.52e+04</td> <td> 3.16e+05</td>
</tr>
<tr>
  <th>long</th>               <td>-1.415e+05</td> <td> 4.58e+04</td> <td>   -3.087</td> <td> 0.002</td> <td>-2.31e+05</td> <td>-5.17e+04</td>
</tr>
<tr>
  <th>sqft_living15</th>      <td>   10.9413</td> <td>    2.911</td> <td>    3.758</td> <td> 0.000</td> <td>    5.235</td> <td>   16.648</td>
</tr>
<tr>
  <th>sqft_lot15</th>         <td>   -0.1623</td> <td>    0.061</td> <td>   -2.655</td> <td> 0.008</td> <td>   -0.282</td> <td>   -0.042</td>
</tr>
<tr>
  <th>renovated</th>          <td>-5.776e+06</td> <td> 7.66e+05</td> <td>   -7.536</td> <td> 0.000</td> <td>-7.28e+06</td> <td>-4.27e+06</td>
</tr>
<tr>
  <th>has_basement</th>       <td>-2.864e+04</td> <td> 4319.474</td> <td>   -6.630</td> <td> 0.000</td> <td>-3.71e+04</td> <td>-2.02e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>      <td>  3.22e+04</td> <td> 1.46e+04</td> <td>    2.213</td> <td> 0.027</td> <td> 3673.608</td> <td> 6.07e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>      <td>-2.569e+04</td> <td>  1.3e+04</td> <td>   -1.974</td> <td> 0.048</td> <td>-5.12e+04</td> <td> -178.813</td>
</tr>
<tr>
  <th>zipcode_98004</th>      <td> 7.188e+05</td> <td> 2.36e+04</td> <td>   30.396</td> <td> 0.000</td> <td> 6.72e+05</td> <td> 7.65e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>      <td> 2.538e+05</td> <td> 2.53e+04</td> <td>   10.030</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 3.03e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>      <td> 2.305e+05</td> <td> 2.07e+04</td> <td>   11.149</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>      <td> 1.958e+05</td> <td> 2.61e+04</td> <td>    7.501</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 2.47e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>      <td> 2.069e+05</td> <td> 2.48e+04</td> <td>    8.348</td> <td> 0.000</td> <td> 1.58e+05</td> <td> 2.56e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>      <td> 1.031e+05</td> <td> 2.22e+04</td> <td>    4.647</td> <td> 0.000</td> <td> 5.96e+04</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>      <td> 4.488e+04</td> <td> 3.23e+04</td> <td>    1.391</td> <td> 0.164</td> <td>-1.83e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>      <td>  9.37e+04</td> <td> 3.55e+04</td> <td>    2.641</td> <td> 0.008</td> <td> 2.42e+04</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>      <td> 4.642e+04</td> <td>  3.5e+04</td> <td>    1.327</td> <td> 0.184</td> <td>-2.21e+04</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>      <td> 3.912e+04</td> <td> 1.93e+04</td> <td>    2.026</td> <td> 0.043</td> <td> 1276.269</td> <td>  7.7e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>      <td>-4.584e+04</td> <td>  1.2e+04</td> <td>   -3.829</td> <td> 0.000</td> <td>-6.93e+04</td> <td>-2.24e+04</td>
</tr>
<tr>
  <th>zipcode_98024</th>      <td> 1.526e+05</td> <td> 3.12e+04</td> <td>    4.897</td> <td> 0.000</td> <td> 9.15e+04</td> <td> 2.14e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>      <td> 1.612e+05</td> <td> 2.12e+04</td> <td>    7.590</td> <td> 0.000</td> <td>  1.2e+05</td> <td> 2.03e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>      <td> 3.675e+04</td> <td> 3.13e+04</td> <td>    1.173</td> <td> 0.241</td> <td>-2.46e+04</td> <td> 9.81e+04</td>
</tr>
<tr>
  <th>zipcode_98029</th>      <td> 1.972e+05</td> <td> 2.42e+04</td> <td>    8.132</td> <td> 0.000</td> <td>  1.5e+05</td> <td> 2.45e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>      <td>  636.3874</td> <td> 1.43e+04</td> <td>    0.045</td> <td> 0.965</td> <td>-2.74e+04</td> <td> 2.87e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>      <td> 4600.7954</td> <td> 1.49e+04</td> <td>    0.309</td> <td> 0.758</td> <td>-2.46e+04</td> <td> 3.38e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>      <td>-6833.3457</td> <td> 1.73e+04</td> <td>   -0.395</td> <td> 0.693</td> <td>-4.07e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>zipcode_98033</th>      <td> 2.996e+05</td> <td> 2.69e+04</td> <td>   11.149</td> <td> 0.000</td> <td> 2.47e+05</td> <td> 3.52e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>      <td> 1.327e+05</td> <td> 2.88e+04</td> <td>    4.605</td> <td> 0.000</td> <td> 7.62e+04</td> <td> 1.89e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>      <td> 4.834e+04</td> <td> 1.61e+04</td> <td>    3.006</td> <td> 0.003</td> <td> 1.68e+04</td> <td> 7.99e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>      <td> 1.244e+06</td> <td>  3.2e+04</td> <td>   38.860</td> <td> 0.000</td> <td> 1.18e+06</td> <td> 1.31e+06</td>
</tr>
<tr>
  <th>zipcode_98040</th>      <td> 4.604e+05</td> <td> 2.09e+04</td> <td>   22.018</td> <td> 0.000</td> <td> 4.19e+05</td> <td> 5.01e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>      <td> 1.096e+04</td> <td> 1.37e+04</td> <td>    0.800</td> <td> 0.424</td> <td>-1.59e+04</td> <td> 3.78e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>      <td> 1.317e+05</td> <td> 2.97e+04</td> <td>    4.433</td> <td> 0.000</td> <td> 7.34e+04</td> <td>  1.9e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>      <td> 1.751e+05</td> <td> 2.74e+04</td> <td>    6.380</td> <td> 0.000</td> <td> 1.21e+05</td> <td> 2.29e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>      <td> 1.536e+05</td> <td> 2.94e+04</td> <td>    5.227</td> <td> 0.000</td> <td>  9.6e+04</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>      <td> 2.392e+04</td> <td> 1.66e+04</td> <td>    1.441</td> <td> 0.150</td> <td>-8618.375</td> <td> 5.65e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>      <td> 6.491e+04</td> <td>  1.8e+04</td> <td>    3.598</td> <td> 0.000</td> <td> 2.95e+04</td> <td>    1e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th>      <td> 1.597e+04</td> <td> 1.57e+04</td> <td>    1.018</td> <td> 0.309</td> <td>-1.48e+04</td> <td> 4.67e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>      <td> 6.302e+04</td> <td> 1.77e+04</td> <td>    3.563</td> <td> 0.000</td> <td> 2.84e+04</td> <td> 9.77e+04</td>
</tr>
<tr>
  <th>zipcode_98065</th>      <td> 9.525e+04</td> <td> 2.74e+04</td> <td>    3.478</td> <td> 0.001</td> <td> 4.16e+04</td> <td> 1.49e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>      <td> -5.57e+04</td> <td> 2.09e+04</td> <td>   -2.671</td> <td> 0.008</td> <td>-9.66e+04</td> <td>-1.48e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>      <td>  8.25e+04</td> <td> 3.21e+04</td> <td>    2.571</td> <td> 0.010</td> <td> 1.96e+04</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>      <td> 1.377e+05</td> <td>  2.6e+04</td> <td>    5.302</td> <td> 0.000</td> <td> 8.68e+04</td> <td> 1.89e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>      <td> 1.417e+05</td> <td>  2.5e+04</td> <td>    5.673</td> <td> 0.000</td> <td> 9.27e+04</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>      <td> 5.934e+04</td> <td> 3.34e+04</td> <td>    1.778</td> <td> 0.075</td> <td>-6086.166</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>      <td>-2.674e+04</td> <td>  1.3e+04</td> <td>   -2.053</td> <td> 0.040</td> <td>-5.23e+04</td> <td>-1209.094</td>
</tr>
<tr>
  <th>zipcode_98102</th>      <td> 4.501e+05</td> <td> 2.77e+04</td> <td>   16.235</td> <td> 0.000</td> <td> 3.96e+05</td> <td> 5.04e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>      <td>  2.62e+05</td> <td>  2.6e+04</td> <td>   10.080</td> <td> 0.000</td> <td> 2.11e+05</td> <td> 3.13e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>      <td> 4.016e+05</td> <td> 2.67e+04</td> <td>   15.033</td> <td> 0.000</td> <td> 3.49e+05</td> <td> 4.54e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>      <td> 9.711e+04</td> <td> 1.93e+04</td> <td>    5.042</td> <td> 0.000</td> <td> 5.94e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th>      <td> 2.667e+05</td> <td> 2.68e+04</td> <td>    9.950</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 3.19e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>      <td> 7.986e+04</td> <td> 2.13e+04</td> <td>    3.752</td> <td> 0.000</td> <td> 3.81e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th>      <td> 4.222e+05</td> <td> 2.77e+04</td> <td>   15.266</td> <td> 0.000</td> <td> 3.68e+05</td> <td> 4.76e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>      <td> 5.576e+05</td> <td> 2.45e+04</td> <td>   22.756</td> <td> 0.000</td> <td>  5.1e+05</td> <td> 6.06e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>      <td> 2.586e+05</td> <td> 2.64e+04</td> <td>    9.782</td> <td> 0.000</td> <td> 2.07e+05</td> <td>  3.1e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>      <td> 2.261e+05</td> <td> 2.15e+04</td> <td>   10.512</td> <td> 0.000</td> <td> 1.84e+05</td> <td> 2.68e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>      <td> 2.343e+05</td> <td> 2.68e+04</td> <td>    8.754</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.87e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>      <td> 1.281e+05</td> <td> 1.88e+04</td> <td>    6.820</td> <td> 0.000</td> <td> 9.13e+04</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>      <td> 4.035e+05</td> <td> 2.61e+04</td> <td>   15.448</td> <td> 0.000</td> <td> 3.52e+05</td> <td> 4.55e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>      <td> 2.824e+05</td> <td> 2.33e+04</td> <td>   12.138</td> <td> 0.000</td> <td> 2.37e+05</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>      <td> 1.243e+05</td> <td> 2.85e+04</td> <td>    4.355</td> <td> 0.000</td> <td> 6.83e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>      <td> 1.408e+05</td> <td> 1.98e+04</td> <td>    7.127</td> <td> 0.000</td> <td> 1.02e+05</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>      <td> 7.636e+04</td> <td> 2.95e+04</td> <td>    2.591</td> <td> 0.010</td> <td> 1.86e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>      <td> 1.903e+05</td> <td> 2.02e+04</td> <td>    9.401</td> <td> 0.000</td> <td> 1.51e+05</td> <td>  2.3e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>      <td> 2.232e+05</td> <td> 2.17e+04</td> <td>   10.306</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>      <td> 6.589e+04</td> <td> 1.81e+04</td> <td>    3.650</td> <td> 0.000</td> <td> 3.05e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>zipcode_98148</th>      <td> 4.184e+04</td> <td> 2.45e+04</td> <td>    1.705</td> <td> 0.088</td> <td>-6261.188</td> <td> 8.99e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>      <td> 5.779e+04</td> <td> 3.06e+04</td> <td>    1.887</td> <td> 0.059</td> <td>-2239.720</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>      <td>  1.62e+04</td> <td> 1.65e+04</td> <td>    0.980</td> <td> 0.327</td> <td>-1.62e+04</td> <td> 4.86e+04</td>
</tr>
<tr>
  <th>zipcode_98168</th>      <td> 4.138e+04</td> <td> 1.75e+04</td> <td>    2.367</td> <td> 0.018</td> <td> 7109.444</td> <td> 7.57e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>      <td> 1.196e+05</td> <td> 3.08e+04</td> <td>    3.889</td> <td> 0.000</td> <td> 5.93e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>      <td> 7276.2285</td> <td> 1.81e+04</td> <td>    0.403</td> <td> 0.687</td> <td>-2.81e+04</td> <td> 4.27e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>      <td> 8379.1848</td> <td> 1.85e+04</td> <td>    0.453</td> <td> 0.651</td> <td>-2.79e+04</td> <td> 4.47e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>      <td>-2.382e+04</td> <td>  1.4e+04</td> <td>   -1.697</td> <td> 0.090</td> <td>-5.13e+04</td> <td> 3687.116</td>
</tr>
<tr>
  <th>zipcode_98199</th>      <td> 3.044e+05</td> <td> 2.54e+04</td> <td>   11.964</td> <td> 0.000</td> <td> 2.55e+05</td> <td> 3.54e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>20806.796</td> <th>  Durbin-Watson:     </th>  <td>   1.995</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>4281101.044</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.158</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>71.471</td>   <th>  Cond. No.          </th>  <td>3.07e+13</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.07e+13. This might indicate that there are<br/>strong multicollinearity or other numerical problems.





    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x211787d75e0>




    
![png](output_54_2.png)
    


Our baseline model's residuals does not meet the normality or the homoscedasticity assumption yet. We need to consider multicollinearity and address the columns that may be compromising the model. Additionally, we need to address outliers.

## Multicollinearity Check

### Correlation Matrices - Before

It looks like there is multicollinearity between the different columns. To address these I will be creating matrices and removing columns that may be causing the issues.


```python
df.drop(['id', 'date'], axis=1, inplace=True)
```


```python
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(),annot=True, mask=mask, cmap='Greens')
```




    <AxesSubplot:>




    
![png](output_60_1.png)
    



```python
mask = np.zeros_like(df.drop('price', axis=1).corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.drop('price', axis=1).corr(),annot=True, mask=mask, cmap='Reds')
```




    <AxesSubplot:>




    
![png](output_61_1.png)
    



```python
df.corr()['price'].abs().sort_values(ascending=False)
```




    price            1.000000
    sqft_living      0.701917
    grade            0.667951
    sqft_above       0.605368
    sqft_living15    0.585241
    bathrooms        0.525906
    view             0.393497
    sqft_basement    0.321108
    bedrooms         0.308787
    lat              0.306692
    waterfront       0.264306
    floors           0.256804
    has_basement     0.178264
    yr_renovated     0.117855
    renovated        0.117543
    sqft_lot         0.089876
    sqft_lot15       0.082845
    yr_built         0.053953
    zipcode          0.053402
    condition        0.036056
    long             0.022036
    Name: price, dtype: float64



Looking at the correlation matrix of the original df (prior to being one hot encoded for simplicity of the visual), there seems to be strong correlations between a multitude of parameters when we take our cut-off point as 0.75. Dropping the 'sqft_living' column will allow us to clear almost all the strong correlations. We are additionally not losing meaningful data since this column is equal to the sum of 'sqft_above' and 'sqft_basement'.


```python
df.drop('sqft_living', axis=1, inplace=True)
```

Since we have engineered a feature called 'has_basement' using 'sqft_basement' there is a high correlation between these columns as well. Since we are still keeping the basement information to a certain degree, we can also drop 'sqft_basement'


```python
df.drop('sqft_basement', axis=1, inplace=True)
```

Similar to above, since we have engineered the column 'renovated' using 'yr_renovated', we can also drop the 'yr_renovated' column.


```python
df.drop('yr_renovated', axis=1, inplace=True)
```

### Correlation Matrix - After


```python
mask = np.zeros_like(df.drop('price', axis=1).corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.drop('price', axis=1).corr(),annot=True, mask=mask, cmap='Reds')
```




    <AxesSubplot:>




    
![png](output_70_1.png)
    


Our correlation matrix looks much better with the exception of 'sqft_above' and 'grade'. As this value is so close to our cut-off value of 0.75, we will be keeping it in to see if it causes any additional issues after adjustments to the model.

# MODEL 

For this project we were asked to specifically use multiple linear regression (MLR) so our model will be an MLR model.


```python
drop_cols = ['id', 'date', 'sqft_living', 'sqft_basement', 'yr_renovated']
df_ohe.drop(drop_cols, axis=1, inplace=True)
```


```python
model_lin_reg(df=df_ohe)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.800</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.799</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1010.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>16:12:08</td>     <th>  Log-Likelihood:    </th> <td>-2.9003e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th>  <td>5.802e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21511</td>      <th>  BIC:               </th>  <td>5.809e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    85</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>-2.422e+07</td> <td>  6.3e+06</td> <td>   -3.845</td> <td> 0.000</td> <td>-3.66e+07</td> <td>-1.19e+07</td>
</tr>
<tr>
  <th>bedrooms</th>      <td>-1.763e+04</td> <td> 1544.455</td> <td>  -11.413</td> <td> 0.000</td> <td>-2.07e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>bathrooms</th>     <td> 4.377e+04</td> <td> 2643.295</td> <td>   16.557</td> <td> 0.000</td> <td> 3.86e+04</td> <td> 4.89e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>      <td>    0.2560</td> <td>    0.039</td> <td>    6.499</td> <td> 0.000</td> <td>    0.179</td> <td>    0.333</td>
</tr>
<tr>
  <th>floors</th>        <td>-6.367e+04</td> <td> 3189.871</td> <td>  -19.961</td> <td> 0.000</td> <td>-6.99e+04</td> <td>-5.74e+04</td>
</tr>
<tr>
  <th>waterfront</th>    <td> 6.937e+05</td> <td> 1.51e+04</td> <td>   46.012</td> <td> 0.000</td> <td> 6.64e+05</td> <td> 7.23e+05</td>
</tr>
<tr>
  <th>view</th>          <td> 6.292e+04</td> <td> 1770.266</td> <td>   35.543</td> <td> 0.000</td> <td> 5.95e+04</td> <td> 6.64e+04</td>
</tr>
<tr>
  <th>condition</th>     <td> 2.922e+04</td> <td> 1965.696</td> <td>   14.866</td> <td> 0.000</td> <td> 2.54e+04</td> <td> 3.31e+04</td>
</tr>
<tr>
  <th>grade</th>         <td> 6.338e+04</td> <td> 1864.517</td> <td>   33.994</td> <td> 0.000</td> <td> 5.97e+04</td> <td>  6.7e+04</td>
</tr>
<tr>
  <th>sqft_above</th>    <td>  190.3529</td> <td>    3.131</td> <td>   60.789</td> <td> 0.000</td> <td>  184.215</td> <td>  196.491</td>
</tr>
<tr>
  <th>yr_built</th>      <td> -861.1302</td> <td>   65.764</td> <td>  -13.094</td> <td> 0.000</td> <td> -990.033</td> <td> -732.228</td>
</tr>
<tr>
  <th>lat</th>           <td> 2.032e+05</td> <td>  6.5e+04</td> <td>    3.125</td> <td> 0.002</td> <td> 7.58e+04</td> <td> 3.31e+05</td>
</tr>
<tr>
  <th>long</th>          <td>-1.284e+05</td> <td> 4.67e+04</td> <td>   -2.749</td> <td> 0.006</td> <td> -2.2e+05</td> <td>-3.69e+04</td>
</tr>
<tr>
  <th>sqft_living15</th> <td>   21.3812</td> <td>    2.944</td> <td>    7.264</td> <td> 0.000</td> <td>   15.612</td> <td>   27.151</td>
</tr>
<tr>
  <th>sqft_lot15</th>    <td>   -0.0880</td> <td>    0.062</td> <td>   -1.418</td> <td> 0.156</td> <td>   -0.210</td> <td>    0.034</td>
</tr>
<tr>
  <th>renovated</th>     <td> 4.037e+04</td> <td> 6547.392</td> <td>    6.166</td> <td> 0.000</td> <td> 2.75e+04</td> <td> 5.32e+04</td>
</tr>
<tr>
  <th>has_basement</th>  <td> 6.225e+04</td> <td> 3093.061</td> <td>   20.124</td> <td> 0.000</td> <td> 5.62e+04</td> <td> 6.83e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th> <td> 3.318e+04</td> <td> 1.48e+04</td> <td>    2.236</td> <td> 0.025</td> <td> 4093.329</td> <td> 6.23e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th> <td>-2.615e+04</td> <td> 1.33e+04</td> <td>   -1.971</td> <td> 0.049</td> <td>-5.22e+04</td> <td> -149.702</td>
</tr>
<tr>
  <th>zipcode_98004</th> <td> 7.234e+05</td> <td> 2.41e+04</td> <td>   30.009</td> <td> 0.000</td> <td> 6.76e+05</td> <td> 7.71e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th> <td> 2.487e+05</td> <td> 2.58e+04</td> <td>    9.653</td> <td> 0.000</td> <td> 1.98e+05</td> <td> 2.99e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th> <td> 2.355e+05</td> <td> 2.11e+04</td> <td>   11.178</td> <td> 0.000</td> <td> 1.94e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th> <td> 1.937e+05</td> <td> 2.66e+04</td> <td>    7.286</td> <td> 0.000</td> <td> 1.42e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th> <td> 2.025e+05</td> <td> 2.53e+04</td> <td>    8.018</td> <td> 0.000</td> <td> 1.53e+05</td> <td> 2.52e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th> <td> 9.554e+04</td> <td> 2.26e+04</td> <td>    4.223</td> <td> 0.000</td> <td> 5.12e+04</td> <td>  1.4e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th> <td> 3.791e+04</td> <td> 3.29e+04</td> <td>    1.154</td> <td> 0.249</td> <td>-2.65e+04</td> <td> 1.02e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th> <td> 8.408e+04</td> <td> 3.61e+04</td> <td>    2.330</td> <td> 0.020</td> <td> 1.33e+04</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th> <td> 4.459e+04</td> <td> 3.56e+04</td> <td>    1.253</td> <td> 0.210</td> <td>-2.52e+04</td> <td> 1.14e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th> <td> 3.069e+04</td> <td> 1.97e+04</td> <td>    1.562</td> <td> 0.118</td> <td>-7832.522</td> <td> 6.92e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th> <td>-4.989e+04</td> <td> 1.22e+04</td> <td>   -4.088</td> <td> 0.000</td> <td>-7.38e+04</td> <td> -2.6e+04</td>
</tr>
<tr>
  <th>zipcode_98024</th> <td> 1.431e+05</td> <td> 3.18e+04</td> <td>    4.504</td> <td> 0.000</td> <td> 8.08e+04</td> <td> 2.05e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th> <td> 1.547e+05</td> <td> 2.16e+04</td> <td>    7.153</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th> <td> 3.521e+04</td> <td> 3.19e+04</td> <td>    1.103</td> <td> 0.270</td> <td>-2.73e+04</td> <td> 9.77e+04</td>
</tr>
<tr>
  <th>zipcode_98029</th> <td> 1.881e+05</td> <td> 2.47e+04</td> <td>    7.614</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th> <td>-1278.3007</td> <td> 1.46e+04</td> <td>   -0.088</td> <td> 0.930</td> <td>-2.99e+04</td> <td> 2.73e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th> <td> 2733.1401</td> <td> 1.52e+04</td> <td>    0.180</td> <td> 0.857</td> <td> -2.7e+04</td> <td> 3.25e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th> <td>-6121.9357</td> <td> 1.76e+04</td> <td>   -0.347</td> <td> 0.728</td> <td>-4.07e+04</td> <td> 2.84e+04</td>
</tr>
<tr>
  <th>zipcode_98033</th> <td> 2.972e+05</td> <td> 2.74e+04</td> <td>   10.857</td> <td> 0.000</td> <td> 2.44e+05</td> <td> 3.51e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th> <td>  1.22e+05</td> <td> 2.93e+04</td> <td>    4.157</td> <td> 0.000</td> <td> 6.45e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th> <td>  4.76e+04</td> <td> 1.64e+04</td> <td>    2.906</td> <td> 0.004</td> <td> 1.55e+04</td> <td> 7.97e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th> <td> 1.254e+06</td> <td> 3.26e+04</td> <td>   38.492</td> <td> 0.000</td> <td> 1.19e+06</td> <td> 1.32e+06</td>
</tr>
<tr>
  <th>zipcode_98040</th> <td> 4.697e+05</td> <td> 2.13e+04</td> <td>   22.035</td> <td> 0.000</td> <td> 4.28e+05</td> <td> 5.12e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th> <td> 9795.9407</td> <td>  1.4e+04</td> <td>    0.702</td> <td> 0.483</td> <td>-1.76e+04</td> <td> 3.72e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th> <td> 1.186e+05</td> <td> 3.03e+04</td> <td>    3.920</td> <td> 0.000</td> <td> 5.93e+04</td> <td> 1.78e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th> <td> 1.633e+05</td> <td> 2.79e+04</td> <td>    5.844</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 2.18e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th> <td> 1.436e+05</td> <td> 2.99e+04</td> <td>    4.794</td> <td> 0.000</td> <td> 8.49e+04</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th> <td> 2.336e+04</td> <td> 1.69e+04</td> <td>    1.381</td> <td> 0.167</td> <td>-9798.967</td> <td> 5.65e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th> <td> 6.749e+04</td> <td> 1.84e+04</td> <td>    3.672</td> <td> 0.000</td> <td> 3.15e+04</td> <td> 1.04e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th> <td> 1.286e+04</td> <td>  1.6e+04</td> <td>    0.805</td> <td> 0.421</td> <td>-1.85e+04</td> <td> 4.42e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th> <td> 5.588e+04</td> <td>  1.8e+04</td> <td>    3.099</td> <td> 0.002</td> <td> 2.05e+04</td> <td> 9.12e+04</td>
</tr>
<tr>
  <th>zipcode_98065</th> <td> 8.765e+04</td> <td> 2.79e+04</td> <td>    3.142</td> <td> 0.002</td> <td>  3.3e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th> <td>-5.806e+04</td> <td> 2.13e+04</td> <td>   -2.730</td> <td> 0.006</td> <td>-9.97e+04</td> <td>-1.64e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th> <td>  7.39e+04</td> <td> 3.27e+04</td> <td>    2.261</td> <td> 0.024</td> <td> 9836.718</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th> <td> 1.277e+05</td> <td> 2.65e+04</td> <td>    4.825</td> <td> 0.000</td> <td> 7.58e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th> <td> 1.295e+05</td> <td> 2.54e+04</td> <td>    5.089</td> <td> 0.000</td> <td> 7.96e+04</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th> <td> 4.529e+04</td> <td>  3.4e+04</td> <td>    1.332</td> <td> 0.183</td> <td>-2.14e+04</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th> <td>-2.867e+04</td> <td> 1.33e+04</td> <td>   -2.161</td> <td> 0.031</td> <td>-5.47e+04</td> <td>-2667.678</td>
</tr>
<tr>
  <th>zipcode_98102</th> <td> 4.422e+05</td> <td> 2.83e+04</td> <td>   15.646</td> <td> 0.000</td> <td> 3.87e+05</td> <td> 4.98e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th> <td> 2.628e+05</td> <td> 2.65e+04</td> <td>    9.927</td> <td> 0.000</td> <td> 2.11e+05</td> <td> 3.15e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th> <td> 3.882e+05</td> <td> 2.72e+04</td> <td>   14.279</td> <td> 0.000</td> <td> 3.35e+05</td> <td> 4.41e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th> <td> 9.546e+04</td> <td> 1.96e+04</td> <td>    4.866</td> <td> 0.000</td> <td>  5.7e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th> <td> 2.629e+05</td> <td> 2.73e+04</td> <td>    9.634</td> <td> 0.000</td> <td> 2.09e+05</td> <td> 3.16e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th> <td>  8.08e+04</td> <td> 2.17e+04</td> <td>    3.731</td> <td> 0.000</td> <td> 3.84e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th> <td> 4.174e+05</td> <td> 2.81e+04</td> <td>   14.844</td> <td> 0.000</td> <td> 3.62e+05</td> <td> 4.73e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th> <td> 5.539e+05</td> <td>  2.5e+04</td> <td>   22.191</td> <td> 0.000</td> <td> 5.05e+05</td> <td> 6.03e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th> <td>  2.56e+05</td> <td> 2.69e+04</td> <td>    9.511</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 3.09e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th> <td> 2.243e+05</td> <td> 2.19e+04</td> <td>   10.240</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.67e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th> <td> 2.321e+05</td> <td> 2.73e+04</td> <td>    8.515</td> <td> 0.000</td> <td> 1.79e+05</td> <td> 2.85e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th> <td> 1.313e+05</td> <td> 1.91e+04</td> <td>    6.862</td> <td> 0.000</td> <td> 9.38e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th> <td> 3.943e+05</td> <td> 2.66e+04</td> <td>   14.841</td> <td> 0.000</td> <td> 3.42e+05</td> <td> 4.46e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th> <td> 2.704e+05</td> <td> 2.37e+04</td> <td>   11.410</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 3.17e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th> <td>  1.21e+05</td> <td> 2.91e+04</td> <td>    4.164</td> <td> 0.000</td> <td>  6.4e+04</td> <td> 1.78e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th> <td> 1.369e+05</td> <td> 2.01e+04</td> <td>    6.804</td> <td> 0.000</td> <td> 9.74e+04</td> <td> 1.76e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th> <td>  7.66e+04</td> <td>    3e+04</td> <td>    2.553</td> <td> 0.011</td> <td> 1.78e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th> <td> 1.855e+05</td> <td> 2.06e+04</td> <td>    8.994</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th> <td> 2.254e+05</td> <td>  2.2e+04</td> <td>   10.231</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.69e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th> <td> 6.678e+04</td> <td> 1.84e+04</td> <td>    3.628</td> <td> 0.000</td> <td> 3.07e+04</td> <td> 1.03e+05</td>
</tr>
<tr>
  <th>zipcode_98148</th> <td>  3.46e+04</td> <td>  2.5e+04</td> <td>    1.382</td> <td> 0.167</td> <td>-1.45e+04</td> <td> 8.37e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th> <td>  5.55e+04</td> <td> 3.12e+04</td> <td>    1.779</td> <td> 0.075</td> <td>-5656.754</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th> <td> 1.449e+04</td> <td> 1.68e+04</td> <td>    0.860</td> <td> 0.390</td> <td>-1.85e+04</td> <td> 4.75e+04</td>
</tr>
<tr>
  <th>zipcode_98168</th> <td>  4.58e+04</td> <td> 1.78e+04</td> <td>    2.573</td> <td> 0.010</td> <td> 1.09e+04</td> <td> 8.07e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th> <td> 1.168e+05</td> <td> 3.13e+04</td> <td>    3.727</td> <td> 0.000</td> <td> 5.54e+04</td> <td> 1.78e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th> <td> 1.105e+04</td> <td> 1.84e+04</td> <td>    0.601</td> <td> 0.548</td> <td> -2.5e+04</td> <td> 4.71e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th> <td> 1.379e+04</td> <td> 1.89e+04</td> <td>    0.731</td> <td> 0.465</td> <td>-2.32e+04</td> <td> 5.08e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th> <td> -2.15e+04</td> <td> 1.43e+04</td> <td>   -1.504</td> <td> 0.133</td> <td>-4.95e+04</td> <td> 6528.999</td>
</tr>
<tr>
  <th>zipcode_98199</th> <td> 3.025e+05</td> <td> 2.59e+04</td> <td>   11.685</td> <td> 0.000</td> <td> 2.52e+05</td> <td> 3.53e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21922.754</td> <th>  Durbin-Watson:     </th>  <td>   1.992</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>5315892.814</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.513</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>79.328</td>   <th>  Cond. No.          </th>  <td>2.84e+08</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.84e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.





    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x21180301d60>




    
![png](output_75_2.png)
    


The sqft_lot15 p-value is higher than our alpha of 0.05 meaning that this coefficient is not statistically significant, so we can go ahead and drop this parameter.


```python
df_ohe.drop('sqft_lot15', axis=1, inplace=True)
```

We also wanted to create a feature to see if having a larger sqft (excluding the basement) would have any effect on the sale price of the home.


```python
df_ohe['has_larger_sqft_than_neighbors'] = df_ohe['sqft_living15'] < df_ohe['sqft_above']
df_ohe['has_larger_sqft_than_neighbors'] = df_ohe['has_larger_sqft_than_neighbors'].astype('int')
df_ohe.drop('sqft_living15', axis=1, inplace=True)
df_ohe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>...</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
      <th>has_larger_sqft_than_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
model_lin_reg(df=df_ohe)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.800</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.800</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1027.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>16:12:18</td>     <th>  Log-Likelihood:    </th> <td>-2.8999e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th>  <td>5.801e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21512</td>      <th>  BIC:               </th>  <td>5.808e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    84</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>-2.425e+07</td> <td> 6.28e+06</td> <td>   -3.863</td> <td> 0.000</td> <td>-3.66e+07</td> <td>-1.19e+07</td>
</tr>
<tr>
  <th>bedrooms</th>                       <td> -1.68e+04</td> <td> 1541.928</td> <td>  -10.899</td> <td> 0.000</td> <td>-1.98e+04</td> <td>-1.38e+04</td>
</tr>
<tr>
  <th>bathrooms</th>                      <td> 4.332e+04</td> <td> 2638.007</td> <td>   16.422</td> <td> 0.000</td> <td> 3.82e+04</td> <td> 4.85e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                       <td>    0.2141</td> <td>    0.030</td> <td>    7.166</td> <td> 0.000</td> <td>    0.156</td> <td>    0.273</td>
</tr>
<tr>
  <th>floors</th>                         <td>-6.274e+04</td> <td> 3179.284</td> <td>  -19.734</td> <td> 0.000</td> <td> -6.9e+04</td> <td>-5.65e+04</td>
</tr>
<tr>
  <th>waterfront</th>                     <td> 6.911e+05</td> <td>  1.5e+04</td> <td>   45.966</td> <td> 0.000</td> <td> 6.62e+05</td> <td> 7.21e+05</td>
</tr>
<tr>
  <th>view</th>                           <td>  6.39e+04</td> <td> 1737.190</td> <td>   36.784</td> <td> 0.000</td> <td> 6.05e+04</td> <td> 6.73e+04</td>
</tr>
<tr>
  <th>condition</th>                      <td> 2.946e+04</td> <td> 1960.975</td> <td>   15.022</td> <td> 0.000</td> <td> 2.56e+04</td> <td> 3.33e+04</td>
</tr>
<tr>
  <th>grade</th>                          <td> 6.282e+04</td> <td> 1829.256</td> <td>   34.343</td> <td> 0.000</td> <td> 5.92e+04</td> <td> 6.64e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                     <td>  213.2508</td> <td>    3.187</td> <td>   66.919</td> <td> 0.000</td> <td>  207.005</td> <td>  219.497</td>
</tr>
<tr>
  <th>yr_built</th>                       <td> -881.6538</td> <td>   65.484</td> <td>  -13.464</td> <td> 0.000</td> <td>-1010.006</td> <td> -753.301</td>
</tr>
<tr>
  <th>lat</th>                            <td> 1.978e+05</td> <td> 6.49e+04</td> <td>    3.048</td> <td> 0.002</td> <td> 7.06e+04</td> <td> 3.25e+05</td>
</tr>
<tr>
  <th>long</th>                           <td>-1.312e+05</td> <td> 4.65e+04</td> <td>   -2.819</td> <td> 0.005</td> <td>-2.22e+05</td> <td>   -4e+04</td>
</tr>
<tr>
  <th>renovated</th>                      <td> 4.136e+04</td> <td> 6531.960</td> <td>    6.332</td> <td> 0.000</td> <td> 2.86e+04</td> <td> 5.42e+04</td>
</tr>
<tr>
  <th>has_basement</th>                   <td> 6.363e+04</td> <td> 3051.663</td> <td>   20.850</td> <td> 0.000</td> <td> 5.76e+04</td> <td> 6.96e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>                  <td> 3.298e+04</td> <td> 1.48e+04</td> <td>    2.228</td> <td> 0.026</td> <td> 3963.897</td> <td>  6.2e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>                  <td>-2.736e+04</td> <td> 1.32e+04</td> <td>   -2.067</td> <td> 0.039</td> <td>-5.33e+04</td> <td>-1414.683</td>
</tr>
<tr>
  <th>zipcode_98004</th>                  <td> 7.254e+05</td> <td>  2.4e+04</td> <td>   30.163</td> <td> 0.000</td> <td> 6.78e+05</td> <td> 7.72e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>                  <td> 2.515e+05</td> <td> 2.57e+04</td> <td>    9.786</td> <td> 0.000</td> <td> 2.01e+05</td> <td> 3.02e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>                  <td> 2.386e+05</td> <td>  2.1e+04</td> <td>   11.363</td> <td> 0.000</td> <td> 1.97e+05</td> <td>  2.8e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>                  <td> 1.964e+05</td> <td> 2.65e+04</td> <td>    7.405</td> <td> 0.000</td> <td> 1.44e+05</td> <td> 2.48e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>                  <td> 2.039e+05</td> <td> 2.52e+04</td> <td>    8.092</td> <td> 0.000</td> <td> 1.55e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>                  <td> 9.662e+04</td> <td> 2.26e+04</td> <td>    4.281</td> <td> 0.000</td> <td> 5.24e+04</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>                  <td> 4.068e+04</td> <td> 3.28e+04</td> <td>    1.241</td> <td> 0.215</td> <td>-2.36e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>                  <td> 8.781e+04</td> <td>  3.6e+04</td> <td>    2.438</td> <td> 0.015</td> <td> 1.72e+04</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>                  <td> 4.763e+04</td> <td> 3.55e+04</td> <td>    1.341</td> <td> 0.180</td> <td> -2.2e+04</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>                  <td> 3.037e+04</td> <td> 1.96e+04</td> <td>    1.548</td> <td> 0.122</td> <td>-8073.985</td> <td> 6.88e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>                  <td>-5.138e+04</td> <td> 1.22e+04</td> <td>   -4.219</td> <td> 0.000</td> <td>-7.53e+04</td> <td>-2.75e+04</td>
</tr>
<tr>
  <th>zipcode_98024</th>                  <td> 1.409e+05</td> <td> 3.17e+04</td> <td>    4.448</td> <td> 0.000</td> <td> 7.88e+04</td> <td> 2.03e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>                  <td> 1.551e+05</td> <td> 2.16e+04</td> <td>    7.190</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>                  <td> 3.818e+04</td> <td> 3.18e+04</td> <td>    1.199</td> <td> 0.231</td> <td>-2.42e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>                  <td> 1.889e+05</td> <td> 2.46e+04</td> <td>    7.667</td> <td> 0.000</td> <td> 1.41e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>                  <td>  366.4616</td> <td> 1.46e+04</td> <td>    0.025</td> <td> 0.980</td> <td>-2.82e+04</td> <td> 2.89e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>                  <td> 1079.4560</td> <td> 1.52e+04</td> <td>    0.071</td> <td> 0.943</td> <td>-2.86e+04</td> <td> 3.08e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>                  <td>-7082.3756</td> <td> 1.76e+04</td> <td>   -0.403</td> <td> 0.687</td> <td>-4.16e+04</td> <td> 2.74e+04</td>
</tr>
<tr>
  <th>zipcode_98033</th>                  <td> 2.993e+05</td> <td> 2.73e+04</td> <td>   10.957</td> <td> 0.000</td> <td> 2.46e+05</td> <td> 3.53e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>                  <td>  1.23e+05</td> <td> 2.93e+04</td> <td>    4.199</td> <td> 0.000</td> <td> 6.56e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>                  <td> 4.753e+04</td> <td> 1.63e+04</td> <td>    2.909</td> <td> 0.004</td> <td> 1.55e+04</td> <td> 7.95e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>                  <td> 1.254e+06</td> <td> 3.25e+04</td> <td>   38.575</td> <td> 0.000</td> <td> 1.19e+06</td> <td> 1.32e+06</td>
</tr>
<tr>
  <th>zipcode_98040</th>                  <td> 4.749e+05</td> <td> 2.13e+04</td> <td>   22.348</td> <td> 0.000</td> <td> 4.33e+05</td> <td> 5.17e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>                  <td> 9389.1280</td> <td> 1.39e+04</td> <td>    0.674</td> <td> 0.500</td> <td>-1.79e+04</td> <td> 3.67e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>                  <td> 1.176e+05</td> <td> 3.02e+04</td> <td>    3.896</td> <td> 0.000</td> <td> 5.84e+04</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>                  <td>  1.65e+05</td> <td> 2.79e+04</td> <td>    5.919</td> <td> 0.000</td> <td>  1.1e+05</td> <td>  2.2e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>                  <td> 1.455e+05</td> <td> 2.99e+04</td> <td>    4.869</td> <td> 0.000</td> <td> 8.69e+04</td> <td> 2.04e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>                  <td> 2.435e+04</td> <td> 1.69e+04</td> <td>    1.443</td> <td> 0.149</td> <td>-8737.813</td> <td> 5.74e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>                  <td> 6.823e+04</td> <td> 1.83e+04</td> <td>    3.720</td> <td> 0.000</td> <td> 3.23e+04</td> <td> 1.04e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th>                  <td> 1.332e+04</td> <td> 1.59e+04</td> <td>    0.835</td> <td> 0.404</td> <td>-1.79e+04</td> <td> 4.46e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>                  <td> 5.721e+04</td> <td>  1.8e+04</td> <td>    3.182</td> <td> 0.001</td> <td>  2.2e+04</td> <td> 9.25e+04</td>
</tr>
<tr>
  <th>zipcode_98065</th>                  <td> 8.945e+04</td> <td> 2.78e+04</td> <td>    3.217</td> <td> 0.001</td> <td>  3.5e+04</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>                  <td>-5.994e+04</td> <td> 2.11e+04</td> <td>   -2.847</td> <td> 0.004</td> <td>-1.01e+05</td> <td>-1.87e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>                  <td> 7.646e+04</td> <td> 3.26e+04</td> <td>    2.345</td> <td> 0.019</td> <td> 1.25e+04</td> <td>  1.4e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>                  <td> 1.308e+05</td> <td> 2.64e+04</td> <td>    4.956</td> <td> 0.000</td> <td> 7.91e+04</td> <td> 1.83e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>                  <td> 1.329e+05</td> <td> 2.54e+04</td> <td>    5.243</td> <td> 0.000</td> <td> 8.32e+04</td> <td> 1.83e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>                  <td> 4.759e+04</td> <td> 3.39e+04</td> <td>    1.403</td> <td> 0.161</td> <td>-1.89e+04</td> <td> 1.14e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>                  <td>-2.821e+04</td> <td> 1.32e+04</td> <td>   -2.133</td> <td> 0.033</td> <td>-5.41e+04</td> <td>-2287.050</td>
</tr>
<tr>
  <th>zipcode_98102</th>                  <td> 4.419e+05</td> <td> 2.82e+04</td> <td>   15.669</td> <td> 0.000</td> <td> 3.87e+05</td> <td> 4.97e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>                  <td>  2.63e+05</td> <td> 2.64e+04</td> <td>    9.959</td> <td> 0.000</td> <td> 2.11e+05</td> <td> 3.15e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>                  <td> 3.893e+05</td> <td> 2.71e+04</td> <td>   14.349</td> <td> 0.000</td> <td> 3.36e+05</td> <td> 4.42e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>                  <td> 9.449e+04</td> <td> 1.96e+04</td> <td>    4.831</td> <td> 0.000</td> <td> 5.62e+04</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th>                  <td> 2.624e+05</td> <td> 2.72e+04</td> <td>    9.638</td> <td> 0.000</td> <td> 2.09e+05</td> <td> 3.16e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>                  <td> 8.161e+04</td> <td> 2.16e+04</td> <td>    3.777</td> <td> 0.000</td> <td> 3.93e+04</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th>                  <td> 4.186e+05</td> <td> 2.81e+04</td> <td>   14.918</td> <td> 0.000</td> <td> 3.64e+05</td> <td> 4.74e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>                  <td> 5.554e+05</td> <td> 2.49e+04</td> <td>   22.297</td> <td> 0.000</td> <td> 5.07e+05</td> <td> 6.04e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>                  <td> 2.565e+05</td> <td> 2.69e+04</td> <td>    9.552</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 3.09e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>                  <td> 2.235e+05</td> <td> 2.18e+04</td> <td>   10.229</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>                  <td> 2.317e+05</td> <td> 2.72e+04</td> <td>    8.525</td> <td> 0.000</td> <td> 1.78e+05</td> <td> 2.85e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>                  <td> 1.315e+05</td> <td> 1.91e+04</td> <td>    6.890</td> <td> 0.000</td> <td> 9.41e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>                  <td> 3.945e+05</td> <td> 2.65e+04</td> <td>   14.881</td> <td> 0.000</td> <td> 3.42e+05</td> <td> 4.46e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>                  <td> 2.713e+05</td> <td> 2.36e+04</td> <td>   11.473</td> <td> 0.000</td> <td> 2.25e+05</td> <td> 3.18e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>                  <td> 1.229e+05</td> <td>  2.9e+04</td> <td>    4.237</td> <td> 0.000</td> <td>  6.6e+04</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>                  <td>  1.36e+05</td> <td> 2.01e+04</td> <td>    6.782</td> <td> 0.000</td> <td> 9.67e+04</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>                  <td> 7.711e+04</td> <td> 2.99e+04</td> <td>    2.576</td> <td> 0.010</td> <td> 1.84e+04</td> <td> 1.36e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>                  <td> 1.841e+05</td> <td> 2.06e+04</td> <td>    8.949</td> <td> 0.000</td> <td> 1.44e+05</td> <td> 2.24e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>                  <td> 2.246e+05</td> <td>  2.2e+04</td> <td>   10.218</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.68e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>                  <td> 6.647e+04</td> <td> 1.84e+04</td> <td>    3.621</td> <td> 0.000</td> <td> 3.05e+04</td> <td> 1.02e+05</td>
</tr>
<tr>
  <th>zipcode_98148</th>                  <td> 3.604e+04</td> <td>  2.5e+04</td> <td>    1.443</td> <td> 0.149</td> <td>-1.29e+04</td> <td>  8.5e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>                  <td> 5.694e+04</td> <td> 3.11e+04</td> <td>    1.829</td> <td> 0.067</td> <td>-4090.436</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>                  <td> 1.463e+04</td> <td> 1.68e+04</td> <td>    0.870</td> <td> 0.384</td> <td>-1.83e+04</td> <td> 4.76e+04</td>
</tr>
<tr>
  <th>zipcode_98168</th>                  <td> 4.695e+04</td> <td> 1.78e+04</td> <td>    2.644</td> <td> 0.008</td> <td> 1.21e+04</td> <td> 8.18e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>                  <td> 1.174e+05</td> <td> 3.13e+04</td> <td>    3.757</td> <td> 0.000</td> <td> 5.62e+04</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>                  <td> 1.199e+04</td> <td> 1.83e+04</td> <td>    0.654</td> <td> 0.513</td> <td> -2.4e+04</td> <td>  4.8e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>                  <td> 1.322e+04</td> <td> 1.88e+04</td> <td>    0.702</td> <td> 0.482</td> <td>-2.37e+04</td> <td> 5.01e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>                  <td>-2.128e+04</td> <td> 1.43e+04</td> <td>   -1.491</td> <td> 0.136</td> <td>-4.92e+04</td> <td> 6692.339</td>
</tr>
<tr>
  <th>zipcode_98199</th>                  <td> 3.036e+05</td> <td> 2.58e+04</td> <td>   11.754</td> <td> 0.000</td> <td> 2.53e+05</td> <td> 3.54e+05</td>
</tr>
<tr>
  <th>has_larger_sqft_than_neighbors</th> <td>-3.716e+04</td> <td> 3131.183</td> <td>  -11.869</td> <td> 0.000</td> <td>-4.33e+04</td> <td> -3.1e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21495.431</td> <th>  Durbin-Watson:     </th>  <td>   1.990</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>4815678.728</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.381</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>75.627</td>   <th>  Cond. No.          </th>  <td>2.47e+08</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.47e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.





    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x211802fc5b0>




    
![png](output_80_2.png)
    


As we can see our residuals still seem to not be fitting the normality and homoscedasticity assumptions. To address this we can move on to the outlier removal process.

### Outlier Removal


```python
#Outlier Removal with the IQR method
#function snippet from Flatiron School Phase #2 Py Files.

def find_outliers_IQR(data):
    """Use Tukey's Method of outlier removal AKA InterQuartile-Range Rule
    and return boolean series where True indicates it is an outlier.
    - Calculates the range between the 75% and 25% quartiles
    - Outliers fall outside upper and lower limits, using a treshold of  1.5*IQR the 75% and 25% quartiles.

    IQR Range Calculation:    
        res = df.describe()
        IQR = res['75%'] -  res['25%']
        lower_limit = res['25%'] - 1.5*IQR
        upper_limit = res['75%'] + 1.5*IQR

    Args:
        data (Series,or ndarray): data to test for outliers.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
    >> good_data = df[~idx_outs].copy()
    
    """
    df_b=data
    res= df_b.describe()

    IQR = res['75%'] -  res['25%']
    lower_limit = res['25%'] - 1.5*IQR
    upper_limit = res['75%'] + 1.5*IQR

    idx_outs = (df_b>upper_limit) | (df_b<lower_limit)

    return idx_outs

```


```python
#Making a copy of df_ohe for the second outlier removal process. Refer to next section.
df_IQR_price = df_ohe.copy()
df_IQR_price
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>...</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
      <th>has_larger_sqft_than_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>2009</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>2014</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2009</td>
      <td>47.5944</td>
      <td>-122.299</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>2004</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2008</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 85 columns</p>
</div>



#### Cleaning Outliers From All Numeric Columns


```python
cols_to_check = ['price', 'bedrooms','bathrooms', 'sqft_lot', 'grade', 'sqft_above', 'condition']
```


```python
for col in cols_to_check:
    df_ohe = df_ohe[find_outliers_IQR(df_ohe[col])==False]
    
df_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>...</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
      <th>has_larger_sqft_than_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>2009</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>2014</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2009</td>
      <td>47.5944</td>
      <td>-122.299</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>2004</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2008</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16556 rows × 85 columns</p>
</div>




```python
plot(df=df_clean, target='price')
```


    
![png](output_88_0.png)
    



```python
model = model_lin_reg(df=df_ohe)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.820</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.819</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   890.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:40:52</td>     <th>  Log-Likelihood:    </th> <td>-2.1045e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 16556</td>      <th>  AIC:               </th>  <td>4.211e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 16471</td>      <th>  BIC:               </th>  <td>4.217e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    84</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>-1.459e+06</td> <td> 4.28e+06</td> <td>   -0.341</td> <td> 0.733</td> <td>-9.85e+06</td> <td> 6.94e+06</td>
</tr>
<tr>
  <th>bedrooms</th>                       <td> 2970.0115</td> <td> 1032.190</td> <td>    2.877</td> <td> 0.004</td> <td>  946.808</td> <td> 4993.215</td>
</tr>
<tr>
  <th>bathrooms</th>                      <td> 2.089e+04</td> <td> 1587.628</td> <td>   13.158</td> <td> 0.000</td> <td> 1.78e+04</td> <td>  2.4e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                       <td>    1.9798</td> <td>    0.268</td> <td>    7.380</td> <td> 0.000</td> <td>    1.454</td> <td>    2.506</td>
</tr>
<tr>
  <th>floors</th>                         <td>-2.331e+04</td> <td> 1931.424</td> <td>  -12.066</td> <td> 0.000</td> <td>-2.71e+04</td> <td>-1.95e+04</td>
</tr>
<tr>
  <th>waterfront</th>                     <td> 2.397e+05</td> <td> 1.88e+04</td> <td>   12.747</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>view</th>                           <td> 4.034e+04</td> <td> 1180.302</td> <td>   34.180</td> <td> 0.000</td> <td>  3.8e+04</td> <td> 4.27e+04</td>
</tr>
<tr>
  <th>condition</th>                      <td> 2.517e+04</td> <td> 1104.208</td> <td>   22.795</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 2.73e+04</td>
</tr>
<tr>
  <th>grade</th>                          <td>   4.6e+04</td> <td> 1243.293</td> <td>   37.000</td> <td> 0.000</td> <td> 4.36e+04</td> <td> 4.84e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                     <td>  139.7277</td> <td>    2.321</td> <td>   60.207</td> <td> 0.000</td> <td>  135.179</td> <td>  144.277</td>
</tr>
<tr>
  <th>yr_built</th>                       <td> -591.2032</td> <td>   38.117</td> <td>  -15.510</td> <td> 0.000</td> <td> -665.916</td> <td> -516.490</td>
</tr>
<tr>
  <th>lat</th>                            <td>-4840.7162</td> <td> 4.02e+04</td> <td>   -0.120</td> <td> 0.904</td> <td>-8.36e+04</td> <td> 7.39e+04</td>
</tr>
<tr>
  <th>long</th>                           <td>-1.989e+04</td> <td> 3.25e+04</td> <td>   -0.613</td> <td> 0.540</td> <td>-8.35e+04</td> <td> 4.37e+04</td>
</tr>
<tr>
  <th>renovated</th>                      <td> 3.521e+04</td> <td> 3896.615</td> <td>    9.037</td> <td> 0.000</td> <td> 2.76e+04</td> <td> 4.28e+04</td>
</tr>
<tr>
  <th>has_basement</th>                   <td> 4.734e+04</td> <td> 1730.499</td> <td>   27.359</td> <td> 0.000</td> <td>  4.4e+04</td> <td> 5.07e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>                  <td> 6751.8751</td> <td> 8013.648</td> <td>    0.843</td> <td> 0.399</td> <td>-8955.740</td> <td> 2.25e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>                  <td>-2498.5176</td> <td> 7128.191</td> <td>   -0.351</td> <td> 0.726</td> <td>-1.65e+04</td> <td> 1.15e+04</td>
</tr>
<tr>
  <th>zipcode_98004</th>                  <td> 5.278e+05</td> <td> 1.51e+04</td> <td>   35.017</td> <td> 0.000</td> <td> 4.98e+05</td> <td> 5.57e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>                  <td> 3.347e+05</td> <td> 1.54e+04</td> <td>   21.725</td> <td> 0.000</td> <td> 3.05e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>                  <td> 2.806e+05</td> <td> 1.28e+04</td> <td>   21.865</td> <td> 0.000</td> <td> 2.55e+05</td> <td> 3.06e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>                  <td>  2.48e+05</td> <td> 1.58e+04</td> <td>   15.711</td> <td> 0.000</td> <td> 2.17e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>                  <td> 2.408e+05</td> <td> 1.52e+04</td> <td>   15.812</td> <td> 0.000</td> <td> 2.11e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>                  <td>  8.48e+04</td> <td> 1.61e+04</td> <td>    5.271</td> <td> 0.000</td> <td> 5.33e+04</td> <td> 1.16e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>                  <td> 1.502e+05</td> <td> 1.98e+04</td> <td>    7.596</td> <td> 0.000</td> <td> 1.11e+05</td> <td> 1.89e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>                  <td> 1.139e+05</td> <td> 2.52e+04</td> <td>    4.518</td> <td> 0.000</td> <td> 6.45e+04</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>                  <td> 1.076e+05</td> <td>  2.2e+04</td> <td>    4.895</td> <td> 0.000</td> <td> 6.45e+04</td> <td> 1.51e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>                  <td> 1843.3567</td> <td> 1.28e+04</td> <td>    0.144</td> <td> 0.886</td> <td>-2.33e+04</td> <td> 2.69e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>                  <td>-1.568e+04</td> <td> 6912.313</td> <td>   -2.269</td> <td> 0.023</td> <td>-2.92e+04</td> <td>-2136.085</td>
</tr>
<tr>
  <th>zipcode_98024</th>                  <td> 1.319e+05</td> <td>  2.4e+04</td> <td>    5.484</td> <td> 0.000</td> <td> 8.47e+04</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>                  <td> 2.319e+05</td> <td> 1.41e+04</td> <td>   16.451</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 2.59e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>                  <td>  1.39e+05</td> <td> 1.92e+04</td> <td>    7.226</td> <td> 0.000</td> <td> 1.01e+05</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>                  <td> 2.395e+05</td> <td> 1.54e+04</td> <td>   15.582</td> <td> 0.000</td> <td> 2.09e+05</td> <td>  2.7e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>                  <td> 7538.6621</td> <td> 8010.111</td> <td>    0.941</td> <td> 0.347</td> <td>-8162.021</td> <td> 2.32e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>                  <td> 1.367e+04</td> <td> 8489.042</td> <td>    1.610</td> <td> 0.107</td> <td>-2970.898</td> <td> 3.03e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>                  <td>-7011.8975</td> <td> 9410.308</td> <td>   -0.745</td> <td> 0.456</td> <td>-2.55e+04</td> <td> 1.14e+04</td>
</tr>
<tr>
  <th>zipcode_98033</th>                  <td> 3.175e+05</td> <td> 1.67e+04</td> <td>   19.007</td> <td> 0.000</td> <td> 2.85e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>                  <td> 1.866e+05</td> <td> 1.78e+04</td> <td>   10.467</td> <td> 0.000</td> <td> 1.52e+05</td> <td> 2.22e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>                  <td> 4.712e+04</td> <td> 1.01e+04</td> <td>    4.652</td> <td> 0.000</td> <td> 2.73e+04</td> <td>  6.7e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>                  <td> 6.561e+05</td> <td> 3.57e+04</td> <td>   18.399</td> <td> 0.000</td> <td> 5.86e+05</td> <td> 7.26e+05</td>
</tr>
<tr>
  <th>zipcode_98040</th>                  <td> 4.384e+05</td> <td> 1.34e+04</td> <td>   32.602</td> <td> 0.000</td> <td> 4.12e+05</td> <td> 4.65e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>                  <td> 1.652e+04</td> <td> 8413.643</td> <td>    1.963</td> <td> 0.050</td> <td>   27.240</td> <td>  3.3e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>                  <td>  1.03e+05</td> <td> 1.99e+04</td> <td>    5.185</td> <td> 0.000</td> <td> 6.41e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>                  <td> 2.576e+05</td> <td>  1.7e+04</td> <td>   15.184</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 2.91e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>                  <td> 2.677e+05</td> <td> 1.94e+04</td> <td>   13.793</td> <td> 0.000</td> <td>  2.3e+05</td> <td> 3.06e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>                  <td> 4.165e+04</td> <td> 9647.913</td> <td>    4.317</td> <td> 0.000</td> <td> 2.27e+04</td> <td> 6.06e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>                  <td>  1.02e+05</td> <td> 1.09e+04</td> <td>    9.359</td> <td> 0.000</td> <td> 8.06e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th>                  <td> 3.764e+04</td> <td> 9421.517</td> <td>    3.995</td> <td> 0.000</td> <td> 1.92e+04</td> <td> 5.61e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>                  <td> 9.328e+04</td> <td> 1.08e+04</td> <td>    8.662</td> <td> 0.000</td> <td> 7.22e+04</td> <td> 1.14e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>                  <td> 1.486e+05</td> <td> 1.79e+04</td> <td>    8.310</td> <td> 0.000</td> <td> 1.14e+05</td> <td> 1.84e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>                  <td> 4.969e+04</td> <td> 1.84e+04</td> <td>    2.703</td> <td> 0.007</td> <td> 1.37e+04</td> <td> 8.57e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>                  <td> 1.613e+05</td> <td> 2.03e+04</td> <td>    7.967</td> <td> 0.000</td> <td> 1.22e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>                  <td> 2.159e+05</td> <td> 1.66e+04</td> <td>   13.019</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.48e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>                  <td> 2.516e+05</td> <td> 1.69e+04</td> <td>   14.896</td> <td> 0.000</td> <td> 2.19e+05</td> <td> 2.85e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>                  <td> 1.486e+05</td> <td> 2.49e+04</td> <td>    5.980</td> <td> 0.000</td> <td> 9.99e+04</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>                  <td>-1.782e+04</td> <td> 7594.534</td> <td>   -2.346</td> <td> 0.019</td> <td>-3.27e+04</td> <td>-2929.013</td>
</tr>
<tr>
  <th>zipcode_98102</th>                  <td> 4.117e+05</td> <td> 1.67e+04</td> <td>   24.706</td> <td> 0.000</td> <td> 3.79e+05</td> <td> 4.44e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>                  <td> 3.265e+05</td> <td>  1.6e+04</td> <td>   20.422</td> <td> 0.000</td> <td> 2.95e+05</td> <td> 3.58e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>                  <td> 3.773e+05</td> <td> 1.64e+04</td> <td>   22.969</td> <td> 0.000</td> <td> 3.45e+05</td> <td> 4.09e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>                  <td> 1.215e+05</td> <td> 1.15e+04</td> <td>   10.595</td> <td> 0.000</td> <td> 9.91e+04</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th>                  <td> 3.233e+05</td> <td> 1.63e+04</td> <td>   19.820</td> <td> 0.000</td> <td> 2.91e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>                  <td> 1.229e+05</td> <td> 1.24e+04</td> <td>    9.887</td> <td> 0.000</td> <td> 9.85e+04</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th>                  <td> 4.129e+05</td> <td> 1.68e+04</td> <td>   24.634</td> <td> 0.000</td> <td>  3.8e+05</td> <td> 4.46e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>                  <td> 4.353e+05</td> <td> 1.52e+04</td> <td>   28.561</td> <td> 0.000</td> <td> 4.05e+05</td> <td> 4.65e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>                  <td> 3.232e+05</td> <td> 1.62e+04</td> <td>   19.899</td> <td> 0.000</td> <td> 2.91e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>                  <td> 2.961e+05</td> <td>  1.3e+04</td> <td>   22.728</td> <td> 0.000</td> <td> 2.71e+05</td> <td> 3.22e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>                  <td> 3.126e+05</td> <td> 1.65e+04</td> <td>   18.945</td> <td> 0.000</td> <td>  2.8e+05</td> <td> 3.45e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>                  <td> 1.693e+05</td> <td> 1.13e+04</td> <td>   15.011</td> <td> 0.000</td> <td> 1.47e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>                  <td> 4.056e+05</td> <td> 1.59e+04</td> <td>   25.484</td> <td> 0.000</td> <td> 3.74e+05</td> <td> 4.37e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>                  <td> 3.107e+05</td> <td> 1.41e+04</td> <td>   22.080</td> <td> 0.000</td> <td> 2.83e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>                  <td> 1.964e+05</td> <td> 1.75e+04</td> <td>   11.220</td> <td> 0.000</td> <td> 1.62e+05</td> <td> 2.31e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>                  <td> 1.911e+05</td> <td> 1.18e+04</td> <td>   16.139</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.14e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>                  <td>  1.55e+05</td> <td> 1.81e+04</td> <td>    8.553</td> <td> 0.000</td> <td>  1.2e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>                  <td> 2.547e+05</td> <td> 1.21e+04</td> <td>   21.058</td> <td> 0.000</td> <td> 2.31e+05</td> <td> 2.78e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>                  <td> 2.453e+05</td> <td> 1.31e+04</td> <td>   18.764</td> <td> 0.000</td> <td>  2.2e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>                  <td> 1.054e+05</td> <td> 1.07e+04</td> <td>    9.876</td> <td> 0.000</td> <td> 8.45e+04</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>zipcode_98148</th>                  <td> 4.602e+04</td> <td> 1.33e+04</td> <td>    3.473</td> <td> 0.001</td> <td>    2e+04</td> <td>  7.2e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>                  <td> 1.424e+05</td> <td> 1.88e+04</td> <td>    7.555</td> <td> 0.000</td> <td> 1.05e+05</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>                  <td> 8.588e+04</td> <td> 9838.761</td> <td>    8.729</td> <td> 0.000</td> <td> 6.66e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98168</th>                  <td> 5.147e+04</td> <td> 1.03e+04</td> <td>    4.997</td> <td> 0.000</td> <td> 3.13e+04</td> <td> 7.17e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>                  <td> 2.054e+05</td> <td> 1.89e+04</td> <td>   10.855</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.43e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>                  <td> 5.611e+04</td> <td> 1.05e+04</td> <td>    5.353</td> <td> 0.000</td> <td> 3.56e+04</td> <td> 7.67e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>                  <td> 3.822e+04</td> <td> 1.04e+04</td> <td>    3.669</td> <td> 0.000</td> <td> 1.78e+04</td> <td> 5.86e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>                  <td>  2.03e+04</td> <td> 7895.317</td> <td>    2.571</td> <td> 0.010</td> <td> 4825.827</td> <td> 3.58e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>                  <td> 3.629e+05</td> <td> 1.57e+04</td> <td>   23.181</td> <td> 0.000</td> <td> 3.32e+05</td> <td> 3.94e+05</td>
</tr>
<tr>
  <th>has_larger_sqft_than_neighbors</th> <td>-1.729e+04</td> <td> 1801.693</td> <td>   -9.599</td> <td> 0.000</td> <td>-2.08e+04</td> <td>-1.38e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1493.906</td> <th>  Durbin-Watson:     </th> <td>   1.993</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5130.230</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.437</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.583</td>  <th>  Cond. No.          </th> <td>5.61e+07</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 5.61e+07. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



    
![png](output_89_1.png)
    



```python
model.pvalues[model.pvalues>0.05]
```




    Intercept        0.733445
    lat              0.904094
    long             0.540053
    zipcode_98002    0.399494
    zipcode_98003    0.725959
    zipcode_98022    0.885568
    zipcode_98030    0.346645
    zipcode_98031    0.107387
    zipcode_98032    0.456204
    dtype: float64



From our p-values, we can see that the latitude and longitude values are insignificant which means that we can go ahead and drop these coefficients.


```python
df_ohe.drop(['lat','long'], axis=1, inplace=True)
```


```python
model = model_lin_reg(df=df_ohe)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.820</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.819</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   912.7</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:40:53</td>     <th>  Log-Likelihood:    </th> <td>-2.1045e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 16556</td>      <th>  AIC:               </th>  <td>4.211e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 16473</td>      <th>  BIC:               </th>  <td>4.217e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    82</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td> 7.453e+05</td> <td> 7.43e+04</td> <td>   10.028</td> <td> 0.000</td> <td>    6e+05</td> <td> 8.91e+05</td>
</tr>
<tr>
  <th>bedrooms</th>                       <td> 2967.3423</td> <td> 1032.116</td> <td>    2.875</td> <td> 0.004</td> <td>  944.283</td> <td> 4990.401</td>
</tr>
<tr>
  <th>bathrooms</th>                      <td> 2.089e+04</td> <td> 1587.508</td> <td>   13.159</td> <td> 0.000</td> <td> 1.78e+04</td> <td>  2.4e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                       <td>    1.9769</td> <td>    0.268</td> <td>    7.371</td> <td> 0.000</td> <td>    1.451</td> <td>    2.503</td>
</tr>
<tr>
  <th>floors</th>                         <td>-2.329e+04</td> <td> 1930.880</td> <td>  -12.061</td> <td> 0.000</td> <td>-2.71e+04</td> <td>-1.95e+04</td>
</tr>
<tr>
  <th>waterfront</th>                     <td> 2.398e+05</td> <td> 1.88e+04</td> <td>   12.752</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>view</th>                           <td> 4.036e+04</td> <td> 1179.978</td> <td>   34.201</td> <td> 0.000</td> <td>  3.8e+04</td> <td> 4.27e+04</td>
</tr>
<tr>
  <th>condition</th>                      <td> 2.516e+04</td> <td> 1103.866</td> <td>   22.796</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 2.73e+04</td>
</tr>
<tr>
  <th>grade</th>                          <td> 4.602e+04</td> <td> 1242.410</td> <td>   37.043</td> <td> 0.000</td> <td> 4.36e+04</td> <td> 4.85e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                     <td>  139.7018</td> <td>    2.320</td> <td>   60.210</td> <td> 0.000</td> <td>  135.154</td> <td>  144.250</td>
</tr>
<tr>
  <th>yr_built</th>                       <td> -591.9980</td> <td>   38.092</td> <td>  -15.541</td> <td> 0.000</td> <td> -666.663</td> <td> -517.333</td>
</tr>
<tr>
  <th>renovated</th>                      <td> 3.518e+04</td> <td> 3896.042</td> <td>    9.030</td> <td> 0.000</td> <td> 2.75e+04</td> <td> 4.28e+04</td>
</tr>
<tr>
  <th>has_basement</th>                   <td> 4.735e+04</td> <td> 1730.358</td> <td>   27.365</td> <td> 0.000</td> <td>  4.4e+04</td> <td> 5.07e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>                  <td> 5621.7555</td> <td> 7796.170</td> <td>    0.721</td> <td> 0.471</td> <td>-9659.580</td> <td> 2.09e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>                  <td>-1734.1376</td> <td> 7011.609</td> <td>   -0.247</td> <td> 0.805</td> <td>-1.55e+04</td> <td>  1.2e+04</td>
</tr>
<tr>
  <th>zipcode_98004</th>                  <td> 5.249e+05</td> <td> 8553.918</td> <td>   61.369</td> <td> 0.000</td> <td> 5.08e+05</td> <td> 5.42e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>                  <td> 3.313e+05</td> <td> 9253.837</td> <td>   35.803</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.49e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>                  <td>  2.77e+05</td> <td> 6768.602</td> <td>   40.926</td> <td> 0.000</td> <td> 2.64e+05</td> <td>  2.9e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>                  <td> 2.439e+05</td> <td> 9008.742</td> <td>   27.077</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.62e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>                  <td> 2.362e+05</td> <td> 7104.393</td> <td>   33.253</td> <td> 0.000</td> <td> 2.22e+05</td> <td>  2.5e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>                  <td> 7.967e+04</td> <td> 1.38e+04</td> <td>    5.785</td> <td> 0.000</td> <td> 5.27e+04</td> <td> 1.07e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>                  <td> 1.466e+05</td> <td> 7965.911</td> <td>   18.408</td> <td> 0.000</td> <td> 1.31e+05</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>                  <td> 1.033e+05</td> <td> 1.35e+04</td> <td>    7.655</td> <td> 0.000</td> <td> 7.69e+04</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>                  <td> 9.954e+04</td> <td> 8480.295</td> <td>   11.737</td> <td> 0.000</td> <td> 8.29e+04</td> <td> 1.16e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>                  <td>-3106.2087</td> <td> 8379.925</td> <td>   -0.371</td> <td> 0.711</td> <td>-1.95e+04</td> <td> 1.33e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>                  <td> -1.38e+04</td> <td> 6195.130</td> <td>   -2.227</td> <td> 0.026</td> <td>-2.59e+04</td> <td>-1654.491</td>
</tr>
<tr>
  <th>zipcode_98024</th>                  <td> 1.233e+05</td> <td> 1.78e+04</td> <td>    6.908</td> <td> 0.000</td> <td> 8.83e+04</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>                  <td> 2.265e+05</td> <td> 7533.833</td> <td>   30.059</td> <td> 0.000</td> <td> 2.12e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>                  <td> 1.364e+05</td> <td> 7089.711</td> <td>   19.235</td> <td> 0.000</td> <td> 1.22e+05</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>                  <td>  2.33e+05</td> <td> 7011.710</td> <td>   33.233</td> <td> 0.000</td> <td> 2.19e+05</td> <td> 2.47e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>                  <td> 5647.8650</td> <td> 7142.783</td> <td>    0.791</td> <td> 0.429</td> <td>-8352.761</td> <td> 1.96e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>                  <td>  1.16e+04</td> <td> 7060.845</td> <td>    1.643</td> <td> 0.100</td> <td>-2241.657</td> <td> 2.54e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>                  <td>-7118.2546</td> <td> 9084.315</td> <td>   -0.784</td> <td> 0.433</td> <td>-2.49e+04</td> <td> 1.07e+04</td>
</tr>
<tr>
  <th>zipcode_98033</th>                  <td>  3.14e+05</td> <td> 6664.977</td> <td>   47.108</td> <td> 0.000</td> <td> 3.01e+05</td> <td> 3.27e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>                  <td> 1.833e+05</td> <td> 6081.409</td> <td>   30.142</td> <td> 0.000</td> <td> 1.71e+05</td> <td> 1.95e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>                  <td> 4.218e+04</td> <td> 6123.375</td> <td>    6.889</td> <td> 0.000</td> <td> 3.02e+04</td> <td> 5.42e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>                  <td> 6.537e+05</td> <td> 3.32e+04</td> <td>   19.684</td> <td> 0.000</td> <td> 5.89e+05</td> <td> 7.19e+05</td>
</tr>
<tr>
  <th>zipcode_98040</th>                  <td> 4.362e+05</td> <td> 8660.795</td> <td>   50.370</td> <td> 0.000</td> <td> 4.19e+05</td> <td> 4.53e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>                  <td> 1.316e+04</td> <td> 6237.511</td> <td>    2.111</td> <td> 0.035</td> <td>  938.149</td> <td> 2.54e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>                  <td> 9.222e+04</td> <td> 8411.472</td> <td>   10.963</td> <td> 0.000</td> <td> 7.57e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>                  <td> 2.528e+05</td> <td> 6211.997</td> <td>   40.696</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 2.65e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>                  <td>  2.61e+05</td> <td> 7611.304</td> <td>   34.286</td> <td> 0.000</td> <td> 2.46e+05</td> <td> 2.76e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>                  <td> 3.957e+04</td> <td> 7202.509</td> <td>    5.494</td> <td> 0.000</td> <td> 2.55e+04</td> <td> 5.37e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>                  <td> 9.924e+04</td> <td> 6610.285</td> <td>   15.013</td> <td> 0.000</td> <td> 8.63e+04</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th>                  <td> 3.462e+04</td> <td> 6422.863</td> <td>    5.391</td> <td> 0.000</td> <td>  2.2e+04</td> <td> 4.72e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>                  <td> 8.987e+04</td> <td> 6561.505</td> <td>   13.696</td> <td> 0.000</td> <td>  7.7e+04</td> <td> 1.03e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>                  <td> 1.395e+05</td> <td> 7348.861</td> <td>   18.986</td> <td> 0.000</td> <td> 1.25e+05</td> <td> 1.54e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>                  <td> 5.291e+04</td> <td>  1.7e+04</td> <td>    3.114</td> <td> 0.002</td> <td> 1.96e+04</td> <td> 8.62e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>                  <td> 1.567e+05</td> <td> 8384.514</td> <td>   18.688</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>                  <td> 2.098e+05</td> <td> 6994.896</td> <td>   29.998</td> <td> 0.000</td> <td> 1.96e+05</td> <td> 2.24e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>                  <td> 2.455e+05</td> <td> 9339.484</td> <td>   26.284</td> <td> 0.000</td> <td> 2.27e+05</td> <td> 2.64e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>                  <td> 1.426e+05</td> <td>  1.6e+04</td> <td>    8.928</td> <td> 0.000</td> <td> 1.11e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>                  <td>-1.948e+04</td> <td> 7070.251</td> <td>   -2.755</td> <td> 0.006</td> <td>-3.33e+04</td> <td>-5617.113</td>
</tr>
<tr>
  <th>zipcode_98102</th>                  <td> 4.111e+05</td> <td> 1.05e+04</td> <td>   39.314</td> <td> 0.000</td> <td> 3.91e+05</td> <td> 4.32e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>                  <td> 3.262e+05</td> <td> 6333.106</td> <td>   51.503</td> <td> 0.000</td> <td> 3.14e+05</td> <td> 3.39e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>                  <td> 3.761e+05</td> <td> 8236.550</td> <td>   45.658</td> <td> 0.000</td> <td>  3.6e+05</td> <td> 3.92e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>                  <td> 1.221e+05</td> <td> 6771.130</td> <td>   18.040</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th>                  <td> 3.236e+05</td> <td> 7348.915</td> <td>   44.036</td> <td> 0.000</td> <td> 3.09e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>                  <td> 1.224e+05</td> <td> 7909.632</td> <td>   15.476</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th>                  <td> 4.129e+05</td> <td> 1.05e+04</td> <td>   39.385</td> <td> 0.000</td> <td> 3.92e+05</td> <td> 4.33e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>                  <td> 4.343e+05</td> <td> 8413.952</td> <td>   51.614</td> <td> 0.000</td> <td> 4.18e+05</td> <td> 4.51e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>                  <td>  3.22e+05</td> <td> 6227.641</td> <td>   51.704</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.34e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>                  <td> 2.972e+05</td> <td> 7015.178</td> <td>   42.366</td> <td> 0.000</td> <td> 2.83e+05</td> <td> 3.11e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>                  <td> 3.129e+05</td> <td> 6336.595</td> <td>   49.383</td> <td> 0.000</td> <td>    3e+05</td> <td> 3.25e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>                  <td> 1.682e+05</td> <td> 6318.881</td> <td>   26.620</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 1.81e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>                  <td> 4.059e+05</td> <td> 8612.640</td> <td>   47.123</td> <td> 0.000</td> <td> 3.89e+05</td> <td> 4.23e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>                  <td> 3.098e+05</td> <td> 7390.263</td> <td>   41.914</td> <td> 0.000</td> <td> 2.95e+05</td> <td> 3.24e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>                  <td>  1.95e+05</td> <td> 6465.387</td> <td>   30.163</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.08e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>                  <td>  1.92e+05</td> <td> 6757.688</td> <td>   28.408</td> <td> 0.000</td> <td> 1.79e+05</td> <td> 2.05e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>                  <td> 1.544e+05</td> <td> 6158.422</td> <td>   25.075</td> <td> 0.000</td> <td> 1.42e+05</td> <td> 1.66e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>                  <td> 2.559e+05</td> <td> 7318.534</td> <td>   34.961</td> <td> 0.000</td> <td> 2.42e+05</td> <td>  2.7e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>                  <td> 2.445e+05</td> <td> 7016.514</td> <td>   34.846</td> <td> 0.000</td> <td> 2.31e+05</td> <td> 2.58e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>                  <td> 1.062e+05</td> <td> 7089.385</td> <td>   14.980</td> <td> 0.000</td> <td> 9.23e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>zipcode_98148</th>                  <td> 4.658e+04</td> <td> 1.22e+04</td> <td>    3.826</td> <td> 0.000</td> <td> 2.27e+04</td> <td> 7.04e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>                  <td> 1.409e+05</td> <td> 6318.872</td> <td>   22.302</td> <td> 0.000</td> <td> 1.29e+05</td> <td> 1.53e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>                  <td> 8.672e+04</td> <td> 7622.316</td> <td>   11.378</td> <td> 0.000</td> <td> 7.18e+04</td> <td> 1.02e+05</td>
</tr>
<tr>
  <th>zipcode_98168</th>                  <td> 5.135e+04</td> <td> 7361.707</td> <td>    6.975</td> <td> 0.000</td> <td> 3.69e+04</td> <td> 6.58e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>                  <td> 2.053e+05</td> <td> 7624.774</td> <td>   26.920</td> <td> 0.000</td> <td>  1.9e+05</td> <td>  2.2e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>                  <td>  5.47e+04</td> <td> 7156.946</td> <td>    7.643</td> <td> 0.000</td> <td> 4.07e+04</td> <td> 6.87e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>                  <td> 3.776e+04</td> <td> 8882.117</td> <td>    4.251</td> <td> 0.000</td> <td> 2.03e+04</td> <td> 5.52e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>                  <td> 2.079e+04</td> <td> 7116.150</td> <td>    2.922</td> <td> 0.003</td> <td> 6844.626</td> <td> 3.47e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>                  <td> 3.638e+05</td> <td> 7274.106</td> <td>   50.009</td> <td> 0.000</td> <td>  3.5e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>has_larger_sqft_than_neighbors</th> <td>-1.728e+04</td> <td> 1801.049</td> <td>   -9.595</td> <td> 0.000</td> <td>-2.08e+04</td> <td>-1.38e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1494.930</td> <th>  Durbin-Watson:     </th> <td>   1.993</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5136.551</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.438</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.585</td>  <th>  Cond. No.          </th> <td>9.83e+05</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 9.83e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



    
![png](output_93_1.png)
    


Thanks to the IQR outlier removal process, our residuals seem to now be better fitting into the normality and homoscedasticity assumptions. 


```python
df_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>...</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
      <th>has_larger_sqft_than_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>2014</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>2004</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2008</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16556 rows × 83 columns</p>
</div>



Since we removed outliers from all numeric columns including our target 'price', we are left with 16556 rows compared to the 21597 we started off with. It makes more sense to remove outliers based on the prices of the homes rather than removing outliers in every single column. This has a potential upside of allowing for there to be more data points and therefore a more accurate model overall.

#### Removing Outliers Based on Price Only


```python
df_IQR_price = df_IQR_price[find_outliers_IQR(df_IQR_price['price'])==False]
df_IQR_price
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>...</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
      <th>has_larger_sqft_than_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>2009</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>2014</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2009</td>
      <td>47.5944</td>
      <td>-122.299</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>2004</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2008</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20439 rows × 85 columns</p>
</div>



As seen above, we are left with approximately 4,000 more data points when we only remove outliers based on price. We still should take a look at the model and whether the residuals have adjusted similarly to the prior model.


```python
model = model_lin_reg(df=df_IQR_price)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.826</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.826</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1152.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:40:55</td>     <th>  Log-Likelihood:    </th> <td>-2.6138e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20439</td>      <th>  AIC:               </th>  <td>5.229e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20354</td>      <th>  BIC:               </th>  <td>5.236e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    84</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>-1.461e+07</td> <td> 3.39e+06</td> <td>   -4.308</td> <td> 0.000</td> <td>-2.13e+07</td> <td>-7.96e+06</td>
</tr>
<tr>
  <th>bedrooms</th>                       <td> -244.3677</td> <td>  850.207</td> <td>   -0.287</td> <td> 0.774</td> <td>-1910.842</td> <td> 1422.106</td>
</tr>
<tr>
  <th>bathrooms</th>                      <td>  1.95e+04</td> <td> 1472.657</td> <td>   13.242</td> <td> 0.000</td> <td> 1.66e+04</td> <td> 2.24e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                       <td>    0.3002</td> <td>    0.017</td> <td>   17.975</td> <td> 0.000</td> <td>    0.268</td> <td>    0.333</td>
</tr>
<tr>
  <th>floors</th>                         <td>-2.479e+04</td> <td> 1761.730</td> <td>  -14.073</td> <td> 0.000</td> <td>-2.82e+04</td> <td>-2.13e+04</td>
</tr>
<tr>
  <th>waterfront</th>                     <td> 1.501e+05</td> <td> 1.31e+04</td> <td>   11.488</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.76e+05</td>
</tr>
<tr>
  <th>view</th>                           <td> 3.731e+04</td> <td> 1052.530</td> <td>   35.448</td> <td> 0.000</td> <td> 3.52e+04</td> <td> 3.94e+04</td>
</tr>
<tr>
  <th>condition</th>                      <td>   2.5e+04</td> <td> 1068.482</td> <td>   23.400</td> <td> 0.000</td> <td> 2.29e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>grade</th>                          <td> 4.831e+04</td> <td> 1030.415</td> <td>   46.879</td> <td> 0.000</td> <td> 4.63e+04</td> <td> 5.03e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                     <td>  131.7060</td> <td>    1.883</td> <td>   69.954</td> <td> 0.000</td> <td>  128.016</td> <td>  135.396</td>
</tr>
<tr>
  <th>yr_built</th>                       <td> -572.7285</td> <td>   36.232</td> <td>  -15.807</td> <td> 0.000</td> <td> -643.747</td> <td> -501.710</td>
</tr>
<tr>
  <th>lat</th>                            <td> 1.545e+05</td> <td> 3.51e+04</td> <td>    4.402</td> <td> 0.000</td> <td> 8.57e+04</td> <td> 2.23e+05</td>
</tr>
<tr>
  <th>long</th>                           <td>-6.569e+04</td> <td> 2.51e+04</td> <td>   -2.621</td> <td> 0.009</td> <td>-1.15e+05</td> <td>-1.66e+04</td>
</tr>
<tr>
  <th>renovated</th>                      <td>  3.36e+04</td> <td> 3726.684</td> <td>    9.017</td> <td> 0.000</td> <td> 2.63e+04</td> <td> 4.09e+04</td>
</tr>
<tr>
  <th>has_basement</th>                   <td> 4.727e+04</td> <td> 1670.349</td> <td>   28.300</td> <td> 0.000</td> <td>  4.4e+04</td> <td> 5.05e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>                  <td> 6083.3919</td> <td> 7819.865</td> <td>    0.778</td> <td> 0.437</td> <td>-9244.173</td> <td> 2.14e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>                  <td>-1.129e+04</td> <td> 6991.183</td> <td>   -1.615</td> <td> 0.106</td> <td> -2.5e+04</td> <td> 2411.862</td>
</tr>
<tr>
  <th>zipcode_98004</th>                  <td> 4.749e+05</td> <td> 1.37e+04</td> <td>   34.573</td> <td> 0.000</td> <td> 4.48e+05</td> <td> 5.02e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>                  <td>  2.94e+05</td> <td> 1.39e+04</td> <td>   21.181</td> <td> 0.000</td> <td> 2.67e+05</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>                  <td> 2.484e+05</td> <td> 1.14e+04</td> <td>   21.769</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>                  <td> 2.177e+05</td> <td> 1.42e+04</td> <td>   15.285</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>                  <td> 2.006e+05</td> <td> 1.36e+04</td> <td>   14.753</td> <td> 0.000</td> <td> 1.74e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>                  <td> 9.965e+04</td> <td>  1.2e+04</td> <td>    8.310</td> <td> 0.000</td> <td> 7.61e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>                  <td> 8.511e+04</td> <td> 1.76e+04</td> <td>    4.825</td> <td> 0.000</td> <td> 5.05e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>                  <td> 8.339e+04</td> <td> 1.94e+04</td> <td>    4.297</td> <td> 0.000</td> <td> 4.54e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>                  <td>  5.97e+04</td> <td> 1.91e+04</td> <td>    3.127</td> <td> 0.002</td> <td> 2.23e+04</td> <td> 9.71e+04</td>
</tr>
<tr>
  <th>zipcode_98022</th>                  <td> 2.401e+04</td> <td> 1.05e+04</td> <td>    2.291</td> <td> 0.022</td> <td> 3470.958</td> <td> 4.45e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>                  <td>-2.911e+04</td> <td> 6450.507</td> <td>   -4.513</td> <td> 0.000</td> <td>-4.18e+04</td> <td>-1.65e+04</td>
</tr>
<tr>
  <th>zipcode_98024</th>                  <td> 1.335e+05</td> <td> 1.72e+04</td> <td>    7.770</td> <td> 0.000</td> <td> 9.98e+04</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>                  <td> 1.777e+05</td> <td> 1.16e+04</td> <td>   15.365</td> <td> 0.000</td> <td> 1.55e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>                  <td> 7.243e+04</td> <td> 1.72e+04</td> <td>    4.221</td> <td> 0.000</td> <td> 3.88e+04</td> <td> 1.06e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>                  <td> 2.089e+05</td> <td> 1.32e+04</td> <td>   15.786</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>                  <td> 1427.2186</td> <td> 7704.548</td> <td>    0.185</td> <td> 0.853</td> <td>-1.37e+04</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>                  <td> 2295.0973</td> <td> 8044.248</td> <td>    0.285</td> <td> 0.775</td> <td>-1.35e+04</td> <td> 1.81e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>                  <td>  -1.8e+04</td> <td> 9302.376</td> <td>   -1.935</td> <td> 0.053</td> <td>-3.62e+04</td> <td>  235.348</td>
</tr>
<tr>
  <th>zipcode_98033</th>                  <td> 2.702e+05</td> <td> 1.48e+04</td> <td>   18.204</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 2.99e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>                  <td>  1.25e+05</td> <td> 1.58e+04</td> <td>    7.906</td> <td> 0.000</td> <td>  9.4e+04</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>                  <td>     5e+04</td> <td> 8718.347</td> <td>    5.735</td> <td> 0.000</td> <td> 3.29e+04</td> <td> 6.71e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>                  <td> 6.029e+05</td> <td> 3.75e+04</td> <td>   16.072</td> <td> 0.000</td> <td> 5.29e+05</td> <td> 6.76e+05</td>
</tr>
<tr>
  <th>zipcode_98040</th>                  <td> 3.904e+05</td> <td> 1.22e+04</td> <td>   32.021</td> <td> 0.000</td> <td> 3.67e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>                  <td> 1.043e+04</td> <td> 7403.975</td> <td>    1.409</td> <td> 0.159</td> <td>-4081.403</td> <td> 2.49e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>                  <td> 1.112e+05</td> <td> 1.62e+04</td> <td>    6.866</td> <td> 0.000</td> <td> 7.94e+04</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>                  <td> 2.092e+05</td> <td>  1.5e+04</td> <td>   13.929</td> <td> 0.000</td> <td>  1.8e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>                  <td> 2.011e+05</td> <td> 1.61e+04</td> <td>   12.466</td> <td> 0.000</td> <td>  1.7e+05</td> <td> 2.33e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>                  <td>  2.19e+04</td> <td> 8989.534</td> <td>    2.436</td> <td> 0.015</td> <td> 4281.690</td> <td> 3.95e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>                  <td> 7.814e+04</td> <td> 9814.522</td> <td>    7.961</td> <td> 0.000</td> <td> 5.89e+04</td> <td> 9.74e+04</td>
</tr>
<tr>
  <th>zipcode_98058</th>                  <td> 2.414e+04</td> <td> 8506.076</td> <td>    2.838</td> <td> 0.005</td> <td> 7466.703</td> <td> 4.08e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>                  <td> 8.623e+04</td> <td> 9615.356</td> <td>    8.968</td> <td> 0.000</td> <td> 6.74e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>                  <td> 1.264e+05</td> <td> 1.49e+04</td> <td>    8.473</td> <td> 0.000</td> <td> 9.72e+04</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>                  <td>   7.2e+04</td> <td> 1.14e+04</td> <td>    6.334</td> <td> 0.000</td> <td> 4.97e+04</td> <td> 9.43e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>                  <td> 1.227e+05</td> <td> 1.76e+04</td> <td>    6.978</td> <td> 0.000</td> <td> 8.82e+04</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>                  <td>  1.88e+05</td> <td> 1.42e+04</td> <td>   13.231</td> <td> 0.000</td> <td>  1.6e+05</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>                  <td> 2.145e+05</td> <td> 1.37e+04</td> <td>   15.671</td> <td> 0.000</td> <td> 1.88e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>                  <td> 1.269e+05</td> <td> 1.83e+04</td> <td>    6.929</td> <td> 0.000</td> <td>  9.1e+04</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>                  <td>-7663.2632</td> <td> 7003.594</td> <td>   -1.094</td> <td> 0.274</td> <td>-2.14e+04</td> <td> 6064.345</td>
</tr>
<tr>
  <th>zipcode_98102</th>                  <td>  3.42e+05</td> <td> 1.55e+04</td> <td>   22.013</td> <td> 0.000</td> <td> 3.12e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>                  <td> 2.507e+05</td> <td> 1.43e+04</td> <td>   17.591</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>                  <td> 3.089e+05</td> <td> 1.49e+04</td> <td>   20.695</td> <td> 0.000</td> <td>  2.8e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>                  <td> 7.308e+04</td> <td> 1.05e+04</td> <td>    6.969</td> <td> 0.000</td> <td> 5.25e+04</td> <td> 9.36e+04</td>
</tr>
<tr>
  <th>zipcode_98107</th>                  <td> 2.481e+05</td> <td> 1.47e+04</td> <td>   16.910</td> <td> 0.000</td> <td> 2.19e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>                  <td> 7.486e+04</td> <td> 1.16e+04</td> <td>    6.476</td> <td> 0.000</td> <td> 5.22e+04</td> <td> 9.75e+04</td>
</tr>
<tr>
  <th>zipcode_98109</th>                  <td> 3.432e+05</td> <td> 1.56e+04</td> <td>   21.980</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>                  <td> 3.687e+05</td> <td>  1.4e+04</td> <td>   26.416</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>                  <td> 2.535e+05</td> <td> 1.45e+04</td> <td>   17.488</td> <td> 0.000</td> <td> 2.25e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>                  <td> 2.366e+05</td> <td> 1.18e+04</td> <td>   20.095</td> <td> 0.000</td> <td> 2.13e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>                  <td> 2.387e+05</td> <td> 1.47e+04</td> <td>   16.278</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 2.67e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>                  <td> 1.236e+05</td> <td> 1.02e+04</td> <td>   12.064</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>                  <td> 3.321e+05</td> <td> 1.45e+04</td> <td>   22.835</td> <td> 0.000</td> <td> 3.04e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>                  <td> 2.458e+05</td> <td> 1.28e+04</td> <td>   19.265</td> <td> 0.000</td> <td> 2.21e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>                  <td> 1.291e+05</td> <td> 1.56e+04</td> <td>    8.261</td> <td> 0.000</td> <td> 9.85e+04</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>                  <td> 1.385e+05</td> <td> 1.08e+04</td> <td>   12.873</td> <td> 0.000</td> <td> 1.17e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>                  <td>  7.66e+04</td> <td> 1.61e+04</td> <td>    4.744</td> <td> 0.000</td> <td>  4.5e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>                  <td> 2.047e+05</td> <td> 1.11e+04</td> <td>   18.520</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>                  <td> 1.918e+05</td> <td> 1.19e+04</td> <td>   16.157</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.15e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>                  <td> 7.139e+04</td> <td> 9838.756</td> <td>    7.256</td> <td> 0.000</td> <td> 5.21e+04</td> <td> 9.07e+04</td>
</tr>
<tr>
  <th>zipcode_98148</th>                  <td> 3.006e+04</td> <td> 1.32e+04</td> <td>    2.273</td> <td> 0.023</td> <td> 4139.841</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>                  <td> 6.584e+04</td> <td> 1.68e+04</td> <td>    3.919</td> <td> 0.000</td> <td> 3.29e+04</td> <td> 9.88e+04</td>
</tr>
<tr>
  <th>zipcode_98166</th>                  <td> 7.007e+04</td> <td> 9010.639</td> <td>    7.777</td> <td> 0.000</td> <td> 5.24e+04</td> <td> 8.77e+04</td>
</tr>
<tr>
  <th>zipcode_98168</th>                  <td> 1.879e+04</td> <td> 9482.888</td> <td>    1.981</td> <td> 0.048</td> <td>  201.759</td> <td> 3.74e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>                  <td> 1.401e+05</td> <td> 1.69e+04</td> <td>    8.267</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>                  <td> 2.552e+04</td> <td> 9804.820</td> <td>    2.602</td> <td> 0.009</td> <td> 6298.076</td> <td> 4.47e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>                  <td> 1.038e+04</td> <td> 9994.135</td> <td>    1.039</td> <td> 0.299</td> <td>-9209.027</td> <td>    3e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>                  <td> 5223.4014</td> <td> 7582.058</td> <td>    0.689</td> <td> 0.491</td> <td>-9638.043</td> <td> 2.01e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>                  <td> 2.926e+05</td> <td>  1.4e+04</td> <td>   20.843</td> <td> 0.000</td> <td> 2.65e+05</td> <td>  3.2e+05</td>
</tr>
<tr>
  <th>has_larger_sqft_than_neighbors</th> <td>-1.718e+04</td> <td> 1719.318</td> <td>   -9.993</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-1.38e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1997.283</td> <th>  Durbin-Watson:     </th> <td>   1.988</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7373.063</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.457</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.797</td>  <th>  Cond. No.          </th> <td>2.38e+08</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.38e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



    
![png](output_100_1.png)
    


Our model has a higher R-sqaured value at 0.826 compared to the prior model at 0.820 and has more data points. We will keep iterating on this model instead of using the prior one due to this reason.


```python
model.pvalues[model.pvalues>0.05]
```




    bedrooms         0.773793
    zipcode_98002    0.436613
    zipcode_98003    0.106306
    zipcode_98030    0.853040
    zipcode_98031    0.775410
    zipcode_98032    0.053031
    zipcode_98042    0.158899
    zipcode_98092    0.273885
    zipcode_98188    0.298986
    zipcode_98198    0.490884
    dtype: float64



Interestingly, we are seeing that bedrooms are no longer a significant coefficient, and therefore not a significant parameter to define the sales price of a home based on our model. We can keep the 'bedrooms' parameter in for the time being to see if anything changes when we scale the model to compare our parameters' effects against each other. A quick note compared to the previous model we had is that 'lat' and 'long' are significant in this model so we are keeping them in.

### Scaling 


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```


```python
df_IQR_price.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront',
           'view', 'condition', 'grade', 'sqft_above', 'yr_built', 'lat', 'long',
           'renovated', 'has_basement', 'zipcode_98002', 'zipcode_98003',
           'zipcode_98004', 'zipcode_98005', 'zipcode_98006', 'zipcode_98007',
           'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014',
           'zipcode_98019', 'zipcode_98022', 'zipcode_98023', 'zipcode_98024',
           'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030',
           'zipcode_98031', 'zipcode_98032', 'zipcode_98033', 'zipcode_98034',
           'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042',
           'zipcode_98045', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055',
           'zipcode_98056', 'zipcode_98058', 'zipcode_98059', 'zipcode_98065',
           'zipcode_98070', 'zipcode_98072', 'zipcode_98074', 'zipcode_98075',
           'zipcode_98077', 'zipcode_98092', 'zipcode_98102', 'zipcode_98103',
           'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108',
           'zipcode_98109', 'zipcode_98112', 'zipcode_98115', 'zipcode_98116',
           'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122',
           'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136',
           'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98155',
           'zipcode_98166', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178',
           'zipcode_98188', 'zipcode_98198', 'zipcode_98199',
           'has_larger_sqft_than_neighbors'],
          dtype='object')




```python
numeric_cols = [col for col in df_IQR_price.columns if (col.startswith('zipcode')==False) & (col.startswith('has')==False) & (col!='price')]
```


```python
numeric_cols
```




    ['bedrooms',
     'bathrooms',
     'sqft_lot',
     'floors',
     'waterfront',
     'view',
     'condition',
     'grade',
     'sqft_above',
     'yr_built',
     'lat',
     'long',
     'renovated']




```python
df_scaled = df_IQR_price.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
df_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>renovated</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>...</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
      <th>has_larger_sqft_than_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>-0.364398</td>
      <td>-1.481158</td>
      <td>-0.223340</td>
      <td>-0.886223</td>
      <td>-0.050015</td>
      <td>-0.268494</td>
      <td>-0.628492</td>
      <td>-0.512196</td>
      <td>-0.726541</td>
      <td>-0.542044</td>
      <td>-0.323955</td>
      <td>-0.306790</td>
      <td>-0.177604</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>-0.364398</td>
      <td>0.278964</td>
      <td>-0.183549</td>
      <td>0.977307</td>
      <td>-0.050015</td>
      <td>-0.268494</td>
      <td>-0.628492</td>
      <td>-0.512196</td>
      <td>0.635417</td>
      <td>-0.679226</td>
      <td>1.163507</td>
      <td>-0.742422</td>
      <td>5.630488</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>-1.465513</td>
      <td>-1.481158</td>
      <td>-0.114614</td>
      <td>-0.886223</td>
      <td>-0.050015</td>
      <td>-0.268494</td>
      <td>-0.628492</td>
      <td>-1.477415</td>
      <td>-1.290585</td>
      <td>-1.296546</td>
      <td>1.283326</td>
      <td>-0.138159</td>
      <td>-0.177604</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>0.736716</td>
      <td>1.335036</td>
      <td>-0.239586</td>
      <td>-0.886223</td>
      <td>-0.050015</td>
      <td>-0.268494</td>
      <td>2.462772</td>
      <td>-0.512196</td>
      <td>-0.905384</td>
      <td>-0.199088</td>
      <td>-0.255892</td>
      <td>-1.262369</td>
      <td>-0.177604</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>-0.364398</td>
      <td>-0.073061</td>
      <td>-0.162603</td>
      <td>-0.886223</td>
      <td>-0.050015</td>
      <td>-0.268494</td>
      <td>-0.628492</td>
      <td>0.453023</td>
      <td>-0.038683</td>
      <td>0.555414</td>
      <td>0.424739</td>
      <td>1.182788</td>
      <td>-0.177604</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
model = model_lin_reg(df=df_scaled)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.826</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.826</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1152.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:40:57</td>     <th>  Log-Likelihood:    </th> <td>-2.6138e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20439</td>      <th>  AIC:               </th>  <td>5.229e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20354</td>      <th>  BIC:               </th>  <td>5.236e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    84</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td> 3.302e+05</td> <td>    1e+04</td> <td>   32.869</td> <td> 0.000</td> <td> 3.11e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>bedrooms</th>                       <td> -221.9276</td> <td>  772.133</td> <td>   -0.287</td> <td> 0.774</td> <td>-1735.370</td> <td> 1291.515</td>
</tr>
<tr>
  <th>bathrooms</th>                      <td> 1.385e+04</td> <td> 1045.849</td> <td>   13.242</td> <td> 0.000</td> <td> 1.18e+04</td> <td> 1.59e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                       <td> 1.201e+04</td> <td>  668.299</td> <td>   17.975</td> <td> 0.000</td> <td> 1.07e+04</td> <td> 1.33e+04</td>
</tr>
<tr>
  <th>floors</th>                         <td> -1.33e+04</td> <td>  945.372</td> <td>  -14.073</td> <td> 0.000</td> <td>-1.52e+04</td> <td>-1.15e+04</td>
</tr>
<tr>
  <th>waterfront</th>                     <td> 7486.4714</td> <td>  651.680</td> <td>   11.488</td> <td> 0.000</td> <td> 6209.126</td> <td> 8763.817</td>
</tr>
<tr>
  <th>view</th>                           <td> 2.384e+04</td> <td>  672.630</td> <td>   35.448</td> <td> 0.000</td> <td> 2.25e+04</td> <td> 2.52e+04</td>
</tr>
<tr>
  <th>condition</th>                      <td> 1.618e+04</td> <td>  691.292</td> <td>   23.400</td> <td> 0.000</td> <td> 1.48e+04</td> <td> 1.75e+04</td>
</tr>
<tr>
  <th>grade</th>                          <td> 5.005e+04</td> <td> 1067.545</td> <td>   46.879</td> <td> 0.000</td> <td>  4.8e+04</td> <td> 5.21e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                     <td> 9.574e+04</td> <td> 1368.559</td> <td>   69.954</td> <td> 0.000</td> <td> 9.31e+04</td> <td> 9.84e+04</td>
</tr>
<tr>
  <th>yr_built</th>                       <td> -1.67e+04</td> <td> 1056.472</td> <td>  -15.807</td> <td> 0.000</td> <td>-1.88e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>lat</th>                            <td>  2.18e+04</td> <td> 4951.117</td> <td>    4.402</td> <td> 0.000</td> <td> 1.21e+04</td> <td> 3.15e+04</td>
</tr>
<tr>
  <th>long</th>                           <td>-9349.0429</td> <td> 3567.175</td> <td>   -2.621</td> <td> 0.009</td> <td>-1.63e+04</td> <td>-2357.092</td>
</tr>
<tr>
  <th>renovated</th>                      <td> 5785.7697</td> <td>  641.636</td> <td>    9.017</td> <td> 0.000</td> <td> 4528.111</td> <td> 7043.429</td>
</tr>
<tr>
  <th>has_basement</th>                   <td> 4.727e+04</td> <td> 1670.349</td> <td>   28.300</td> <td> 0.000</td> <td>  4.4e+04</td> <td> 5.05e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>                  <td> 6083.3919</td> <td> 7819.865</td> <td>    0.778</td> <td> 0.437</td> <td>-9244.173</td> <td> 2.14e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>                  <td>-1.129e+04</td> <td> 6991.183</td> <td>   -1.615</td> <td> 0.106</td> <td> -2.5e+04</td> <td> 2411.862</td>
</tr>
<tr>
  <th>zipcode_98004</th>                  <td> 4.749e+05</td> <td> 1.37e+04</td> <td>   34.573</td> <td> 0.000</td> <td> 4.48e+05</td> <td> 5.02e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>                  <td>  2.94e+05</td> <td> 1.39e+04</td> <td>   21.181</td> <td> 0.000</td> <td> 2.67e+05</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>                  <td> 2.484e+05</td> <td> 1.14e+04</td> <td>   21.769</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>                  <td> 2.177e+05</td> <td> 1.42e+04</td> <td>   15.285</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>                  <td> 2.006e+05</td> <td> 1.36e+04</td> <td>   14.753</td> <td> 0.000</td> <td> 1.74e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>                  <td> 9.965e+04</td> <td>  1.2e+04</td> <td>    8.310</td> <td> 0.000</td> <td> 7.61e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>                  <td> 8.511e+04</td> <td> 1.76e+04</td> <td>    4.825</td> <td> 0.000</td> <td> 5.05e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>                  <td> 8.339e+04</td> <td> 1.94e+04</td> <td>    4.297</td> <td> 0.000</td> <td> 4.54e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>                  <td>  5.97e+04</td> <td> 1.91e+04</td> <td>    3.127</td> <td> 0.002</td> <td> 2.23e+04</td> <td> 9.71e+04</td>
</tr>
<tr>
  <th>zipcode_98022</th>                  <td> 2.401e+04</td> <td> 1.05e+04</td> <td>    2.291</td> <td> 0.022</td> <td> 3470.958</td> <td> 4.45e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>                  <td>-2.911e+04</td> <td> 6450.507</td> <td>   -4.513</td> <td> 0.000</td> <td>-4.18e+04</td> <td>-1.65e+04</td>
</tr>
<tr>
  <th>zipcode_98024</th>                  <td> 1.335e+05</td> <td> 1.72e+04</td> <td>    7.770</td> <td> 0.000</td> <td> 9.98e+04</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>                  <td> 1.777e+05</td> <td> 1.16e+04</td> <td>   15.365</td> <td> 0.000</td> <td> 1.55e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>                  <td> 7.243e+04</td> <td> 1.72e+04</td> <td>    4.221</td> <td> 0.000</td> <td> 3.88e+04</td> <td> 1.06e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>                  <td> 2.089e+05</td> <td> 1.32e+04</td> <td>   15.786</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>                  <td> 1427.2186</td> <td> 7704.548</td> <td>    0.185</td> <td> 0.853</td> <td>-1.37e+04</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>                  <td> 2295.0973</td> <td> 8044.248</td> <td>    0.285</td> <td> 0.775</td> <td>-1.35e+04</td> <td> 1.81e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>                  <td>  -1.8e+04</td> <td> 9302.376</td> <td>   -1.935</td> <td> 0.053</td> <td>-3.62e+04</td> <td>  235.348</td>
</tr>
<tr>
  <th>zipcode_98033</th>                  <td> 2.702e+05</td> <td> 1.48e+04</td> <td>   18.204</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 2.99e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>                  <td>  1.25e+05</td> <td> 1.58e+04</td> <td>    7.906</td> <td> 0.000</td> <td>  9.4e+04</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>                  <td>     5e+04</td> <td> 8718.347</td> <td>    5.735</td> <td> 0.000</td> <td> 3.29e+04</td> <td> 6.71e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>                  <td> 6.029e+05</td> <td> 3.75e+04</td> <td>   16.072</td> <td> 0.000</td> <td> 5.29e+05</td> <td> 6.76e+05</td>
</tr>
<tr>
  <th>zipcode_98040</th>                  <td> 3.904e+05</td> <td> 1.22e+04</td> <td>   32.021</td> <td> 0.000</td> <td> 3.67e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>                  <td> 1.043e+04</td> <td> 7403.975</td> <td>    1.409</td> <td> 0.159</td> <td>-4081.403</td> <td> 2.49e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>                  <td> 1.112e+05</td> <td> 1.62e+04</td> <td>    6.866</td> <td> 0.000</td> <td> 7.94e+04</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>                  <td> 2.092e+05</td> <td>  1.5e+04</td> <td>   13.929</td> <td> 0.000</td> <td>  1.8e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>                  <td> 2.011e+05</td> <td> 1.61e+04</td> <td>   12.466</td> <td> 0.000</td> <td>  1.7e+05</td> <td> 2.33e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>                  <td>  2.19e+04</td> <td> 8989.534</td> <td>    2.436</td> <td> 0.015</td> <td> 4281.690</td> <td> 3.95e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>                  <td> 7.814e+04</td> <td> 9814.522</td> <td>    7.961</td> <td> 0.000</td> <td> 5.89e+04</td> <td> 9.74e+04</td>
</tr>
<tr>
  <th>zipcode_98058</th>                  <td> 2.414e+04</td> <td> 8506.076</td> <td>    2.838</td> <td> 0.005</td> <td> 7466.703</td> <td> 4.08e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>                  <td> 8.623e+04</td> <td> 9615.356</td> <td>    8.968</td> <td> 0.000</td> <td> 6.74e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>                  <td> 1.264e+05</td> <td> 1.49e+04</td> <td>    8.473</td> <td> 0.000</td> <td> 9.72e+04</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>                  <td>   7.2e+04</td> <td> 1.14e+04</td> <td>    6.334</td> <td> 0.000</td> <td> 4.97e+04</td> <td> 9.43e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>                  <td> 1.227e+05</td> <td> 1.76e+04</td> <td>    6.978</td> <td> 0.000</td> <td> 8.82e+04</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>                  <td>  1.88e+05</td> <td> 1.42e+04</td> <td>   13.231</td> <td> 0.000</td> <td>  1.6e+05</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>                  <td> 2.145e+05</td> <td> 1.37e+04</td> <td>   15.671</td> <td> 0.000</td> <td> 1.88e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>                  <td> 1.269e+05</td> <td> 1.83e+04</td> <td>    6.929</td> <td> 0.000</td> <td>  9.1e+04</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>                  <td>-7663.2632</td> <td> 7003.594</td> <td>   -1.094</td> <td> 0.274</td> <td>-2.14e+04</td> <td> 6064.345</td>
</tr>
<tr>
  <th>zipcode_98102</th>                  <td>  3.42e+05</td> <td> 1.55e+04</td> <td>   22.013</td> <td> 0.000</td> <td> 3.12e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>                  <td> 2.507e+05</td> <td> 1.43e+04</td> <td>   17.591</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>                  <td> 3.089e+05</td> <td> 1.49e+04</td> <td>   20.695</td> <td> 0.000</td> <td>  2.8e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>                  <td> 7.308e+04</td> <td> 1.05e+04</td> <td>    6.969</td> <td> 0.000</td> <td> 5.25e+04</td> <td> 9.36e+04</td>
</tr>
<tr>
  <th>zipcode_98107</th>                  <td> 2.481e+05</td> <td> 1.47e+04</td> <td>   16.910</td> <td> 0.000</td> <td> 2.19e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>                  <td> 7.486e+04</td> <td> 1.16e+04</td> <td>    6.476</td> <td> 0.000</td> <td> 5.22e+04</td> <td> 9.75e+04</td>
</tr>
<tr>
  <th>zipcode_98109</th>                  <td> 3.432e+05</td> <td> 1.56e+04</td> <td>   21.980</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>                  <td> 3.687e+05</td> <td>  1.4e+04</td> <td>   26.416</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>                  <td> 2.535e+05</td> <td> 1.45e+04</td> <td>   17.488</td> <td> 0.000</td> <td> 2.25e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>                  <td> 2.366e+05</td> <td> 1.18e+04</td> <td>   20.095</td> <td> 0.000</td> <td> 2.13e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>                  <td> 2.387e+05</td> <td> 1.47e+04</td> <td>   16.278</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 2.67e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>                  <td> 1.236e+05</td> <td> 1.02e+04</td> <td>   12.064</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>                  <td> 3.321e+05</td> <td> 1.45e+04</td> <td>   22.835</td> <td> 0.000</td> <td> 3.04e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>                  <td> 2.458e+05</td> <td> 1.28e+04</td> <td>   19.265</td> <td> 0.000</td> <td> 2.21e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>                  <td> 1.291e+05</td> <td> 1.56e+04</td> <td>    8.261</td> <td> 0.000</td> <td> 9.85e+04</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>                  <td> 1.385e+05</td> <td> 1.08e+04</td> <td>   12.873</td> <td> 0.000</td> <td> 1.17e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>                  <td>  7.66e+04</td> <td> 1.61e+04</td> <td>    4.744</td> <td> 0.000</td> <td>  4.5e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>                  <td> 2.047e+05</td> <td> 1.11e+04</td> <td>   18.520</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>                  <td> 1.918e+05</td> <td> 1.19e+04</td> <td>   16.157</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.15e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>                  <td> 7.139e+04</td> <td> 9838.756</td> <td>    7.256</td> <td> 0.000</td> <td> 5.21e+04</td> <td> 9.07e+04</td>
</tr>
<tr>
  <th>zipcode_98148</th>                  <td> 3.006e+04</td> <td> 1.32e+04</td> <td>    2.273</td> <td> 0.023</td> <td> 4139.841</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>                  <td> 6.584e+04</td> <td> 1.68e+04</td> <td>    3.919</td> <td> 0.000</td> <td> 3.29e+04</td> <td> 9.88e+04</td>
</tr>
<tr>
  <th>zipcode_98166</th>                  <td> 7.007e+04</td> <td> 9010.639</td> <td>    7.777</td> <td> 0.000</td> <td> 5.24e+04</td> <td> 8.77e+04</td>
</tr>
<tr>
  <th>zipcode_98168</th>                  <td> 1.879e+04</td> <td> 9482.888</td> <td>    1.981</td> <td> 0.048</td> <td>  201.759</td> <td> 3.74e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>                  <td> 1.401e+05</td> <td> 1.69e+04</td> <td>    8.267</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>                  <td> 2.552e+04</td> <td> 9804.820</td> <td>    2.602</td> <td> 0.009</td> <td> 6298.076</td> <td> 4.47e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>                  <td> 1.038e+04</td> <td> 9994.135</td> <td>    1.039</td> <td> 0.299</td> <td>-9209.027</td> <td>    3e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>                  <td> 5223.4014</td> <td> 7582.058</td> <td>    0.689</td> <td> 0.491</td> <td>-9638.043</td> <td> 2.01e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>                  <td> 2.926e+05</td> <td>  1.4e+04</td> <td>   20.843</td> <td> 0.000</td> <td> 2.65e+05</td> <td>  3.2e+05</td>
</tr>
<tr>
  <th>has_larger_sqft_than_neighbors</th> <td>-1.718e+04</td> <td> 1719.318</td> <td>   -9.993</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-1.38e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1997.283</td> <th>  Durbin-Watson:     </th> <td>   1.988</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7373.063</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.457</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.797</td>  <th>  Cond. No.          </th> <td>    291.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    
![png](output_110_1.png)
    


Even after scaling, the bedrooms coefficient seems to be insignificant. This could be due to the actual sqft mattering more and being a better predictor of sales price. We can go ahead and drop the bedrooms column and take a look at the coefficients for further insight.


```python
df_scaled.drop('bedrooms', axis=1, inplace=True)
model = model_lin_reg(df=df_scaled)
```


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.826</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.826</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1166.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Apr 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:40:58</td>     <th>  Log-Likelihood:    </th> <td>-2.6138e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20439</td>      <th>  AIC:               </th>  <td>5.229e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20355</td>      <th>  BIC:               </th>  <td>5.236e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    83</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td> 3.303e+05</td> <td>    1e+04</td> <td>   32.877</td> <td> 0.000</td> <td> 3.11e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>bathrooms</th>                      <td> 1.377e+04</td> <td> 1013.577</td> <td>   13.590</td> <td> 0.000</td> <td> 1.18e+04</td> <td> 1.58e+04</td>
</tr>
<tr>
  <th>sqft_lot</th>                       <td> 1.202e+04</td> <td>  667.475</td> <td>   18.011</td> <td> 0.000</td> <td> 1.07e+04</td> <td> 1.33e+04</td>
</tr>
<tr>
  <th>floors</th>                         <td>-1.329e+04</td> <td>  944.113</td> <td>  -14.078</td> <td> 0.000</td> <td>-1.51e+04</td> <td>-1.14e+04</td>
</tr>
<tr>
  <th>waterfront</th>                     <td> 7489.9003</td> <td>  651.556</td> <td>   11.495</td> <td> 0.000</td> <td> 6212.797</td> <td> 8767.003</td>
</tr>
<tr>
  <th>view</th>                           <td> 2.385e+04</td> <td>  672.182</td> <td>   35.482</td> <td> 0.000</td> <td> 2.25e+04</td> <td> 2.52e+04</td>
</tr>
<tr>
  <th>condition</th>                      <td> 1.617e+04</td> <td>  690.800</td> <td>   23.406</td> <td> 0.000</td> <td> 1.48e+04</td> <td> 1.75e+04</td>
</tr>
<tr>
  <th>grade</th>                          <td> 5.007e+04</td> <td> 1062.780</td> <td>   47.117</td> <td> 0.000</td> <td>  4.8e+04</td> <td> 5.22e+04</td>
</tr>
<tr>
  <th>sqft_above</th>                     <td> 9.563e+04</td> <td> 1314.489</td> <td>   72.748</td> <td> 0.000</td> <td> 9.31e+04</td> <td> 9.82e+04</td>
</tr>
<tr>
  <th>yr_built</th>                       <td>-1.667e+04</td> <td> 1051.449</td> <td>  -15.855</td> <td> 0.000</td> <td>-1.87e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>lat</th>                            <td> 2.181e+04</td> <td> 4950.717</td> <td>    4.406</td> <td> 0.000</td> <td> 1.21e+04</td> <td> 3.15e+04</td>
</tr>
<tr>
  <th>long</th>                           <td>-9339.4918</td> <td> 3566.940</td> <td>   -2.618</td> <td> 0.009</td> <td>-1.63e+04</td> <td>-2348.002</td>
</tr>
<tr>
  <th>renovated</th>                      <td> 5791.9948</td> <td>  641.256</td> <td>    9.032</td> <td> 0.000</td> <td> 4535.081</td> <td> 7048.909</td>
</tr>
<tr>
  <th>has_basement</th>                   <td> 4.717e+04</td> <td> 1636.087</td> <td>   28.833</td> <td> 0.000</td> <td>  4.4e+04</td> <td> 5.04e+04</td>
</tr>
<tr>
  <th>zipcode_98002</th>                  <td> 6079.6824</td> <td> 7819.678</td> <td>    0.777</td> <td> 0.437</td> <td>-9247.516</td> <td> 2.14e+04</td>
</tr>
<tr>
  <th>zipcode_98003</th>                  <td>-1.128e+04</td> <td> 6990.820</td> <td>   -1.613</td> <td> 0.107</td> <td> -2.5e+04</td> <td> 2426.563</td>
</tr>
<tr>
  <th>zipcode_98004</th>                  <td> 4.749e+05</td> <td> 1.37e+04</td> <td>   34.573</td> <td> 0.000</td> <td> 4.48e+05</td> <td> 5.02e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>                  <td> 2.939e+05</td> <td> 1.39e+04</td> <td>   21.180</td> <td> 0.000</td> <td> 2.67e+05</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>                  <td> 2.484e+05</td> <td> 1.14e+04</td> <td>   21.768</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>                  <td> 2.176e+05</td> <td> 1.42e+04</td> <td>   15.283</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>                  <td> 2.005e+05</td> <td> 1.36e+04</td> <td>   14.751</td> <td> 0.000</td> <td> 1.74e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>                  <td> 9.968e+04</td> <td>  1.2e+04</td> <td>    8.313</td> <td> 0.000</td> <td> 7.62e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98011</th>                  <td> 8.508e+04</td> <td> 1.76e+04</td> <td>    4.824</td> <td> 0.000</td> <td> 5.05e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>                  <td> 8.342e+04</td> <td> 1.94e+04</td> <td>    4.299</td> <td> 0.000</td> <td> 4.54e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>                  <td> 5.968e+04</td> <td> 1.91e+04</td> <td>    3.126</td> <td> 0.002</td> <td> 2.23e+04</td> <td> 9.71e+04</td>
</tr>
<tr>
  <th>zipcode_98022</th>                  <td> 2.404e+04</td> <td> 1.05e+04</td> <td>    2.294</td> <td> 0.022</td> <td> 3502.443</td> <td> 4.46e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>                  <td> -2.91e+04</td> <td> 6450.225</td> <td>   -4.512</td> <td> 0.000</td> <td>-4.17e+04</td> <td>-1.65e+04</td>
</tr>
<tr>
  <th>zipcode_98024</th>                  <td> 1.335e+05</td> <td> 1.72e+04</td> <td>    7.771</td> <td> 0.000</td> <td> 9.99e+04</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>                  <td> 1.777e+05</td> <td> 1.16e+04</td> <td>   15.366</td> <td> 0.000</td> <td> 1.55e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>                  <td> 7.239e+04</td> <td> 1.72e+04</td> <td>    4.219</td> <td> 0.000</td> <td> 3.88e+04</td> <td> 1.06e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>                  <td> 2.089e+05</td> <td> 1.32e+04</td> <td>   15.786</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>                  <td> 1409.0238</td> <td> 7704.114</td> <td>    0.183</td> <td> 0.855</td> <td>-1.37e+04</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>                  <td> 2269.2724</td> <td> 8043.564</td> <td>    0.282</td> <td> 0.778</td> <td>-1.35e+04</td> <td>  1.8e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>                  <td>-1.804e+04</td> <td> 9301.251</td> <td>   -1.939</td> <td> 0.053</td> <td>-3.63e+04</td> <td>  195.624</td>
</tr>
<tr>
  <th>zipcode_98033</th>                  <td> 2.701e+05</td> <td> 1.48e+04</td> <td>   18.203</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 2.99e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>                  <td>  1.25e+05</td> <td> 1.58e+04</td> <td>    7.904</td> <td> 0.000</td> <td>  9.4e+04</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>                  <td> 5.001e+04</td> <td> 8718.135</td> <td>    5.736</td> <td> 0.000</td> <td> 3.29e+04</td> <td> 6.71e+04</td>
</tr>
<tr>
  <th>zipcode_98039</th>                  <td> 6.029e+05</td> <td> 3.75e+04</td> <td>   16.071</td> <td> 0.000</td> <td> 5.29e+05</td> <td> 6.76e+05</td>
</tr>
<tr>
  <th>zipcode_98040</th>                  <td> 3.904e+05</td> <td> 1.22e+04</td> <td>   32.021</td> <td> 0.000</td> <td> 3.66e+05</td> <td> 4.14e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>                  <td> 1.043e+04</td> <td> 7403.795</td> <td>    1.408</td> <td> 0.159</td> <td>-4085.059</td> <td> 2.49e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>                  <td> 1.111e+05</td> <td> 1.62e+04</td> <td>    6.865</td> <td> 0.000</td> <td> 7.94e+04</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>                  <td> 2.092e+05</td> <td>  1.5e+04</td> <td>   13.927</td> <td> 0.000</td> <td>  1.8e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>                  <td> 2.012e+05</td> <td> 1.61e+04</td> <td>   12.472</td> <td> 0.000</td> <td>  1.7e+05</td> <td> 2.33e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>                  <td> 2.192e+04</td> <td> 8989.182</td> <td>    2.438</td> <td> 0.015</td> <td> 4297.280</td> <td> 3.95e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>                  <td> 7.812e+04</td> <td> 9814.117</td> <td>    7.960</td> <td> 0.000</td> <td> 5.89e+04</td> <td> 9.74e+04</td>
</tr>
<tr>
  <th>zipcode_98058</th>                  <td> 2.411e+04</td> <td> 8505.148</td> <td>    2.834</td> <td> 0.005</td> <td> 7436.354</td> <td> 4.08e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>                  <td>  8.62e+04</td> <td> 9614.416</td> <td>    8.965</td> <td> 0.000</td> <td> 6.74e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>                  <td> 1.264e+05</td> <td> 1.49e+04</td> <td>    8.474</td> <td> 0.000</td> <td> 9.72e+04</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>                  <td> 7.211e+04</td> <td> 1.14e+04</td> <td>    6.348</td> <td> 0.000</td> <td> 4.98e+04</td> <td> 9.44e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>                  <td> 1.227e+05</td> <td> 1.76e+04</td> <td>    6.978</td> <td> 0.000</td> <td> 8.82e+04</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>                  <td>  1.88e+05</td> <td> 1.42e+04</td> <td>   13.230</td> <td> 0.000</td> <td>  1.6e+05</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>                  <td> 2.145e+05</td> <td> 1.37e+04</td> <td>   15.671</td> <td> 0.000</td> <td> 1.88e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>                  <td> 1.269e+05</td> <td> 1.83e+04</td> <td>    6.929</td> <td> 0.000</td> <td>  9.1e+04</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>                  <td>-7660.1869</td> <td> 7003.428</td> <td>   -1.094</td> <td> 0.274</td> <td>-2.14e+04</td> <td> 6067.095</td>
</tr>
<tr>
  <th>zipcode_98102</th>                  <td>  3.42e+05</td> <td> 1.55e+04</td> <td>   22.018</td> <td> 0.000</td> <td> 3.12e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>                  <td> 2.507e+05</td> <td> 1.43e+04</td> <td>   17.593</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>                  <td> 3.089e+05</td> <td> 1.49e+04</td> <td>   20.694</td> <td> 0.000</td> <td>  2.8e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>                  <td> 7.308e+04</td> <td> 1.05e+04</td> <td>    6.970</td> <td> 0.000</td> <td> 5.25e+04</td> <td> 9.36e+04</td>
</tr>
<tr>
  <th>zipcode_98107</th>                  <td> 2.482e+05</td> <td> 1.47e+04</td> <td>   16.914</td> <td> 0.000</td> <td> 2.19e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>                  <td> 7.488e+04</td> <td> 1.16e+04</td> <td>    6.478</td> <td> 0.000</td> <td> 5.22e+04</td> <td> 9.75e+04</td>
</tr>
<tr>
  <th>zipcode_98109</th>                  <td> 3.433e+05</td> <td> 1.56e+04</td> <td>   21.986</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>                  <td> 3.688e+05</td> <td>  1.4e+04</td> <td>   26.420</td> <td> 0.000</td> <td> 3.41e+05</td> <td> 3.96e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>                  <td> 2.535e+05</td> <td> 1.45e+04</td> <td>   17.490</td> <td> 0.000</td> <td> 2.25e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>                  <td> 2.366e+05</td> <td> 1.18e+04</td> <td>   20.103</td> <td> 0.000</td> <td> 2.14e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>                  <td> 2.387e+05</td> <td> 1.47e+04</td> <td>   16.282</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 2.67e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>                  <td> 1.236e+05</td> <td> 1.02e+04</td> <td>   12.067</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>                  <td> 3.321e+05</td> <td> 1.45e+04</td> <td>   22.841</td> <td> 0.000</td> <td> 3.04e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>                  <td> 2.459e+05</td> <td> 1.28e+04</td> <td>   19.268</td> <td> 0.000</td> <td> 2.21e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>                  <td> 1.291e+05</td> <td> 1.56e+04</td> <td>    8.260</td> <td> 0.000</td> <td> 9.85e+04</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>                  <td> 1.386e+05</td> <td> 1.08e+04</td> <td>   12.884</td> <td> 0.000</td> <td> 1.17e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>                  <td> 7.658e+04</td> <td> 1.61e+04</td> <td>    4.743</td> <td> 0.000</td> <td> 4.49e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>                  <td> 2.047e+05</td> <td>  1.1e+04</td> <td>   18.531</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.26e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>                  <td> 1.918e+05</td> <td> 1.19e+04</td> <td>   16.160</td> <td> 0.000</td> <td> 1.69e+05</td> <td> 2.15e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>                  <td> 7.139e+04</td> <td> 9838.519</td> <td>    7.257</td> <td> 0.000</td> <td> 5.21e+04</td> <td> 9.07e+04</td>
</tr>
<tr>
  <th>zipcode_98148</th>                  <td> 3.009e+04</td> <td> 1.32e+04</td> <td>    2.275</td> <td> 0.023</td> <td> 4163.796</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>                  <td> 6.579e+04</td> <td> 1.68e+04</td> <td>    3.917</td> <td> 0.000</td> <td> 3.29e+04</td> <td> 9.87e+04</td>
</tr>
<tr>
  <th>zipcode_98166</th>                  <td> 7.008e+04</td> <td> 9010.377</td> <td>    7.778</td> <td> 0.000</td> <td> 5.24e+04</td> <td> 8.77e+04</td>
</tr>
<tr>
  <th>zipcode_98168</th>                  <td>  1.88e+04</td> <td> 9482.544</td> <td>    1.983</td> <td> 0.047</td> <td>  216.730</td> <td> 3.74e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>                  <td> 1.401e+05</td> <td> 1.69e+04</td> <td>    8.267</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>                  <td> 2.549e+04</td> <td> 9804.095</td> <td>    2.600</td> <td> 0.009</td> <td> 6270.930</td> <td> 4.47e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>                  <td> 1.035e+04</td> <td> 9993.181</td> <td>    1.035</td> <td> 0.301</td> <td>-9241.852</td> <td> 2.99e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>                  <td> 5236.4108</td> <td> 7581.752</td> <td>    0.691</td> <td> 0.490</td> <td>-9624.433</td> <td> 2.01e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>                  <td> 2.926e+05</td> <td>  1.4e+04</td> <td>   20.850</td> <td> 0.000</td> <td> 2.65e+05</td> <td>  3.2e+05</td>
</tr>
<tr>
  <th>has_larger_sqft_than_neighbors</th> <td>-1.719e+04</td> <td> 1718.780</td> <td>  -10.003</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-1.38e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1998.403</td> <th>  Durbin-Watson:     </th> <td>   1.987</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7377.212</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.458</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.797</td>  <th>  Cond. No.          </th> <td>    284.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    
![png](output_112_1.png)
    


# iNTERPRET

## Final Scaled Coefficients


```python
coeffs = model.params.sort_values().to_frame('coeffs')
coeffs['abs'] = coeffs['coeffs'].abs()
coeffs.sort_values('abs', ascending=False, inplace=True)
```


```python
coeffs[~coeffs.index.str.startswith('zipcode')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coeffs</th>
      <th>abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>330273.764168</td>
      <td>330273.764168</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>95626.887282</td>
      <td>95626.887282</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>50074.835277</td>
      <td>50074.835277</td>
    </tr>
    <tr>
      <th>has_basement</th>
      <td>47173.818402</td>
      <td>47173.818402</td>
    </tr>
    <tr>
      <th>view</th>
      <td>23850.371935</td>
      <td>23850.371935</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>21810.707948</td>
      <td>21810.707948</td>
    </tr>
    <tr>
      <th>has_larger_sqft_than_neighbors</th>
      <td>-17192.309109</td>
      <td>17192.309109</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>-16670.283873</td>
      <td>16670.283873</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>16169.003737</td>
      <td>16169.003737</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>13774.717849</td>
      <td>13774.717849</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>-13290.789002</td>
      <td>13290.789002</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>12021.914526</td>
      <td>12021.914526</td>
    </tr>
    <tr>
      <th>long</th>
      <td>-9339.491807</td>
      <td>9339.491807</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>7489.900314</td>
      <td>7489.900314</td>
    </tr>
    <tr>
      <th>renovated</th>
      <td>5791.994762</td>
      <td>5791.994762</td>
    </tr>
  </tbody>
</table>
</div>



When compared to each other, we are seeing that the top 3 parameters that affect the sales price of a home are the total sqft above ground (total living sqft, excluding any basements), the grade of construction/finishes as well as whether the house had a basement or not.


```python
df_sqft_above = df_IQR_price.loc[:,['price','sqft_above']]
```


```python
df_sqft_above = df_sqft_above.groupby('sqft_above').mean().reset_index()
```

## Data Visualizations


```python
sns.lmplot(x='sqft_above', y='price', data=df_sqft_above)
```




    <seaborn.axisgrid.FacetGrid at 0x2117fef2a30>




    
![png](output_121_1.png)
    



```python
print(df_sqft_above['sqft_above'].min())
print(df_sqft_above['sqft_above'].max())
```

    370
    5710
    


```python
def categorize(x):
    if (x<1000) & (x > 0):
        val = 'Up to 1000 SF'
    elif (x>=1000) & (x<2000):
        val = '1000-2000 SF'
    elif (x>=2000) & (x<3000):
        val = '2000-3000 SF'
    elif (x>=3000) & (x<4000):
        val = '3000 - 4000 SF'
    elif (x>=4000) & (x<5000):
        val = '4000 - 5000 SF'
    else:
        val = '5000+ SF'
    return val
```


```python
df_sqft_above['Category'] = df_sqft_above['sqft_above'].map(lambda x: categorize(x))
```


```python
df_sqft_above['Category'].value_counts()
```




    2000-3000 SF      272
    1000-2000 SF      263
    3000 - 4000 SF    151
    Up to 1000 SF      78
    4000 - 5000 SF     60
    5000+ SF           12
    Name: Category, dtype: int64




```python
mean_price_per_cat = df_sqft_above.groupby('Category')['price'].mean().reset_index()
```


```python
mean_price_per_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000-2000 SF</td>
      <td>403132.561189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-3000 SF</td>
      <td>496939.230583</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3000 - 4000 SF</td>
      <td>694294.614732</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4000 - 5000 SF</td>
      <td>908895.025397</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5000+ SF</td>
      <td>907491.666667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Up to 1000 SF</td>
      <td>321905.992281</td>
    </tr>
  </tbody>
</table>
</div>




```python
order = ['Up to 1000 SF', '1000-2000 SF', '2000-3000 SF', '3000 - 4000 SF', '4000 - 5000 SF', '5000+ SF']
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(figsize=(10,7))
    sns.barplot(data = df_sqft_above, x='Category', y= 'price', order=order, ci=68)
    ax.set_xlabel('Sqft of Above (Excluding Basement)')
    ax.set_ylabel('Average Price ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')));
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticks(range(0,1000000,100000))
```


    
![png](output_128_0.png)
    



```python
with plt.style.context('seaborn-whitegrid'):
    sns.catplot(data = df_sqft_above, x='Category', y='price', aspect=1.5, order=order)
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_yticks(range(0,1200000,100000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylabel('Sale Price($)')
    fig.set_size_inches(10, 7)
    ax.set_xlabel('Sqft Above (Excluding Basement)')
    ax.set_title('Relationship of Sale Price with Above Ground Sqft')
#     sns.pointplot(x= 'Category', y='price', data=mean_price_per_cat, order=order)
```


    
![png](output_129_0.png)
    


As can be seen above, as the square footage of the house increased sale price of the home also tended to increase with it. Even though there is a spread of price at each category of square footage and therefore some overlaps between them, there is a clear positive trend between sale price and square footage above ground.


```python
df_IQR_price['has_basement'] = df_IQR_price['has_basement'].astype(bool)
```

    <ipython-input-655-8ecde1b6317e>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_IQR_price['has_basement'] = df_IQR_price['has_basement'].astype(bool)
    


```python
from matplotlib.ticker import FuncFormatter
with plt.style.context('seaborn-whitegrid'):

    sns.lmplot(x='sqft_above', y='price', data=df_IQR_price, hue='has_basement', 
               aspect=2, scatter_kws=dict(alpha=0.5), palette=sns.color_palette("Spectral_r"))
    ax = plt.gca()
    ax.set_xlabel('Sqft Above Ground (Excluding Basement)')
    ax.set_ylabel('Sale Price ($)')
    ax.set_title("Relationship of Sale Price with Above Ground Sqft and Having a Basement")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.xaxis.set_ticks(range(0,6500,500))
    ax.set_ylim(0, 1300000)
    plt.gcf().set_size_inches(12,7);
```


    
![png](output_132_0.png)
    


When we look at the relationship between square footage above ground and the home's sale price again, but add in a second parameter to define whether the house had a basement or not, the houses with a basement have a slightly more positive relationship with the sale price. So if a house had a basement, it tended to have a slightly higher price than a comparable home. This is visible from the difference of slopes between the two lines shown above where the green line is diverging from the blue line in a positive way.


```python
df_grade = df_IQR_price.loc[:,['price','grade']]
df_grade
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360000.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400000.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402101.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400000.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325000.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>20439 rows × 2 columns</p>
</div>




```python
def categorize_grade(x):
    if (x<7):
        val = 'Below Average'
    elif (x<=8) & (x>6):
        val = 'Average'
    else:
        val = 'Above Average'
    
    return val
```


```python
df_grade['Category'] = df_grade['grade'].map(lambda x: categorize_grade(x))
```


```python
df_grade['Category'].value_counts()
```




    Average          14905
    Above Average     3227
    Below Average     2307
    Name: Category, dtype: int64




```python
df_grade
df_grade_sqft = pd.concat([df_grade, pd.DataFrame(df_IQR_price['sqft_above'])], axis=1)
df_grade_sqft
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>grade</th>
      <th>Category</th>
      <th>sqft_above</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>7</td>
      <td>Average</td>
      <td>1180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>7</td>
      <td>Average</td>
      <td>2170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>6</td>
      <td>Below Average</td>
      <td>770</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>7</td>
      <td>Average</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>8</td>
      <td>Average</td>
      <td>1680</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360000.0</td>
      <td>8</td>
      <td>Average</td>
      <td>1530</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400000.0</td>
      <td>8</td>
      <td>Average</td>
      <td>2310</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402101.0</td>
      <td>7</td>
      <td>Average</td>
      <td>1020</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400000.0</td>
      <td>8</td>
      <td>Average</td>
      <td>1600</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325000.0</td>
      <td>7</td>
      <td>Average</td>
      <td>1020</td>
    </tr>
  </tbody>
</table>
<p>20439 rows × 4 columns</p>
</div>




```python
mean_price_per_grade = df_grade.groupby('Category')['price'].mean().reset_index()
mean_price_per_grade
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Above Average</td>
      <td>726319.069724</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Average</td>
      <td>450805.678497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Below Average</td>
      <td>294856.879931</td>
    </tr>
  </tbody>
</table>
</div>




```python
with plt.style.context('seaborn-whitegrid'):
    order=['Below Average', 'Average', 'Above Average']
    fig, ax = plt.subplots(figsize=(12,7))
    sns.barplot(data = mean_price_per_grade, x='Category', y= 'price', order=order, 
                palette=sns.color_palette("hls", 14))
    ax.set_xlabel('Grade of Finishes/Construction')
    ax.set_ylabel('Average Sale Price ($)')
    ax.set_title('Average Sale Price per Grade of Finishes and Construction')
    ax.set_ylim(0,1200000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')));
    ax.set_yticks(range(0,1300000,100000))
```


    
![png](output_140_0.png)
    


Another interesting, but also expected relationship is between the grade of finishes and the sale price. Houses that had higher grades of finishes and a better construction quality sold for higher prices.


```python
from matplotlib.ticker import FuncFormatter
with plt.style.context('seaborn-whitegrid'):

    sns.lmplot(x='sqft_above', y='price', data=df_grade_sqft, hue='Category', 
               aspect=2, scatter_kws=dict(alpha=0.7), palette=sns.color_palette("hls", 14))
    ax = plt.gca()
    ax.set_xlabel('Sqft Above Ground (Excluding Basement)')
    ax.set_ylabel('Sale Price ($)')
    ax.set_title("Relationship of Sale Price with Above Ground Sqft and Grade")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.xaxis.set_ticks(range(0,6500,500))
    ax.set_ylim(0, 1300000)
    plt.gcf().set_size_inches(12,7);
```


    
![png](output_142_0.png)
    


When grade is plotted with square footage above ground, the relationship shown in the previous visual becomes even more apparent. As can be seen from the different colored data points, above average homes tended to have a higher sale price. The regression lines for average and above average have a higher positive slope compared to the below average homes, which means that as the square footage of a home increases, the higher graded homes will tend to have higher prices. 


```python
from matplotlib.ticker import FuncFormatter
with plt.style.context('seaborn-whitegrid'):

    sns.lmplot(x='sqft_above', y='price', data=df_IQR_price, hue='grade', 
               aspect=2, scatter_kws=dict(alpha=0.5), palette=sns.color_palette("Spectral_r"))
    ax = plt.gca()
    ax.set_xlabel('Sqft Above Ground (Excluding Basement)')
    ax.set_ylabel('Sale Price ($)')
    ax.set_title("Relationship of Sale Price with Above Ground Sqft and Having a Basement")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.xaxis.set_ticks(range(0,6500,500))
    ax.set_ylim(0, 1300000)
    plt.gcf().set_size_inches(12,7);
```

# CONCLUSIONS & RECOMMENDATIONS

Evaluate how well your work solves the stated business problem.

***
Questions to consider:
* How do you interpret the results?
* How well does your model fit your data? How much better is this than your baseline model?
* How confident are you that your results would generalize beyond the data you have?
* How confident are you that this model would benefit the business if put into use?
***

Provide your conclusions about the work you've done, including any limitations or next steps.

***
Questions to consider:
* What would you recommend the business do as a result of this work?
* What are some reasons why your analysis might not fully solve the business problem?
* What else could you do in the future to improve this project?
***

Even though renovations are usually a lot of effort and stressful to lots of homeowners, they may help increase the property's value. To sum up, our analysis for King County, Washington showed the following:

- Increasing the square footage above ground tends to increase the house's value.
- Focusing on the grade of finishes and the quality of construction as a whole tends to pay dividends when it comes to selling the house.
- Having a basement is the third most effective parameter in increasing a home's sale price. 

Given more time and information about what the homeowner's renovation budget would be, we would have wanted to analyze whether these top 3 parameters would truly be the most effective in bringing a net value increase since a renovation such as adding a basement to a home would be very costly and may not end up returning a net value increase. Additionally, the construction costs in the state of Washington may be higher than other states due to factors such as permitting, material costs, logistical challenges etc. which may effect the net value increase as well. Furthermore, having information about whether the homeowner is thinking about living in the renovated house or renting it out would allow us to fine tune our analysis and bring more valuable insight.
