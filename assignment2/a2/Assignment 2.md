# Task A: Data Wrangling and Analysis on ARD Dataset

# A1. Dataset size

### How many rows and columns exist in this dataset? 


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
data = pd.read_csv('Australian_Road_Deaths.csv',encoding='utf-8')
# read  'monthly_smartcard_replacements.csv'
#A1
data.shape 
#.shape function returns the rows and columns of the data
print('The data has rows number:',data.shape[0])
print('The data has columns number:',data.shape[1])
```

    The data has rows number: 9140
    The data has columns number: 22
    

# A2. The number of unique values in some columns

### Count the number of unique values for National Remoteness Areas, SA4 Name 2016, National LGA Name 2017, and National Road Type in this dataset.


```python
data[['National Remoteness Areas','SA4 Name 2016','National LGA Name 2017','National Road Type']].nunique()
```




    National Remoteness Areas      5
    SA4 Name 2016                 88
    National LGA Name 2017       500
    National Road Type             9
    dtype: int64



The number of unique values of National Remoteness Areas are 5,SA4 Name 2016 are 88, National LGA Name 2017 are 500, National Road Type are 9.

# A3. Missing values and duplicates

### There are some missing values: Unspecified, Undetermined, and blank (NaN) represent missing values.
### 1. How many rows contain missing values (Unspecified or Undetermined or blank) in this dataset? 


```python
data['YYYYMM']=pd.to_datetime(data['YYYYMM'],format='%Y%m') #to_datetime Convert the type of the "YYYYMM" column to date-time format.
data['month']=data['YYYYMM'].dt.month # dt.month get the month of date
data['year']=data['YYYYMM'].dt.year # dt.year get the year of date
```


```python
data = data.replace('Unspecified',np.nan) # Use Replace to replace 'Unspecified'  with the null value 'NaN'
data = data.replace('Undetermined',np.nan)# Use Replace to replace 'Undetermined' with the null value 'NaN'
df = data.loc[data.isnull().T.any()]  
# Isnull () determines whether an element in the data isnull; T is the transpose; Any () determines whether the row has a null value.
df
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
      <th>Crash ID</th>
      <th>State</th>
      <th>YYYYMM</th>
      <th>Day of week</th>
      <th>Time</th>
      <th>Crash Type</th>
      <th>Bus Involvement</th>
      <th>Heavy Rigid Truck Involvement</th>
      <th>Articulated Truck Involvement</th>
      <th>Road User</th>
      <th>...</th>
      <th>National Remoteness Areas</th>
      <th>SA4 Name 2016</th>
      <th>National LGA Name 2017</th>
      <th>National Road Type</th>
      <th>Christmas Period</th>
      <th>Easter Period</th>
      <th>Age Group</th>
      <th>Time of day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20212133</td>
      <td>Vic</td>
      <td>2021-09-01</td>
      <td>Sunday</td>
      <td>0:30:00</td>
      <td>Single</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Motorcycle rider</td>
      <td>...</td>
      <td>Inner Regional Australia</td>
      <td>Melbourne - Outer East</td>
      <td>Yarra Ranges (S)</td>
      <td>Arterial Road</td>
      <td>No</td>
      <td>No</td>
      <td>26_to_39</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20214022</td>
      <td>SA</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>23:31:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Pedestrian</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Adelaide - North</td>
      <td>Playford (C)</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>26_to_39</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20212096</td>
      <td>Vic</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>23:00:00</td>
      <td>Single</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Car passenger</td>
      <td>...</td>
      <td>Inner Regional Australia</td>
      <td>Hume</td>
      <td>Wangaratta (RC)</td>
      <td>Access Road</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20212145</td>
      <td>Vic</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>22:25:00</td>
      <td>Single</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Car driver</td>
      <td>...</td>
      <td>Outer Regional Australia</td>
      <td>Hume</td>
      <td>Wangaratta (RC)</td>
      <td>Arterial Road</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20212075</td>
      <td>Vic</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>5:15:00</td>
      <td>Single</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Motorcycle rider</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Melbourne - South East</td>
      <td>Casey (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
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
      <th>9135</th>
      <td>20142068</td>
      <td>Vic</td>
      <td>2014-01-01</td>
      <td>Monday</td>
      <td>18:20:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car passenger</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>26_to_39</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9136</th>
      <td>20141285</td>
      <td>NSW</td>
      <td>2014-01-01</td>
      <td>Tuesday</td>
      <td>20:50:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car driver</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>65_to_74</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9137</th>
      <td>20143125</td>
      <td>Qld</td>
      <td>2014-01-01</td>
      <td>Friday</td>
      <td>1:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Motorcycle pillion Car passenger</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9138</th>
      <td>20143065</td>
      <td>Qld</td>
      <td>2014-01-01</td>
      <td>Friday</td>
      <td>10:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Car driver</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>26_to_39</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9139</th>
      <td>20141099</td>
      <td>NSW</td>
      <td>2014-01-01</td>
      <td>Wednesday</td>
      <td>13:24:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car driver</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>2302 rows × 24 columns</p>
</div>



There are 2302 rows contain missing values in this dataset

### 2. List the months with no missing values in them.


```python
df['month'].unique() # The month is obtained according to the table containing missing value
```




    array([ 9,  8,  7,  6,  5,  4,  3,  2,  1, 12, 11, 10], dtype=int64)



There are missing values in the data every month

### 3. Remove the records with missing values.


```python
# data.dropna(inplace = True)
data = data.dropna(how ='any',axis =0)
# How ='any 'to delete rows (columns) that contain missing values;Axis =0 or axis='index 'deletes rows with missing values
data
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
      <th>Crash ID</th>
      <th>State</th>
      <th>YYYYMM</th>
      <th>Day of week</th>
      <th>Time</th>
      <th>Crash Type</th>
      <th>Bus Involvement</th>
      <th>Heavy Rigid Truck Involvement</th>
      <th>Articulated Truck Involvement</th>
      <th>Road User</th>
      <th>...</th>
      <th>National Remoteness Areas</th>
      <th>SA4 Name 2016</th>
      <th>National LGA Name 2017</th>
      <th>National Road Type</th>
      <th>Christmas Period</th>
      <th>Easter Period</th>
      <th>Age Group</th>
      <th>Time of day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>20213034</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>4:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Motorcycle rider</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Brisbane - South</td>
      <td>Brisbane (C)</td>
      <td>Busway</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20213026</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Wednesday</td>
      <td>23:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car passenger</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Ipswich</td>
      <td>Ipswich (C)</td>
      <td>National or State Highway</td>
      <td>No</td>
      <td>No</td>
      <td>0_to_16</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20213092</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>2:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car driver</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Logan - Beaudesert</td>
      <td>Logan (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20214053</td>
      <td>SA</td>
      <td>2021-09-01</td>
      <td>Thursday</td>
      <td>21:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car driver</td>
      <td>...</td>
      <td>Inner Regional Australia</td>
      <td>Adelaide - Central and Hills</td>
      <td>Adelaide Hills (DC)</td>
      <td>Sub-Arterial Road</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>11</th>
      <td>20213178</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Sunday</td>
      <td>21:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Motorcycle rider</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Gold Coast</td>
      <td>Gold Coast (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
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
      <th>9106</th>
      <td>20144083</td>
      <td>SA</td>
      <td>2014-01-01</td>
      <td>Friday</td>
      <td>11:10:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Car passenger</td>
      <td>...</td>
      <td>Outer Regional Australia</td>
      <td>South Australia - South East</td>
      <td>The Coorong (DC)</td>
      <td>National or State Highway</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9112</th>
      <td>20145108</td>
      <td>WA</td>
      <td>2014-01-01</td>
      <td>Wednesday</td>
      <td>11:47:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Motorcycle rider</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Perth - South East</td>
      <td>Belmont (C)</td>
      <td>National or State Highway</td>
      <td>Yes</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9121</th>
      <td>20144022</td>
      <td>SA</td>
      <td>2014-01-01</td>
      <td>Monday</td>
      <td>9:35:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Pedestrian</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Adelaide - North</td>
      <td>Tea Tree Gully (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>75_or_older</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9129</th>
      <td>20145072</td>
      <td>WA</td>
      <td>2014-01-01</td>
      <td>Tuesday</td>
      <td>21:30:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Car driver</td>
      <td>...</td>
      <td>Remote Australia</td>
      <td>Western Australia - Outback (South)</td>
      <td>Esperance (S)</td>
      <td>National or State Highway</td>
      <td>No</td>
      <td>No</td>
      <td>75_or_older</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>9131</th>
      <td>20144007</td>
      <td>SA</td>
      <td>2014-01-01</td>
      <td>Tuesday</td>
      <td>20:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Pedestrian</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Adelaide - North</td>
      <td>Playford (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>75_or_older</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>6838 rows × 24 columns</p>
</div>



### 4. Remove duplicates as well after removing the missing values



```python
data = data.drop_duplicates()
data.reset_index()
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
      <th>index</th>
      <th>Crash ID</th>
      <th>State</th>
      <th>YYYYMM</th>
      <th>Day of week</th>
      <th>Time</th>
      <th>Crash Type</th>
      <th>Bus Involvement</th>
      <th>Heavy Rigid Truck Involvement</th>
      <th>Articulated Truck Involvement</th>
      <th>...</th>
      <th>National Remoteness Areas</th>
      <th>SA4 Name 2016</th>
      <th>National LGA Name 2017</th>
      <th>National Road Type</th>
      <th>Christmas Period</th>
      <th>Easter Period</th>
      <th>Age Group</th>
      <th>Time of day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>20213034</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>4:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Brisbane - South</td>
      <td>Brisbane (C)</td>
      <td>Busway</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>20213026</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Wednesday</td>
      <td>23:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Ipswich</td>
      <td>Ipswich (C)</td>
      <td>National or State Highway</td>
      <td>No</td>
      <td>No</td>
      <td>0_to_16</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>20213092</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Saturday</td>
      <td>2:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Logan - Beaudesert</td>
      <td>Logan (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>20214053</td>
      <td>SA</td>
      <td>2021-09-01</td>
      <td>Thursday</td>
      <td>21:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Inner Regional Australia</td>
      <td>Adelaide - Central and Hills</td>
      <td>Adelaide Hills (DC)</td>
      <td>Sub-Arterial Road</td>
      <td>No</td>
      <td>No</td>
      <td>17_to_25</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>20213178</td>
      <td>Qld</td>
      <td>2021-09-01</td>
      <td>Sunday</td>
      <td>21:00:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Gold Coast</td>
      <td>Gold Coast (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Night</td>
      <td>9</td>
      <td>2021</td>
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
      <th>6817</th>
      <td>9106</td>
      <td>20144083</td>
      <td>SA</td>
      <td>2014-01-01</td>
      <td>Friday</td>
      <td>11:10:00</td>
      <td>Multiple</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>...</td>
      <td>Outer Regional Australia</td>
      <td>South Australia - South East</td>
      <td>The Coorong (DC)</td>
      <td>National or State Highway</td>
      <td>No</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>6818</th>
      <td>9112</td>
      <td>20145108</td>
      <td>WA</td>
      <td>2014-01-01</td>
      <td>Wednesday</td>
      <td>11:47:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Perth - South East</td>
      <td>Belmont (C)</td>
      <td>National or State Highway</td>
      <td>Yes</td>
      <td>No</td>
      <td>40_to_64</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>6819</th>
      <td>9121</td>
      <td>20144022</td>
      <td>SA</td>
      <td>2014-01-01</td>
      <td>Monday</td>
      <td>9:35:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Adelaide - North</td>
      <td>Tea Tree Gully (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>75_or_older</td>
      <td>Day</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>6820</th>
      <td>9129</td>
      <td>20145072</td>
      <td>WA</td>
      <td>2014-01-01</td>
      <td>Tuesday</td>
      <td>21:30:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Remote Australia</td>
      <td>Western Australia - Outback (South)</td>
      <td>Esperance (S)</td>
      <td>National or State Highway</td>
      <td>No</td>
      <td>No</td>
      <td>75_or_older</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>6821</th>
      <td>9131</td>
      <td>20144007</td>
      <td>SA</td>
      <td>2014-01-01</td>
      <td>Tuesday</td>
      <td>20:00:00</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>Major Cities of Australia</td>
      <td>Adelaide - North</td>
      <td>Playford (C)</td>
      <td>Local Road</td>
      <td>No</td>
      <td>No</td>
      <td>75_or_older</td>
      <td>Night</td>
      <td>1</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>6822 rows × 25 columns</p>
</div>



# A4. Number of crashes in each month
### List the number of crashes in each month. In which two months are the number of crashes at their largest? 


```python
pd.value_counts(data['month'],sort = True)
```




    3     654
    8     637
    7     596
    1     593
    4     575
    12    565
    6     556
    5     554
    9     531
    10    530
    11    517
    2     514
    Name: month, dtype: int64



March and August had the highest number of crashes

# A5. Investigating crashes over different months for specific road user
### Now look at the Road User and YYYYMM columns and answer the following questions
### 1. Compute the average number of crashes against Month for car drivers. To do this,
### a. Extract Year and Month as separate columns
### b. Compute the number of crashes by both Year and Month for car drivers
### c. Based on task A5-1-b result, compute again the average number of crashes against Month. For each month, the average number of crashes is calculated over different years for which we have collected data for. 


```python
temp = data[['year','Road User','month']]
temp = temp[temp['Road User']=='Car driver']
pd.DataFrame(temp.groupby(['year','month']).size().reset_index(name='crashes num'))
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
      <th>year</th>
      <th>month</th>
      <th>crashes num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>2</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2021</td>
      <td>5</td>
      <td>33</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2021</td>
      <td>6</td>
      <td>41</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2021</td>
      <td>7</td>
      <td>43</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2021</td>
      <td>8</td>
      <td>39</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2021</td>
      <td>9</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 3 columns</p>
</div>



### 2. Draw a chart showing the average number of crashes over different months computed in task A5-1.


```python
temp.groupby(['month']).size().plot(figsize=(20,16))
plt.xlabel('Month')
plt.ylabel('Number of crashes ')
plt.suptitle('The average number of crashes over different months')
plt.show()
```


    
![png](output_25_0.png)
    


### 3. Discuss any interesting point in the chart

The number of car accidents is highest in March and August, and lowest in February. The whole figure is shaped like a cat's head.

# A6. Exploring Speed, National Road Type, and Age

### Now look at the Speed, National Road Type, and Age columns and answer the following questions

### 1. Draw a chart showing the average speed against National Road Type for car drivers


```python
A6 = data[['Speed','Road User','National Road Type','Age']]
A61 = A6[A6['Road User']=='Car driver']
A61 = pd.DataFrame(A61.groupby(['National Road Type'])['Speed'].mean())
A61.plot(figsize=(20,16))
plt.xlabel('National Road Type')
plt.ylabel('The average speed')
plt.suptitle('The average speed against National Road Type for car drivers')
```




    Text(0.5, 0.98, 'The average speed against National Road Type for car drivers')




    
![png](output_31_1.png)
    


### 2. Due to measurement error, there are some counter-intuitive values in Age column.Identify those values and replace them with zero.


```python
data.loc[data.Age<=18,'Age'] = 0
data.loc[data.Age>=90,'Age'] = 0
```

Set the measurement error to the age range of less than 18 years and more than 90 years, and replace these data with zero.

# A7. Relationship between Age, Speed, and Driving Experiences

### 1. Compute pairwise correlation of columns, Age, Speed, and Driving Experiences for vehicle drivers (such as Motorcycle rider). Which two features have the highest linear association? 


```python
data['Road User'].unique()
```




    array(['Motorcycle rider', 'Car passenger', 'Car driver', 'Pedal cyclist',
           'Pedestrian', 'Other vehicle driver',
           'Motorcycle pillion Car passenger'], dtype=object)




```python
A7 = data[['Age','Speed','Driving experience','Road User']]
A7_1 = A7[A7['Road User'].isin(['Motorcycle rider','Car driver','Pedal cyclist','Other vehicle driver'])]
# Collect all qualified road users
A7_1
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
      <th>Age</th>
      <th>Speed</th>
      <th>Driving experience</th>
      <th>Road User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>19</td>
      <td>41</td>
      <td>3</td>
      <td>Motorcycle rider</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>53</td>
      <td>12</td>
      <td>Car driver</td>
    </tr>
    <tr>
      <th>10</th>
      <td>24</td>
      <td>140</td>
      <td>7</td>
      <td>Car driver</td>
    </tr>
    <tr>
      <th>11</th>
      <td>52</td>
      <td>71</td>
      <td>29</td>
      <td>Motorcycle rider</td>
    </tr>
    <tr>
      <th>13</th>
      <td>32</td>
      <td>97</td>
      <td>11</td>
      <td>Car driver</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9093</th>
      <td>34</td>
      <td>130</td>
      <td>14</td>
      <td>Motorcycle rider</td>
    </tr>
    <tr>
      <th>9094</th>
      <td>26</td>
      <td>21</td>
      <td>6</td>
      <td>Car driver</td>
    </tr>
    <tr>
      <th>9105</th>
      <td>45</td>
      <td>125</td>
      <td>14</td>
      <td>Car driver</td>
    </tr>
    <tr>
      <th>9112</th>
      <td>46</td>
      <td>142</td>
      <td>15</td>
      <td>Motorcycle rider</td>
    </tr>
    <tr>
      <th>9129</th>
      <td>84</td>
      <td>74</td>
      <td>43</td>
      <td>Car driver</td>
    </tr>
  </tbody>
</table>
<p>4626 rows × 4 columns</p>
</div>




```python
A7_1.corr()
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
      <th>Age</th>
      <th>Speed</th>
      <th>Driving experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>-0.004005</td>
      <td>0.783145</td>
    </tr>
    <tr>
      <th>Speed</th>
      <td>-0.004005</td>
      <td>1.000000</td>
      <td>-0.007659</td>
    </tr>
    <tr>
      <th>Driving experience</th>
      <td>0.783145</td>
      <td>-0.007659</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The correlation between a variable and itself is one，Age and Driving experience have the strongest linear correlation.

### 2. Now let's look at the relationship between the number of crashes and Driving Experiences.To do this, first compute the number of crashes against Driving Experiences for vehicle drivers and plot the values of these two features against each other. Is there any relationship between these two features? Describe it.


```python
A7_1.groupby(['Driving experience']).size().plot(figsize=(20,16))
plt.ylabel('Number of crashes')
plt.title('Number of crashes against Driving Experiences for vehicle drivers')
```




    Text(0.5, 1.0, 'Number of crashes against Driving Experiences for vehicle drivers')




    
![png](output_42_1.png)
    


# A8. Investigating yearly trend of crash
### We will now investigate the trend in the crash over years. For this, you will need to compute the number of crashes by year.


```python
A8 = data.groupby('year').size().reset_index(name='size')
A8
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
      <th>year</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>301</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>939</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>733</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>1108</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>1173</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>926</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>685</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Fit a linear regression using Python to this data (The number of crashes over different years) and plot the linear fit. 



```python
plt.scatter(A8['year'],A8['size'])
plt.show()
```


    
![png](output_46_0.png)
    



```python
slope, intercept, r_value, p_value, std_err = linregress(A8['year'],A8['size'])
```


```python
print("slope: %f     intercept: %f" % (slope, intercept))
print("r-value: %f" % r_value)
print("p-value: %f" % p_value)
print("std-err: %f" % std_err)
```

    slope: 48.738095     intercept: -97476.357143
    r-value: 0.430510
    p-value: 0.286985
    std-err: 41.715525
    


```python
line = [slope*xi + intercept for xi in A8['year']]
# We can then plot the 'line': 

plt.plot(A8['year'],line,'r-', linewidth=2)
# And add the original data points to the same plot:

plt.scatter(A8['year'], A8['size'])
plt.show() 
```


    
![png](output_49_0.png)
    


### 2. Use the linear fit to predict the number of crashes in 2022.



```python
slope*2022+intercept
```




    1072.0714285714348



The predicted number of collisions in 2022 is 1072

### 3. Can you think of a better model that well captures the trend of yearly crash? Develop a new model and explain why it is better suited for this task.



```python
#import required packages
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#provide data
np.random.seed(0)
x = A8['year'].to_numpy()
y = A8['size'].to_numpy()

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

#create polynomial regression
polynomial_features= PolynomialFeatures(degree= 20)
x_poly = polynomial_features.fit_transform(x)


model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='b')

plt.show()
```

    122.94365591029292
    0.7753620650654187
    


    
![png](output_54_1.png)
    


This new model is better than the original model because the trend in the data is up and down, while the original model can only see a large overall trend, and cannot accurately represent the short-term changes in quantity.  It can also be seen from the initial scatter plot that the trend of the data is not static, which may also be because the year and number of collisions are not highly correlated and not closely correlated.  Therefore, the polynomial model fits the data better than the linear model.

### 4. Use your new model to predict the number of crashes in 2022.


```python
model.predict(polynomial_features.fit_transform([[2022]]))
```




    array([[318.85863018]])



# A9. Filling in missing values
### Rather than replacing some counter-intuitive values with zero in task A6, use a better (e.g., model-based) approach to fill in the counter-intuitive values


```python
A9 = data.groupby(['Age']).size().reset_index(name = 'size')
plt.scatter(A9['Age'],A9['size'])
```




    <matplotlib.collections.PathCollection at 0x2ee2f5cc430>




    
![png](output_59_1.png)
    


# Task B: Decision Tree Classification on Song Popularity Dataset and K-means Clustering on Other Data

# B1. Classification


```python
data = pd.read_csv('song_data.csv',encoding='utf-8')
```


```python
data.head()
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
      <th>song_name</th>
      <th>song_popularity</th>
      <th>song_duration_ms</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>audio_mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>audio_valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Boulevard of Broken Dreams</td>
      <td>4</td>
      <td>262333</td>
      <td>0.005520</td>
      <td>0.496</td>
      <td>0.682</td>
      <td>0.000029</td>
      <td>8</td>
      <td>0.0589</td>
      <td>-4.095</td>
      <td>1</td>
      <td>0.0294</td>
      <td>167.060</td>
      <td>4</td>
      <td>0.474</td>
    </tr>
    <tr>
      <th>1</th>
      <td>In The End</td>
      <td>4</td>
      <td>216933</td>
      <td>0.010300</td>
      <td>0.542</td>
      <td>0.853</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.1080</td>
      <td>-6.407</td>
      <td>0</td>
      <td>0.0498</td>
      <td>105.256</td>
      <td>4</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seven Nation Army</td>
      <td>4</td>
      <td>231733</td>
      <td>0.008170</td>
      <td>0.737</td>
      <td>0.463</td>
      <td>0.447000</td>
      <td>0</td>
      <td>0.2550</td>
      <td>-7.828</td>
      <td>1</td>
      <td>0.0792</td>
      <td>123.881</td>
      <td>4</td>
      <td>0.324</td>
    </tr>
    <tr>
      <th>3</th>
      <td>By The Way</td>
      <td>4</td>
      <td>216933</td>
      <td>0.026400</td>
      <td>0.451</td>
      <td>0.970</td>
      <td>0.003550</td>
      <td>0</td>
      <td>0.1020</td>
      <td>-4.938</td>
      <td>1</td>
      <td>0.1070</td>
      <td>122.444</td>
      <td>4</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How You Remind Me</td>
      <td>3</td>
      <td>223826</td>
      <td>0.000954</td>
      <td>0.447</td>
      <td>0.766</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.1130</td>
      <td>-5.065</td>
      <td>1</td>
      <td>0.0313</td>
      <td>172.011</td>
      <td>4</td>
      <td>0.574</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = data.iloc[:,1:]
# Just take the column of numbers
```


```python
X = df.iloc[:,1:]
y = df.iloc[:,:1]
# y take the song_popularity column, X take the other column
```


```python
X
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
      <th>song_duration_ms</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>audio_mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>audio_valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>262333</td>
      <td>0.005520</td>
      <td>0.496</td>
      <td>0.682</td>
      <td>0.000029</td>
      <td>8</td>
      <td>0.0589</td>
      <td>-4.095</td>
      <td>1</td>
      <td>0.0294</td>
      <td>167.060</td>
      <td>4</td>
      <td>0.474</td>
    </tr>
    <tr>
      <th>1</th>
      <td>216933</td>
      <td>0.010300</td>
      <td>0.542</td>
      <td>0.853</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.1080</td>
      <td>-6.407</td>
      <td>0</td>
      <td>0.0498</td>
      <td>105.256</td>
      <td>4</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>231733</td>
      <td>0.008170</td>
      <td>0.737</td>
      <td>0.463</td>
      <td>0.447000</td>
      <td>0</td>
      <td>0.2550</td>
      <td>-7.828</td>
      <td>1</td>
      <td>0.0792</td>
      <td>123.881</td>
      <td>4</td>
      <td>0.324</td>
    </tr>
    <tr>
      <th>3</th>
      <td>216933</td>
      <td>0.026400</td>
      <td>0.451</td>
      <td>0.970</td>
      <td>0.003550</td>
      <td>0</td>
      <td>0.1020</td>
      <td>-4.938</td>
      <td>1</td>
      <td>0.1070</td>
      <td>122.444</td>
      <td>4</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>223826</td>
      <td>0.000954</td>
      <td>0.447</td>
      <td>0.766</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.1130</td>
      <td>-5.065</td>
      <td>1</td>
      <td>0.0313</td>
      <td>172.011</td>
      <td>4</td>
      <td>0.574</td>
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
    </tr>
    <tr>
      <th>18830</th>
      <td>159645</td>
      <td>0.893000</td>
      <td>0.500</td>
      <td>0.151</td>
      <td>0.000065</td>
      <td>11</td>
      <td>0.1110</td>
      <td>-16.107</td>
      <td>1</td>
      <td>0.0348</td>
      <td>113.969</td>
      <td>4</td>
      <td>0.300</td>
    </tr>
    <tr>
      <th>18831</th>
      <td>205666</td>
      <td>0.765000</td>
      <td>0.495</td>
      <td>0.161</td>
      <td>0.000001</td>
      <td>11</td>
      <td>0.1050</td>
      <td>-14.078</td>
      <td>0</td>
      <td>0.0301</td>
      <td>94.286</td>
      <td>4</td>
      <td>0.265</td>
    </tr>
    <tr>
      <th>18832</th>
      <td>182211</td>
      <td>0.847000</td>
      <td>0.719</td>
      <td>0.325</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.1250</td>
      <td>-12.222</td>
      <td>1</td>
      <td>0.0355</td>
      <td>130.534</td>
      <td>4</td>
      <td>0.286</td>
    </tr>
    <tr>
      <th>18833</th>
      <td>352280</td>
      <td>0.945000</td>
      <td>0.488</td>
      <td>0.326</td>
      <td>0.015700</td>
      <td>3</td>
      <td>0.1190</td>
      <td>-12.020</td>
      <td>1</td>
      <td>0.0328</td>
      <td>106.063</td>
      <td>4</td>
      <td>0.323</td>
    </tr>
    <tr>
      <th>18834</th>
      <td>193533</td>
      <td>0.911000</td>
      <td>0.640</td>
      <td>0.381</td>
      <td>0.000254</td>
      <td>4</td>
      <td>0.1040</td>
      <td>-11.790</td>
      <td>1</td>
      <td>0.0302</td>
      <td>91.490</td>
      <td>4</td>
      <td>0.581</td>
    </tr>
  </tbody>
</table>
<p>18835 rows × 13 columns</p>
</div>




```python
y
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
      <th>song_popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>18830</th>
      <td>3</td>
    </tr>
    <tr>
      <th>18831</th>
      <td>3</td>
    </tr>
    <tr>
      <th>18832</th>
      <td>2</td>
    </tr>
    <tr>
      <th>18833</th>
      <td>3</td>
    </tr>
    <tr>
      <th>18834</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>18835 rows × 1 columns</p>
</div>



### 1. Divide the data set into a 75% training set and a 25% testing set using only the features relevant for classification.


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)
```

### 2. Use feature scaling and train a decision tree model.


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
```




    DecisionTreeClassifier()



### 3. Using the test set, predict using the decision tree and compute the confusion matrix and the accuracy of classification.


```python
print('accuracy:',classifier.score(X_test,y_test))
```

    accuracy: 0.4890634954342748
    


```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
```


```python
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
```


```python
import seaborn as sns
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  
con_mat_norm = np.around(con_mat_norm, decimals=2)

plt.figure(figsize=(10, 10))
sns.heatmap(con_mat_norm,linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)
plt.ylim(0,5)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title(' confusion matrix')
plt.show()
```


    
![png](output_77_0.png)
    


### 4. Discuss your findings from the confusion matrix and accuracy. You should consider other performance metrics you learnt in lecture 7 to answer this question.


```python
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

f1_score(y_test,y_pred,average='micro')
```




    0.4890634954342748




```python
precision_score(y_test,y_pred,average='weighted')
```




    0.47834121759148124




```python
recall_score(y_test,y_pred,average='macro')
```




    0.4866103883014675



The best classification accuracy of class 5 is 88%, while the accuracy of class 1 and Class 2 is low

# B2. Clustering

# dataset：https://www.kaggle.com/datasets/madhurpant/world-population-data


```python
df2=pd.read_csv('height_weight_data.csv')
```


```python
df2.head()
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
      <th>country</th>
      <th>male_height</th>
      <th>female_height</th>
      <th>male_weight</th>
      <th>female_weight</th>
      <th>male_bmi</th>
      <th>female_bmi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Netherlands</td>
      <td>184</td>
      <td>170</td>
      <td>87.9</td>
      <td>73.2</td>
      <td>26.1</td>
      <td>25.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Montenegro</td>
      <td>183</td>
      <td>170</td>
      <td>90.4</td>
      <td>75.3</td>
      <td>27.0</td>
      <td>26.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Estonia</td>
      <td>182</td>
      <td>168</td>
      <td>89.9</td>
      <td>73.7</td>
      <td>27.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Denmark</td>
      <td>182</td>
      <td>169</td>
      <td>86.8</td>
      <td>70.2</td>
      <td>26.3</td>
      <td>24.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bosnia and Herzegovina</td>
      <td>182</td>
      <td>167</td>
      <td>87.1</td>
      <td>70.6</td>
      <td>26.4</td>
      <td>25.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df2.iloc[:,1:]
# Just take the column of numbers
```


```python
from pandas.plotting import scatter_matrix
scatter_matrix(df2.iloc[:,[0,2]])
plt.show() 
```


    
![png](output_88_0.png)
    



```python
X = df2[["male_height", "male_weight"]]  
X
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
      <th>male_height</th>
      <th>male_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>184</td>
      <td>87.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>183</td>
      <td>90.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>182</td>
      <td>89.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>182</td>
      <td>86.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>182</td>
      <td>87.1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>164</td>
      <td>60.5</td>
    </tr>
    <tr>
      <th>122</th>
      <td>164</td>
      <td>69.1</td>
    </tr>
    <tr>
      <th>123</th>
      <td>163</td>
      <td>62.5</td>
    </tr>
    <tr>
      <th>124</th>
      <td>162</td>
      <td>59.5</td>
    </tr>
    <tr>
      <th>125</th>
      <td>159</td>
      <td>53.9</td>
    </tr>
  </tbody>
</table>
<p>126 rows × 2 columns</p>
</div>




```python
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

Xx=X.values

plt.scatter(Xx[:,0], Xx[:,1], c=y_pred) 

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="*", s=250, c=[0,1], edgecolors="k") 

plt.xlabel("Male Height")
plt.ylabel("Male Weight")
plt.title("KMeans k=2")
plt.show() 
```


    
![png](output_90_0.png)
    



```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

Xx=X.values

plt.scatter(Xx[:,0], Xx[:,1], c=y_pred) 

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="*", s=250, c=[0,1,2], edgecolors="k") 

plt.xlabel("Male Height")
plt.ylabel("Male Weight")
plt.title("KMeans k=3")
plt.show() 
```


    
![png](output_91_0.png)
    



```python
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_pred = kmeans.predict(X)

Xx=X.values

plt.scatter(Xx[:,0], Xx[:,1], c=y_pred) 

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="*", s=250, c=[0,1,2,3], edgecolors="k") 

plt.xlabel("Male Height")
plt.ylabel("Male Weight")
plt.title("KMeans k=4")
plt.show() 
```


    
![png](output_92_0.png)
    



```python
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_pred = kmeans.predict(X)

Xx=X.values

plt.scatter(Xx[:,0], Xx[:,1], c=y_pred) 


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="*", s=250, c=[0,1,2,3,4], edgecolors="k") 

plt.xlabel("Male Height")
plt.ylabel("Male Weight")
plt.title("KMeans k=5")
plt.show() 
```


    
![png](output_93_0.png)
    


Kmeans clustered the data distribution into 5 classes. When the number of clusters was 2, the intra-group distance was minimized and the inter-group distance was maximized, so the Kmeans clustering effect was the best when the number of clusters was 2
