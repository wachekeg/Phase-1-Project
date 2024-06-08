# Analysis of Aircraft Incident Frequencies by Make and Model: Identifying Low-Risk Aircraft for Commercial and Private Operations

![Plane](Plane.jpg)

## Introduction

This project analyzes Aviation Data collected between 1962 and 2023 to come up with useful insights to guide a company towards finding the lowest risk aircrafts for a new endeavor.

## Business Problem

Our company is poised to enter the aviation industry by purchasing and managing aircraft for both private and commercial usage, with the goal of expanding its portfolio. Given the inherent dangers associated with aviation, it is imperative to comprehend these risks in order to make well-informed judgments. In order to ensure a safe and successful launch into the market, this project aims to deliver a data-driven examination of aviation safety with a particular focus on finding the lowest-risk aircraft.

## Objective

The main objectives of this task are:

1. **Data Preparation**
2. **Risk Analysis and Identification**
3. **Reporting and Recommendations**

## Data Understanding 

The data used in this project is a dataset from the **National Transportation Safety Board** that includes aviation accident data from 1962 to 2023 about civil aviation accidents and selected incidents in the United States and international waters.

The dataset contains 31 attributes and 90348 records. 

For the purpose of this analysis, the data was filtered to remain with the attributes and records that were relevant. The cleaned and filtered dataset used has 11 attributes and 61724 records.

Here is a breakdown of the attributes and what they represent:

- Event.Date - The date of incident occurence
- Location - The location of incident occurence
- Injury.Severity - The overall severity of the injuries resulting from the incident
- Make - The manufacturer or brand that produces the aircraft
- Model - The specific version or type of aircraft produced by a particular manufacturer
- Purpose.of.flight - The reason for which the flight was being conducted at the time of the incident
- Total.Fatal.Injuries - The number of fatalities resulting from the incident
- Total.Serious.Injuries - The number of serious injuries sustained in the incident
- Total.Minor.Injuries - The number of minor injuries resulting from the incident
- Total.Uninjured - The number of individuals who were not injured in the incident
- Broad.phase.of.flight - The general phase of flight during which the incident occurred

## 1. Data Preparation

### 1.1 Data Importation


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Load the data from its directory
aviation = pd.read_csv('Aviation_Data.csv', encoding='latin1', low_memory=False)
#aviation.head(10)
```


```python
# Examine the data to identify its shape, and how many missing values are therein
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 90348 entries, 0 to 90347
    Data columns (total 31 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Id                88889 non-null  object 
     1   Investigation.Type      90348 non-null  object 
     2   Accident.Number         88889 non-null  object 
     3   Event.Date              88889 non-null  object 
     4   Location                88837 non-null  object 
     5   Country                 88663 non-null  object 
     6   Latitude                34382 non-null  object 
     7   Longitude               34373 non-null  object 
     8   Airport.Code            50132 non-null  object 
     9   Airport.Name            52704 non-null  object 
     10  Injury.Severity         87889 non-null  object 
     11  Aircraft.damage         85695 non-null  object 
     12  Aircraft.Category       32287 non-null  object 
     13  Registration.Number     87507 non-null  object 
     14  Make                    88826 non-null  object 
     15  Model                   88797 non-null  object 
     16  Amateur.Built           88787 non-null  object 
     17  Number.of.Engines       82805 non-null  float64
     18  Engine.Type             81793 non-null  object 
     19  FAR.Description         32023 non-null  object 
     20  Schedule                12582 non-null  object 
     21  Purpose.of.flight       82697 non-null  object 
     22  Air.carrier             16648 non-null  object 
     23  Total.Fatal.Injuries    77488 non-null  float64
     24  Total.Serious.Injuries  76379 non-null  float64
     25  Total.Minor.Injuries    76956 non-null  float64
     26  Total.Uninjured         82977 non-null  float64
     27  Weather.Condition       84397 non-null  object 
     28  Broad.phase.of.flight   61724 non-null  object 
     29  Report.Status           82505 non-null  object 
     30  Publication.Date        73659 non-null  object 
    dtypes: float64(5), object(26)
    memory usage: 21.4+ MB
    

### 1.2 Data Cleaning

This sub category will include:

- Removing irrelevant columns
- Handling of missing data
- Data Transformation

- Removing irrelevant columns


```python
# Remove irrelevant columns to remain witha smaller dataset
irrelevant_columns = ['Publication.Date', 'Report.Status', 'Engine.Type', 'Number.of.Engines', 'Amateur.Built', 'Registration.Number', 'Aircraft.damage',
                      'Airport.Name', 'Airport.Code', 'Accident.Number', 'Investigation.Type', 'Event.Id', 'Country', 'Weather.Condition']
aviation = aviation.drop(columns=irrelevant_columns)
```

- Handling of missing data

From the dataset, there are several columns with a high percentage of missing values. A threshold of 50% shall be used to drop these columns


```python
# Use a threshold of 50% to drop columns with a higher percentage of missing values
threshold = 50
missing_percentage = aviation.isnull().mean() *100
columns_to_drop = missing_percentage[missing_percentage > threshold].index
aviation = aviation.drop(columns=columns_to_drop)
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 90348 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88837 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88826 non-null  object 
     4   Model                   88797 non-null  object 
     5   Purpose.of.flight       82697 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   61724 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 7.6+ MB
    


```python
#Drop the missing rows in the 'Event.Date' column to remain with relevant data

aviation = aviation.dropna(axis=0, subset= ['Event.Date'])
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88837 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88826 non-null  object 
     4   Model                   88797 non-null  object 
     5   Purpose.of.flight       82697 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   61724 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    

Having fewer relevant columns to work with, each column with missing values can now be handled individually. 

The 'Broad.Phase.of.Flight' has many missing records. However, dropping these records will result in losing alot of meaningful data, including the date of incidence occurence. The missing values will thus be replaced with 'Unknown' for the purpose of data completeness


```python
# Replace the missing values in the column with 'Unknown'
aviation['Broad.phase.of.flight'] = aviation['Broad.phase.of.flight'].fillna('Unknown')
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88837 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88826 non-null  object 
     4   Model                   88797 non-null  object 
     5   Purpose.of.flight       82697 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   88889 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    

Next, the NaN values in the 'Purpose.of.flight' column will be replaced with the string 'Unknown' to ensure data completeness. These will not be dropped since they are few, and will not interfere with the integrity of the data. The same will be done for the 'Location', 'Model', 'Injury.Severity' and 'Make' columns.


```python
# Replace the missing values in the column with 'Unknown'
aviation['Purpose.of.flight'] = aviation['Purpose.of.flight'].fillna('Unknown')
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88837 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88826 non-null  object 
     4   Model                   88797 non-null  object 
     5   Purpose.of.flight       88889 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   88889 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    


```python
# Replace the missing values in the column with 'Unknown'
aviation['Model'] = aviation['Model'].fillna('Unknown')
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88837 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88826 non-null  object 
     4   Model                   88889 non-null  object 
     5   Purpose.of.flight       88889 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   88889 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    


```python
# Replace the missing values in the column with 'Unknown'
aviation['Location'] = aviation['Location'].fillna('Unknown')
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88889 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88826 non-null  object 
     4   Model                   88889 non-null  object 
     5   Purpose.of.flight       88889 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   88889 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    


```python
# Replace the missing values in the column with 'Unknown'
aviation['Make'] = aviation['Make'].fillna('Unknown')
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88889 non-null  object 
     2   Injury.Severity         87889 non-null  object 
     3   Make                    88889 non-null  object 
     4   Model                   88889 non-null  object 
     5   Purpose.of.flight       88889 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   88889 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    


```python
# Replace the missing values in the column with 'Unknown'
aviation['Injury.Severity'] = aviation['Injury.Severity'].fillna('Unknown')
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Event.Date              88889 non-null  object 
     1   Location                88889 non-null  object 
     2   Injury.Severity         88889 non-null  object 
     3   Make                    88889 non-null  object 
     4   Model                   88889 non-null  object 
     5   Purpose.of.flight       88889 non-null  object 
     6   Total.Fatal.Injuries    77488 non-null  float64
     7   Total.Serious.Injuries  76379 non-null  float64
     8   Total.Minor.Injuries    76956 non-null  float64
     9   Total.Uninjured         82977 non-null  float64
     10  Broad.phase.of.flight   88889 non-null  object 
    dtypes: float64(4), object(7)
    memory usage: 8.1+ MB
    

Next, missing values in the 'Total.Fatal.Injuries', 'Total.Serious.Injuries', 'Total.Minor.Injuries', and 'Total.Uninjured' columns will be handled by replacing them with the median for their respective columns


```python
# Compute the median for each of the 4 columns
median_fatal_injuries = aviation['Total.Fatal.Injuries'].median()
median_serious_injuries = aviation['Total.Serious.Injuries'].median()
median_minor_injuries = aviation['Total.Minor.Injuries'].median()
median_uninjuired = aviation['Total.Uninjured'].median()
```


```python
# Impute the median values for the 4 columns
aviation['Total.Fatal.Injuries'] = aviation['Total.Fatal.Injuries'].fillna(median_fatal_injuries)
aviation['Total.Serious.Injuries'] = aviation['Total.Serious.Injuries'].fillna(median_serious_injuries)
aviation['Total.Minor.Injuries'] = aviation['Total.Minor.Injuries'].fillna(median_minor_injuries)
aviation['Total.Uninjured'] = aviation['Total.Uninjured'].fillna(median_uninjuired)
```

- Data Transformation


```python
#Convert 'Event.Date' to datetime
aviation['Event.Date'] = pd.to_datetime(aviation['Event.Date'])
aviation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 88889 entries, 0 to 90347
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   Event.Date              88889 non-null  datetime64[ns]
     1   Location                88889 non-null  object        
     2   Injury.Severity         88889 non-null  object        
     3   Make                    88889 non-null  object        
     4   Model                   88889 non-null  object        
     5   Purpose.of.flight       88889 non-null  object        
     6   Total.Fatal.Injuries    88889 non-null  float64       
     7   Total.Serious.Injuries  88889 non-null  float64       
     8   Total.Minor.Injuries    88889 non-null  float64       
     9   Total.Uninjured         88889 non-null  float64       
     10  Broad.phase.of.flight   88889 non-null  object        
    dtypes: datetime64[ns](1), float64(4), object(6)
    memory usage: 8.1+ MB
    


```python
#Export the cleaned data to excel for visualization in Tableau
aviation.to_excel('cleaned_aviation_data.xlsx', index=False)
```

## 2.0 Risk Analysis and Identification

Now to be able to establish which aircrafts are the lowest risk for the company to begin this endeavor, we need to carry out some analysis

In particular, this session will look into:

- Analysis of Injury Severity by Aircraft Make and Model

- Survivability Analysis by Aircraft Make

- Incident Frequency by Aircraft Make

- Time Trend Analysis of Incidents

- Injury Severity by Broad Phase of Flight

- Analysis of risks associated with Purpose of Flight


### 2.1 Analysis of injury severity by Aircraft Make and Model

This analysis aims to find out which makes and models of aircraft have the highest and lowest rates of major, minor, and fatal injuries.


```python
# Calculate the sum of injuries by make and model and store in a new dataframe
injury_info = aviation.groupby(['Make', 'Model']).agg({
    'Total.Fatal.Injuries': 'sum',
    'Total.Serious.Injuries': 'sum',
    'Total.Minor.Injuries': 'sum'
}).reset_index()

injury_info.head()
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
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107.5 Flying Corporation</td>
      <td>One Design DR 107</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200</td>
      <td>G103</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>177MF LLC</td>
      <td>PITTS MODEL 12</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1977 Colfer-chan</td>
      <td>STEEN SKYBOLT</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1st Ftr Gp</td>
      <td>FOCKE-WULF 190</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create a new column in the new dataframe to show the total injuries
injury_info['Total_Injuries'] = injury_info[['Total.Fatal.Injuries', 'Total.Serious.Injuries', 'Total.Minor.Injuries']].sum(axis=1)
#injury_info.head()
```


```python
#Sort by 'Total_Injuries' to determine the aircrafts with the lowest risk

low_risk_models = injury_info.sort_values(by = 'Total_Injuries')
low_risk_models.head(10)
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
      <th>Make</th>
      <th>Model</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Total_Injuries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10093</th>
      <td>HENDRICKS GEORGE DAVID JR</td>
      <td>RV8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5377</th>
      <td>CONSOLIDATED AERONAUTICS INC.</td>
      <td>LAKE LA 4 200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12062</th>
      <td>Let</td>
      <td>L-23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8949</th>
      <td>Flurry</td>
      <td>AVENTURA II</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5374</th>
      <td>CONSOLIDATED AERONAUTICS</td>
      <td>LAKE LA4-200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5373</th>
      <td>CONSOLIDATED  AERONAUTICS INC.</td>
      <td>LAKE LA-4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5372</th>
      <td>CONRAD THEODORE J</td>
      <td>THORP T-18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5371</th>
      <td>CONNER LEROY</td>
      <td>RV6A</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8950</th>
      <td>Fly Baby</td>
      <td>1-A</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15468</th>
      <td>Pitts</td>
      <td>SPECIAL 51C</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Sort by 'Total_Injuries' to determine the aircrafts makes and models with the highest risk

high_risk_models = injury_info.sort_values(by = 'Total_Injuries', ascending = False).head(10)
```


```python
#Export the data to excel for visualization in Tableau
low_risk_models.to_excel('aviation_risk_data.xlsx', index=False)
```


```python
# Plotting the data
# High Risk Models
plt.figure(figsize=(12, 8))

plt.barh(high_risk_models['Make'] + ' ' + high_risk_models['Model'], high_risk_models['Total_Injuries'], color='red')
plt.xlabel('Total Injuries')
plt.ylabel('Make and Model')
plt.title('Top 10 High Risk Aircraft Makes and Models')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest risk on top

plt.savefig('Top 10 High Risk Aircraft Makes and Models.jpg', format='jpg')
```


    
![png](output_44_0.png)
    


This analysis shows that various models of the Cessna and Piper makes are high risk, in that they have caused the most number of injuries. Certain traits, such as older designs, higher usage rates, or involvement in particular flight types (e.g., commercial or high-stress contexts), may be shared by high-risk models. This will be further investigated.

This analysis also demonstrates that various models of the Cirrus and Cernia makes have no recorded injuries, which means they are very low risk makes. In low-stress or private flight circumstances, where the likelihood of severe incidents is lower, low-risk models might be utilized more frequently. It is also assumed that these models may be better maintained than the high-risk ones, or they may be more recent versions with enhanced safety measures.

### 2.2 Survivability Analysis by Aircraft Make 

This analysis aims to find out which aircraft makes have the best survivability rates—that is, the highest percentage of passengers who escape injury unharmed.


```python
#Calculate the statistics of uninjured persons, by make and model

survivor_info = aviation.groupby(['Make'])['Total.Uninjured'].sum().reset_index()
```


```python
# Sort the survivor_info dataframe by 'Total.Uninjured'

highest_survivability_aircrafts = survivor_info.sort_values(by = 'Total.Uninjured', ascending = False).head(15)
```


```python
#Export the data to excel for visualization in Tableau
highest_survivability_aircrafts.to_excel('highest_survivability_aircrafts.xlsx', index = False)
```


```python
# Plot the aircraft makes with the highest survivability rates
plt.figure(figsize=(12, 8))
plt.barh(highest_survivability_aircrafts['Make'] , highest_survivability_aircrafts['Total.Uninjured'], color='skyblue', edgecolor = 'brown')

plt.xlabel('Total Uninjured Passengers')
plt.ylabel('Aircraft Make')
plt.title('Top 15 Aircraft with the highest survivability')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
plt.show()

plt.savefig('Top 15 Aircraft with the highest survivability.jpg', format='jpg')
```


    
![png](output_50_0.png)
    



    <Figure size 640x480 with 0 Axes>


This analysis shows that Boeing has the highest survivability rates, and the previous analysis has not listed it as one of the top 10 high risk aircraft makes. 

This implies that the crew and passengers are well protected by their aircraft designs in the event of an incident. This may be due to superior safety features, robust construction, and effective emergency systems.

### 2.3 Incident Frequency by Aircraft Make

With this analysis, we will identify which aircraft makes are involved in the fewest or most incidents.


```python
# Group by 'Make' and count incidents
# Convert the Series to a DataFrame and name the count column Incident.Count.

plane_make_incident_counts = aviation.groupby(['Make']).size().reset_index(name='Incident.Count')
plane_make_incident_counts.head()
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
      <th>Make</th>
      <th>Incident.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107.5 Flying Corporation</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>177MF LLC</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1977 Colfer-chan</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1st Ftr Gp</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sort by the Incident.Count column in descending order to bring the models with the most incidents to the top.
most_incidents_aircraft_makes = plane_make_incident_counts.sort_values(by = 'Incident.Count', ascending = False).head(15)
```


```python
# Plot using matplotlib
plt.figure(figsize=(12, 8))
plt.barh(most_incidents_aircraft_makes['Make'] , most_incidents_aircraft_makes['Incident.Count'], color='green')

plt.xlabel('Incident Count')
plt.ylabel('Aircraft Make and Model')
plt.title('Top 15 Aircraft Makes with the highest Incident Counts')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
plt.show()

plt.savefig('Top 15 Aircraft Makes with the highest Incident Counts.jpg', format='jpg')
```


    
![png](output_55_0.png)
    



    <Figure size 640x480 with 0 Axes>


This analysis demonstrates that the Cessna and Piper aircraft makes have the most incident counts. From the first analysis, these two makes were also observed to be the most high risk aircraft makes. For the new venture the company plans to go into, these two models may not be the best to be considered.


```python
# Sort by the Incident.Count column in ascending order to bring the models with the fewest incidents to the top.
fewest_incidents_aircraft_make = plane_make_incident_counts.sort_values(by = 'Incident.Count')
fewest_incidents_aircraft_make
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
      <th>Make</th>
      <th>Incident.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107.5 Flying Corporation</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5170</th>
      <td>Mingess-bennett</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5169</th>
      <td>Miner</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5168</th>
      <td>Mince</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5167</th>
      <td>Miltenberger</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5608</th>
      <td>PIPER</td>
      <td>2841</td>
    </tr>
    <tr>
      <th>936</th>
      <td>Beech</td>
      <td>4330</td>
    </tr>
    <tr>
      <th>1320</th>
      <td>CESSNA</td>
      <td>4922</td>
    </tr>
    <tr>
      <th>5795</th>
      <td>Piper</td>
      <td>12029</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>Cessna</td>
      <td>22227</td>
    </tr>
  </tbody>
</table>
<p>8237 rows × 2 columns</p>
</div>




```python
#Export the data to excel for visualization in Tableau
fewest_incidents_aircraft_make.to_excel('fewest_incidents_aircraft_makes.xlsx', index=False)
```


```python
#Export the data to excel for visualization in Tableau
most_incidents_aircraft_makes.to_excel('Aircraft_makes_with_most_incidents.xlsx', index = False)
```

### 2.4 Time Trend Analysis of Incidents

With this analysis, we shall examine the temporal patterns to determine the number of people involved in incidents per year, and how this has changed over time


```python
#Extract the year from the 'Event.Date' column
aviation['Year'] = aviation['Event.Date'].dt.year
```


```python
# Filter the DataFrame to include only incidents from 1982 onwards
aviation_filtered = aviation[aviation['Year'] >= 1982]

#Count incidents per year
temporal_incident_info = aviation_filtered.groupby(['Year']).size().reset_index(name='Incident.Count')
```


```python
temporal_incident_info.to_excel('temporal_incident_info.xlsx', index = False)
```


```python
# Plot the temporal incident information using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(temporal_incident_info['Year'], temporal_incident_info['Incident.Count'], marker='o')

plt.xlabel('Year')
plt.ylabel('Incident Count')
plt.title('Incident Count Over Time')
plt.grid(True)
plt.show()

plt.savefig('Incident Count Over Time.jpg', format='jpg')
```


    
![png](output_64_0.png)
    



    <Figure size 640x480 with 0 Axes>


The trend analysis shows that airplane incidents have been on a gradual decline over the years. This can be attributed to various factors including advanced technology that enhances crew and passenger safety, better maintenance strategies, improved air traffic management, among other factors.

### 2.5 Injury Severity by Broad Phase of Flight

This analysis will investigate which flight phases—such as takeoff, cruise, and landing—seem to cause the greatest injuries.


```python
#Calculate injury statistics by Phase of Flight
phase_info = aviation.groupby('Broad.phase.of.flight').agg({
    'Total.Fatal.Injuries': 'sum',
    'Total.Serious.Injuries': 'sum',
    'Total.Minor.Injuries': 'sum'
}).reset_index()

phase_info.head()
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
      <th>Broad.phase.of.flight</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Approach</td>
      <td>3842.0</td>
      <td>1920.0</td>
      <td>2526.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Climb</td>
      <td>1762.0</td>
      <td>606.0</td>
      <td>962.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cruise</td>
      <td>6173.0</td>
      <td>2183.0</td>
      <td>4531.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Descent</td>
      <td>913.0</td>
      <td>473.0</td>
      <td>998.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Go-around</td>
      <td>587.0</td>
      <td>388.0</td>
      <td>622.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create a new column in the new dataframe to show the total injuries
phase_info['Total.Injuries'] = phase_info[['Total.Fatal.Injuries', 'Total.Serious.Injuries', 'Total.Minor.Injuries']].sum(axis=1)
```


```python
#Sort by the Total.Injuries column in ascending order.
phase_info = phase_info.sort_values(by = 'Total.Injuries')
# # Filter the DataFrame to exclude records with 'Unknown' values
phase_info = phase_info[phase_info['Broad.phase.of.flight'] != 'Unknown']
phase_info
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
      <th>Broad.phase.of.flight</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Total.Injuries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Other</td>
      <td>85.0</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>119.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Taxi</td>
      <td>102.0</td>
      <td>111.0</td>
      <td>501.0</td>
      <td>714.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Standing</td>
      <td>161.0</td>
      <td>241.0</td>
      <td>397.0</td>
      <td>799.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Go-around</td>
      <td>587.0</td>
      <td>388.0</td>
      <td>622.0</td>
      <td>1597.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Descent</td>
      <td>913.0</td>
      <td>473.0</td>
      <td>998.0</td>
      <td>2384.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Climb</td>
      <td>1762.0</td>
      <td>606.0</td>
      <td>962.0</td>
      <td>3330.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Landing</td>
      <td>518.0</td>
      <td>1234.0</td>
      <td>3209.0</td>
      <td>4961.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Approach</td>
      <td>3842.0</td>
      <td>1920.0</td>
      <td>2526.0</td>
      <td>8288.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Maneuvering</td>
      <td>5323.0</td>
      <td>1912.0</td>
      <td>1980.0</td>
      <td>9215.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Takeoff</td>
      <td>4304.0</td>
      <td>3138.0</td>
      <td>4955.0</td>
      <td>12397.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cruise</td>
      <td>6173.0</td>
      <td>2183.0</td>
      <td>4531.0</td>
      <td>12887.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Export the data to excel for visualization in Tableau
phase_info.to_excel('phase_info.xlsx', index = False)
```


```python
#Visualize the number of injuries that have occurred in various phases of flight
# Plot using matplotlib
plt.figure(figsize=(12, 6))
plt.barh(phase_info['Broad.phase.of.flight'], phase_info['Total.Injuries'], color='orange', edgecolor='black')

plt.xlabel('Total Injuries')
plt.ylabel('Phase of Flight')
plt.title('Proportion of injuries by Phase of flight')
plt.xticks(rotation=45)
plt.show()

plt.savefig('Proportion of injuries by Phase of flight.jpg', format='jpg')
```


    
![png](output_71_0.png)
    



    <Figure size 640x480 with 0 Axes>


The analysis to understand the relation between phase of flight and total injuries reveals that most passenger injuries occur during the cruise and take-off phases. A fairly large portion of injuries also occur in the maneuvering and approach phases. 

This analysis therefore higlights the importance of focusing on safety during the takeoff, cruise, maneuvering and approach phases of flight.

### 2.6 Analysis of risks associated with Purpose of Flight

This analysis looks into the risk associated with different purposes of flights (e.g., commercial, private).


```python
#Calculate injury statistics by Purpose of Flight
purpose_info = aviation.groupby('Purpose.of.flight').agg({
    'Total.Fatal.Injuries': 'sum',
    'Total.Serious.Injuries': 'sum',
    'Total.Minor.Injuries': 'sum'
}).reset_index()
```


```python
#Create a new column in the new dataframe to show the total injuries
purpose_info['Total.Injuries'] = purpose_info[['Total.Fatal.Injuries', 'Total.Serious.Injuries', 'Total.Minor.Injuries']].sum(axis=1)
purpose_info
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
      <th>Purpose.of.flight</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Total.Injuries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ASHO</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aerial Application</td>
      <td>549.0</td>
      <td>595.0</td>
      <td>794.0</td>
      <td>1938.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aerial Observation</td>
      <td>414.0</td>
      <td>318.0</td>
      <td>321.0</td>
      <td>1053.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Air Drop</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Air Race show</td>
      <td>42.0</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Air Race/show</td>
      <td>34.0</td>
      <td>21.0</td>
      <td>10.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Banner Tow</td>
      <td>19.0</td>
      <td>31.0</td>
      <td>10.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Business</td>
      <td>2313.0</td>
      <td>881.0</td>
      <td>1106.0</td>
      <td>4300.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Executive/corporate</td>
      <td>598.0</td>
      <td>143.0</td>
      <td>192.0</td>
      <td>933.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>External Load</td>
      <td>39.0</td>
      <td>28.0</td>
      <td>36.0</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ferry</td>
      <td>386.0</td>
      <td>112.0</td>
      <td>216.0</td>
      <td>714.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Firefighting</td>
      <td>37.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Flight Test</td>
      <td>130.0</td>
      <td>90.0</td>
      <td>86.0</td>
      <td>306.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Glider Tow</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Instructional</td>
      <td>1913.0</td>
      <td>1532.0</td>
      <td>2062.0</td>
      <td>5507.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Other Work Use</td>
      <td>511.0</td>
      <td>414.0</td>
      <td>541.0</td>
      <td>1466.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PUBL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PUBS</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Personal</td>
      <td>18762.0</td>
      <td>10611.0</td>
      <td>12959.0</td>
      <td>42332.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Positioning</td>
      <td>635.0</td>
      <td>241.0</td>
      <td>303.0</td>
      <td>1179.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Public Aircraft</td>
      <td>406.0</td>
      <td>189.0</td>
      <td>244.0</td>
      <td>839.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Public Aircraft - Federal</td>
      <td>41.0</td>
      <td>24.0</td>
      <td>31.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Public Aircraft - Local</td>
      <td>13.0</td>
      <td>49.0</td>
      <td>19.0</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Public Aircraft - State</td>
      <td>23.0</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Skydiving</td>
      <td>234.0</td>
      <td>90.0</td>
      <td>57.0</td>
      <td>381.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Unknown</td>
      <td>23062.0</td>
      <td>5944.0</td>
      <td>8425.0</td>
      <td>37431.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Export the data to excel for visualization in Tableau
purpose_info.to_excel('purpose_info.xlsx', index = False)
```


```python
#Sort by the Total.Injuries column in descending order.
purpose_info = purpose_info.sort_values(by = 'Total.Injuries', ascending = False).head(10)
#Filter the DataFrame to exclude records with 'Unknown' values
purpose_info = purpose_info[purpose_info['Purpose.of.flight'] != 'Unknown']
purpose_info
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
      <th>Purpose.of.flight</th>
      <th>Total.Fatal.Injuries</th>
      <th>Total.Serious.Injuries</th>
      <th>Total.Minor.Injuries</th>
      <th>Total.Injuries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>Personal</td>
      <td>18762.0</td>
      <td>10611.0</td>
      <td>12959.0</td>
      <td>42332.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Instructional</td>
      <td>1913.0</td>
      <td>1532.0</td>
      <td>2062.0</td>
      <td>5507.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Business</td>
      <td>2313.0</td>
      <td>881.0</td>
      <td>1106.0</td>
      <td>4300.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aerial Application</td>
      <td>549.0</td>
      <td>595.0</td>
      <td>794.0</td>
      <td>1938.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Other Work Use</td>
      <td>511.0</td>
      <td>414.0</td>
      <td>541.0</td>
      <td>1466.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Positioning</td>
      <td>635.0</td>
      <td>241.0</td>
      <td>303.0</td>
      <td>1179.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aerial Observation</td>
      <td>414.0</td>
      <td>318.0</td>
      <td>321.0</td>
      <td>1053.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Executive/corporate</td>
      <td>598.0</td>
      <td>143.0</td>
      <td>192.0</td>
      <td>933.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Public Aircraft</td>
      <td>406.0</td>
      <td>189.0</td>
      <td>244.0</td>
      <td>839.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualize the top 10 purpose of flight that have had most injuries

# Plot using matplotlib
plt.figure(figsize=(12, 6))
plt.bar(purpose_info['Purpose.of.flight'], purpose_info['Total.Injuries'], color='skyblue', edgecolor='grey')

plt.xlabel('Purpose of Flight')
plt.ylabel('Total Injuries')
plt.title('Top 10 Purposes of Flight vs Total Injuries')
plt.xticks(rotation=45)
plt.show()

plt.savefig('Top 10 Purposes of Flight vs Total Injuries.jpg', format='jpg')
```


    
![png](output_78_0.png)
    



    <Figure size 640x480 with 0 Axes>


When compared to other flight purposes, personal flights have a much higher total injury rate (42,332).
This implies that personal flights may be the riskiest, most likely as a result of pilots' differing skill levels, less stringent safety regulations, or a greater number of flights.


# 3. Conclusion

From the analysis conducted on the aviation data, the following were observed:

   1. Cessna and Piper aircraft makes are associated with the highest injury and icident rates
   2. Boeing aircraft have higher survivability rates and are not listed among the top 10 high-risk aircraft makes. 
   3.  The trend analysis shows a gradual decline in airplane incidents over the years, reflecting improvements in aviation technology, safety protocols, and regulatory oversight.
   4. Most injuries occur during Cruise and Take-Off phases, indicating critical points where enhanced safety measures are necessary. 
   5. Personal and Instructional Flight categories have the highest total injuries, indicating they are the most risky.

# 4. Recommendation

1. **Focus on Low-Risk Models**

For starting the new business endeavor, focusing on acquiring and operating low-risk models is advisable. These models have demonstrated better safety records, lower incidents of severe injuries and high survivability rates. A great example is Boeing. Aircraft makes to steer clear of include Cessna and Piper since these are associated with the highest incidents and injuries.

2. **Risk Management for Private Flights**

Since part of the new endeavor is to purpase and operate aircrafts for private purposes, it is of importance that extra attention be given to risk management strategies when operating private flights, seeing as personal flights are the most risky. 
This may include investing in advanced safety equipment, rigorous training programs, and strict safety protocols.

3. **Enhanced Safety Protocols**

It will be essential to instill a strong safety culture in the company. This covers routine incident investigations, safety audits, and incident reviews.


```python

```
