#!/usr/bin/env python
# coding: utf-8

# # DSC 200 Final project

# In[ ]:





# This project involves the analysis of a climate projection of temperature change across the Earth under a 'middle-of-the-road' future scenario in which global mean temperatures reach more than 2 degrees centigrade above the pre-industrial. You will read in the data, analyze it, and visualize it in a variety of ways. You will also write a small command line interface to make the analysis more interactive. 
# 
# We will be using data created by the NorESM2 climate model and processed and as part of the ClimateBench [dataset](https://zenodo.org/records/7064308), described in this [paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002954). **All the files you will need for this project are available in the `public` folder on DataHub though.**

# ### Table of contents:
# 
#  1. [Read in CO2 emissions data for historical + future scenario](#1-read-in-historical-and-future-estimated-co2-emissions-data) [5 points]
#  2. [Read in temperature data](#2-read-in-the-temperature-data) [5 points]
#  3. [Create a simple regression model](#3-create-a-simple-regression-model-of-global-warming) [15 points]
#  4. [Extend this to a regional temperature model, by region, and by state](#4-extend-this-to-a-regional-temperature-model-by-region-and-by-state) [15 points]
#  5. [Plot the regression coefficients for each country](#5-plot-the-regression-coefficients-for-each-country) [5 points]
#  6. [Do an analysis of your choosing](#6-do-an-analysis-of-your-choosing) [15 points]
#  7. [Make a command line interface](#7-make-a-command-line-interface-to-a-prediction-script) [20 points]
#  
# ### Other requirements:
#  You will also be graded on Documentation and commenting, coding style, and code quality:
#  - Documentation should be in the form of a Jupyter notebook, and should include a description of the data, the analysis, and the results. [10 points]
#  - The code should be well documented, and should follow the PEP8 coding style. [5 points]
#  - The code should be well organized, and should be broken up into functions and classes as appropriate. For full marks try to use no for-loops in your code. [5 points]
# 
# Be sure to read the question and reach out to the instructor or TA if you have any questions.
# 
# ### Total points: 100 (30% of total), plus midterm makeup
#  - Note, the midterm grade is still capped at 100%
# 
# ### Deadline: Wednesday December 10th 11:59pm
# 

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ## 1. Read in historical and future (estimated) CO2 emissions data
# 
# Monthly CO2 emissions data is available since 1850 globally for each industrial sector. We want to use annual average totals of the emissions for all sectors for our analysis. Since CO2 has a very long lifetime in the atmosphere (1000's of years), we can assume that the total amount of anthropogenic CO2 in the atmosphere is the cumulative sum of all emissions since 1850. This is what we will use for our analysis.
# 
# To read this data do **either** Q1a (to get 5 points plus additional makeup points for the midterm) **or** Q1b (to get 5 points for this project)

# In[2]:


input_path = 'public/'


# ### 1a. OPTIONAL: Create interpolated cumulative CO2 from the raw data using Pandas
# 
# To gain (up to) 20% additional marks for your midterm makeup (capped at 100%), you can create a new column in the CO2 emissions data that is the cumulative CO2 emissions. 

# In[3]:


# These input files provide CO2 emissions data for the historical period (1850-2014) and the future period (2015-2100). They should all be concatenated into a single file.

historical_input_files = ['CO2_emissions/CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_200001-201412.csv',
                          'CO2_emissions/CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_195001-199912.csv',
                          'CO2_emissions/CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_190001-194912.csv',
                          'CO2_emissions/CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_185101-189912.csv']

future_input_file = 'CO2_emissions/CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.csv'


# First you will need to read, concatenate and process the raw CSV files, and sum over the `sector` and `month` columns to get an annual total.

# In[4]:


# YOUR CODE HERE

# load and concatenate the data
df_hist = pd.concat([pd.read_csv(input_path + f) for f in historical_input_files], ignore_index=True)

# load future data
df_future = pd.read_csv(input_path + future_input_file)

# combine historical and future data
df_complete = pd.concat([df_hist, df_future], ignore_index=True)

# convert the complete DataFrame to an Xarray Dataset for 2+ dimensions of data
ds = df_complete.set_index(["year", "month", "sector"]).to_xarray()

ds

# raise NotImplementedError()


# In[5]:


# sum over the sector and month columns to get an annual total
annual = ds["global_total"].sum(dim=("month", "sector"))

# add to existing dataset
ds = ds.assign(annual_emissions=annual)

ds


# **Note**, the future data is only provided every five years so that will need linearly interpolated to get annual values.

# In[6]:


# YOUR CODE HERE

#expand the dataset to hold data expansion
ds = ds.reindex(year=np.arange(1851, 2101))

# linearly interpolate the gappy annual data to fill in missing years (1851-2100)
ds_interpolated = annual.interp(year=np.arange(1851, 2101))

# calculate the cumulative co2 emissions from the continuous annual data
cumulative_co2_interpolated = ds_interpolated.cumsum(dim="year")

# add to dataset
ds = ds.assign(cumulative_co2=cumulative_co2_interpolated)

# raise NotImplementedError()


# Now, divide by 1e6 to get the units in GtC (Giga tonnes of carbon) and calculate the cumulative sum.

# In[7]:


# YOUR CODE HERE

# convert to giga tonnes 
annual_gtc = ds_interpolated / 1e6

# calculate the cumulative sum 
annual_gtc_cumsum = annual_gtc.cumsum(dim="year")

# add to dataset
ds = ds.assign(cumulative_co2_GtC=annual_gtc_cumsum)

ds

# raise NotImplementedError()


# Check the data against the existing `combined_co2.csv` and save it to use for the rest of the project.

# In[8]:


# YOUR CODE HERE

# convert to dataframe
df_calculated = ds['cumulative_co2_GtC'].to_dataframe().reset_index()

# load existing file to dataframe
df_existing = pd.read_csv("combined_co2.csv")

# merge the two DataFrames for comparison
df_comparison = df_calculated.merge(df_existing, on='year', how='outer').dropna()

# comparing floats adjust for float error
float_error = 1e-6

# create a new comparison column
df_comparison['difference'] = df_comparison['cumulative_co2_GtC'] - df_comparison['cumulative_CO2']

# filter values with difference
values_different = values_different = df_comparison[df_comparison['difference'] > float_error]

# if no difference save
if len(values_different) == 0:
    df_calculated = df_calculated.rename(columns={
        'cumulative_co2_GtC': 'cumulative_CO2'
    }
)
    df_calculated.to_csv("combined_co2.csv", index=False)
    print("no errors found, adding new data to combined_co2.csv")
else:
    print(values_different)

# raise NotImplementedError()


# ### 1b. Otherwise just read in the cumulative CO2 emissions data from the provided file

# In[9]:


pre_processed_input_file = input_path+'cumulative_co2.csv'

# Read the input files


# In[10]:


# YOUR CODE HERE
# raise NotImplementedError()


# ## 2. Read in the temperature data
# 
# Note, this temperature change as modelled by the NorESM2 climate model relative to the pre-industrial period. It's purely driven by the prescribed emissions, so it won't perfect represent the actual temperatures we experienced in a given year (which are subject to chaotic fluctuations), but it's a good model.

# In[11]:


temperature_input_file = input_path+'global_temperature.nc'

# Read the input files
## Note, the variable name in the netcdf file is 'tas' (not 'temperature')


# In[12]:


# YOUR CODE HERE
# load into xarray dataset
ds_temperature = xr.open_dataset(temperature_input_file)

# rename the temperature 'tas' to 'temperature' for clarity
ds_temperature = ds_temperature.rename({'tas': 'temperature'})

ds_temperature
# raise NotImplementedError()


# And take the global mean. Don't forget to calculate and include weights for the latitude of each grid cell.

# In[13]:


# YOUR CODE HERE

# calculate the latitude weights (radians)
weights = np.cos(np.deg2rad(ds_temperature['lat']))

# apply weights
temp_weighted = ds_temperature['temperature'].weighted(weights)

# calculate weighted mean 
ds_global_temp = temp_weighted.mean(dim=['lat', 'lon'])

# add back to dataset
ds_temperature = ds_temperature.assign(global_mean=ds_global_temp)

ds_temperature
# raise NotImplementedError()


# ## 3. Create a simple regression model of global warming
# 
# Global warming can be surprisingly well predicted just using a linear model of cumulative CO2 emissions. This is because the CO2 emissions are the primary driver of global warming, and the CO2 stays in the atmosphere for a long time (see e.g. https://www.nature.com/articles/ngeo3031).
# 
# To get global temperature as a function of cumulative CO2. You can use the `LinearRegression` class from `sklearn.linear_model`, with documentation provided [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). You should only need to use the `fit` and `predict` methods. The `fit` method takes two arguments, the first is the input data, and the second is the output data. The `predict` method takes one argument, the input data.
# 
# Alternatively, you can also use the `statsmodels` package to get more detailed statistics on the regression. See [here](https://www.statsmodels.org/stable/regression.html) for documentation.
# 
# Since we're only aiming to create an interpolation model, we don't need to worry too much about keeping a test set aside. We can just use all the data to train the model. You could also use a train-test split if you want to.

# In[14]:


import json
import os

def update_coefficients_json_by_key(key_id,slope,intercept,r_squared,filename):
    """
    Reads an existing JSON file, merges a single new set of coefficient data 
    under a specified key (country/global ID), and overwrites the file.

    Args:
        key_id: The unique identifier for the entry .
        slope: The calculated slope coefficient.
        intercept: The calculated intercept coefficient.
        r_squared: The calculated R-squared value.
        filename: The name of the JSON file to update.
    """
    
    # structure the new entry
    new_entry = {
        key_id: {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        }
    }
    
    # intialize the existing data
    existing_data = {}

    # load existing data if possible
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: File '{filename}' exists but is empty or invalid JSON. Starting fresh.")

    # add or update the existing data with the new entry
    existing_data.update(new_entry)

    # save file
    try:
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f"Data for '{key_id}' successfully merged and saved to: '{filename}'")

    except Exception as e:
        print(f"Error saving JSON file '{filename}': {e}")


# In[15]:


# YOUR CODE HERE
import json
import numpy as np
from sklearn.linear_model import LinearRegression



# align the 2 data sets
X_data_aligned, Y_data_aligned = xr.align(
    ds['cumulative_co2_GtC'],
    ds_temperature['global_mean'],
    join='inner', # Only keep years present in both datasets
    fill_value=np.nan # Use NaN for missing data (though inner join prevents this)
)

# convert to 2d numpy array
X = X_data_aligned.values.reshape(-1, 1)

# assign y
y = Y_data_aligned.values

# create + train
model = LinearRegression()
model.fit(X, y)

#extract + save the intercepts(Part 7)
save_file_name = "trained_linear_coefficients.json"

# extract coefficients from the fitted model
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y) 

update_coefficients_json_by_key("Global",slope,intercept,r_squared,save_file_name)


# Plot global mean temperature as a function of cumulative CO2 emissions, along with the regression fit
# 

# In[18]:


# YOUR CODE HERE

# regression fit values
Y_pred = model.predict(X)

# initialize plot
plt.figure(figsize=(10, 6))

# create scatter plot
plt.scatter(
    X.flatten(), 
    y, 
    label='Observed Data (Aligned Years)', 
    color='midnightblue', 
    alpha=0.5
)

# add prediciton line
plt.plot(
    X.flatten(), 
    Y_pred, 
    color='orangered', 
    linewidth=3, 
    label=f'Linear Fit (TCRE={model.coef_[0]:.4f} °C/GtC)'
)

# label axiis
plt.xlabel('Cumulative CO2 Emissions (GtC)')
plt.ylabel('Global Mean Temperature (°C)')
plt.title('Global Mean Temperature vs. Cumulative CO2 Emissions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

# show plot
plt.show()
# raise NotImplementedError()


# Plot global mean temperature as a function of year, along with the regression fit

# In[21]:


# YOUR CODE HERE

# extract years
years = Y_data_aligned['time'].values 

# extract temps
Y_observed = Y_data_aligned.values

# intialize plot
plt.figure(figsize=(12, 6))

# plot observed temp
plt.plot(years, Y_observed, label='Observed Global Mean Temp', color='midnightblue', alpha=0.7, linewidth=2)

# plot prediction/regression fit
plt.plot(years, Y_pred, color='orangered', linestyle='--', linewidth=2, 
         label='Regression Fit')

# title + labels
plt.xlabel('Year')
plt.ylabel('Global Mean Temperature (°C)')
plt.title('Observed Global Warming vs. Linear Trend Predicted by CO2')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')

# show plot
plt.show()
# raise NotImplementedError()


# Where does the prediction do well? Where does it do less well? Visualise the residuals.

# In[24]:


# YOUR CODE HERE

# ----------where does it do well-----------------
# => predicts long term trends for 25-50 years

# ----------where it doesnt do well--------------
# => short term yearly predictions
# => over predicited global warming during the 1950s - 1980

# calculate residuals
residuals = Y_observed - Y_pred

# initialize plot 
plt.figure(figsize=(12, 6))

# plot residuals vs. Year
# basis line (0 error).
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

# plot residuals
plt.scatter(years, residuals, color='forestgreen', alpha=0.9)

# labels + titles
plt.xlabel('Year')
plt.ylabel('Residuals (°C) [Observed - Predicted]')
plt.title('Residuals of Temperature Mean Prediction')
plt.grid(True, linestyle=':', alpha=0.6)

# show plot
plt.show()

# raise NotImplementedError()


# ## 4. Extend this to a regional temperature model, by region, and by state

# While the relationship between global temperature and cumulative CO2 emissions is very linear, the relationship between regional temperature and cumulative CO2 emissions is less so. This is because the regional temperature is affected by other factors, such as the regional distribution of land and ocean, and the regional distribution of CO2 emissions. Nevertheless, let's see how well it can do

# Read in the country mask which is a 2D array of the same size as the temperature data, with each grid cell containing the country code of the country that grid cell is in.

# In[25]:


country_mask_file = input_path+'country_mask.nc'


# Average the spatial coordinates into countries so that you end up with a dataset that has dimensionality of the number of countries by the number of time points.

# In[26]:


# YOUR CODE HERE

# import data into xarray
ds_mask = xr.open_dataset(country_mask_file)

ds_mask

#rename the abritray variable name for easier reference
ds_mask = ds_mask.rename({'__xarray_dataarray_variable__': 'country_code'})

ds_mask

# raise NotImplementedError()


# In[27]:


# YOUR CODE HERE
# extract data array by grouping
country_mask_da = ds_mask['country_code']

# calculate mean based off country code, since multi-dimensional use stacked
ds_country_means = (
    ds_temperature['temperature']
    .groupby(country_mask_da)
    .mean(dim='stacked_lat_lon') 
)

# convert the resulting DataArray back to a Dataset
ds_country_means = ds_country_means.to_dataset(name='country_mean_temp')

# rename country code to countries for easy ref
ds_country_means = ds_country_means.rename({'country_code': 'country'})

#rename time to year for dataset snyc for co2 emissions
ds_country_means = ds_country_means.rename({'time': 'year'})

ds_co2 = ds[['cumulative_co2_GtC']]

ds_country_means = ds_country_means.merge(ds_co2)

ds_country_means
# raise NotImplementedError()


# Plot a bar chart of the warming in each country in 2023. Note, the temperature data is baselined to 1850.

# In[29]:


# YOUR CODE HERE

# get baseline mean temp I am assuming that since the first data point was 1851 it means that it was the year 1850 - 1851 such that 1850 was the baseline
temp_base = ds_country_means.sel(year=1851)['country_mean_temp']

# get 2023 temp
temp_current = ds_country_means.sel(year=2023)['country_mean_temp']

#calculate warming
warming_2023 = temp_current - temp_base

# convert to df
df_warming = warming_2023.to_dataframe(name='warming_degC').reset_index()

# intialize plot
plt.figure(figsize=(20, 6))

# plot
plt.bar(
    df_warming['country'],
    df_warming['warming_degC'],
    color='teal'
)

# labeling and formatting
plt.xlabel('Country')
plt.ylabel('Warming Since 1851 (°C)')
plt.title('Country Warming from 1851 to 2023')
plt.xticks(rotation=90, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# show plot
plt.show()
# raise NotImplementedError()


# Calculate a linear regression model for each country along with the R^2 value. Plot the R^2 values for each country as a bar chart.

# In[30]:


# YOUR CODE HERE
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

# define regression for lambda in df
def linregress_country(data):
    """
    Performs linear regression (Temperature ~ Time) for a single country.
    Returns the R-squared value and the slope (warming rate).
    """

    # drop NaN
    required_cols = ['cumulative_co2_GtC', 'country_mean_temp']
    data = data.dropna(subset=required_cols)

    # define year as x
    x = data['cumulative_co2_GtC'].values
    # defin temp as y
    y = data['country_mean_temp'].values

    #reshape for sklearn
    X = x.reshape(-1, 1)

    # initialize and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # extract coefficients + r^2
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    # return r^2 + slope + intercepts
    return pd.Series({
        'r_squared': r_squared,
        'slope': slope,           
        'intercept': intercept    
    })


# In[31]:


# YOUR CODE HERE

# save trained linear coefficents (part 7)

def save_coefficients_to_json(df_results, filename):
    """
    Reads the existing coefficient JSON file (preserving entries), 
    merges all country-level coefficients from the DataFrame into it, and saves 
    the combined structure.
    
    Args:
        df_results: DataFrame containing 'country', 'slope', 
                                    'intercept', and 'r_squared' for countries.
        filename: The name of the file to save the JSON data to.
    """
    
    # initialize storage for all
    existing_data = {}

    # update storage with entries
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: File '{filename}' exists but is invalid JSON. Starting country list fresh.")

    # set the country as the key index
    df_indexed = df_results.set_index('country', drop=True).copy()
    
    # convert df to dictionary
    country_data_to_save = df_indexed.to_dict(orient='index')
    
    # add or update storage with new data
    existing_data.update(country_data_to_save)
    data_to_save = existing_data

    try:
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        
        print(f"Total entries saved/updated: {len(data_to_save)}")
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")


# In[40]:


# YOUR CODE HERE

# flatten dataset into pandas dataframe
df_regress = ds_country_means.to_dataframe().reset_index()

# group by country and apply the regression function
df_results = df_regress.groupby('country')[['cumulative_co2_GtC', 'country_mean_temp']].apply(linregress_country).reset_index()

# save trained linear coefficents (part 7)
save_file_name = "trained_linear_coefficients.json"
save_coefficients_to_json(df_results,save_file_name)

# initialize plot
plt.figure(figsize=(25, 7))

# plot bars
plt.bar(df_results['country'], df_results['r_squared'], color='cadetblue')

# label + formatting
plt.xlabel('Country Code')
plt.ylabel(r'$R^{2}$ Value') 
plt.title(r'$R^{2}$ for Linear Warming Trend (Temperature vs. cumulative CO2 emissions)')
plt.xticks(rotation=90, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# show plot
plt.show()

#not sure why there is a quesiton on the bottom but no answer box:

#sort the ascending by error
df_results = df_results.sort_values(by='r_squared', ascending=False)

#select 3 from the top
top_least_error = df_results.head(3)
print("Linear regression works really well with:")
print(top_least_error)

#select 3 from the bottom
worst_error = df_results.tail(3)
print("Linear regression works not that well with:")
print(worst_error)

# seems like trend for being good for southern hemisphere countries but not northern

# raise NotImplementedError()


# For which countries does the linear assumption work well, and where does it work less well?

# YOUR ANSWER HERE

# ## 5. Plot the regression coefficients for each country
# 
# 
# Which five countries are most sensitive to CO2 emissions and hence warming the fastest?

# In[41]:


# YOUR CODE HERE

#sort the warming data ascending by warming degress
df_warming = df_results.sort_values(by='slope', ascending=False)

#select 5 from the top
top_5_warming = df_warming.head(5)

print("Top 5 Countries most effected by CO2 Emissions:")
print(top_5_warming)

# raise NotImplementedError()


# ## 6. Do an analysis of your choosing
# 
# Maybe dig into the changes in one particular country, or look at changes in the variability of temperature. Perhaps look at the chances of exceeding certain temperature limits. 

# In[43]:


# YOUR CODE HERE

# I am choosing to analyze south korea

df_korea = ds_country_means.sel(country = "South_Korea")

# initialize plot
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Year')

# plot temperature
color_temp = 'tab:red'
ax1.plot(df_korea['year'], df_korea['country_mean_temp'], color=color_temp, label='Mean Temperature')
ax1.tick_params(axis='y', labelcolor=color_temp)

# Add horizontal line at 0 for baseline reference
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# plot world co2 level
ax2 = ax1.twinx()  
color_co2 = 'tab:blue'
ax2.plot(df_korea['year'], df_korea['cumulative_co2_GtC'], color=color_co2, linestyle='-', label='Cumulative CO2')
ax2.tick_params(axis='y', labelcolor=color_co2)
ax2.set_ylim(bottom=0) # Ensure cumulative emissions start from 0

# title + labels
ax2.set_ylabel('Cumulative CO2 Emissions (GtC)', color=color_co2)
ax1.set_ylabel('Country Mean Temp (°C)', color=color_temp)
plt.title('South Korea Mean Temperature Anomaly and Cumulative CO2 Over Time')

# Combine and display legends for both axes
lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

#raise NotImplementedError()


# In[44]:


# YOUR CODE HERE

# Noticed some interesting trends, for one, korea went into a massive temperature downfall after 1950

# find the lowest mean temp
lowest_temp_value = df_korea['country_mean_temp'].min()

# find the year of the lowest mean temp
year_of_lowest_temp = df_korea['country_mean_temp'].idxmin('year').item()

print(f"Year of Lowest Temperature: {year_of_lowest_temp}")
print(f"Absolute Lowest Temperature Value: {lowest_temp_value:.4f} °C")


# In[45]:


# YOUR CODE HERE

# start of data
start_year = 1850

# just before the start of the industrial revolution of south kroea
end_year = 1960


# isolate analysis years;
pre_war = df_korea.sel(year=slice(start_year, end_year))

# Now, convert the filtered Xarray object to a Pandas DataFrame for plotting
# The rest of your plotting code (which is Pandas/Matplotlib based) will now work.
pre_war_pd = pre_war.to_dataframe().reset_index()

# 2. Plotting the trend (Use df_filtered_pd instead of df_filtered)
fig, ax = plt.subplots(figsize=(10, 5))

# Plot Temperature Anomaly
ax.plot(pre_war_pd['year'], pre_war_pd['country_mean_temp'], color='tab:red', linewidth=2)

# Add horizontal line at 0 for baseline reference
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Calculate and plot the linear regression trend line for this period
# Prepare data for linear regression using numpy polyfit
X = pre_war_pd['year'].values
y = pre_war_pd['country_mean_temp'].values

# Fit a first-degree polynomial (linear trend)
trend_coeffs = np.polyfit(X, y, 1)
trend_line = np.poly1d(trend_coeffs)

# The slope (trend_coeffs[0]) is multiplied by 10 to get the rate per decade
slope_per_decade = trend_coeffs[0] * 10

# an ice age is classified as 8 C (this was the last ice age)
# currently the average temp in the world is 15C and korea has an average temp of 13
# by this account where korea is 2C less than the world, the world's temp in 1850 
# was 13C so that means korea was around 11C. The difference need at that point would be 3C
temp_change_needed = 3.0 # °C

# Assume a hypothetical cooling rate equal in magnitude to the calculated warming rate.
annual_rate = abs(trend_coeffs[0])

# Calculate years required
years_to_ice_age = temp_change_needed / annual_rate

print(f"Calculated Slope (1850-1960): {slope_per_decade:.4f} °C/decade")
print(f"Hypothetical Years to reach 8°C: {years_to_ice_age:.0f} years")
# The slope (trend_coeffs[0]) is multiplied by 10 to get the rate per decade
slope_per_decade = trend_coeffs[0] * 10 

# Plot the trend line
ax.plot(X, trend_line(X), color='black', linestyle='-', linewidth=1, label=f'Linear Trend ({slope_per_decade:.3f} °C/decade)')

# 3. Final Formatting
ax.set_title(f'South Korea Mean Temperature Anomaly Trend: {start_year} to {end_year}')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Temperature Anomaly (°C)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='best')

plt.tight_layout()




# In[122]:


# The industrial revolution started int the 1960s with the rise of the asian tigers.
start_year = 1960

# to end of data
end_year = 2050
# 1. Use .sel() with slice() to filter the DataArray by the 'year' coordinate.
# This works directly on the Xarray object.
industrial_selection = df_korea.sel(year=slice(start_year, end_year))

# Now, convert the filtered Xarray object to a Pandas DataFrame for plotting
# The rest of your plotting code (which is Pandas/Matplotlib based) will now work.
industrial_selection_pd = industrial_selection.to_dataframe().reset_index()

# 2. Plotting the trend (Use df_filtered_pd instead of df_filtered)
fig, ax = plt.subplots(figsize=(10, 5))

# Plot Temperature Anomaly
ax.plot(industrial_selection_pd['year'], industrial_selection_pd['country_mean_temp'], color='tab:red', linewidth=2)

# Add horizontal line at 0 for baseline reference
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Calculate and plot the linear regression trend line for this period
# Prepare data for linear regression using numpy polyfit
X = industrial_selection_pd['year'].values
y = industrial_selection_pd['country_mean_temp'].values

# Fit a first-degree polynomial (linear trend)
trend_coeffs = np.polyfit(X, y, 1)
trend_line = np.poly1d(trend_coeffs)

# The slope (trend_coeffs[0]) is multiplied by 10 to get the rate per decade
slope_per_decade = trend_coeffs[0] * 10

# human body temp is 37
# currently the average temp in the world is 15C and korea has an average temp of 13
# by this account where korea is 2C less than the world, the world's temp in 1850 
# was 13C so that means korea was around 11C. The difference need at that point would be 3C
temp_change_needed = 37 - 11 # °C

# Assume a hypothetical cooling rate equal in magnitude to the calculated warming rate.
annual_rate = abs(trend_coeffs[0])

# Calculate years required
years_to_human_body_temp = temp_change_needed / annual_rate


print(f"Calculated Slope (1960-2015): {slope_per_decade:.4f} °C/decade")
print(f"Hypothetical Years to reach 37°C: {years_to_human_body_temp:.0f} years")


# The slope (trend_coeffs[0]) is multiplied by 10 to get the rate per decade
slope_per_decade = trend_coeffs[0] * 10 

# Plot the trend line
ax.plot(X, trend_line(X), color='black', linestyle='-', linewidth=1, label=f'Linear Trend ({slope_per_decade:.3f} °C/decade)')

# 3. Final Formatting
ax.set_title(f'South Korea Mean Temperature Anomaly Trend: {start_year} to {end_year}')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Temperature Anomaly (°C)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='best')

plt.tight_layout()


# In[ ]:





# ## 7. Make a command line interface to a prediction script 
# 
# The inputs should include the name of a country (or global mean) and cumulative CO2 emissions. It should return the predicted temperature change relative to 1850. You can use the `argparse` package to do this. See [here](https://docs.python.org/3/howto/argparse.html) for documentation. Be sure to check for valid inputs.
# 
# Also provide the option to save the predictions to a CSV file.
# 
# This script should use the regression coefficients learned in the previous step so it doesn't have to use the full model output each time. You could store them in a numpy file, a pandas CSV file, or even JSON. 

# In[46]:


# YOUR CODE HERE

# raise NotImplementedError()


# In[47]:


get_ipython().system('python predict_temp.py South_Korea 1500.0')


# ---
# ### No action is required below. Your answers for the project should be written **above this line only**.  
# *(Do **not** modify or delete the following “YOUR ANSWER HERE” block.)*
# 
# You may see a block below that says **"YOUR ANSWER HERE"**. This block is **not for you to fill in** — it is used internally by the TAs to grade the documentation, coding style, and code quality portions of your project.
# 
# - Documentation and Explanation — 10 points  
# - Code Documentation & Style — 5 points  
# - Code Organization & Structure — 5 points  
# 

# YOUR ANSWER HERE

# YOUR ANSWER HERE

# YOUR ANSWER HERE
