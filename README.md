# Predicting Global Temperature from Cumulative CO2 Emissions

This project explores the relationship between cumulative CO2 emissions and global temperature change using climate model projections. The goal is to investigate how well global temperature trends can be explained by historical and projected anthropogenic CO2 emissions.

The analysis is based on climate simulation data from the NorESM2 climate model, provided through the ClimateBench dataset.

## Project Overview

CO2 emissions are the primary driver of long-term global temperature change. Because CO2 remains in the atmosphere for centuries, cumulative emissions provide a useful predictor of global warming. In this project, we construct a simple regression model to analyze the relationship between cumulative emissions and modeled global temperature.

The workflow includes:

- Processing historical and projected CO2 emissions data
- Computing cumulative CO2 emissions over time
- Extracting global temperature data from climate model outputs
- Calculating a latitude-weighted global mean temperature
- Fitting a regression model to estimate temperature change as a function of cumulative emissions

## Data Sources

The project uses climate model data from the **NorESM2 climate model** included in the **ClimateBench dataset**.

Main datasets used:

- Historical and projected CO2 emissions data (1850–2100)
- Global surface temperature projections from NorESM2

The temperature data is provided as a NetCDF file and processed using the `xarray` library.

## Methodology

### 1. CO2 emissions processing

Monthly CO2 emissions data from multiple industrial sectors is combined and aggregated to produce annual global emissions totals.

Steps:

1. Load multiple CSV files containing emissions data
2. Concatenate historical and projected emissions datasets
3. Aggregate emissions across sectors and months
4. Compute cumulative CO2 emissions from 1850 onward

### 2. Temperature data processing

Temperature data from the climate model is stored in NetCDF format.

Processing steps:

- Load the dataset using `xarray`
- Compute latitude-based weights using the cosine of latitude
- Calculate the weighted global mean temperature

### 3. Regression model

A simple linear regression model is used to estimate the relationship between cumulative CO2 emissions and global temperature.

The model is implemented using `scikit-learn`:

Temperature = β₀ + β₁ × (Cumulative CO₂)

This provides a simple approximation of how global temperature responds to accumulated greenhouse gas emissions.

### 4. Visualization

The project includes visualizations of:

- cumulative CO2 emissions over time
- global temperature trends
- regression relationship between emissions and temperature

## Technologies Used

- Python
- Pandas
- Xarray
- NumPy
- Matplotlib
- Scikit-learn

## Possible Extensions

This project could be extended by:

- exploring nonlinear climate response models
- comparing results across multiple climate models
- incorporating additional forcing factors such as methane or aerosols
