# loading libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope


# Defining directories
dir = os.getcwd()
# Creating a directory for the data insights
os.makedirs(os.path.join(dir, 'Raw Data', 'Permit Data Insights'), exist_ok=True)


chunks = []

permit_path = os.path.join(dir, 'Raw Data', 'BPS_Compiled_File_202412.csv')

for chunk in pd.read_csv(permit_path, encoding='latin1', chunksize=10**6):
    chunks.append(chunk)

permit_raw = pd.concat(chunks, ignore_index=True)
zillow_rent = pd.read_csv(os.path.join(dir, 'Raw Data', 'Zip_zori_uc_sfrcondomfr_sm_month (1).csv'))
zillow_rent_seasonal = pd.read_csv(os.path.join(dir, 'Raw Data', 'Zip_zori_uc_sfrcondomfr_sm_sa_month (1).csv'))
# Load shapefiles
shapefile_2020 = gpd.read_file(r"C:\Users\...\...\JHU\HomeEconomics\Shapefiles\Files 2020\tl_2020_us_zcta520.shp")
shapefile_2010 = gpd.read_file(r"C:\Users\...\...\JHU\HomeEconomics\Shapefiles\Files 2020\tl_2020_us_zcta520.shp")


# Getting insights about the dataset

col_names = permit_raw.columns

# # # # Saving Col Names for Review # # # #
# data_insight_paths = os.path.join(dir, 'Raw Data', 'Permit Data Insights')
# pd.Series(col_names).to_csv(os.path.join(data_insight_paths, 'col_names.txt'), index=False)

#################
permit_raw['ZIP_CODE'].isna().value_counts() # False - 7.8 Mn , True - 3.05 Mn
#################

#################
permit_raw['LOCATION_TYPE'].value_counts() 
## Place       9707208 , County       744651
## Metro        334604, State         47570, 
## Micro         12816 , Division       8388, 
## Region         3728 , Country         932
#################

#################
# --> I will first filter the data for place only
# --> Since data from 1990 to 2000 is limited to 3000 entries, we are dropping 1999 and before
# --> 'Place' is the base denomination. I will group 'Place' by 'Zip Code' and then by 'County' for further analysis
# --> There are multiple 'Place' for a 'Zip Code'; In other words, 'Zip Code' is not unique
# --> For the first phase, I will transfer the dataset onto CSV and work on a different python file
################

# Isolating permit raw for County and year
place = permit_raw[permit_raw['LOCATION_TYPE'] == 'Place']
place = place[place['YEAR'] >= 2000]

place.to_csv(os.path.join(dir, 'Raw Data', 'Semi_Datasets', 'place.csv'), index=False)

place.to_csv('place.csv', index=False)

# Cleaning the dates
### Dataset from 2000 to 2024


# Clenaing the dates
### -> 99 is used for values that are pulled on a yearly basis
place['SURVEY_DATE_tr'] = np.where(place['MONTH'] == 99, np.nan, place['YEAR'].astype(str) + '-' + place['MONTH'].astype(str).str.zfill(2))

place['SURVEY_DATE_tr'] = pd.to_datetime(place['SURVEY_DATE_tr'], format='%Y-%m', errors='coerce').dt.strftime('%Y-%m')

place.dropna(subset = ['SURVEY_DATE_tr'], inplace = True)

place['SURVEY_DATE_tr'].value_counts().sort_index() ## Ensuring that the dates are cleaned


######################### Grouping values by Zip Code into a new data fram #########################

#### Cleaning the zip codes column ####

# Drop rows where ZIP_CODE_int length is greater than 5 characters
place_clean = place[place['ZIP_CODE'].str.len() <= 7]
place_clean = place[place['ZIP_CODE'].str.len() == 5] # approx 300 zip codes with a length of less than 5 characters for zip code

# converting zip code to string from float
place_clean['ZIP_CODE_int'] = place_clean['ZIP_CODE'].str[:5]
place_clean['ZIP_CODE_int'] = place_clean['ZIP_CODE_int'].astype(str)

# Dropping rows where ZIP_CODE_int length is greater than 5 characters

place_val_cols = place_clean[['ZIP_CODE_int', 'SURVEY_DATE_tr', 'BLDGS_1_UNIT','BLDGS_2_UNITS','BLDGS_3_4_UNITS',
                        'BLDGS_5_UNITS','TOTAL_UNITS']].reset_index(drop=True)

# place_val_cols[['ZIP_CODE_int']].value_counts() # checking to test accuracy and balance of the dataset

### Grouping by Zip Code

place_summarized = place_val_cols.groupby(['ZIP_CODE_int', 'SURVEY_DATE_tr']).sum().reset_index()

# place_summarized.head() ## Checking the data

### Transforming the data to get a wide format

place_summarized_unit1_t = place_summarized.pivot(index='ZIP_CODE_int', columns='SURVEY_DATE_tr', values='BLDGS_1_UNIT').fillna(0)
place_summarized_unit2_t = place_summarized.pivot(index='ZIP_CODE_int', columns='SURVEY_DATE_tr', values='BLDGS_2_UNITS').fillna(0)
place_summarized_unit3to4_t = place_summarized.pivot(index='ZIP_CODE_int', columns='SURVEY_DATE_tr', values='BLDGS_3_4_UNITS').fillna(0)
place_summarized_unit5_t = place_summarized.pivot(index='ZIP_CODE_int', columns='SURVEY_DATE_tr', values='BLDGS_5_UNITS').fillna(0)
place_summarized_total_t = place_summarized.pivot(index='ZIP_CODE_int', columns='SURVEY_DATE_tr', values='TOTAL_UNITS').fillna(0)

# -> Each approx 13.6K rows of data and 275 cols

### Identifying core objects for further analysis
months_in_data = np.sort(place_val_cols['SURVEY_DATE_tr'].unique())
months_in_data_post15 = months_in_data[months_in_data >= '2016-01']

###  Extracting semi datasets for easier processing
# os.makedirs(os.path.join(dir, 'Raw Data', 'Semi_Datasets'), exist_ok=True)
save_path = os.path.join(dir, 'Raw Data', 'Semi_Datasets')

place_summarized_unit1_t.to_csv(os.path.join(save_path, 'place_summarized_unit1_t.csv'), index=True)
place_summarized_unit2_t.to_csv(os.path.join(save_path, 'place_summarized_unit2_t.csv'), index=True)
place_summarized_unit3to4_t.to_csv(os.path.join(save_path, 'place_summarized_unit3to4_t.csv'), index=True)
place_summarized_unit5_t.to_csv(os.path.join(save_path, 'place_summarized_unit5_t.csv'), index=True)
place_summarized_total_t.to_csv(os.path.join(save_path, 'place_summarized_total_t.csv'), index=True)

####### Cleaning unit dataset #######
def clean_dataset(df, months_in_data_post15):
    permit_sums = df.cumsum(axis=1)  # Calculating cumulative permits issued
    monthly_growth = permit_sums.pct_change(axis=1) * 100  # Calculating monthly growth in permits issued
    monthly_growth.fillna(0, inplace=True)  # Replacing NaN with 0
    monthly_growth.replace([np.inf, -np.inf], 100, inplace=True)  # Replacing inf with 100
    monthly_growth_filtered = monthly_growth.loc[:, monthly_growth.columns.isin(months_in_data_post15)]
    return monthly_growth_filtered

# Applying the function to each dataset
unit1_monthly_growth_filtered = clean_dataset(place_summarized_unit1_t, months_in_data_post15)
unit2_monthly_growth_filtered = clean_dataset(place_summarized_unit2_t, months_in_data_post15)
unit3to4_monthly_growth_filtered = clean_dataset(place_summarized_unit3to4_t, months_in_data_post15)
unit5_monthly_growth_filtered = clean_dataset(place_summarized_unit5_t, months_in_data_post15)
total_monthly_growth_filtered = clean_dataset(place_summarized_total_t, months_in_data_post15)


###### because function is not working #######

unit1_monthly_growth_filtered = place_summarized_unit1_t.cumsum(axis=1)
unit2_monthly_growth_filtered = place_summarized_unit2_t.cumsum(axis=1)
unit3to4_monthly_growth_filtered = place_summarized_unit3to4_t.cumsum(axis=1)
unit5_monthly_growth_filtered = place_summarized_unit5_t.cumsum(axis=1)
total_monthly_growth_filtered = place_summarized_total_t.cumsum(axis=1)

unit1_monthly_growth = unit1_monthly_growth_filtered.pct_change(axis=1) * 100
unit2_monthly_growth = unit2_monthly_growth_filtered.pct_change(axis=1) * 100
unit3to4_monthly_growth = unit3to4_monthly_growth_filtered.pct_change(axis=1) * 100
unit5_monthly_growth = unit5_monthly_growth_filtered.pct_change(axis=1) * 100
total_monthly_growth = total_monthly_growth_filtered.pct_change(axis=1) * 100

unit1_monthly_growth.fillna(0, inplace=True)
unit2_monthly_growth.fillna(0, inplace=True)
unit3to4_monthly_growth.fillna(0, inplace=True)
unit5_monthly_growth.fillna(0, inplace=True)
total_monthly_growth.fillna(0, inplace=True)

unit1_monthly_growth.replace([np.inf, -np.inf], 100, inplace=True)
unit2_monthly_growth.replace([np.inf, -np.inf], 100, inplace=True)
unit3to4_monthly_growth.replace([np.inf, -np.inf], 100, inplace=True)
unit5_monthly_growth.replace([np.inf, -np.inf], 100, inplace=True)
total_monthly_growth.replace([np.inf, -np.inf], 100, inplace=True)

# Create a new directory for processed segmented data
processed_path = os.path.join(dir, 'Raw Data', 'Processed_Segmented_Data')
os.makedirs(processed_path, exist_ok=True)

# Export the processed monthly growth data
unit1_monthly_growth.to_csv(os.path.join(processed_path, 'unit1_monthly_growth.csv'), index=True)
unit2_monthly_growth.to_csv(os.path.join(processed_path, 'unit2_monthly_growth.csv'), index=True)
unit3to4_monthly_growth.to_csv(os.path.join(processed_path, 'unit3to4_monthly_growth.csv'), index=True)
unit5_monthly_growth.to_csv(os.path.join(processed_path, 'unit5_monthly_growth.csv'), index=True)
total_monthly_growth.to_csv(os.path.join(processed_path, 'total_monthly_growth.csv'), index=True)


# Calculate progressive sums by ZIP code for each building type
progressive_sums = {
    'unit1': place_summarized.sort_values('SURVEY_DATE_tr').groupby('ZIP_CODE_int')['BLDGS_1_UNIT'].cumsum(),
    'unit2': place_summarized.sort_values('SURVEY_DATE_tr').groupby('ZIP_CODE_int')['BLDGS_2_UNITS'].cumsum(),
    'unit3to4': place_summarized.sort_values('SURVEY_DATE_tr').groupby('ZIP_CODE_int')['BLDGS_3_4_UNITS'].cumsum(), 
    'unit5': place_summarized.sort_values('SURVEY_DATE_tr').groupby('ZIP_CODE_int')['BLDGS_5_UNITS'].cumsum(),
    'total': place_summarized.sort_values('SURVEY_DATE_tr').groupby('ZIP_CODE_int')['TOTAL_UNITS'].cumsum()
}

# Create DataFrames with progressive sums
for unit_type, prog_sum in progressive_sums.items():
    place_summarized[f'{unit_type}_progressive_sum'] = prog_sum

# Export progressive sums to CSV
progressive_sums_path = os.path.join(processed_path, 'progressive_sums')
os.makedirs(progressive_sums_path, exist_ok=True)

place_summarized.to_csv(os.path.join(progressive_sums_path, 'all_progressive_sums.csv'), index=False)

total_monthly_growth_filtered.to_csv(os.path.join(processed_path, 'total_monthly_growth_filtered.csv'), index=True)
########################## 
# Switching files to understanding data.py
########################## 


############################# PLOTTING (testing)#################################
##### Plotting the growth through time #####



# Load state and national boundary shapefiles
states = gpd.read_file(r"C:\Users\svij2\...\JHU\HomeEconomics\Shapefiles\Country and State\cb_2018_us_state_500k.shp")

# Calculate the average monthly change in permits for 2016 columns
unit1_avg_monthly_change_2016v20 = ((place_summarized_total_t['2020-01'] - place_summarized_total_t['2016-01']) / place_summarized_total_t['2016-01'] * 100).reset_index().replace([np.inf, -np.inf], 100).replace(np.nan, 0)

unit1_avg_monthly_change_2016v20.rename(columns={'ZIP_CODE_int':'ZCTA5CE20', 0:'mean'}, inplace=True)
unit1_avg_monthly_change_2016v20['ZCTA5CE20'] = unit1_avg_monthly_change_2016v20['ZCTA5CE20'].astype(str)
# Merge the average monthly change data with the shapefile data
shapefile_2020_m = shapefile_2020.merge(unit1_avg_monthly_change_2016v20, on='ZCTA5CE20', how='inner')
shapefile_2010_m = shapefile_2010.merge(unit1_avg_monthly_change_2016v20, on='ZCTA5CE20', how='inner')

# Plotting the data
# Plot pre-2020 data
fig, ax = plt.subplots(figsize=(12, 8))
shapefile_2010_m.plot(column='mean', cmap='OrRd', linewidth=0, edgecolor=None,
                      legend=True, ax=ax, vmin=-5, vmax=10)

# Add state boundaries
states.geometry.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

plt.title('Average Monthly Change in Permits (2016-2020)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-125, -66)
plt.ylim(25, 50)
# plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'Average Monthly Change in Permits (2016-2020).png'))
plt.show()

#############################
# Using the total units 
#############################

total_monthly_growth_filtered.head()

# Perform anomaly detection focusing on the last 12 months of data

# Get the last 6 months of data
last_6_months = total_monthly_growth_filtered.select_dtypes(include=['float64', 'int64']).iloc[:, -6:]

# Calculate the mean growth rate for each ZIP code over the last 12 months
mean_growth_rates = last_6_months.mean(axis=1)

# Standardize the mean growth rates
scaler = StandardScaler()
growth_rates_scaled = scaler.fit_transform(mean_growth_rates.values.reshape(-1, 1))

# Perform anomaly detection using Robust Covariance Estimation
outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
outliers = outlier_detector.fit_predict(growth_rates_scaled)

# Add results to the original dataframe
total_monthly_growth_filtered['Mean_Growth_Last_6M'] = mean_growth_rates
total_monthly_growth_filtered['Is_Anomaly'] = outliers



# Get additional location information for anomalous ZIPs
anomalous_zips = total_monthly_growth_filtered[total_monthly_growth_filtered['Is_Anomaly'] == -1]

anomalous_zips['ZIP_CODE_int'] = anomalous_zips['ZIP_CODE_int'].astype(str)
place_clean['ZIP_CODE_int'] = place_clean['ZIP_CODE_int'].astype(str)

anomalous_zips_info = pd.merge(
    anomalous_zips,
    place_clean[['ZIP_CODE_int','COUNTY_NAME', 'STATE_NAME']].drop_duplicates(),
    on='ZIP_CODE_int',
    how='inner'
)

anomalous_zips_info

# Create summary text
summary_text = []
summary_text.append("Anomaly Detection Summary (Last 6 Months):")
summary_text.append(f"Number of ZIPs with anomalous growth: {len(total_monthly_growth_filtered[total_monthly_growth_filtered['Is_Anomaly'] == -1])}")
summary_text.append(f"Number of ZIPs with normal growth: {len(total_monthly_growth_filtered[total_monthly_growth_filtered['Is_Anomaly'] == 1])}")
summary_text.append("\nStatistics for Anomalous ZIPs:")
summary_text.append(f"Average growth rate: {anomalous_zips['Mean_Growth_Last_6M'].mean():.2f}%")
summary_text.append(f"Min growth rate: {anomalous_zips['Mean_Growth_Last_6M'].min():.2f}%")
summary_text.append(f"Max growth rate: {anomalous_zips['Mean_Growth_Last_6M'].max():.2f}%")

# Add state-level anomaly counts
summary_text.append("\nAnomalies by State:")
state_counts = anomalous_zips_info['STATE_NAME'].value_counts()
for state, count in state_counts.items():
    summary_text.append(f"{state}: {count} anomalous ZIPs")

summary_text.append("\nLocation Details for Anomalous ZIPs:")

# Add location details
location_details = anomalous_zips_info[['ZIP_CODE_int', 'COUNTY_NAME', 'STATE_NAME', 'Mean_Growth_Last_6M']].sort_values('Mean_Growth_Last_6M', ascending=False)
for _, row in location_details.iterrows():
    summary_text.append(f"ZIP: {row['ZIP_CODE_int']}, County: {row['COUNTY_NAME']}, State: {row['STATE_NAME']}, Growth: {row['Mean_Growth_Last_6M']:.2f}%")

# Write to file
with open(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'anomaly_detection_summary.txt'), 'w') as f:
    f.write('\n'.join(summary_text))

########################################################################
################ Comparing with Rents using Zillow Rents ################
########################################################################

# Merge total monthly growth data with Zillow rent data
# First ensure RegionName is numeric type to match ZIP_CODE
zillow_rent_seasonal['RegionName'] = pd.to_numeric(zillow_rent_seasonal['RegionName'], errors='coerce')
zillow_rent_seasonal.rename(columns={'RegionName': 'ZIP_CODE_int'}, inplace=True)
zillow_rent_seasonal['ZIP_CODE_int'] = zillow_rent_seasonal['ZIP_CODE_int'].astype(str)
total_monthly_growth_filtered.reset_index(inplace=True)

# Get columns from 2015-01-31 onwards
rent_cols = [col for col in zillow_rent_seasonal.columns if col == 'ZIP_CODE_int' or 
            (pd.to_datetime(col, format='%Y-%m-%d', errors='coerce') >= pd.to_datetime('2015-01-31'))]

rent_w_permit_gr = pd.merge(
    total_monthly_growth_filtered.add_suffix('_permit'),
    zillow_rent_seasonal[rent_cols].add_suffix('_rent'),
    left_on='ZIP_CODE_int_permit',
    right_on='ZIP_CODE_int_rent',
    how='inner'
)

zillow_rent_a = zillow_rent_seasonal.copy()

# Get first 7 chars of rent date columns only (excluding ZIP_CODE_int)
rent_cols = [col[:7] if col != 'ZIP_CODE_int' else col for col in rent_cols]

# Rename only the rent date columns in zillow_rent_a to match 
for col in zillow_rent_a.columns:
    if col != 'ZIP_CODE_int':
        zillow_rent_a.rename(columns={col: col[:7]}, inplace=True)

total_monthly_permit_agg = place_summarized_total_t.cumsum(axis=1)

total_monthly_permit_agg = total_monthly_permit_agg.loc[:, total_monthly_permit_agg.columns.isin(months_in_data_post15)]

total_monthly_permit_agg.reset_index(inplace=True)

rent_w_permit = pd.merge(
    total_monthly_permit_agg.add_suffix('_permit'),
    zillow_rent_a[rent_cols].add_suffix('_rent'),
    left_on='ZIP_CODE_int_permit',
    right_on='ZIP_CODE_int_rent',
    how='inner'
)

######################################
# Correlogram
######################################  

df = rent_w_permit.copy()

df = df[~(df[permit_dates] < 50).all(axis=1)] # Filtering out those zips that have less than 50 permits in the last 9 years
# Remove columns before 2016 for both permit and rent data
df = df[[col for col in df.columns if col == 'ZIP_CODE_int_permit' or col == 'ZIP_CODE_int_rent' or 
         (pd.to_datetime(col.split('_')[0], format='%Y-%m') >= pd.to_datetime('2016-01-01'))]]

# Extract permit and rent columns (excluding ZIP code)
permit_cols = [col for col in df.columns if col.endswith('_permit') and col != 'ZIP_CODE_int_permit']
rent_cols = [col for col in df.columns if col.endswith('_rent')]

rent_cols.remove('ZIP_CODE_int_rent')

# Sort columns by date (assuming the date is the first part of the column name in YYYY-MM format)
permit_dates = sorted(permit_cols, key=lambda x: pd.to_datetime(x.split('_')[0], format='%Y-%m'))
rent_dates = sorted(rent_cols, key=lambda x: pd.to_datetime(x.split('_')[0], format='%Y-%m'))

# Create separate DataFrames for permits and rents with datetime indices for columns
permits_df = df[permit_dates].copy()
permits_df.columns = pd.to_datetime([col.split('_')[0] for col in permit_dates], format='%Y-%m')

rent_df = df[rent_dates].copy()
rent_df.columns = pd.to_datetime([col.split('_')[0] for col in rent_dates], format='%Y-%m')

# Calculate percentage changes along the time axis (axis=1)
# Multiply by 100 if you prefer percentages
permit_pct = permits_df.pct_change(axis=1) * 100
rent_pct = rent_df.pct_change(axis=1) * 100

# -> Relationship between permit issued in time t and lagged rents 

max_lag = 36
lag_corrs = []

for lag in range(max_lag + 1):
    # For a given lag, shift the rent series so that:
    # rent_pct[t] becomes aligned with permit_pct[t] from lagged perspective.
    # Here, shifting with a negative lag moves rent data to the left,
    # meaning rent change at time t will be compared with permit change at time t-lag.
    shifted_rent = rent_pct.shift(-lag, axis=1)
    
    row_corrs = []
    # Iterate over each row (e.g., each ZIP code)
    for i in range(df.shape[0]):
        permit_series = permit_pct.iloc[i].dropna()
        rent_series = shifted_rent.iloc[i].dropna()
        # Find common time points after shifting
        common_index = permit_series.index.intersection(rent_series.index)
        if len(common_index) > 1:
            r = permit_series.loc[common_index].corr(rent_series.loc[common_index])
            row_corrs.append(r)
    # Average the correlations across rows (ZIP codes)
    lag_corrs.append(np.nanmean(row_corrs))

# Plot the correlations as a bar graph (correlogram style)
plt.figure(figsize=(12, 6))
plt.bar(range(max_lag + 1), lag_corrs, color='skyblue')
plt.xlabel('Lag (months)')
plt.ylabel('Average correlation')
plt.title('Correlation between % Change in Permits and % Change in Rent (Lagged)')
plt.xticks(range(0, max_lag + 1, 3))
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'Correlation between % Change in Permits and % Change in Rent (Lagged).png'))
plt.show()


# -> Relationship between rents in time t and lagged permits

max_lag = 36
lag_corrs = []

for lag in range(max_lag + 1):
    shifted_permit = permit_pct.shift(-lag, axis=1)
    
    row_corrs = []
    for i in range(df.shape[0]):
        # For each row (e.g., each ZIP code), compare rent at time t with permit at time t+lag
        rent_series = rent_pct.iloc[i].dropna()
        permit_series = shifted_permit.iloc[i].dropna()
        
        # Find common time indices after shifting
        common_index = rent_series.index.intersection(permit_series.index)
        if len(common_index) > 1:
            r = rent_series.loc[common_index].corr(permit_series.loc[common_index])
            row_corrs.append(r)
    
    lag_corrs.append(np.nanmean(row_corrs))

# Plot the reversed correlation as a bar graph
plt.figure(figsize=(12, 6))
plt.bar(range(max_lag + 1), lag_corrs, color='lightgreen')
plt.xlabel('Lag (months)')
plt.ylabel('Average correlation')
plt.title("Correlation between % Change in Rent and % Change in Permits (Lagged)")
plt.xticks(range(0, max_lag + 1, 3))
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'Correlation between % Change in Rent and % Change in Permits (Lagged).png'))
plt.show()

########################################################
# Relationship between permits for MFH and rents
########################################################

place_summarized_total_t.columns













