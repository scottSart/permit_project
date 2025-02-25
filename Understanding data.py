import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import pgeocode
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

###########################
#### Identifying paths ####
###########################
dir = os.getcwd()

progressive_sums_path = os.path.join(dir, 'Raw Data', 'Processed_Segmented_Data', 'progressive_sums')

unit_specific_path = os.path.join(dir, 'Raw Data', 'Processed_Segmented_Data')

summarized_path = os.path.join(dir, 'Raw Data', 'Semi_Datasets')

zip_code_shapefile = pd.read_excel(r"C:\Users\...\...\JHU\HomeEconomics\Shapefiles\ZIPCODES Mapping\ZIP_Locale_Detail.xls")

#### loading the progressive sum cumulative data ####

progressive_sums_cumulative = pd.read_csv(os.path.join(progressive_sums_path, 'all_progressive_sums.csv'))


#### Total monthly growth dataset ####

total_monthly_growth = pd.read_csv(os.path.join(unit_specific_path, 'total_monthly_growth_filtered.csv'))
## pct change dataset

unit_1 = pd.read_csv(os.path.join(unit_specific_path, 'unit1_monthly_growth.csv'))
unit_2 = pd.read_csv(os.path.join(unit_specific_path, 'unit2_monthly_growth.csv'))
unit_3to4 = pd.read_csv(os.path.join(unit_specific_path, 'unit3to4_monthly_growth.csv'))
unit_5 = pd.read_csv(os.path.join(unit_specific_path, 'unit5_monthly_growth.csv'))

## summarized dataset

unit_1_summarized = pd.read_csv(os.path.join(summarized_path, 'place_summarized_unit1_t.csv'))
unit_2_summarized = pd.read_csv(os.path.join(summarized_path, 'place_summarized_unit2_t.csv'))
unit_3to4_summarized = pd.read_csv(os.path.join(summarized_path, 'place_summarized_unit3to4_t.csv'))
unit_5_summarized = pd.read_csv(os.path.join(summarized_path, 'place_summarized_unit5_t.csv'))
total_summarized = pd.read_csv(os.path.join(summarized_path, 'place_summarized_total_t.csv'))


###########################
# Playing with data to get insights
###########################

progressive_sums_cumulative['ZIP_CODE_int'] = progressive_sums_cumulative['ZIP_CODE_int'].astype(str).str.zfill(5)
zip_code_shapefile['PHYSICAL ZIP'] = zip_code_shapefile['PHYSICAL ZIP'].astype(str).str.zfill(5)


total_monthly_growth['ZIP_CODE_int'].astype(str).str.len().value_counts() 
## There are 1531, 4 digit zip codes and 12138, 5 digit zip codes

# Converting each 4 digit to 5 by adding a 0 to the front
total_monthly_growth['ZIP_CODE_int'] = total_monthly_growth['ZIP_CODE_int'].astype(str).str.zfill(5)

# grouping permits issued by quarter

total_summarized['ZIP_CODE_int'] = total_summarized['ZIP_CODE_int'].astype(str).str.zfill(5)


# Convert year-month columns to quarters and sum
# Get all columns except ZIP_CODE_int which contains the time series data
time_cols = [col for col in total_summarized.columns if col != 'ZIP_CODE_int']

# Create empty dataframe with ZIP codes
quarterly_total = pd.DataFrame(total_summarized['ZIP_CODE_int'])



# Process each quarter
for year in range(2000, 2025): 
	for quarter in range(1, 5):
		# Get months in this quarter
		if quarter == 1:
			months = [f'{year}-01', f'{year}-02', f'{year}-03']
		elif quarter == 2:
			months = [f'{year}-04', f'{year}-05', f'{year}-06']
		elif quarter == 3:
			months = [f'{year}-07', f'{year}-08', f'{year}-09']
		else:
			months = [f'{year}-10', f'{year}-11', f'{year}-12']
			
		# Only include months that exist in the dataset
		valid_months = [m for m in months if m in time_cols]
		
		if valid_months:  # If we have any data for this quarter
			quarter_name = f'{year}-Q{quarter}'
			quarterly_total[quarter_name] = total_summarized[valid_months].sum(axis=1)

# Get zip codes that have non-zero values across all quarters
# Filter out rows that are all zeros
non_zero_data = quarterly_total[~(quarterly_total.drop('ZIP_CODE_int', axis=1) == 0).all(axis=1)]


######################
# Plotting quarterly values over time for totals
######################
# Get the mean permits per quarter across non-zero rows
quarterly_mean = non_zero_data.drop('ZIP_CODE_int', axis=1).mean()

# Create the plot
plt.figure(figsize=(15, 8))

# Plot individual zip code lines in gray
for _, row in non_zero_data.iterrows():
    plt.plot(row.drop('ZIP_CODE_int'), color='gray', alpha=0.1)

# Plot the mean line in black
plt.plot(quarterly_mean, color='black', linewidth=2, label='Average Permits')

plt.title('Quarterly Building Permits by ZIP Code (2000-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Permits')
plt.ylim(0, 1000)

# Modify x-axis labels to show only years
x_ticks = plt.gca().get_xticks()
x_labels = plt.gca().get_xticklabels()
new_labels = []
prev_year = None
for label in x_labels:
    year = label.get_text().split('-')[0]
    if year != prev_year:
        new_labels.append(year)
        prev_year = year
    else:
        new_labels.append('')
plt.xticks(x_ticks, new_labels, rotation=90)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'quarterly_permits_trend.png'))
plt.show()
# Save the plot
plt.close()

######################
# Creating blocked heatmap of permit growth by year for 2022-24
######################


progressive_sums_cumulative['SURVEY_DATE_tr'] = pd.to_datetime(progressive_sums_cumulative['SURVEY_DATE_tr'])

# Convert to quarterly periods
progressive_sums_cumulative['Quarter'] = progressive_sums_cumulative['SURVEY_DATE_tr'].dt.to_period('Q')

# Group by quarter and sum the permit columns
quarterly_sums = progressive_sums_cumulative.groupby('Quarter')[
    ['BLDGS_1_UNIT', 'BLDGS_2_UNITS', 'BLDGS_3_4_UNITS', 'BLDGS_5_UNITS']
].sum()

quarterly_sums.rename(columns={'BLDGS_1_UNIT': '1 Unit', 'BLDGS_2_UNITS': '2 Units', 'BLDGS_3_4_UNITS': '3-4 Units', 'BLDGS_5_UNITS': '5+ Units'}, inplace=True)

quarterly_sums
# Create the stacked bar chart
ax = quarterly_sums.plot(kind='bar', stacked=True, figsize=(10, 6))

# Customize x-axis: show the year only for the first quarter of each year, blank otherwise.
tick_labels = [str(q.year) if q.quarter == 1 else '' for q in quarterly_sums.index]
ax.set_xticks(range(len(quarterly_sums.index)))
ax.set_xticklabels(tick_labels, rotation=0)

ax.set_title("Quarterly Sums of Permits Issued by Building Type")
ax.set_xlabel("Quarter")
ax.set_ylabel("Number of Permits")
plt.tight_layout()
# plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'quarterly_permits_by_unit_type.png'))
# plt.show()

########################################
# Identifying key areas of growth for each unit type
########################################

shapefile_2010 = gpd.read_file(r"C:\Users\...\...\JHU\HomeEconomics\Shapefiles\Files 2020\tl_2020_us_zcta520.shp")

# shapefile_2010.columns  # Columns are 'ZCTA5CE20', 'GEOID20', 'CLASSFP20', 'MTFCC20', 'FUNCSTAT20', 'ALAND20','AWATER20', 'INTPTLAT20', 'INTPTLON20', 'geometry'


zip_code_shapefile.rename(columns={'PHYSICAL ZIP': 'ZIP_CODE_int'}, inplace=True)

progressive_sums_cumulative = progressive_sums_cumulative.merge(zip_code_shapefile, on='ZIP_CODE_int', how='inner')

progressive_sums_cumulative.columns


df = progressive_sums_cumulative.copy()

# Convert dates to pandas datetime
df['SURVEY_DATE_tr'] = pd.to_datetime(df['SURVEY_DATE_tr'])

# Define "last 6 months" cutoff from the latest date in your dataset
max_date = df['SURVEY_DATE_tr'].max()
cutoff_date = max_date - pd.DateOffset(months=6)

# Split into baseline vs. last 6 months
baseline = df[(df['SURVEY_DATE_tr'] >= '2000-01-01') & (df['SURVEY_DATE_tr'] < cutoff_date)]
last_6_months = df[df['SURVEY_DATE_tr'] >= cutoff_date]

# Columns of interest
cols = ['BLDGS_1_UNIT', 'BLDGS_2_UNITS', 'BLDGS_3_4_UNITS', 'BLDGS_5_UNITS']

# Group by state and sum building columns
baseline_state_sums = baseline.groupby('PHYSICAL STATE')[cols].sum()
last6_state_sums = last_6_months.groupby('PHYSICAL STATE')[cols].sum()

# Calculate total permits
baseline_state_sums['baseline_total'] = baseline_state_sums[cols].sum(axis=1)
last6_state_sums['last6_total'] = last6_state_sums[cols].sum(axis=1)

# Merge for growth calculation
merged = baseline_state_sums[['baseline_total']].merge(
    last6_state_sums[['last6_total']],
    left_index=True,
    right_index=True,
    how='outer'
).fillna(0)

# Calculate percentage growth
merged['pct_growth'] = (
    (merged['last6_total']) 
    / (merged['baseline_total'].replace({0: pd.NA}))  # avoid division by 0
    * 100
)
merged['pct_growth'] = merged['pct_growth'].fillna(0)  # fill if baseline was 0

merged = merged.sort_values('pct_growth', ascending=False)

# Identify top 5 states by growth
top5 = merged.sort_values('pct_growth', ascending=False).head(5)
print("\nTop 5 'Heating Up' States (by % growth in permits):")
print(top5)

# Plot bar chart of % growth for top 3
fig, ax = plt.subplots(figsize=(6,4))
merged['pct_growth'].plot(kind='bar', ax=ax)
ax.set_title('Top States by Permit Growth (Last 6 Months vs. Baseline)')
ax.set_ylabel('Growth (%)')
ax.set_xlabel('State')
plt.tight_layout()
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'top_states_by_permit_growth.png'))
plt.show()



################################
# Identifying the dominant segment for each state
################################


# Columns of interest
cols = ['BLDGS_1_UNIT', 'BLDGS_2_UNITS', 'BLDGS_3_4_UNITS', 'BLDGS_5_UNITS']

# Sum across each segment for baseline vs. last 6 months, grouped by state
base_sum = baseline.groupby('PHYSICAL STATE')[cols].sum().add_suffix('_base')
last_sum = last_6_months.groupby('PHYSICAL STATE')[cols].sum().add_suffix('_last6')

# Merge
merged = base_sum.join(last_sum, how='outer').fillna(0)
merged['STATE'] = merged.index

# Compute growth for each of the four segments
for c in cols:
    base_col = c + '_base'
    last_col = c + '_last6'
    growth_col = c + '_growth'
    merged[growth_col] = (
        (merged[last_col]) 
        / merged[base_col].replace(0, pd.NA)  # to avoid division by zero
        * 100
    )
    merged[growth_col] = merged[growth_col].fillna(0)

# Determine which segment had the highest growth per state
def find_dominant_segment(row):
    segs = {
        '1_unit': row['BLDGS_1_UNIT_growth'],
        '2_units': row['BLDGS_2_UNITS_growth'],
        '3_4_units': row['BLDGS_3_4_UNITS_growth'],
        '5_units': row['BLDGS_5_UNITS_growth']
    }
    return max(segs, key=segs.get)

merged['dominant_segment'] = merged.apply(find_dominant_segment, axis=1)

states_gdf = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2019/STATE/tl_2019_us_state.zip"
)

# Filter to the contiguous states + DC, if desired
contiguous_states = [
    'AL','AR','AZ','CA','CO','CT','DE','FL','GA','IA','ID','IL','IN','KS','KY',
    'LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM',
    'NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA',
    'WI','WV','WY','DC'
]
states_gdf = states_gdf[states_gdf['STUSPS'].isin(contiguous_states)]

# Merge in the "dominant_segment" info

joined_gdf = states_gdf.merge(
    merged[['STATE','dominant_segment']], 
    left_on='STUSPS', 
    right_on='STATE', 
    how='left'
)

seg_to_int = {
    '1_unit': 1,
    '2_units': 2,
    '3_4_units': 3,
    '5_units': 4
}
joined_gdf['segment_int'] = joined_gdf['dominant_segment'].map(seg_to_int)

# Build a discrete colormap with 4 colors in the specified order:
# 1_unit -> blue, 2_units -> green, 3_4_units -> lightyellow, 5_units -> brown
cmap = ListedColormap(["blue", "yellow", "red", "brown"]) 
# Create boundaries so that 0.5-1.5 maps to "blue", 1.5-2.5 -> "green", etc.
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)


fig, ax = plt.subplots(figsize=(15, 10))

# Plot the states, coloring by segment_int
joined_gdf.plot(
    column='segment_int',
    cmap=cmap,
    norm=norm,
    legend=False,  # We'll build a custom legend below
    missing_kwds={'color': 'lightgrey'},
    edgecolor='black',
    linewidth=0.5,
    ax=ax
)

# Define small states that we'll label with an arrow instead of direct text
# Adjust offsets as needed to get the arrows placed well
arrow_positions = {
    'DC': (4.0, 0.0), 
    'DE': (3, -1),
    'RI': (2, 0.5),
    'CT': (2, -0.5),
    'NJ': (3, -2),
    'MD': (3, -3)
    # Add others if necessary
}

for idx, row in joined_gdf.iterrows():
    centroid = row.geometry.centroid
    st = row['STATE']
    
    if st in arrow_positions:
        dx, dy = arrow_positions[st]
        ax.annotate(
            st, 
            xy=(centroid.x, centroid.y),
            xytext=(centroid.x + dx, centroid.y + dy),
            arrowprops=dict(arrowstyle='->', linewidth=0.5),
            ha='left',
            va='center',
            fontsize=8
        )
    else:
        ax.text(
            centroid.x, centroid.y, st,
            ha='center', va='center',
            fontsize=8
        )

ax.set_title("Dominant Permit Growth Segment by State (Last 6 Months vs Baseline)", pad=20)
ax.axis('off')

# Build a custom legend
legend_labels = [ 
    ("1-unit Growth Dominant", "blue"),
    ("2-units Growth Dominant", "yellow"),
    ("3-4 Units Growth Dominant", "red"),
    ("5+ Units Growth Dominant", "brown")
]
patches = [mpatches.Patch(color=color, label=label) for (label, color) in legend_labels]
ax.legend(handles=patches, loc='lower left', title="Dominant Segment")

plt.tight_layout()
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'dominant_segment_by_state.png'))
plt.show()


########################################
# Identifying the dominant segment for each zip code
########################################
# 1) Start from your main DataFrame

df = progressive_sums_cumulative.copy()

# 2) Convert date column if needed
df['SURVEY_DATE_tr'] = pd.to_datetime(df['SURVEY_DATE_tr'])

# 3) Identify cutoff for "last 6 months"
max_date = df['SURVEY_DATE_tr'].max()
cutoff_date = max_date - pd.DateOffset(months=6)

# 4) Split into baseline vs last 6 months
baseline = df[(df['SURVEY_DATE_tr'] >= '2000-01-01') & (df['SURVEY_DATE_tr'] < cutoff_date)]
last_6_months = df[df['SURVEY_DATE_tr'] >= cutoff_date]

# 5) Columns of interest
cols = ['BLDGS_1_UNIT', 'BLDGS_2_UNITS', 'BLDGS_3_4_UNITS', 'BLDGS_5_UNITS']

# 6) Sum each segment by ZIP_CODE_int for baseline & last 6 months
base_sum = baseline.groupby('ZIP_CODE_int', as_index=False)[cols].sum()
base_sum.rename(columns={c: c + '_base' for c in cols}, inplace=True)

last_sum = last_6_months.groupby('ZIP_CODE_int', as_index=False)[cols].sum()
last_sum.rename(columns={c: c + '_last6' for c in cols}, inplace=True)

# 7) Merge them into one DataFrame
merged = pd.merge(
    base_sum,
    last_sum,
    on='ZIP_CODE_int',
    how='outer'
).fillna(0)

# 8) Calculate % growth for each segment
for c in cols:
    base_col = c + '_base'
    last_col = c + '_last6'
    growth_col = c + '_growth'
    merged[growth_col] = (
        (merged[last_col]) /
        merged[base_col].replace(0, pd.NA)
        * 100
    )
    # Fill NaNs (where baseline was zero) with 0 or another placeholder
    merged[growth_col] = merged[growth_col].fillna(0)

# 9) Determine which segment had the highest growth per ZIP
def find_dominant_segment(row):
    segs = {
        '1_unit': row['BLDGS_1_UNIT_growth'],
        '2_units': row['BLDGS_2_UNITS_growth'],
        '3_4_units': row['BLDGS_3_4_UNITS_growth'],
        '5_units': row['BLDGS_5_UNITS_growth']
    }
    return max(segs, key=segs.get)

merged['dominant_segment'] = merged.apply(find_dominant_segment, axis=1)

# 10) Load your ZIP (ZCTA) shapefile. For example:
#     shapefile_2010 has 'ZCTA5CE20' as ZIP code field, rename to 'ZIP_CODE'
shapefile_2010 = shapefile_2010.rename(columns={'ZCTA5CE20': 'ZIP_CODE'}).copy()

# Ensure both sides match types. If your shapefile is int, do:
shapefile_2010['ZIP_CODE'] = shapefile_2010['ZIP_CODE'].astype(int)
merged['ZIP_CODE_int'] = merged['ZIP_CODE_int'].astype(int)

# 11) Merge shapefile with permit-growth data
zip_merged_gdf = shapefile_2010.merge(
    merged[['ZIP_CODE_int', 'dominant_segment']],
    left_on='ZIP_CODE',
    right_on='ZIP_CODE_int',
    how='left'
)

# 13) Map each dominant segment to a numeric code for coloring
seg_to_int = {
    '1_unit': 1,
    '2_units': 2,
    '3_4_units': 3,
    '5_units': 4
}
zip_merged_gdf['segment_int'] = zip_merged_gdf['dominant_segment'].map(seg_to_int)

# 14) Create a discrete colormap: blue for 1_unit, green for 2_units, lightyellow for 3-4 units, brown for 5_units
cmap = ListedColormap(["blue", "yellow", "red", "brown"])
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)

# 15) Plot the ZIP-level map focusing on the mainland USA with xlims/ylims
fig, ax = plt.subplots(figsize=(20, 12))

zip_merged_gdf.plot(
    column='segment_int',
    cmap=cmap,
    norm=norm,
    legend=False,          # Custom legend below
    missing_kwds={'color': 'lightgrey'},
    linewidth=0,
    ax=ax
)


# Set x and y limits for the mainland USA
ax.set_xlim([-125, -66])
ax.set_ylim([24, 50])

states_gdf = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2019/STATE/tl_2019_us_state.zip")
contiguous_states = [
    'AL','AR','AZ','CA','CO','CT','DE','FL','GA','IA','ID','IL','IN','KS','KY',
    'LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM',
    'NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA',
    'WI','WV','WY','DC'
]
states_gdf = states_gdf[states_gdf['STUSPS'].isin(contiguous_states)]
states_gdf.boundary.plot(ax=ax, color='black', linewidth=1)

ax.set_title("Dominant Permit Growth Segment by ZIP (Last 6 Months vs Baseline)", pad=20)
ax.axis('off')

# 16) Create a custom legend for the segments
legend_labels = [
    ("1-unit Growth Dominant", "blue"),
    ("2-units Growth Dominant", "yellow"),
    ("3-4 Units Growth Dominant", "red"),
    ("5+ Units Growth Dominant", "brown")
]
patches = [mpatches.Patch(color=color, label=label) for (label, color) in legend_labels]
ax.legend(handles=patches, loc='lower left', title="Dominant Segment")

plt.tight_layout()
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'dominant_segment_by_zip.png'))
plt.show()

########################################
# Breaking down the visualization by region
########################################


high_pop_states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']

# Subset the states GeoDataFrame to only these states.
states_high = states_gdf[states_gdf['STUSPS'].isin(high_pop_states)]

# Create subplots: 2 rows x 5 columns
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
axes = axes.flatten()

for i, state_abbr in enumerate(high_pop_states):
    ax = axes[i]
    # Subset the state's geometry
    state_geom = states_high[states_high['STUSPS'] == state_abbr]
    
    # Get bounding box for the state with a small buffer
    minx, miny, maxx, maxy = state_geom.total_bounds
    buffer_x = (maxx - minx) * 0.1
    buffer_y = (maxy - miny) * 0.1
    ax.set_xlim(minx - buffer_x, maxx + buffer_x)
    ax.set_ylim(miny - buffer_y, maxy + buffer_y)
    
    # Subset the ZIP-code GeoDataFrame to those that intersect the state's geometry
    zip_subset = zip_merged_gdf[zip_merged_gdf.intersects(state_geom.unary_union)]
    
    # Plot the ZIP codes for the current state
    zip_subset.plot(
        column='segment_int',
        cmap=cmap,
        norm=norm,
        legend=False,
        missing_kwds={'color': 'lightgrey'},
        linewidth=0,
        ax=ax
    )
    
    # Overlay the state border
    state_geom.boundary.plot(ax=ax, color='black', linewidth=1)
    
    ax.set_title(state_abbr)
    ax.axis('off')

# Build a common legend for the four dominant segments
legend_labels = [
    ("1-unit Growth Dominant", "blue"),
    ("2-units Growth Dominant", "yellow"),
    ("3-4 Units Growth Dominant", "red"),
    ("5+ Units Growth Dominant", "brown")
]
patches = [mpatches.Patch(color=color, label=label) for (label, color) in legend_labels]
fig.legend(handles=patches, loc='lower center', ncol=4, title="Dominant Permit Growth Segment")

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(os.path.join(dir, 'Raw Data', 'Permit Data Insights', 'dominant_segment_by_zip_by_region.png'))
plt.show()



