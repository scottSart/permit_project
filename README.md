# Permit Data Analysis

## Overview

This project processes and analyzes permit data to extract insights about trends in residential construction permits. The pipeline consists of two main scripts: `loading.py` and `Understanding Data.py`, which handle data processing, cleaning, transformation, visualization, and analysis.

---

## 1. loading.py

### What I Do:
- **Library & Directory Setup:**
  - Import libraries (`pandas`, `numpy`, `matplotlib`, `geopandas`, etc.).
  - Define working directories and create necessary folders:
    - `Raw Data/Permit Data Insights`
    - `Raw Data/Semi_Datasets`
    - `Raw Data/Processed_Segmented_Data`
  - Use `os.makedirs(exist_ok=True)` to ensure folders exist.

- **Data Loading & Cleaning:**
  - Load a large permits CSV file in chunks alongside Zillow rent data.
  - Inspect dataset integrity (e.g., missing ZIP codes, location types).
  - Filter records to include only `type="Place"` from the year 2000 onward.

- **Date & ZIP Code Formatting:**
  - Convert date fields to `YYYY-MM` format.
  - Ensure ZIP codes are consistent 5-digit strings.

- **Data Grouping & Transformation:**
  - Aggregate permit data by ZIP code and date.
  - Pivot data into separate tables by building type (1-unit, 2-units, 3–4 units, 5+ units).
  - Compute cumulative sums and monthly growth percentages.

- **Data Export:**
  - Save processed data in structured folders:
    - `Raw Data/Processed_Segmented_Data/` (cumulative sums, growth calculations).
    - `Raw Data/Semi_Datasets/` (pivot tables and intermediate summaries).

---

## 2. Understanding Data.py

### What I Do:
- **Data Loading:**
  - Load processed CSV files from `loading.py` (cumulative sums, monthly growth, summarized datasets).

- **ZIP Code Consistency:**
  - Ensure all ZIP codes are consistently formatted as 5-digit strings.

- **Quarterly Aggregation & Trend Analysis:**
  - Aggregate permit data by quarter (2000–2024).
  - Generate visualizations:
    - Individual ZIP code trends (plotted in gray).
    - Average trend line (plotted in black).

- **Stacked Bar Charts:**
  - Show quarterly permit sums by building type.

- **Growth Analysis:**
  - Compare a baseline period to the last six months.
  - Compute percentage growth in permits per state and building type.
  - Identify:
    - **Top states** with highest growth.
    - **Dominant building type** with highest growth per state and ZIP code.

- **Mapping Visualizations:**
  - Merge growth data with shapefiles (state and ZIP code level).
  - Generate:
    - **State-level maps** (highlighting dominant permit growth segments).
    - **ZIP-level maps** (with subplots for high-population states).

- **Plot Export:**
  - Save all visualizations in `Raw Data/Permit Data Insights/` using `plt.savefig()`.

---

## 3. File Storage Methods

### Directory Structure:
- **Base Directory**: Uses the current working directory.
- **Subdirectories**:
  - `Raw Data/Permit Data Insights/` → Contains visualizations.
  - `Raw Data/Semi_Datasets/` → Stores intermediate pivoted/summarized CSVs.
  - `Raw Data/Processed_Segmented_Data/` → Holds cleaned data, cumulative sums, and growth calculations.

### Saving Data:
- **CSV Exports**: Use `DataFrame.to_csv()` with or without index as needed.
- **Processed Files**: Store cumulative sums and monthly growth percentages in designated folders.

### Plot Exports:
- **All plots** saved in `Raw Data/Permit Data Insights/` with clear filenames:
  - `quarterly_permits_trend.png`
  - `dominant_segment_by_state.png`
  - Other analysis-specific visualizations.

---

### Usage
1. Run `loading.py` to process and clean permit data.
2. Run `Understanding Data.py` to perform analysis and generate visualizations.
