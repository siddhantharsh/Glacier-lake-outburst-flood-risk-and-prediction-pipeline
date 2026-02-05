# GLOF Risk Prediction Framework for Himalayan Glacial Lakes

This repository presents a machine‑learning pipeline designed to assess the likelihood of **Glacial Lake Outburst Floods (GLOFs)** across the Himalayan region. By integrating satellite remote sensing, digital elevation models, and glacier inventories, the framework predicts which glacial lakes pose the greatest hazard. The workflow spans from raw geospatial data ingestion to an interactive web map for risk visualization.

## Core goals

- **Detect and catalog Himalayan glacial lakes** using multi‑temporal Landsat imagery, SRTM DEMs, and the GLIMS glacier database.
- **Derive lake‑specific attributes** such as surface area, elevation, expansion trends, glacier adjacency, and topographic slope.
- **Address incomplete observations** with Multiple Imputation by Chained Equations (MICE) to fill gaps in the satellite record.
- **Build and evaluate predictive models** that classify lakes based on prior GLOF occurrences.
- **Deliver an interactive hazard viewer** enabling stakeholders to explore lake‑level risk estimates.

## Repository layout

| Directory/File | Purpose |
|----------------|---------|
| **`Notebooks/`** | Jupyter notebooks covering data ingestion, feature engineering, exploratory analysis, model training, and deployment. Highlights: `Final_LakeFeatures.ipynb` (geospatial processing), `ML_implementation.ipynb` (modeling), `Model_deployment.ipynb` (Streamlit app). |
| **`CSVs/`** | Processed datasets ready for machine learning (e.g., `df_final_with_laketypesorted_pre.csv`, `ml_pos_y_drop.csv`). |
| **`model.pkl`** | Serialized logistic regression model trained on engineered features. |
| **`requirements.txt`** | Python package dependencies (scikit‑learn, pandas, miceforest, folium, streamlit, etc.). |
| **`Midstates Research Symposium Poster.png`** | Visual summary of methodology and findings. |

## Input datasets

The pipeline ingests several geospatial products:

1. **Landsat 5/7/8/9 Surface Reflectance**: Used to compute the **Normalized Difference Water Index (NDWI)** for lake boundary extraction. Landsat 7’s SLC gap (≈2003–2009) introduces missing data.
2. **SRTM DEM (USGS/SRTMGL1_003)**: Supplies elevation for lakes and glaciers.
3. **GLIMS Glacier Inventory**: Glacier polygons for calculating contact, distance, elevation contrast, and slope.
4. **IHR Glacial Lake Atlas**: A catalog of ~5,000 Himalayan lakes, merged with GLOF records to generate positive (historical GLOF) and negative samples.
5. **Historical GLOF database**: Compiled from literature and the IHR atlas; positive labels correspond to lakes with documented outbursts.

## Geospatial workflow and feature generation

### Lake detection via Google Earth Engine

`Final_LakeFeatures.ipynb` contains Earth Engine (EE) functions that automatically detect lake polygons for given coordinates and years. These functions run entirely on EE servers, ensuring scalability across thousands of lakes.

- **`detect_lake_from_point(point, year, search_radius=3000, ndwi_thresh=0.3)`**: Buffers a point by `search_radius` (default 3 km), retrieves fall‑season Landsat imagery for `year ± 1`, applies cloud/snow masking, computes NDWI, thresholds water, and vectorizes the result. Returns the largest water polygon with area in hectares.
- **`detect_lake_pre_glof(point, year, search_radius=3000, ndwi_thresh=0.3)`**: Similar to above but composites imagery from `year-3` to `year-1` to capture the lake before a GLOF event.
- **`df_to_fc(df)`**: Converts a pandas DataFrame of lake points into an EE `FeatureCollection`.
- **`detect_area_nonglof(feature)` / `detect_area_glof(feature)`**: Wrappers that apply the above functions to each feature in a collection and attach `Lake_area_calculated_ha`.

### Expansion dynamics

Expansion rates are expressed as hectares per year and capture lake growth over time:

- **`calc_expansion_glof(feature)`** (5‑ and 10‑year variants): Detects lake area at `year` and at `year-5`/`year-10`, then computes `(area_t2 - area_t1) / years`.
- **`calc_expansion_nonglof(feature)`**: Same logic for non‑GLOF lakes.

### Glacier proximity metrics

Glacier adjacency is a critical risk factor. Server‑side functions extract these metrics:

- **`get_nearest_glacier(lake_geom, buffer_km=50)`**: Finds glaciers within a 50 km buffer. Returns contact status, nearest distance, lake and glacier elevations, slope, glacier area, and GLIMS IDs.
- **`_glacier_metrics_for_lake_polygon(lake, buffer_km=50)`**: Enhances robustness by counting touching glaciers, summing their areas, and computing slope via DEM.
- **`annotate_glacier_metrics_from_polygons(lakes_fc, buffer_km=50)`**: Applies the above to a `FeatureCollection` of lake polygons, appending glacier attributes.

These functions automate complex spatial analyses without manual GIS steps.

## Engineered feature set

After processing, datasets (e.g., `df_final_with_laketypesorted_pre.csv`) contain the following fields:

| Feature | Description |
|--------|-------------|
| `Longitude`, `Latitude` | Coordinates of the lake centroid. |
| `Year_final` | Reference year for each record (typically the most recent observation for negative lakes or the GLOF event year for positive lakes). |
| `Lake_area_calculated_ha` | Lake area computed from NDWI‑derived polygons. |
| `Elevation_m` / `lake_elev_m` | Elevation of the lake (from SRTM). |
| `glacier_area_ha` / `glacier_area_km2` | Combined area of touching glaciers (ha / km²). |
| `slope_glac_to_lake` | Rise/run slope between the lake and the nearest glacier. |
| `glacier_touch_count` | Number of glacier polygons intersecting the lake. |
| `nearest_glacier_dist_m` | Distance to the closest glacier if not touching (metres). |
| `glacier_elev_m` | Elevation of the nearest glacier (SRTM). |
| `5y_expansion_rate`, `10y_expansion_rate` | Expansion rate in hectares per year over 5 y and 10 y periods. |
| `Lake_type` / `Lake_type_simplified` | Categorical labels describing lake dam type (moraine, ice, other).  One‑hot encoding is applied using `pandas.get_dummies()`:contentReference[oaicite:16]{index=16}. |
| `is_supraglacial` | Boolean flag for lakes on glacier surfaces. |
| `glacier_contact` | Binary indicator: `True` if the lake touches a glacier, `False` otherwise. |
| `GLOF` | Target label: 1 if a historical GLOF occurred, 0 otherwise. |

## Missing data treatment

Satellite gaps and DEM limitations lead to missing values. The ML notebook (`ML_implementation.ipynb`) employs **MICE** (`miceforest`) to impute these gaps, preserving dataset integrity for modeling.

## Modeling pipeline

1. **Data preparation**: One‑hot encode categoricals, split imputed data (80/20), and use stratified 5‑fold CV.
2. **Baseline evaluation**: Train logistic regression, random forest, gradient boosting, and SVM. Logistic regression achieves the highest recall (~0.92) with modest precision (~0.50).
3. **Hyperparameter tuning**: Adjust class weights to prioritize recall. The tuned logistic regression reaches recall ≈ 0.92 and F1 ≈ 0.76.
4. **Model persistence**: Serialize the final model to `model.pkl` for use in the Streamlit app.

## Interactive risk visualization

`Model_deployment.ipynb` illustrates how to build an interactive hazard map:

- Load `model.pkl` and compute per‑lake probabilities.
- Render a `folium.Map` centered on the Himalayas. Circle markers are colored by risk (green → low, yellow → moderate, red → high).
- Wrap the map in a `streamlit` app (`app.py`) for user interaction and lake detail pop‑ups.

## Findings and constraints

- **Performance**: The tuned logistic regression balances recall and precision (recall ≈ 0.92, F1 ≈ 0.76) and is selected for deployment.
- **Key drivers**: Glacier contact count, distance to nearest glacier, glacier elevation, lake expansion rates, and slope are top predictors. Rapidly expanding lakes near or touching glaciers are flagged as higher risk.
- **Limitations**: Landsat 7 SLC gaps and NDWI misclassification can miss small lakes; historical GLOF records are incomplete; DEM resolution constrains slope estimates in steep terrain; annual snapshots may overlook short‑term changes.

## Quick start guide

1. **Install packages**: Create a virtual environment and run `pip install -r requirements.txt`.
2. **Reproduce features**: Run `Final_LakeFeatures.ipynb` in a Jupyter environment with Earth Engine access (`ee.Authenticate()`, `ee.Initialize()`). Adjust project IDs if needed.
3. **Train the model**: Execute `ML_implementation.ipynb` to impute data, encode features, train models, and save the final logistic regression.
4. **Launch the viewer**: Run `streamlit run app.py` (after generating `app.py` from `Model_deployment.ipynb`) to explore the interactive hazard map.

Made by Siddhant Harsh
