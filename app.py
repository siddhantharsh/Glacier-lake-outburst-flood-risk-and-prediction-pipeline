# ===================================================
# app.py — Hosted-style Interactive GLOF Hazard Map
# ===================================================
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
import pandas as pd

# ---------------------------------------------------
# 1. Page setup
# ---------------------------------------------------
st.set_page_config(page_title="Himalayan GLOF Hazard Map", layout="wide")
st.title("Predicting Glacial Lake Outburst Flood (GLOF) Risks Across the Himalayas")
st.markdown("""This interactive map visualizes over 2,000 glacial lakes across the Himalayas with machine-learning-based hazard probability estimates.
        Lake features were derived from remote sensing data and global glacial lake inventories, helping identify lakes with a high potential for future
        outburst events.""")

# ---------------------------------------------------
# 2. Load dataset (cached for speed)
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hazard_probabilities.csv")
    except FileNotFoundError:
        return pd.read_csv("CSVs/hazard_probabilities.csv")

hazard_df = load_data()

# ---------------------------------------------------
# 3. Sidebar map selector
# ---------------------------------------------------
st.sidebar.header("Map Settings")

base_maps = {
    "CartoDB Positron": "CartoDB.Positron",
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Dark Matter": "CartoDB.DarkMatter",
    "Esri Satellite": "Esri.WorldImagery",
    "Esri NatGeo": "Esri.NatGeoWorldMap",
    "OpenTopoMap": "OpenTopoMap"
}

selected_map = st.sidebar.selectbox(
    "Choose Base Map Style:",
    options=list(base_maps.keys()),
    index=0
)

zoom_level = st.sidebar.slider("Zoom Level", 5, 12, 7)

# ---------------------------------------------------
# 4. Create map (once only)
# ---------------------------------------------------
m = folium.Map(
    location=[28.2, 87.0],
    zoom_start=zoom_level,
    tiles=base_maps[selected_map],
    attr="© OpenStreetMap contributors"
)

# Continuous color scale
colormap = LinearColormap(
    ['green', 'yellow', 'orange', 'red', 'darkred'],
    vmin=0, vmax=1,
    caption='Hazard Probability'
).add_to(m)

# Marker clustering for speed
marker_cluster = MarkerCluster().add_to(m)

# ---------------------------------------------------
# 5. Add all lakes as clustered markers
# ---------------------------------------------------
area_col = "Lake_area_calculated_ha"

for _, row in hazard_df.iterrows():
    prob = row["Hazard_Prob"]
    area = row.get(area_col, 0.5)
    color = colormap(prob)
    radius = max(3, min(area / 30, 10))

    popup_html = f"""
    <b>Lake Information</b><br>
    <b>Latitude:</b> {row.get('Latitude', float('nan')):.4f}<br>
    <b>Longitude:</b> {row.get('Longitude', float('nan')):.4f}<br>
    <b>Lake Area (ha):</b> {row.get('Lake_area_calculated_ha', 0):.2f}<br>
    <b>Elevation (m):</b> {row.get('Elevation_m', 0):.0f}<br>
    <b>Lake Type:</b> {row.get('Lake_type_simplified', 'N/A')}<br>
    <b>Glacier Area (ha):</b> {row.get('glacier_area_ha', 0):.2f}<br>
    <b>Nearest Glacier Dist (m):</b> {row.get('nearest_glacier_dist_m', 0):.0f}<br>
    <b>5-yr Expansion Rate (ha/yr):</b> {row.get('5y_expansion_rate', 0):.3f}<br>
    <b>10-yr Expansion Rate (ha/yr):</b> {row.get('10y_expansion_rate', 0):.3f}<br>
    <b>Observed GLOF:</b> {row.get('GLOF', 'N/A')}<br>
    <b><font color='{color}'>Hazard Probability:</font></b> {prob:.2f}
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=radius,
        fill=True,
        fill_color=color,
        color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=350)
    ).add_to(marker_cluster)

# ---------------------------------------------------
# 6. save to static HTML
# ---------------------------------------------------
m.save("hazard_map.html")

# ---------------------------------------------------
# 7. Render pre-saved HTML (no lag)
# ---------------------------------------------------
with open("hazard_map.html", "r", encoding="utf-8") as f:
    map_html = f.read()

st.components.v1.html(map_html, height=750, width=1300)