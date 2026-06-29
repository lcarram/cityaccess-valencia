import folium
import plotly.express as px


def service_map(services, boundaries=None):
    m = folium.Map(location=[39.47, -0.376], zoom_start=12, tiles="cartodbpositron")
    if boundaries is not None:
        folium.GeoJson(boundaries.to_json(), name="Neighbourhoods", style_function=lambda _: {"fillOpacity": 0.05, "weight": 1}).add_to(m)
    colors = {
        "healthcare": "red",
        "education": "blue",
        "libraries": "purple",
        "sports facilities": "green",
        "social services": "orange",
        "culture": "darkred",
    }
    for _, row in services.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=colors.get(row["category"], "gray"),
            fill=True,
            popup=f"{row['service_name']} ({row['category']})",
        ).add_to(m)
    return m


def choropleth(boundaries, data, column, legend_name):
    gdf = boundaries.merge(data[["neighbourhood", column]], on="neighbourhood", how="left")
    m = folium.Map(location=[39.47, -0.376], zoom_start=12, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=["neighbourhood", column],
        key_on="feature.properties.neighbourhood",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=legend_name,
    ).add_to(m)
    folium.GeoJson(gdf.to_json(), tooltip=folium.GeoJsonTooltip(fields=["neighbourhood", column])).add_to(m)
    return m


def radar(df, neighbourhoods, columns):
    long = df[df["neighbourhood"].isin(neighbourhoods)].melt("neighbourhood", value_vars=columns)
    return px.line_polar(long, r="value", theta="variable", color="neighbourhood", line_close=True)
