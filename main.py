import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import folium
from geopy.geocoders import Nominatim
import time


df = pd.read_csv("Air_Quality.csv")
df["Air_quality_status"] = df["Data Value"].apply(lambda x: "Good" if x <= 20 else "Bad")
columns_to_drop = ['Unique ID', 'Indicator ID', 'Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Join ID', 'Message', 'Time Period']
df = df.drop(columns=columns_to_drop)
df["Start_Date"] = pd.to_datetime(df['Start_Date'])
df["year"] = df["Start_Date"].dt.year
df["month"] = df["Start_Date"].dt.month

def Season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["month"].apply(Season)


label_encoder = LabelEncoder()
df["Geo Place Name_Label"] = label_encoder.fit_transform(df["Geo Place Name"])
df["Season_Label"] = label_encoder.fit_transform(df["Season"])
df["Target"] = df["Air_quality_status"].map({"Good": 0, "Bad": 1})


features = ["Data Value", "Geo Place Name_Label", "year", "month", "Season_Label"]
X = df[features]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


df['Prediction'] = model.predict(X)
df['Prediction_Label'] = df['Prediction'].map({0: "Good", 1: "Bad"})


avg_prediction = df.groupby("Geo Place Name")["Prediction"].mean().reset_index()
avg_prediction["Predicted_Label"] = avg_prediction["Prediction"].apply(lambda x: "Good" if x < 0.5 else "Bad")


geolocator = Nominatim(user_agent="air_quality_mapping")
unique_places = avg_prediction["Geo Place Name"].dropna().unique()

location_dict = {"Place": [], "Latitude": [], "Longitude": [], "Prediction": []}

for place in unique_places:
    try:
        location = geolocator.geocode(f"{place}, New York City, NY")
        if location:
            pred_label = avg_prediction.loc[avg_prediction["Geo Place Name"] == place, "Predicted_Label"].values[0]
            location_dict["Place"].append(place)
            location_dict["Latitude"].append(location.latitude)
            location_dict["Longitude"].append(location.longitude)
            location_dict["Prediction"].append(pred_label)
        else:
            continue
    except:
        continue
    time.sleep(1)  # Respect API rate limits

location_df = pd.DataFrame(location_dict)

# Map with color
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

for _, row in location_df.iterrows():
    color = "green" if row["Prediction"] == "Good" else "red"
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        popup=f"{row['Place']}: {row['Prediction']}",
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6
    ).add_to(nyc_map)

nyc_map.save("NYC_Air_Quality_Predicted_Colored_Map.html")
print(" Map saved with color-coded predictions!")
