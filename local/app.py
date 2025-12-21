import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

dataset = pd.read_csv("local/electric_vehicles_spec_2025.csv.csv")

#-----------------------------------------------#
### Dataset Description and Feature Selection ###
#-----------------------------------------------#
numeric_features = [
    "top_speed_kmh",
    "battery_capacity_kWh",
    "number_of_cells",
    "torque_nm",
    "efficiency_wh_per_km",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "length_mm",
    "width_mm",
    "height_mm"
]

categorical_features = [
    "battery_type",
    "fast_charge_port",
    "drivetrain"
]

# brand, model, segment, car_body_type, source_url feature'ları modele dahil edilemez.


for col in numeric_features:
    if col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce") # errors="coerce" ile kirli veriyi NaN’a çevirme

X = dataset[numeric_features + categorical_features]
y = dataset["range_km"]

#-----------------------------#
### Preprocessing Pipelines ###
#-----------------------------#

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ELASTICNET + FEATURE SELECTION

elastic_net = ElasticNet(
    alpha=0.01,
    l1_ratio=0.5,
    random_state=42
)

feature_selector = SelectFromModel(
    estimator=elastic_net,
    threshold="median"
)

# aşağıdakiler düzeltilmeli
# range_km’yi proxy target olarak kullanma
# Menzili açıklamayan feature’lar segmentasyonda da zayıftır varsayımı

#------------------#
### PCA + KMEANS ###         
#------------------#

pca = PCA(
    n_components=0.80,
    random_state=42
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Pipelines
segmentation_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("feature_selection", feature_selector),
    ("pca", pca),
    ("kmeans", kmeans)
])

analysis_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("feature_selection", feature_selector),
    ("pca", pca)
])