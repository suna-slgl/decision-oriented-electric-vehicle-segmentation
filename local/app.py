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
from sklearn.metrics import silhouette_score, silhouette_samples


# =========================
# Load dataset
# =========================
dataset = pd.read_csv(
    "/content/electric-vehicle-specifications-dataset-2025/electric_vehicles_spec_2025.csv.csv"
)


# =========================
# Define feature sets
# =========================
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
    "height_mm",
]

categorical_features = [
    "battery_type",
    "fast_charge_port",
    "drivetrain",
]

# Not: brand, model, segment, car_body_type, source_url gibi alanlar modele dahil edilmedi.


# =========================
# Coerce numeric columns (dirty strings -> NaN)
# =========================
for col in numeric_features:
    if col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")


# =========================
# Build X (features) and y (proxy target for supervised feature selection)
# =========================
X = dataset[numeric_features + categorical_features]
y = dataset["range_km"]  # ElasticNet feature selection için proxy target


# =========================
# Define preprocessing (impute + scale / impute + onehot)
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# =========================
# Define ElasticNet-based feature selection (supervised)
# =========================
elastic_net = ElasticNet(
    alpha=0.01,
    l1_ratio=0.5,
    random_state=42,
)

feature_selector = SelectFromModel(
    estimator=elastic_net,
    threshold="median",
)

# =========================
# Define PCA + KMeans
# =========================
pca = PCA(
    n_components=0.80,
    random_state=42,
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)

# =========================
# Build pipelines (training vs analysis)
# =========================
# Segmentasyon pipeline'ı: preprocess -> feature selection -> pca -> kmeans
segmentation_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("feature_selection", feature_selector),
    ("pca", pca),
    ("kmeans", kmeans),
])

# Analiz pipeline'ı: kmeans yok (PCA uzayı üretmek için)
analysis_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("feature_selection", feature_selector),
    ("pca", pca),
])

# =========================
# Fit segmentation pipeline and assign cluster labels to dataset
# =========================
segmentation_pipeline.fit(X, y)
dataset["ev_segments"] = segmentation_pipeline.predict(X)

segment_counts = dataset["ev_segments"].value_counts()
segment_percent = (dataset["ev_segments"].value_counts(normalize=True) * 100).round(2)

print("Segment counts:\n", segment_counts)
print("\nSegment percentages (%):\n", segment_percent)


# =========================
# Compute segment profiles (interpretable summary statistics)
# =========================
profile_cols = [
    "range_km",
    "battery_capacity_kWh",
    "top_speed_kmh",
    "torque_nm",
    "efficiency_wh_per_km",
    "acceleration_0_100_s",
    "length_mm",
]

segment_profiles = dataset.groupby("ev_segments")[profile_cols].mean().round(2)
print("\nSegment profiles (mean):\n", segment_profiles)


# =========================
# Human-readable segment names (business labels)
# =========================
segment_names = {
    0: "Premium / Yüksek Performanslı Elektrikli Araçlar",
    1: "Ana Akım / Dengeli Elektrikli Araçlar",
    2: "Fayda Odaklı / Düşük Performanslı Elektrikli Araçlar",
    3: "Şehir İçi / Ekonomik Elektrikli Araçlar",
}


# =========================
# Silhouette analysis (segment reliability matrix)
# =========================
# Not: Burada aynı dönüşüm zinciriyle PCA uzayı üretilir.
X_pca = analysis_pipeline.fit_transform(X, y)

# Etiketleri segmentation pipeline içindeki KMeans'ten alıyoruz (gerçek segment etiketleri)
labels = segmentation_pipeline.named_steps["kmeans"].labels_

silhouette_values = silhouette_samples(X_pca, labels)

silhouette_df = pd.DataFrame({
    "ev_segment": labels,
    "silhouette_skoru": silhouette_values,
})

silhouette_matrix = silhouette_df.groupby("ev_segment").agg(
    ortalama_silhouette=("silhouette_skoru", "mean"),
    minimum_silhouette=("silhouette_skoru", "min"),
    maksimum_silhouette=("silhouette_skoru", "max"),
    gozlem_sayisi=("silhouette_skoru", "count"),
).round(3)

print("\nSilhouette matrix:\n", silhouette_matrix)