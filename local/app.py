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

# Load dataset
dataset = pd.read_csv("electric_vehicles_spec_2025.csv.csv")

# Define feature sets
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

# brand, model, segment, car_body_type, source_url gibi alanlar modele dahil edilmedi.

for col in numeric_features:
    if col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")


X = dataset[numeric_features + categorical_features]
y = dataset["range_km"]  # ElasticNet feature selection için proxy target

# ====================================================================================================

def build_preprocessor(numeric_features, categorical_features):
    # PREPROCESSING PIPELINES
    
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

    return preprocessor

def build_feature_selector():
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

    return feature_selector

def compute_correlation(X, y, preprocessor, feature_selector, sample_size=None, random_state=42):
# Modele giren feature'lar üzerinde Korelasyon ölçümü

    # a) Preprocessing (ölçekleme + kategorik dönüşüm)
    X_preprocessed = preprocessor.fit_transform(X)

    # b) Feature selection
    feature_selector.fit(X_preprocessed, y)
    X_selected = feature_selector.transform(X_preprocessed)

    # c) Korelasyon hesaplaması öncesi boyut kontrolü (çok fazla feature varsa örnekleme)
    if sample_size is not None and X_selected.shape[1] > sample_size:
        rng = np.random.RandomState(random_state)
        selected_columns = rng.choice(
            X_selected.shape[1],
            size=sample_size,
            replace=False
        )
        X_selected = X_selected[:, selected_columns]

    # d) Pearson Korelasyon Matrisi
    corr_matrix = pd.DataFrame(X_selected).corr()

    print("ÖZELLİKLER ARASI KORELASYON MATRİSİ:")
    display(corr_matrix.round(3))


    # e) Korelasyon Özet İstatistikleri (üst üçgen → her feature çifti bir kez sayılır)
    abs_corr = corr_matrix.abs()
    upper_triangle = abs_corr.where(
        np.triu(np.ones(abs_corr.shape), k=1).astype(bool)
    )

    max_abs_correlation = upper_triangle.max().max()
    num_pairs_gt_070 = (upper_triangle > 0.70).sum().sum()
    num_pairs_gt_090 = (upper_triangle > 0.90).sum().sum()

    correlation_summary = {
        "En yüksek mutlak korelasyon (|r|)": round(float(max_abs_correlation), 3)
        if pd.notna(max_abs_correlation) else None,
        "Güçlü korelasyonlu feature çifti sayısı (|r| > 0.70)": int(num_pairs_gt_070),
        "Çok güçlü korelasyonlu feature çifti sayısı (|r| > 0.90)": int(num_pairs_gt_090),
        "Korelasyon hesabına giren feature sayısı": int(corr_matrix.shape[0]),
    }

    # Özet istatistikleri tablo haline getir
    summary_df = pd.DataFrame(
        list(correlation_summary.items()),
        columns=["Ölçüt", "Değer"]
    )


    print("\nKORELASYON ÖZET İSTATİSTİKLERİ:")
    display(
    summary_df
        .style
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
    )
    return corr_matrix, correlation_summary



def build_pca():
    # PCA

    pca = PCA(
        n_components=0.95, # 0.50, 0.75 ve 0.95 denendi
        random_state=42
    )
    return pca

def build_kmeans():
    # HÜCRE 6.2: KMEANS

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    return kmeans

def build_pipelines(preprocessor, feature_selector, pca, kmeans):
    # Pipelines
    # Bu fonksiyon, daha önce ayrı ayrı tanımladığın tüm bileşenleri (preprocessing, feature selection, PCA, KMeans) alıp, scikit-learn’in çalıştırabileceği iki adet pipeline nesnesi hâline getirir.

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

    return segmentation_pipeline, analysis_pipeline

def fit_predict_segments(dataset, X, y, segmentation_pipeline):
    # FIT & PREDICT

    segmentation_pipeline.fit(X, y)

    dataset["ev_segments"] = segmentation_pipeline.predict(X)

    dataset["ev_segments"].value_counts()

    return dataset

def compute_segment_profiles(dataset):
    # K-Means ile oluşturulan 4 adet segmentin özelliklerine göre etiketlendirilmesi
    
    profile_cols = [
        "range_km",             # kullanım kapasitesi
        "battery_capacity_kWh", # enerji büyüklüğü
        "top_speed_kmh",        # performans
        "torque_nm",            # performans
        "efficiency_wh_per_km", # ekonomik/şehir içi uygunluk
        "acceleration_0_100_s", # sportiflik (küçük = hızlı)
        "length_mm"             # araç sınıfı (küçük ↔ büyük)
    ]

    return dataset.groupby("ev_segments")[profile_cols].mean().round(2)

segment_names = {
    0: "Premium / Yüksek Performanslı Elektrikli Araçlar",     # Büyük, güçlü, uzun menzilli, pahalı olması çok muhtemel.
    1: "Ana Akım / Dengeli Elektrikli Araçlar",                # Dengeli, ana akım, aile kullanımı.
    2: "Fayda Odaklı / Düşük Performanslı Elektrikli Araçlar", # Performansı düşük, verimlilik odaklı değil, muhtemelen eski veya ağır platform.
    3: "Şehir İçi / Ekonomik Elektrikli Araçlar"               # Küçük, verimli, şehir içi odaklı.
}

def compute_silhouette_matrix(X, y, analysis_pipeline, segmentation_pipeline):
    # SILHOUETTE SKORU HESAPLAMA

    # PCA uzayını, KMeans öncesi pipeline'dan al
    X_pca = analysis_pipeline.fit_transform(X, y)

    # Pipeline içindeki KMeans etiketlerini kullan (YENİ KMeans YOK)
    labels = segmentation_pipeline.named_steps["kmeans"].labels_

    # Her gözlem için silhouette skoru
    silhouette_values = silhouette_samples(X_pca, labels)

    # DataFrame oluştur
    silhouette_df = pd.DataFrame({
        "ev_segment": labels,
        "silhouette_skoru": silhouette_values
    })

    # Segment bazlı sonuç matrisi
    silhouette_matrix = silhouette_df.groupby("ev_segment").agg(
        ortalama_silhouette=("silhouette_skoru", "mean"),
        minimum_silhouette=("silhouette_skoru", "min"),
        maksimum_silhouette=("silhouette_skoru", "max"),
        gozlem_sayisi=("silhouette_skoru", "count")
    ).round(3)

    return silhouette_matrix

# ====================================================================================================

# DEBUG (Not: Colab'de farklı bir hücreye yaz)

preprocessor = build_preprocessor(numeric_features, categorical_features)
feature_selector = build_feature_selector()
pca = build_pca()
kmeans = build_kmeans()

segmentation_pipeline, analysis_pipeline = build_pipelines(
    preprocessor, feature_selector, pca, kmeans
)

correlation_matrix, correlation_summary = compute_correlation(
    X=X,
    y=y,
    preprocessor=preprocessor,
    feature_selector=feature_selector,
    sample_size=None,
    random_state=42
)

dataset = fit_predict_segments(dataset, X, y, segmentation_pipeline)

silhouette_matrix = compute_silhouette_matrix(X, y, analysis_pipeline, segmentation_pipeline)
silhouette_matrix

# ====================================================================================================

# PCA olmadan ElasticNet sonuçları ile K-Means uygulaması

# 1) PCA'sız pipeline tanımları
def build_pipelines_no_pca(preprocessor, feature_selector, kmeans):
    segmentation_pipeline_no_pca = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("feature_selection", feature_selector),
        ("kmeans", kmeans)
    ])

    analysis_pipeline_no_pca = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("feature_selection", feature_selector)
    ])

    return segmentation_pipeline_no_pca, analysis_pipeline_no_pca


# 2) Fit & Predict (PCA'sız)
def fit_predict_segments_no_pca(dataset, X, y, segmentation_pipeline_no_pca):
    segmentation_pipeline_no_pca.fit(X, y)
    dataset["ev_segments_no_pca"] = segmentation_pipeline_no_pca.predict(X)
    return dataset


# 3) Silhouette matrix (PCA'sız uzayda)
def compute_silhouette_matrix_no_pca(X, y, analysis_pipeline_no_pca, segmentation_pipeline_no_pca):
    # PCA yok: feature_selection sonrası uzay
    X_fs = analysis_pipeline_no_pca.fit_transform(X, y)

    # KMeans etiketleri (yeni kmeans yok)
    labels = segmentation_pipeline_no_pca.named_steps["kmeans"].labels_

    # Silhouette skorları
    silhouette_values = silhouette_samples(X_fs, labels)

    silhouette_df = pd.DataFrame({
        "ev_segment": labels,
        "silhouette_skoru": silhouette_values
    })

    silhouette_matrix = silhouette_df.groupby("ev_segment").agg(
        ortalama_silhouette=("silhouette_skoru", "mean"),
        minimum_silhouette=("silhouette_skoru", "min"),
        maksimum_silhouette=("silhouette_skoru", "max"),
        gozlem_sayisi=("silhouette_skoru", "count")
    ).round(3)

    return silhouette_matrix


preprocessor = build_preprocessor(numeric_features, categorical_features)
feature_selector = build_feature_selector()
kmeans = build_kmeans()

segmentation_pipeline_no_pca, analysis_pipeline_no_pca = build_pipelines_no_pca(
    preprocessor, feature_selector, kmeans
)

dataset = fit_predict_segments_no_pca(dataset, X, y, segmentation_pipeline_no_pca)

sil_matrix_no_pca = compute_silhouette_matrix_no_pca(
    X, y, analysis_pipeline_no_pca, segmentation_pipeline_no_pca
)

sil_matrix_no_pca
