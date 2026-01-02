# Decision-Oriented Electric Vehicle Segmentation with ElasticNet, PCA, and KMeans
A technically interpretable electric vehicle segmentation pipeline combining ElasticNet-based feature selection, PCA dimensionality reduction, and KMeans clustering, with segment quality evaluated via silhouette analysis.

## Proje Özeti

**Decision-Oriented Electric Vehicle Segmentation**, elektrikli araçların (EV) teknik ve fonksiyonel özelliklerine göre veri odaklı, karar verme süreçlerine yardımcı olacak şekilde otomatik segmentasyonunu amaçlayan bir analiz ve modelleme çalışmasıdır. Proje, çok boyutlu makine öğrenimi teknikleriyle gelişmiş segmentasyon mimarisi kurarak, sektörel pazar analizinden ürün konumlandırmasına ve stratejik karar destek sistemlerine kadar geniş bir uygulama yelpazesi hedefler.

---

## Mevcut Mimari ve Yöntemler

Sistem, bütünleşik ve tekrarlanabilir bir makine öğrenimi boru hattı (ML pipeline) üzerinde aşağıdaki adımlardan oluşur:

### 1. Veri Yükleme ve İlk Temizlik

- Elektrikli araçlara ait veri seti pandas kullanılarak içe aktarılır.
- Çalışmada **kullanılacak nitelikler** teknik (ör. batarya kapasitesi, hız, tork, verimlilik, akselerasyon) ve kategorik (ör. batarya tipi, güç aktarım sistemi) olarak ayrılır.
- Sayısal biçime uygunluk ve eksik değerlerin coerce edilmesi sağlanır.

### 2. Özellik İşleme (Feature Engineering)

- **Sayısal Özellikler**: Eksik veri için medyan ile tamamlama ve standart skor normalizasyonu (`StandardScaler`).
- **Kategorik Özellikler**: Mod ile doldurma ve one-hot encoding (`OneHotEncoder` ile, sparse_output=False).
- Bütün ön işlemler **ColumnTransformer** ile entegre biçimde yürütülür.

### 3. Özellik Seçimi (Feature Selection)

- **ElasticNet** regresyon modeliyle hedef değişken (range_km) üzerinden anlamlı özniteliklerin seçimi.
- **SelectFromModel** kullanılarak yüksek katkı sağlayan özniteliklerin otomatik seçimi.

### 4. Boyut İndirgeme (Dimensionality Reduction)

- **Principal Component Analysis (PCA)** ile %95 varyans korunacak şekilde özellik uzayının boyutu indirgenir (isteğe bağlı).

### 5. Segmentasyon (Clustering)

- **K-Means** algoritmasıyla (k=4) EV veri örnekleri teknik ve performans parametrelerine göre segmentlere ayrılır.
- Her segmentin istatistiksel profili çıkarılır ve segmentler; Premium, Ana Akım, Fayda Odaklı ve Şehir İçi/Ekonomik olarak tanımlanır.

### 6. Değerlendirme ve Validasyon

- **Silhouette Score** ve ilgili istatistikler ile segmentasyonun yapısal tutarlılığı ölçülür.
- Özellikler arası **korelasyon matrisi** hesaplanıp üst üçgende yüksek kolerasyon çiftleri özetlenir.
- Farklı pipeline varyantları ile PCA’lı ve PCA’sız segmentasyon analizi karşılaştırılır.

---

## Kullanılan Araçlar ve Teknolojiler

- **Python 3.x** (temel çalışma ortamı)
- **pandas, numpy** (veri işlemleri ve matematiksel işlemler)
- **scikit-learn** (makine öğrenimi altyapısı)
    - `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ElasticNet`, `SelectFromModel`, `PCA`, `KMeans`, `silhouette_score`, `silhouette_samples`
- **matplotlib** (analitik görselleştirme)
- _(Opsiyonel)_ **Jupyter Notebook** (etkileşimli analiz ve raporlama)

---

## Uygulama Akışı

1. `electric_vehicles_spec_2025.csv.csv` dosyası pandas ile yüklenir.
2. Teknik ve kategorik öznitelikler için ön işleme pipeline’ı hazırlanır.
3. ElasticNet tabanlı otomatik özellik seçimi ile en anlamlı girdiler belirlenir.
4. PCA ile boyut indirgeme uygulanır (ya da doğrudan özelliklerle devam edilir).
5. Segmentasyon pipeline’ı ile dört ana EV segmenti çıkarılır.
6. Silhouette analizleri ve segment profilleri yorumlanıp raporlanır.

---

## Akademik/Endüstriyel Katkı ve Kullanım Alanları

- Elektrikli otomotiv pazarında ürün stratejisi ve pazar segmentasyonu çalışmaları
- Çoklu kriterli karar verme (MCDM) ve optimizasyon yaklaşımlarına teknik veri tabanı sağlama
- Veri bilimi ile pazarlama, Ar-Ge ve mühendislik ekipleri arasında köprü kurma
- Alternatif segmentasyon mimarilerinin validasyonu ve karşılaştırılması

---

## Kurulum

```bash
git clone https://github.com/suna-slgl/decision-oriented-electric-vehicle-segmentation.git
cd decision-oriented-electric-vehicle-segmentation
pip install -r requirements.txt
```

## Kullanım

```bash
python app.py
```
Sonuçlar rapor ve görselleştirme biçiminde çıktı olarak oluşturulacaktır.

---

## Lisans

MIT Lisanslıdır.

---

Daha fazla teknik detaya, algoritmik akışa ve örnek çıktılara ulaşmak için `app.py` dosyasını inceleyiniz.
