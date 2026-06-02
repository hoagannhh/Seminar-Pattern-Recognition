# 🌆 Wikicities Population — Dự đoán dân số thành phố từ Knowledge Graph

> **Seminar môn Nhận dạng mẫu (Pattern Recognition)**  
> Trường Đại học — Năm 3, Học kỳ 3

Dự án này xây dựng pipeline Machine Learning để **dự đoán dân số các thành phố thế giới** dựa trên dữ liệu trích xuất từ **DBpedia Knowledge Graph** thông qua SPARQL. Bài toán được đặt dạng **hồi quy (regression)** trên không gian log-scale.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cách lấy dữ liệu](#-cách-lấy-dữ-liệu)
- [Hướng dẫn chạy](#-hướng-dẫn-chạy)
  - [Chạy trên máy local](#chạy-trên-máy-local)
  - [Chạy trên Google Colab](#chạy-trên-google-colab)
- [Mô tả các file](#-mô-tả-các-file)
- [Pipeline tổng thể](#-pipeline-tổng-thể)
- [Features được sử dụng](#-features-được-sử-dụng)
- [Models](#-models)
- [Kết quả](#-kết-quả)
- [Yêu cầu môi trường](#-yêu-cầu-môi-trường)

---

## 🎯 Tổng quan

| Mục | Chi tiết |
|-----|----------|
| **Bài toán** | Hồi quy — Dự đoán `log(population)` của thành phố |
| **Nguồn dữ liệu** | [DBpedia](https://dbpedia.org) qua SPARQL endpoint |
| **Số lượng mẫu** | ~10.000 thành phố toàn cầu |
| **Target** | `log_population = log1p(population)` |
| **Metric chính** | RMSE, MAE, R² (trên log scale) |
| **Models** | Linear Regression, Ridge, Random Forest, XGBoost, LightGBM |

---

## 📁 Cấu trúc thư mục

```
Seminar-Pattern-Recognition/
│
├── 📓 notebooks/                   # Jupyter notebooks theo từng bước
│   ├── 01_data_collection.ipynb    # Thu thập dữ liệu từ DBpedia SPARQL
│   ├── 02_eda.ipynb                # Phân tích khám phá dữ liệu (EDA)
│   ├── 03_feature_set1.ipynb       # Thực nghiệm Feature Set 1 (baseline)
│   ├── 04_feature_set2.ipynb       # Thực nghiệm Feature Set 2 (mở rộng)
│   └── 05_stability_final.ipynb    # Đánh giá ổn định & so sánh cuối cùng
│
├── 🐍 src/                         # Module Python tái sử dụng
│   ├── data_collection.py          # Hàm thu thập dữ liệu qua SPARQL
│   ├── feature_engineering.py      # Hàm biến đổi & tạo feature
│   └── evaluation.py               # Hàm đánh giá model & vẽ đồ thị
│
├── 📊 data/                        # Dữ liệu (không commit file lớn lên Git)
│   ├── raw/                        # Dữ liệu thô từ DBpedia
│   │   └── wikicities_raw.csv      # ~10.000 thành phố, download từ DBpedia
│   ├── processed/                  # Dữ liệu sau làm sạch & tiền xử lý
│   └── features/                   # Feature matrix sẵn sàng để train model
│
├── 🤖 models/                      # Model đã train (joblib / pickle)
│
├── 📈 reports/                     # Biểu đồ, bảng kết quả xuất ra
│
├── setup_colab.py                  # Script cài đặt môi trường trên Colab
├── requirements.txt                # Danh sách thư viện cần cài thêm
├── .env                            # Cấu hình (endpoint, seed,...)
└── .gitignore                      # Bỏ qua file lớn & secret
```

---

## 📥 Cách lấy dữ liệu

Dữ liệu được thu thập **tự động** từ [DBpedia SPARQL Endpoint](https://dbpedia.org/sparql) — không cần đăng ký tài khoản hay API key.

### Cách 1: Chạy notebook `01_data_collection.ipynb` *(Khuyến nghị)*

```
notebooks/01_data_collection.ipynb
```

Notebook này sẽ:
1. Kết nối đến `https://dbpedia.org/sparql`
2. Truy vấn tất cả entity thuộc lớp `dbo:City` có trường `dbo:populationTotal`
3. Lấy các thuộc tính: tên, dân số, diện tích, độ cao, tọa độ, quốc gia, vùng, mật độ dân số, năm thành lập, abstract
4. Phân trang tự động (1000 bản ghi/trang) với retry khi timeout
5. Lưu kết quả vào `data/raw/wikicities_raw.csv`

> ⚠️ **Lưu ý**: DBpedia SPARQL endpoint có thể chậm hoặc timeout. Script đã có cơ chế retry và `sleep_sec` giữa các trang. Tổng thời gian thu thập ~10.000 bản ghi khoảng **15–30 phút**.

### Cách 2: Dùng file CSV có sẵn

File `data/raw/wikicities_raw.csv` (~900 KB) đã được thu thập sẵn. Nếu bạn clone repo về, có thể bỏ qua bước thu thập và chạy thẳng từ notebook `02_eda.ipynb`.

### Cấu hình thu thập (`.env`)

```ini
SPARQL_ENDPOINT=https://dbpedia.org/sparql
SPARQL_PAGE_SIZE=1000        # Số bản ghi mỗi trang
SPARQL_MAX_CITIES=10000      # Giới hạn tổng số thành phố
SPARQL_SLEEP_SEC=2.0         # Nghỉ 2 giây giữa các request
RANDOM_SEED=42
TEST_SIZE=0.2
```

---

## 🚀 Hướng dẫn chạy

### Chạy trên máy local

**1. Clone repo**
```bash
git clone https://github.com/hoagannhh/Seminar-Pattern-Recognition.git
cd Seminar-Pattern-Recognition
```

**2. Tạo virtual environment & cài thư viện**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> 💡 Các thư viện đã có sẵn trong Colab (numpy, pandas, matplotlib, scikit-learn, scipy) **không** có trong `requirements.txt` để tránh cài trùng. Trên máy local, cần cài thêm thủ công:
> ```bash
> pip install numpy pandas matplotlib scikit-learn scipy
> ```

**3. Chạy Jupyter**
```bash
jupyter notebook
```

**4. Chạy notebooks theo thứ tự:**

| Thứ tự | Notebook | Mô tả |
|--------|----------|-------|
| 1 | `01_data_collection.ipynb` | Thu thập dữ liệu (có thể bỏ qua nếu dùng CSV sẵn) |
| 2 | `02_eda.ipynb` | EDA — hiểu phân phối, missing values, tương quan |
| 3 | `03_feature_set1.ipynb` | Train & đánh giá Feature Set 1 |
| 4 | `04_feature_set2.ipynb` | Train & đánh giá Feature Set 2 |
| 5 | `05_stability_final.ipynb` | So sánh tổng thể & phân tích ổn định |

---

### Chạy trên Google Colab

**Bước 1**: Tạo GitHub Personal Access Token (PAT)
- Vào [github.com/settings/tokens](https://github.com/settings/tokens) → Generate new token (classic)
- Tick quyền: `repo`
- Copy token

**Bước 2**: Mở notebook trên Colab, chạy cell đầu tiên:
```python
exec(open("setup_colab.py").read())
# hoặc nếu chưa có file:
# !wget https://raw.githubusercontent.com/hoagannhh/Seminar-Pattern-Recognition/main/setup_colab.py
# exec(open("setup_colab.py").read())
```

Script `setup_colab.py` sẽ tự động:
- Lấy GitHub token từ **Colab Secrets** (🔑 icon bên trái) → biến môi trường → nhập tay
- Clone/pull repo về `/content/Seminar-Pattern-Recognition`
- Thêm `src/` vào `sys.path`
- Cài `requirements.txt`
- Tạo các thư mục cần thiết
- Kiểm tra GPU

> 💡 **Khuyến nghị**: Lưu token vào **Colab Secrets** với key `GITHUB_TOKEN` để không phải nhập lại mỗi session.

---

## 📝 Mô tả các file

### Notebooks

#### `01_data_collection.ipynb`
Thu thập dữ liệu thô từ DBpedia SPARQL. Bao gồm:
- Đếm tổng số thành phố trong KG
- Phân trang và thu thập song song
- Đếm số triples (numTriples) của từng entity — đại diện cho độ phong phú thông tin trong KG
- Lưu `data/raw/wikicities_raw.csv`

#### `02_eda.ipynb`
Phân tích khám phá dữ liệu (EDA). Bao gồm:
- Thống kê tổng quan (shape, dtypes, missing values)
- Phân phối của `population` và `log_population`
- Phân phối địa lý (tọa độ lat/lon)
- Tương quan giữa các features và target
- Top quốc gia, phân bố theo continent
- Scatter plots, heatmap correlation

#### `03_feature_set1.ipynb`
Thực nghiệm với **Feature Set 1** (baseline thuần số):
- Các features: `log_area`, `log_elevation`, `log_populationDensity`, `log_numTriples`, `lat`, `lon`
- Models: Linear Regression, Ridge, Random Forest, XGBoost, LightGBM
- Cross-validation 5-fold
- So sánh RMSE, MAE, R²

#### `04_feature_set2.ipynb`
Thực nghiệm với **Feature Set 2** (mở rộng với categorical và interaction):
- Thêm: Target Rate Encoding cho `country_name`, `region_name`
- Thêm: Interaction features (`area_x_density`, `lat_abs`, `triple_per_attr`,...)
- Feature importance với SHAP
- So sánh với Feature Set 1

#### `05_stability_final.ipynb`
Đánh giá tổng thể và tính ổn định:
- Chạy CV với 10 random seeds khác nhau
- Boxplot phân phối RMSE để kiểm tra variance
- So sánh cuối cùng giữa tất cả Feature Set và Model
- Kết luận và hướng phát triển

---

### Module `src/`

#### `src/data_collection.py`
Các hàm thu thập dữ liệu:

| Hàm | Mô tả |
|-----|-------|
| `sparql_query(query)` | Gửi SPARQL query, trả về JSON |
| `get_total_cities()` | Đếm tổng số thành phố trong DBpedia |
| `fetch_page(offset, limit)` | Lấy một trang dữ liệu |
| `collect_all_cities(max_cities)` | Thu thập toàn bộ với phân trang & retry |
| `get_triple_count(uri)` | Đếm số triples của một entity |
| `add_triple_counts(df)` | Thêm cột `numTriples` vào DataFrame |

#### `src/feature_engineering.py`
Các hàm tạo và biến đổi features:

| Hàm | Mô tả |
|-----|-------|
| `rare_category_grouper(series, min_freq)` | Gộp các giá trị categorical hiếm thành `"OTHER"` |
| `log_transform_cols(df, cols)` | Tạo cột `log_{col} = log1p(col)` |
| `target_rate_encoding(df_train, df_apply, cat_cols, target_col)` | Target Rate Encoding (smoothed) tránh data leakage |
| `add_computable_features(df)` | Tạo interaction features (area×density, lat_abs,...) |
| `build_feature_matrix(df, num_cols, tre_cols)` | Xây dựng feature matrix cuối (impute median) |

#### `src/evaluation.py`
Các hàm đánh giá và vẽ đồ thị:

| Hàm | Mô tả |
|-----|-------|
| `rmse(y_true, y_pred)` | Tính RMSE |
| `mae(y_true, y_pred)` | Tính MAE |
| `r2(y_true, y_pred)` | Tính R² |
| `evaluate_cv(model, X, y, cv)` | Cross-validation, trả về dict metrics |
| `compare_models(models, X, y)` | So sánh nhiều model, trả về DataFrame |
| `get_feature_importance(model, feature_names)` | Trích xuất feature importance |
| `plot_feature_importance(importance, top_n)` | Vẽ bar chart importance |
| `stability_cv(model, X, y, n_runs)` | CV nhiều lần với seed khác nhau |
| `plot_stability(stability_df)` | Boxplot tính ổn định |
| `compare_stability(results)` | So sánh stability nhiều model/feature set |

---

### File cấu hình

#### `setup_colab.py`
Script thiết lập môi trường Google Colab tự động. Chạy một lần đầu mỗi session mới.

#### `requirements.txt`
Các thư viện cần cài thêm (ngoài những gì Colab đã có sẵn):

| Thư viện | Mục đích |
|----------|----------|
| `SPARQLWrapper` | Giao tiếp với SPARQL endpoint |
| `category-encoders` | Target encoding nâng cao |
| `xgboost` | Model XGBoost |
| `lightgbm` | Model LightGBM |
| `shap` | Giải thích model (SHAP values) |
| `python-dotenv` | Load `.env` config |
| `tqdm` | Progress bar |
| `joblib` | Lưu/load model |
| `seaborn` | Biểu đồ thống kê |
| `plotly` | Biểu đồ tương tác |

#### `.env`
File cấu hình tham số hệ thống. **Không commit lên Git** (đã có trong `.gitignore`).

---

## 🔄 Pipeline tổng thể

```
DBpedia SPARQL
     │
     ▼
[01] Data Collection          → data/raw/wikicities_raw.csv
     │
     ▼
[02] EDA                      → Hiểu dữ liệu, phát hiện outlier
     │
     ▼
[03/04] Feature Engineering   → data/features/
     │   ├── Log transform
     │   ├── Rare category grouping
     │   ├── Target Rate Encoding
     │   └── Interaction features
     │
     ▼
     Model Training (CV 5-fold)
     │   ├── Linear Regression
     │   ├── Ridge Regression
     │   ├── Random Forest
     │   ├── XGBoost
     │   └── LightGBM
     │
     ▼
[05] Stability Analysis       → reports/
     └── So sánh RMSE qua 10 random seeds
```

---

## 🔧 Features được sử dụng

### Feature Set 1 — Baseline (numeric thuần)
| Feature | Mô tả |
|---------|-------|
| `log_area` | log diện tích thành phố (km²) |
| `log_elevation` | log độ cao so với mực nước biển |
| `log_populationDensity` | log mật độ dân số |
| `log_numTriples` | log số triples trong KG (độ phong phú thông tin) |
| `lat`, `lon` | Tọa độ địa lý |
| `abstractLen` | Độ dài đoạn abstract trong Wikipedia |

### Feature Set 2 — Mở rộng
Bao gồm Feature Set 1, cộng thêm:

| Feature | Mô tả |
|---------|-------|
| `country_name_tre` | Target Rate Encoding của quốc gia |
| `region_name_tre` | Target Rate Encoding của vùng |
| `area_x_density` | Tích diện tích × mật độ (proxy quy mô đô thị) |
| `lat_abs` | Khoảng cách tuyệt đối từ xích đạo |
| `has_region` | Binary: có thông tin vùng hay không |
| `is_recent` | Binary: năm thành lập ≥ 1800 |
| `triple_per_attr` | Tỉ lệ triples / số thuộc tính (mật độ thông tin KG) |

---

## 🤖 Models

| Model | Thư viện | Ghi chú |
|-------|----------|---------|
| Linear Regression | scikit-learn | Baseline đơn giản nhất |
| Ridge Regression | scikit-learn | L2 regularization |
| Random Forest | scikit-learn | Ensemble cây quyết định |
| XGBoost | xgboost | Gradient boosting mạnh |
| LightGBM | lightgbm | Gradient boosting nhanh, hiệu quả |

---

## 📊 Kết quả

> Kết quả chi tiết xem trong `notebooks/05_stability_final.ipynb` và `reports/`.

Metric đánh giá trên `log_population` (RMSE thấp hơn = tốt hơn):

- **RMSE ≈ 0.85–0.90** → sai số trung bình ~0.85 trên log scale (tương đương ~2.3× trên scale gốc)
- LightGBM và XGBoost cho kết quả tốt nhất
- Feature Set 2 cải thiện đáng kể so với Feature Set 1 nhờ Target Rate Encoding

---

## 🛠️ Yêu cầu môi trường

- **Python** ≥ 3.10
- **Google Colab** (khuyến nghị) hoặc máy local với Jupyter
- Kết nối Internet (để truy vấn DBpedia SPARQL)

```bash
pip install -r requirements.txt
# Thêm trên local (đã có sẵn trong Colab):
pip install numpy pandas matplotlib scikit-learn scipy jupyter
```

---

## 👥 Tác giả

| Họ tên | GitHub |
|--------|--------|
| Hoàng Anh | [@hoagannhh](https://github.com/hoagannhh) |

---

## 📄 License

Dự án phục vụ mục đích học thuật (seminar môn Nhận dạng mẫu).  
Dữ liệu từ DBpedia được cấp phép theo [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
