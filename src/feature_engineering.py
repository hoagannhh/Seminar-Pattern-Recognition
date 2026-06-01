# ============================================================
#  src/feature_engineering.py  —  wikicities-population
#
#  Các hàm feature engineering dùng chung cho các notebooks.
# ============================================================

import numpy as np
import pandas as pd
from typing import List


def rare_category_grouper(series: pd.Series, min_freq: int = 10, other_label: str = "OTHER") -> pd.Series:
    """
    Gộp các giá trị categorical xuất hiện ít hơn min_freq lần thành other_label.

    Parameters
    ----------
    series    : pd.Series  — cột categorical cần xử lý
    min_freq  : int        — ngưỡng tần suất tối thiểu (mặc định 10)
    other_label : str      — nhãn thay thế (mặc định "OTHER")

    Returns
    -------
    pd.Series với các giá trị hiếm đã được thay bằng other_label.

    Example
    -------
    >>> df['country_name'] = rare_category_grouper(df['country_name'], min_freq=10)
    """
    freq = series.value_counts()
    rare_values = freq[freq < min_freq].index
    return series.where(~series.isin(rare_values), other=other_label)


def log_transform_cols(df: pd.DataFrame, cols: List[str], shift: float = 1.0) -> pd.DataFrame:
    """
    Tạo cột log_{col} = log1p(col) cho mỗi cột trong cols.
    Bỏ qua (không tạo) nếu cột không tồn tại trong df.

    Parameters
    ----------
    df    : pd.DataFrame  — dataframe gốc (không bị sửa đổi)
    cols  : list[str]     — danh sách tên cột cần transform
    shift : float         — giá trị cộng thêm trước log (mặc định 1.0, tức log1p)

    Returns
    -------
    pd.DataFrame mới với các cột log_{col} được thêm vào.

    Example
    -------
    >>> df_fe = log_transform_cols(df, ['area', 'numTriples', 'abstractLen'])
    """
    df_out = df.copy()
    for col in cols:
        if col not in df_out.columns:
            continue
        # Clamp về 0 trước khi log để tránh log âm
        df_out[f"log_{col}"] = np.log1p(np.maximum(df_out[col].fillna(0), 0))
    return df_out


# ── Target Rate Encoding ─────────────────────────────────────

def target_rate_encoding(df_train: pd.DataFrame, df_apply: pd.DataFrame,
                         cat_cols: List[str], target_col: str,
                         smoothing: float = 10.0,
                         global_mean: float = None) -> pd.DataFrame:
    """
    Target Rate Encoding (smoothed mean encoding) cho biến categorical.

    Công thức:  encoded = (n_i * mean_i + m * global_mean) / (n_i + m)
    với m = smoothing factor (regularization).

    Tính encoding từ df_train, áp dụng lên df_apply (tránh data leakage).

    Parameters
    ----------
    df_train   : pd.DataFrame  — tập train để tính thống kê
    df_apply   : pd.DataFrame  — tập cần encode (có thể = df_train hoặc val/test)
    cat_cols   : list[str]     — danh sách cột categorical cần encode
    target_col : str           — tên cột target (log_population)
    smoothing  : float         — hệ số làm mịn (mặc định 10)
    global_mean: float | None  — nếu None sẽ tính từ df_train

    Returns
    -------
    pd.DataFrame mới (copy của df_apply) với cột {col}_tre thay thế cột gốc.

    Example
    -------
    >>> df_tr = target_rate_encoding(df_train, df_train, ['country_name'], 'log_population')
    >>> df_va = target_rate_encoding(df_train, df_val,   ['country_name'], 'log_population')
    """
    if global_mean is None:
        global_mean = df_train[target_col].mean()

    df_out = df_apply.copy()

    for col in cat_cols:
        if col not in df_train.columns:
            continue
        stats = (
            df_train.groupby(col)[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "cat_mean", "count": "cat_count"})
        )
        stats["tre"] = (
            (stats["cat_count"] * stats["cat_mean"] + smoothing * global_mean)
            / (stats["cat_count"] + smoothing)
        )
        mapping = stats["tre"].to_dict()
        df_out[f"{col}_tre"] = df_out[col].map(mapping).fillna(global_mean)

    return df_out


# ── Computable (Interaction) Features ───────────────────────

def add_computable_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các feature tính toán từ các cột đã có.

    Features được tạo:
    - area_x_density  : area × populationDensity  (proxy tổng quy mô)
    - log_area_x_density : log của trên
    - lat_abs         : |lat| — khoảng cách từ xích đạo
    - lon_abs         : |lon|
    - has_region      : 1 nếu region_name không null
    - is_recent       : 1 nếu foundingYear >= 1800
    - triple_per_attr : numTriples / (numAttributes + 1) — mật độ thông tin KG

    Parameters
    ----------
    df : pd.DataFrame — dataframe sau log_transform_cols

    Returns
    -------
    pd.DataFrame mới với các cột computable được thêm vào.

    Example
    -------
    >>> df_fe = add_computable_features(df_fe)
    """
    df_out = df.copy()

    if "area" in df_out.columns and "populationDensity" in df_out.columns:
        df_out["area_x_density"] = (
            df_out["area"].fillna(0) * df_out["populationDensity"].fillna(0)
        )
        df_out["log_area_x_density"] = np.log1p(
            np.maximum(df_out["area_x_density"], 0)
        )

    if "lat" in df_out.columns:
        df_out["lat_abs"] = df_out["lat"].abs()

    if "lon" in df_out.columns:
        df_out["lon_abs"] = df_out["lon"].abs()

    if "region_name" in df_out.columns:
        df_out["has_region"] = df_out["region_name"].notna().astype(int)

    if "foundingYear" in df_out.columns:
        df_out["is_recent"] = (df_out["foundingYear"].fillna(0) >= 1800).astype(int)

    if "numTriples" in df_out.columns and "numAttributes" in df_out.columns:
        df_out["triple_per_attr"] = (
            df_out["numTriples"] / (df_out["numAttributes"].fillna(0) + 1)
        )
        df_out["log_triple_per_attr"] = np.log1p(
            np.maximum(df_out["triple_per_attr"], 0)
        )

    return df_out


def build_feature_matrix(df: pd.DataFrame,
                          num_cols: List[str],
                          tre_cols: List[str] = None) -> pd.DataFrame:
    """
    Xây dựng ma trận feature cuối cùng từ các cột numeric và TRE.
    Impute median cho các giá trị thiếu.

    Parameters
    ----------
    df       : pd.DataFrame  — dataframe đã qua transform
    num_cols : list[str]     — các cột numeric cần giữ
    tre_cols : list[str]     — các cột TRE (kết thúc bằng _tre)

    Returns
    -------
    pd.DataFrame sẵn sàng đưa vào model (không có NaN).
    """
    cols = [c for c in num_cols if c in df.columns]
    if tre_cols:
        cols += [c for c in tre_cols if c in df.columns]

    X = df[cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X
