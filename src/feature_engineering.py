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
