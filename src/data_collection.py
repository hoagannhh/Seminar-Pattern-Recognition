# ============================================================
#  src/data_collection.py  —  wikicities-population
#
#  Các hàm thu thập dữ liệu từ DBpedia SPARQL endpoint.
#  Được tách ra từ 01_data_collection.ipynb để tái sử dụng.
# ============================================================

import time
import math
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
PAGE_SIZE = 1000
DEFAULT_TIMEOUT = 60

QUERY_TEMPLATE = """
PREFIX dbo:  <http://dbpedia.org/ontology/>
PREFIX dbp:  <http://dbpedia.org/property/>
PREFIX geo:  <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dct:  <http://purl.org/dc/terms/>

SELECT DISTINCT
  ?city ?name ?population ?area ?elevation ?lat ?lon
  ?country ?region ?populationDensity ?foundingYear ?abstract
WHERE {{
  ?city a dbo:City .
  ?city dbo:populationTotal ?population .
  OPTIONAL {{ ?city rdfs:label ?name      FILTER (lang(?name) = 'en') }}
  OPTIONAL {{ ?city dbo:area ?area }}
  OPTIONAL {{ ?city dbo:elevation ?elevation }}
  OPTIONAL {{ ?city geo:lat ?lat }}
  OPTIONAL {{ ?city geo:long ?lon }}
  OPTIONAL {{ ?city dbo:country ?country }}
  OPTIONAL {{ ?city dbo:region  ?region  }}
  OPTIONAL {{ ?city dbo:populationDensity ?populationDensity }}
  OPTIONAL {{ ?city dbp:foundingYear ?foundingYear }}
  OPTIONAL {{ ?city dbo:abstract ?abstract FILTER (lang(?abstract) = 'en') }}
}}
ORDER BY ?city
LIMIT  {limit}
OFFSET {offset}
"""

COUNT_QUERY = """
PREFIX dbo:  <http://dbpedia.org/ontology/>
SELECT (COUNT(DISTINCT ?city) AS ?total)
WHERE {
  ?city a dbo:City .
  ?city dbo:populationTotal ?population .
}
"""

TRIPLE_COUNT_TEMPLATE = """
SELECT (COUNT(*) AS ?numTriples)
WHERE {{
  <{uri}> ?p ?o .
}}
"""


def sparql_query(query: str, endpoint: str = SPARQL_ENDPOINT,
                 timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Gửi SPARQL query và trả về JSON response."""
    resp = requests.get(
        endpoint,
        params={"query": query, "format": "application/sparql-results+json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_total_cities(endpoint: str = SPARQL_ENDPOINT) -> int:
    """Trả về tổng số thành phố trong DBpedia."""
    data = sparql_query(COUNT_QUERY, endpoint)
    return int(data["results"]["bindings"][0]["total"]["value"])


def fetch_page(offset: int, limit: int = PAGE_SIZE,
               endpoint: str = SPARQL_ENDPOINT) -> list[dict]:
    """Lấy một trang dữ liệu thành phố từ SPARQL endpoint."""
    query = QUERY_TEMPLATE.format(limit=limit, offset=offset)
    data = sparql_query(query, endpoint)
    rows = []
    for b in data["results"]["bindings"]:
        rows.append({
            "uri":               b.get("city", {}).get("value"),
            "name":              b.get("name", {}).get("value"),
            "population":        b.get("population", {}).get("value"),
            "area":              b.get("area", {}).get("value"),
            "elevation":         b.get("elevation", {}).get("value"),
            "lat":               b.get("lat", {}).get("value"),
            "lon":               b.get("lon", {}).get("value"),
            "country":           b.get("country", {}).get("value"),
            "region":            b.get("region", {}).get("value"),
            "populationDensity": b.get("populationDensity", {}).get("value"),
            "foundingYear":      b.get("foundingYear", {}).get("value"),
            "abstract":          b.get("abstract", {}).get("value"),
        })
    return rows


def collect_all_cities(max_cities: Optional[int] = None,
                       page_size: int = PAGE_SIZE,
                       endpoint: str = SPARQL_ENDPOINT,
                       sleep_sec: float = 1.0) -> pd.DataFrame:
    """
    Thu thập toàn bộ thành phố từ DBpedia theo pagination.

    Parameters
    ----------
    max_cities : int | None  — Giới hạn số thành phố (None = lấy tất cả)
    page_size  : int         — Số bản ghi mỗi trang
    endpoint   : str         — SPARQL endpoint URL
    sleep_sec  : float       — Thời gian nghỉ giữa các request

    Returns
    -------
    pd.DataFrame với các cột raw từ DBpedia.
    """
    total = get_total_cities(endpoint)
    if max_cities:
        total = min(total, max_cities)

    n_pages = math.ceil(total / page_size)
    all_rows = []
    failed_pages = []

    for page in range(n_pages):
        offset = page * page_size
        try:
            rows = fetch_page(offset, min(page_size, total - offset), endpoint)
            all_rows.extend(rows)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"  Page {page} failed: {e}")
            failed_pages.append(page)

    if failed_pages:
        print(f"Retry {len(failed_pages)} failed pages...")
        for page in failed_pages:
            offset = page * page_size
            try:
                rows = fetch_page(offset, min(page_size, total - offset), endpoint)
                all_rows.extend(rows)
                time.sleep(sleep_sec * 2)
            except Exception as e:
                print(f"  Page {page} still failed: {e}")

    df = pd.DataFrame(all_rows)
    num_cols = ["population", "area", "elevation", "lat", "lon",
                "populationDensity", "foundingYear"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["country", "region"]:
        if col in df.columns:
            df[col + "_name"] = df[col].str.split("/").str[-1].str.replace("_", " ")

    return df


def get_triple_count(uri: str, endpoint: str = SPARQL_ENDPOINT,
                     timeout: int = 30) -> int:
    """Đếm số triples của một entity trong KG."""
    try:
        query = TRIPLE_COUNT_TEMPLATE.format(uri=uri)
        data = sparql_query(query, endpoint, timeout)
        return int(data["results"]["bindings"][0]["numTriples"]["value"])
    except Exception:
        return 0


def add_triple_counts(df: pd.DataFrame, uri_col: str = "uri",
                      endpoint: str = SPARQL_ENDPOINT,
                      sleep_sec: float = 0.1) -> pd.DataFrame:
    """
    Thêm cột numTriples vào DataFrame bằng cách query từng entity.
    Chậm — chỉ dùng khi cần refresh dữ liệu.
    """
    df = df.copy()
    counts = []
    for uri in df[uri_col]:
        counts.append(get_triple_count(uri, endpoint))
        time.sleep(sleep_sec)
    df["numTriples"] = counts
    return df
