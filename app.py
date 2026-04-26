from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# -----------------------
# LOAD DATA (OPTION 1 FIX)
# -----------------------
DATA_PATH = os.environ.get("DATA_PATH", "ads_base.csv")
df = pd.read_csv(DATA_PATH)

# -----------------------
# TYPE CLEANING
# -----------------------
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["KM"] = pd.to_numeric(df["KM"], errors="coerce")
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

# -----------------------
# NORMALISE TEXT FIELDS
# -----------------------
df["Brand"] = df["Brand"].astype(str).str.strip()
df["Model"] = df["Model"].astype(str).str.strip()
df["Category"] = df["Category"].astype(str).str.strip()

# =========================================================
# YEARS
# =========================================================
@app.route("/years", methods=["GET"])
def get_years():
    years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    return jsonify(years)

# =========================================================
# BRANDS
# =========================================================
@app.route("/brands", methods=["GET"])
def get_brands():
    year = request.args.get("year")
    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") == year]
        except:
            pass

    return jsonify(sorted(filtered["Brand"].dropna().unique().tolist()))

# =========================================================
# MODELS
# =========================================================
@app.route("/models", methods=["GET"])
def get_models():
    year = request.args.get("year")
    brand = request.args.get("brand")

    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") == year]
        except:
            pass

    if brand not in [None, "", "null"]:
        filtered = filtered[
            filtered["Brand"].astype(str).str.strip().str.lower()
            == str(brand).strip().lower()
        ]

    return jsonify(sorted(filtered["Model"].dropna().unique().tolist()))

# =========================================================
# CATEGORIES
# =========================================================
@app.route("/categories", methods=["GET"])
def get_categories():
    year = request.args.get("year")
    brand = request.args.get("brand")
    model = request.args.get("model")

    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") == year]
        except:
            pass

    if brand not in [None, "", "null"]:
        filtered = filtered[
            filtered["Brand"].astype(str).str.strip().str.lower()
            == str(brand).strip().lower()
        ]

    if model not in [None, "", "null"]:
        filtered = filtered[
            filtered["Model"].astype(str).str.strip().str.lower()
            == str(model).strip().lower()
        ]

    return jsonify(sorted(filtered["Category"].dropna().unique().tolist()))

# =========================================================
# VALUATION ENGINE
# =========================================================
def get_valuation(df, year, brand, model, category):
    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[
                pd.to_numeric(filtered["Year"], errors="coerce") == year
            ]
        except:
            pass

    if brand not in [None, "", "null"]:
        filtered = filtered[
            filtered["Brand"].astype(str).str.strip().str.lower()
            == str(brand).strip().lower()
        ]

    if model not in [None, "", "null"]:
        filtered = filtered[
            filtered["Model"].astype(str).str.strip().str.lower()
            == str(model).strip().lower()
        ]

    if category not in [None, "", "null"]:
        filtered = filtered[
            filtered["Category"].astype(str).str.strip().str.lower()
            == str(category).strip().lower()
        ]

    if filtered.empty:
        return {
            "median_price": None,
            "min_price": None,
            "max_price": None,
            "scatter": []
        }

    median_price = filtered["Price"].median()
    min_price = filtered["Price"].min()
    max_price = filtered["Price"].max()

    latest_date = filtered["DATE"].max()

    filtered["status"] = filtered["DATE"].apply(
        lambda x: "active" if x == latest_date else "removed"
    )

    scatter = filtered[["KM", "Price", "status"]].dropna().to_dict(orient="records")

    return {
        "median_price": float(median_price) if pd.notna(median_price) else None,
        "min_price": float(min_price) if pd.notna(min_price) else None,
        "max_price": float(max_price) if pd.notna(max_price) else None,
        "scatter": scatter
    }

# =========================================================
# API ENDPOINT
# =========================================================
@app.route("/get_valuation", methods=["POST"])
def valuation():
    data = request.json

    result = get_valuation(
        df,
        data.get("year"),
        data.get("brand"),
        data.get("model"),
        data.get("category")
    )

    return jsonify(result)

# =========================================================
# HEALTH CHECK
# =========================================================
@app.route("/")
def home():
    return "Car Valuation API is running"

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)