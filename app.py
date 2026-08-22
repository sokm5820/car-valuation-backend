from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import requests
import io
import threading
import time
import json

import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)
CORS(app)

# -----------------------
# GOOGLE SHEETS CONNECTION
# -----------------------

GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

credentials = Credentials.from_service_account_info(
    json.loads(GOOGLE_CREDENTIALS),
    scopes=SCOPES
)

gc = gspread.authorize(credentials)

sheet = gc.open("North Cyprus Vehicle Leads").sheet1

# -----------------------
# LOAD DATA (GITHUB CSV SOURCE - SAFE VERSION)
# -----------------------
CSV_URL = "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/ads_base.csv"

df = pd.DataFrame()  # safe default

# 🔥 ADDED: readiness flag
DATA_READY = False

def load_data():
    global df, DATA_READY
    try:
        r = requests.get(CSV_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        DATA_READY = True
        print("CSV loaded successfully")
    except Exception as e:
        print("CSV LOAD FAILED:", e)
        df = pd.DataFrame()
        DATA_READY = False

load_data()

# -----------------------
# 🔥 ADDED: AUTO REFRESH EVERY 12 HOURS
# -----------------------
def refresh_data_loop():
    while True:
        time.sleep(12 * 60 * 60)  # 12 hours
        print("Refreshing CSV data from GitHub...")
        load_data()

threading.Thread(target=refresh_data_loop, daemon=True).start()

# -----------------------
# TYPE CLEANING
# -----------------------
def safe_prepare_dataframe():
    global df

    if df is None or df.empty:
        return

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["KM"] = pd.to_numeric(df["KM"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df["Brand"] = df["Brand"].astype(str).str.strip()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()

safe_prepare_dataframe()

# =========================================================
# AI BUYING ASSISTANT - CURRENT MARKET DATA
# =========================================================

MARKET_CSV_URL = "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/market_base.csv"

market_df = pd.DataFrame()
MARKET_READY = False


def load_market_data():
    global market_df, MARKET_READY

    try:
        r = requests.get(MARKET_CSV_URL, timeout=15)
        r.raise_for_status()

        new_market_df = pd.read_csv(
            io.StringIO(r.text),
            low_memory=False
        )

        # -----------------------
        # REQUIRED COLUMNS
        # -----------------------
        required_columns = [
            "Brand",
            "Model",
            "Year",
            "Price",
            "KM",
            "Company",
            "Location",
            "Transmission",
            "Color",
            "Image",
            "Link"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in new_market_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"market_base.csv missing columns: {missing_columns}"
            )

        # -----------------------
        # NUMERIC TYPES
        # -----------------------
        new_market_df["Year"] = pd.to_numeric(
            new_market_df["Year"],
            errors="coerce"
        )

        new_market_df["Price"] = pd.to_numeric(
            new_market_df["Price"],
            errors="coerce"
        )

        new_market_df["KM"] = pd.to_numeric(
            new_market_df["KM"],
            errors="coerce"
        )

        # -----------------------
        # TEXT TYPES
        # -----------------------
        text_columns = [
            "Brand",
            "Model",
            "Company",
            "Location",
            "Transmission",
            "Color",
            "Image",
            "Link"
        ]

        for col in text_columns:
            new_market_df[col] = (
                new_market_df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # Only replace the live dataframe after
        # the new file has loaded successfully.
        market_df = new_market_df
        MARKET_READY = True

        print(
            f"Market CSV loaded successfully: "
            f"{len(market_df)} listings"
        )

    except Exception as e:
        print("MARKET CSV LOAD FAILED:", e)

        # IMPORTANT:
        # If an old successful dataset already exists,
        # leave it running rather than wiping it.
        if market_df is None or market_df.empty:
            market_df = pd.DataFrame()
            MARKET_READY = False


load_market_data()


# =========================================================
# MARKET DATA AUTO REFRESH EVERY 12 HOURS
# =========================================================

def refresh_market_data_loop():
    while True:
        time.sleep(12 * 60 * 60)

        print("Refreshing market CSV data from GitHub...")

        load_market_data()


threading.Thread(
    target=refresh_market_data_loop,
    daemon=True
).start()

# =========================================================
# YEARS
# =========================================================
@app.route("/years", methods=["GET"])
def get_years():
    if not DATA_READY or df is None or df.empty:
        return jsonify([])
    years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    return jsonify(years)

# =========================================================
# BRANDS
# =========================================================
@app.route("/brands", methods=["GET"])
def get_brands():
    if not DATA_READY or df is None or df.empty:
        return jsonify([])

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
    if not DATA_READY or df is None or df.empty:
        return jsonify([])

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
    if not DATA_READY or df is None or df.empty:
        return jsonify([])

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
    if not DATA_READY or df is None or df.empty:
        return {
            "median_price": None,
            "min_price": None,
            "max_price": None,
            "scatter": []
        }

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
# AI BUYING ASSISTANT - MARKET SEARCH ENGINE
# =========================================================

def market_search(
    budget=None,
    min_budget=None,
    brands=None,
    exclude_brands=None,
    models=None,
    exclude_models=None,
    locations=None,
    exclude_locations=None,
    companies=None,
    exclude_companies=None,
    transmissions=None,
    colors=None,
    min_year=None,
    max_year=None,
    min_km=None,
    max_km=None,
    limit=20
):
    if not MARKET_READY or market_df is None or market_df.empty:
        return {
            "success": False,
            "error": "MARKET_DATA_NOT_READY",
            "count": 0,
            "returned": 0,
            "results": []
        }

    filtered = market_df.copy()

    # -----------------------
    # HELPERS
    # -----------------------

    def normalize(value):
        return str(value).strip().casefold()

    def normalize_list(values):
        if not values:
            return []

        if not isinstance(values, list):
            values = [values]

        return [
            normalize(value)
            for value in values
            if value not in [None, ""]
        ]

    def contains_any(series, values):
        values = normalize_list(values)

        if not values:
            return pd.Series(
                True,
                index=series.index
            )

        normalized_series = (
            series
            .fillna("")
            .astype(str)
            .map(normalize)
        )

        mask = pd.Series(
            False,
            index=series.index
        )

        for value in values:
            mask = mask | normalized_series.str.contains(
                value,
                regex=False,
                na=False
            )

        return mask

    # -----------------------
    # PRICE
    # -----------------------

    if budget not in [None, ""]:
        filtered = filtered[
            filtered["Price"] <= float(budget)
        ]

    if min_budget not in [None, ""]:
        filtered = filtered[
            filtered["Price"] >= float(min_budget)
        ]

    # -----------------------
    # BRAND
    # -----------------------

    if brands:
        wanted = set(
            normalize_list(brands)
        )

        filtered = filtered[
            filtered["Brand"]
            .map(normalize)
            .isin(wanted)
        ]

    if exclude_brands:
        unwanted = set(
            normalize_list(exclude_brands)
        )

        filtered = filtered[
            ~filtered["Brand"]
            .map(normalize)
            .isin(unwanted)
        ]

    # -----------------------
    # MODEL
    # -----------------------

    if models:
        filtered = filtered[
            contains_any(
                filtered["Model"],
                models
            )
        ]

    if exclude_models:
        filtered = filtered[
            ~contains_any(
                filtered["Model"],
                exclude_models
            )
        ]

    # -----------------------
    # LOCATION
    # -----------------------

    if locations:
        filtered = filtered[
            contains_any(
                filtered["Location"],
                locations
            )
        ]

    if exclude_locations:
        filtered = filtered[
            ~contains_any(
                filtered["Location"],
                exclude_locations
            )
        ]

    # -----------------------
    # COMPANY / DEALERSHIP
    # -----------------------

    if companies:
        filtered = filtered[
            contains_any(
                filtered["Company"],
                companies
            )
        ]

    if exclude_companies:
        filtered = filtered[
            ~contains_any(
                filtered["Company"],
                exclude_companies
            )
        ]

    # -----------------------
    # TRANSMISSION
    # -----------------------

    if transmissions:
        filtered = filtered[
            contains_any(
                filtered["Transmission"],
                transmissions
            )
        ]

    # -----------------------
    # COLOR
    # -----------------------

    if colors:
        filtered = filtered[
            contains_any(
                filtered["Color"],
                colors
            )
        ]

    # -----------------------
    # YEAR
    # -----------------------

    if min_year not in [None, ""]:
        filtered = filtered[
            filtered["Year"] >= int(min_year)
        ]

    if max_year not in [None, ""]:
        filtered = filtered[
            filtered["Year"] <= int(max_year)
        ]

    # -----------------------
    # MILEAGE
    # -----------------------
    # Cars with missing KM remain in the dataset normally.
    #
    # BUT if the user specifically requests a mileage
    # limit, a car with unknown KM cannot be said to meet it.

    if max_km not in [None, ""]:
        filtered = filtered[
            filtered["KM"].notna()
            & (filtered["KM"] <= float(max_km))
        ]

    if min_km not in [None, ""]:
        filtered = filtered[
            filtered["KM"].notna()
            & (filtered["KM"] >= float(min_km))
        ]

    # -----------------------
    # SORT
    # -----------------------

    filtered = filtered.sort_values(
        ["Price", "Year"],
        ascending=[True, False]
    )

    total_count = len(filtered)

    # -----------------------
    # LIMIT
    # -----------------------

    try:
        limit = int(limit)
    except:
        limit = 20

    limit = max(
        1,
        min(limit, 100)
    )

    results_df = filtered.head(limit).copy()

    # -----------------------
    # JSON-SAFE RESULTS
    # -----------------------

    results = []

    for _, row in results_df.iterrows():

        results.append({
            "brand": row["Brand"],
            "model": row["Model"],

            "year": (
                int(row["Year"])
                if pd.notna(row["Year"])
                else None
            ),

            "price": (
                float(row["Price"])
                if pd.notna(row["Price"])
                else None
            ),

            "km": (
                int(row["KM"])
                if pd.notna(row["KM"])
                else None
            ),

            "company": row["Company"],
            "location": row["Location"],
            "transmission": row["Transmission"],
            "color": row["Color"],
            "image": row["Image"],
            "link": row["Link"]
        })

    return {
        "success": True,
        "count": total_count,
        "returned": len(results),
        "results": results
    }


# =========================================================
# AI BUYING ASSISTANT - MARKET SEARCH API
# =========================================================

@app.route("/api/search", methods=["POST"])
def api_market_search():

    try:
        data = request.json or {}

        result = market_search(
            budget=data.get("budget"),
            min_budget=data.get("min_budget"),

            brands=data.get("brands"),
            exclude_brands=data.get("exclude_brands"),

            models=data.get("models"),
            exclude_models=data.get("exclude_models"),

            locations=data.get("locations"),
            exclude_locations=data.get("exclude_locations"),

            companies=data.get("companies"),
            exclude_companies=data.get("exclude_companies"),

            transmissions=data.get("transmissions"),
            colors=data.get("colors"),

            min_year=data.get("min_year"),
            max_year=data.get("max_year"),

            min_km=data.get("min_km"),
            max_km=data.get("max_km"),

            limit=data.get("limit", 20)
        )

        return jsonify(result)

    except (TypeError, ValueError) as e:
        print("INVALID MARKET SEARCH:", e)

        return jsonify({
            "success": False,
            "error": "INVALID_SEARCH_PARAMETERS",
            "count": 0,
            "returned": 0,
            "results": []
        }), 400

    except Exception as e:
        print("MARKET SEARCH FAILED:", e)

        return jsonify({
            "success": False,
            "error": "SEARCH_FAILED",
            "count": 0,
            "returned": 0,
            "results": []
        }), 500


# =========================================================
# AI BUYING ASSISTANT - HEALTH CHECK
# =========================================================

@app.route("/api/market_health", methods=["GET"])
def market_health():

    return jsonify({
        "status": "ok" if MARKET_READY else "loading",
        "ready": MARKET_READY,
        "rows": len(market_df)
    })

# =========================================================
# LEAD SUBMISSION
# =========================================================
@app.route("/submit_lead", methods=["POST"])
def submit_lead():
    try:
        data = request.json or {}

        # -----------------------
        # GET SUBMITTED DATA
        # -----------------------
        year = data.get("year")
        brand = data.get("brand")
        model = data.get("model")
        category = data.get("category")

        valuation = data.get("valuation")
        min_price = data.get("min_price")
        max_price = data.get("max_price")

        name = str(data.get("name", "")).strip()
        phone = str(data.get("phone", "")).strip()
        consent = data.get("consent")

        # -----------------------
        # VALIDATION
        # -----------------------

        if not name:
            return jsonify({
                "success": False,
                "error": "NAME_REQUIRED"
            }), 400

        if not phone:
            return jsonify({
                "success": False,
                "error": "PHONE_REQUIRED"
            }), 400

        if consent is not True:
            return jsonify({
                "success": False,
                "error": "CONSENT_REQUIRED"
            }), 400

        if not year or not brand or not model or not category:
            return jsonify({
                "success": False,
                "error": "VEHICLE_INFO_INCOMPLETE"
            }), 400

        # -----------------------
        # DATES
        # -----------------------

        submitted_at = pd.Timestamp.now()

        expires_at = submitted_at + pd.Timedelta(days=90)

        # -----------------------
        # ADD LEAD TO GOOGLE SHEET
        # -----------------------

        sheet.append_row([
            submitted_at.strftime("%Y-%m-%d %H:%M:%S"),
            year,
            brand,
            model,
            category,
            valuation,
            min_price,
            max_price,
            name,
            phone,
            "TRUE",
            expires_at.strftime("%Y-%m-%d %H:%M:%S")
        ])

        # -----------------------
        # SUCCESS
        # -----------------------

        return jsonify({
            "success": True
        })

    except Exception as e:
        print("LEAD SUBMISSION FAILED:", e)

        return jsonify({
            "success": False,
            "error": "SUBMISSION_FAILED"
        }), 500

# =========================================================
# HEALTH CHECK (UPDATED - COLD START SAFE ENDPOINT)
# =========================================================
@app.route("/")
def home():
    return "Car Valuation API is running"

@app.route("/api/health")
def health():
    return {
        "status": "ok" if DATA_READY else "loading",
        "ready": DATA_READY,
        "rows": len(df)
    }

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)