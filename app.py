from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import requests
import io
import threading
import time
import json

# AI interpreter configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

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
            "Category",
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
            "Category",
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
# AI BUYING ASSISTANT - NATURAL LANGUAGE INTERPRETER
# =========================================================

AI_FILTER_KEYS = {
    "budget",
    "min_budget",
    "brands",
    "exclude_brands",
    "models",
    "exclude_models",
    "categories",
    "exclude_categories",
    "locations",
    "exclude_locations",
    "companies",
    "exclude_companies",
    "transmissions",
    "colors",
    "min_year",
    "max_year",
    "min_km",
    "max_km",
}


def compact_market_context():
    """
    Give the model enough live vocabulary to map user language onto
    values that actually exist in market_base.csv without sending the
    entire dataset to the model.
    """
    if not MARKET_READY or market_df is None or market_df.empty:
        return {}

    def unique_values(column, limit=None):
        values = sorted(
            market_df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        return values[:limit] if limit else values

    return {
        "brands": unique_values("Brand"),
        "locations": unique_values("Location"),
        "transmissions": unique_values("Transmission"),
        "companies": unique_values("Company", 250),
    }


def extract_response_text(payload):
    """
    Extract the text returned by the Responses API without requiring
    the OpenAI Python package.
    """
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")

    return ""


def sanitize_ai_filters(raw_filters):
    """
    Never trust model output directly. Only allow fields supported by
    market_search and normalize empty values away.
    """
    if not isinstance(raw_filters, dict):
        return {}

    clean = {}

    for key, value in raw_filters.items():
        if key not in AI_FILTER_KEYS:
            continue

        if value in [None, "", [], {}]:
            continue

        clean[key] = value

    return clean


def interpret_market_query(message, current_filters=None, language="TR"):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY_NOT_CONFIGURED")

    current_filters = current_filters or {}
    market_context = compact_market_context()

    instructions = """
You are the query interpreter for a North Cyprus vehicle-market assistant.

Your job is NOT to answer the user and NOT to invent listings.
Convert the user's latest message into structured changes to a vehicle search.

The deterministic search engine supports ONLY these filter fields:
budget, min_budget, brands, exclude_brands, models, exclude_models,
categories, exclude_categories, locations, exclude_locations,
companies, exclude_companies, transmissions, colors,
min_year, max_year, min_km, max_km.

Important rules:
- Return JSON only.
- "filters" must contain ONLY constraints expressed or clearly modified
  by the latest user message.
- Do not repeat old filters merely because they appear in current_filters.
- Use null/empty omission rather than guessing.
- Prices are GBP.
- Convert "15k", "15 bin", "15 thousand" to 15000.
- Convert mileage expressions similarly.
- "Bireysel" means a private/individual seller.
- If the user wants galleries/dealers, set exclude_companies to ["Bireysel"].
- If the user wants private sellers, set companies to ["Bireysel"].
- If the user says either/both seller types are fine, put
  "seller_mode": "both" so the application can clear the prior seller filter.
- If the user explicitly removes a previous constraint, put its field name
  in "clear_filters".
- Preserve real market spellings when they are supplied in market_context.
- Do not translate brand/model names.
- Do not turn subjective ideas such as reliable, sporty, economical,
  family-friendly, small, luxurious, or good value into unsupported hard
  filters. Put those concepts in "preferences".
- Use short canonical preference tags whenever possible so they persist cleanly across turns:
  vehicle_type:SUV, vehicle_type:crossover, vehicle_type:pickup, vehicle_type:small_car,
  vehicle_type:motorcycle, use_case:commute, use_case:family, priority:economy,
  priority:reliability, priority:performance.
- If the user explicitly says they have no brand/model preference, add "any_brand_model".
- If they explicitly say they have no year or mileage restriction, add "any_year_km".
- If they explicitly say any vehicle type is fine, add "any_vehicle_type".
- "preferences" is for useful soft intent that the deterministic filters
  cannot represent yet.
- "needs_clarification" should only be true when the latest request cannot
  safely be represented without asking the user something.
- "clarification_question" should be short and in the user's language.
"""

    user_payload = {
        "language": language,
        "latest_message": message,
        "current_filters": current_filters,
        "market_context": market_context,
    }

    schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "budget": {"type": ["number", "null"]},
                    "min_budget": {"type": ["number", "null"]},
                    "brands": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_brands": {"type": ["array", "null"], "items": {"type": "string"}},
                    "models": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_models": {"type": ["array", "null"], "items": {"type": "string"}},
                    "categories": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_categories": {"type": ["array", "null"], "items": {"type": "string"}},
                    "locations": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_locations": {"type": ["array", "null"], "items": {"type": "string"}},
                    "companies": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_companies": {"type": ["array", "null"], "items": {"type": "string"}},
                    "transmissions": {"type": ["array", "null"], "items": {"type": "string"}},
                    "colors": {"type": ["array", "null"], "items": {"type": "string"}},
                    "min_year": {"type": ["integer", "null"]},
                    "max_year": {"type": ["integer", "null"]},
                    "min_km": {"type": ["number", "null"]},
                    "max_km": {"type": ["number", "null"]},
                },
                "required": [
                    "budget", "min_budget", "brands", "exclude_brands",
                    "models", "exclude_models", "categories", "exclude_categories",
                    "locations", "exclude_locations", "companies", "exclude_companies",
                    "transmissions", "colors", "min_year", "max_year",
                    "min_km", "max_km"
                ],
            },
            "clear_filters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "seller_mode": {
                "type": ["string", "null"],
                "enum": ["individual", "gallery", "both", None],
            },
            "preferences": {
                "type": "array",
                "items": {"type": "string"},
            },
            "needs_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
        },
        "required": [
            "filters",
            "clear_filters",
            "seller_mode",
            "preferences",
            "needs_clarification",
            "clarification_question",
        ],
        "additionalProperties": False,
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "instructions": instructions,
            "input": json.dumps(user_payload, ensure_ascii=False),
        },
        timeout=30,
    )

    response.raise_for_status()

    response_payload = response.json()
    response_text = extract_response_text(response_payload)

    if not response_text:
        raise ValueError("AI_INTERPRETER_EMPTY_RESPONSE")

    interpreted = json.loads(response_text)
    interpreted["filters"] = sanitize_ai_filters(
        interpreted.get("filters", {})
    )

    # Remove null values emitted because the strict schema requires every
    # filter property to be present.
    interpreted["filters"] = {
        key: value
        for key, value in interpreted["filters"].items()
        if value is not None
    }

    return interpreted


@app.route("/api/interpret", methods=["POST"])
def api_interpret_market_query():
    try:
        data = request.json or {}

        message = str(data.get("message", "")).strip()

        if not message:
            return jsonify({
                "success": False,
                "error": "MESSAGE_REQUIRED"
            }), 400

        result = interpret_market_query(
            message=message,
            current_filters=data.get("current_filters") or {},
            language=str(data.get("language", "TR")).upper(),
        )

        return jsonify({
            "success": True,
            **result
        })

    except requests.HTTPError as e:
        status_code = (
            e.response.status_code
            if e.response is not None
            else None
        )

        response_text = (
            e.response.text[:2000]
            if e.response is not None
            else ""
        )

        print(
            "OPENAI INTERPRETER HTTP ERROR:",
            status_code,
            response_text,
            flush=True
        )

        return jsonify({
            "success": False,
            "error": "AI_INTERPRETER_FAILED",
            "openai_status": status_code,
            "openai_message": response_text
        }), 502

    except RuntimeError as e:
        print("AI INTERPRETER CONFIG ERROR:", e, flush=True)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 503

    except Exception as e:
        print("AI INTERPRETER FAILED:", repr(e), flush=True)

        return jsonify({
            "success": False,
            "error": "AI_INTERPRETER_FAILED"
        }), 500


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
    categories=None,
    exclude_categories=None,
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
    # CATEGORY / VARIANT
    # -----------------------

    if categories:
        filtered = filtered[
            contains_any(
                filtered["Category"],
                categories
            )
        ]

    if exclude_categories:
        filtered = filtered[
            ~contains_any(
                filtered["Category"],
                exclude_categories
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
            "category": row["Category"],

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
# AI BUYING ASSISTANT - CONVERSATION / GROUNDED RESPONSE
# =========================================================

def apply_interpretation_to_filters(previous_filters, interpretation):
    """
    Merge the AI's incremental filter changes into the current search state.
    This mirrors the frontend behavior, but keeps the full assistant endpoint
    self-contained and deterministic.
    """
    next_filters = dict(previous_filters or {})
    incoming = sanitize_ai_filters(interpretation.get("filters", {}))

    next_filters.update(incoming)

    seller_mode = interpretation.get("seller_mode")

    if seller_mode == "individual":
        next_filters["companies"] = ["Bireysel"]
        next_filters.pop("exclude_companies", None)

    elif seller_mode == "gallery":
        next_filters["exclude_companies"] = ["Bireysel"]
        next_filters.pop("companies", None)

    elif seller_mode == "both":
        next_filters.pop("companies", None)
        next_filters.pop("exclude_companies", None)

    for key in interpretation.get("clear_filters", []) or []:
        if key in AI_FILTER_KEYS:
            next_filters.pop(key, None)

    return next_filters


def merge_preferences(previous_preferences, new_preferences):
    """
    Keep soft user preferences across turns without duplicating them.
    """
    merged = []

    for value in list(previous_preferences or []) + list(new_preferences or []):
        value = str(value).strip()

        if value and value.casefold() not in {
            existing.casefold() for existing in merged
        }:
            merged.append(value)

    return merged


def _preference_flags(preferences):
    prefs = {str(p).strip().casefold() for p in (preferences or []) if str(p).strip()}
    return {
        "has_vehicle_type": any(p.startswith("vehicle_type:") for p in prefs) or "any_vehicle_type" in prefs,
        "has_use_case": any(p.startswith("use_case:") for p in prefs),
        "any_brand_model": "any_brand_model" in prefs,
        "any_year_km": "any_year_km" in prefs,
    }


def guided_narrowing_question(filters, preferences, count, language="TR"):
    """Ask at most one high-value buying question before recommending."""
    flags = _preference_flags(preferences)

    has_budget = filters.get("budget") not in [None, ""]
    has_brand_model = bool(
        filters.get("brands") or filters.get("models") or
        filters.get("exclude_brands") or filters.get("exclude_models") or
        flags["any_brand_model"]
    )
    has_age_mileage = bool(
        filters.get("min_year") not in [None, ""] or
        filters.get("max_year") not in [None, ""] or
        filters.get("min_km") not in [None, ""] or
        filters.get("max_km") not in [None, ""] or
        flags["any_year_km"]
    )
    has_focus = has_brand_model or flags["has_vehicle_type"] or flags["has_use_case"]

    questions = {
        "TR": {
            "budget": "Maksimum bütçeniz nedir?",
            "focus": "Nasıl bir araç arıyorsunuz — örneğin küçük otomobil, SUV, crossover, pick-up veya motosiklet — ya da ağırlıklı kullanım amacınız nedir?",
            "age_km": "Minimum model yılı veya maksimum kilometre sınırınız var mı?",
            "type_or_brand": "Araç tipi veya marka/model konusunda özellikle istediğiniz ya da istemediğiniz bir şey var mı?",
        },
        "EN": {
            "budget": "What's your maximum budget?",
            "focus": "What kind of vehicle are you after — for example a small car, SUV, crossover, pickup or motorcycle — or what will you mainly use it for?",
            "age_km": "Do you have a minimum year or maximum mileage in mind?",
            "type_or_brand": "Any vehicle type, brand or model you particularly want or want to avoid?",
        },
        "RU": {
            "budget": "Какой у вас максимальный бюджет?",
            "focus": "Какой тип транспорта вы ищете — например компактный автомобиль, SUV, кроссовер, пикап или мотоцикл — и для чего в основном будете его использовать?",
            "age_km": "Есть ли минимальный год выпуска или максимальный пробег?",
            "type_or_brand": "Есть ли тип, марка или модель, которые вы особенно хотите или хотите исключить?",
        },
    }
    q = questions.get(language, questions["TR"])

    if not has_budget:
        return q["budget"]

    # Broad searches should first establish what the buyer actually needs.
    if int(count or 0) > 80 and not has_focus:
        return q["focus"]

    # Once purpose/type is known, age/mileage usually narrows the market most usefully.
    if int(count or 0) > 120 and not has_age_mileage:
        return q["age_km"]

    # If the market is still very broad, ask one final optional preference question.
    if int(count or 0) > 220 and not has_brand_model and not flags["has_vehicle_type"]:
        return q["type_or_brand"]

    return None


def _model_market_median(item):
    """Return a robust same-brand/model market median where enough comparisons exist."""
    if market_df is None or market_df.empty:
        return None, 0

    brand = str(item.get("brand") or "").strip().casefold()
    model = str(item.get("model") or "").strip().casefold()
    if not brand or not model:
        return None, 0

    comps = market_df[
        market_df["Brand"].fillna("").astype(str).str.strip().str.casefold().eq(brand) &
        market_df["Model"].fillna("").astype(str).str.strip().str.casefold().eq(model)
    ]
    prices = pd.to_numeric(comps["Price"], errors="coerce")
    prices = prices[prices > 0].dropna()
    if len(prices) < 3:
        return None, len(prices)
    return float(prices.median()), len(prices)


def is_suspicious_recommendation_price(item):
    """
    Keep questionable listings searchable, but stop obvious price-normalisation
    anomalies from being used as recommendation evidence.
    """
    try:
        price = float(item.get("price"))
    except (TypeError, ValueError):
        return True

    if price <= 0:
        return True

    median, n = _model_market_median(item)
    if median and n >= 3 and median >= 2500:
        # A listing at less than 40% of its own model's market median is too
        # anomalous to use as recommendation evidence without verification.
        if price < median * 0.40:
            return True

    return False


def select_assistant_candidates(results, filters, max_candidates=24):
    """Choose a representative recommendation sample from the full filtered market."""
    if not results:
        return []

    clean = [item for item in results if not is_suspicious_recommendation_price(item)]
    if not clean:
        clean = list(results)

    # Avoid exact same-price seller clusters dominating the sample as a second guard.
    cluster_counts = {}
    for item in clean:
        company = str(item.get("company") or "").strip().casefold()
        try:
            price = float(item.get("price"))
        except (TypeError, ValueError):
            continue
        key = (company, price)
        cluster_counts[key] = cluster_counts.get(key, 0) + 1

    de_clustered = []
    for item in clean:
        company = str(item.get("company") or "").strip().casefold()
        try:
            price = float(item.get("price"))
        except (TypeError, ValueError):
            continue
        if cluster_counts.get((company, price), 0) >= 4 and price <= 4000:
            continue
        de_clustered.append(item)

    if de_clustered:
        clean = de_clustered

    clean = sorted(clean, key=lambda x: float(x.get("price") or 0))
    if len(clean) <= max_candidates:
        return clean

    # Sample evenly across the whole filtered price distribution rather than
    # handing the model only the cheapest rows.
    chosen = []
    seen = set()
    for i in range(max_candidates):
        idx = round(i * (len(clean) - 1) / max(max_candidates - 1, 1))
        item = clean[idx]
        identity = item.get("link") or (
            item.get("brand"), item.get("model"), item.get("year"), item.get("price")
        )
        if identity not in seen:
            chosen.append(item)
            seen.add(identity)

    return chosen[:max_candidates]

def generate_grounded_market_answer(
    message,
    language,
    filters,
    preferences,
    search_result,
):
    """
    Generate a conversational answer grounded in the deterministic
    market-search result. Current listings, prices, mileage, locations,
    sellers, and counts may ONLY come from search_result.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY_NOT_CONFIGURED")

    count = int(search_result.get("count", 0) or 0)
    results = search_result.get("results", []) or []
    candidates = select_assistant_candidates(results, filters)

    if count == 0:
        fallback = {
            "TR": "Bu kriterlere uyan aktif ilan bulamadım. İsterseniz kriterlerden birini esnetebiliriz.",
            "EN": "I couldn't find an active listing matching those criteria. We can loosen one of the filters if you like.",
            "RU": "Я не нашёл активных объявлений по этим критериям. Можно немного ослабить один из фильтров.",
        }

        return fallback.get(language, fallback["TR"])

    instructions = """
You are the conversational intelligence layer for a North Cyprus vehicle-buying assistant.

Use the supplied deterministic market data as the only source of truth for current listings,
prices, mileage, locations, sellers, transmissions and result counts. Never invent listing facts.
General model-level automotive knowledge may be used cautiously for soft preferences such as
economy or reliability, but never present it as verified condition or performance of an individual listing.

STYLE — IMPORTANT:
- Answer in the requested language.
- Be brief: normally 1-3 short sentences. Only expand if the user asks for detail.
- Do NOT output URLs, links, markdown links, or raw listing URLs. Links are UI data only.
- Do NOT dump a list of listings just because matches exist.
- Mention at most 3 models/listings, and only when doing so genuinely helps answer the latest question.
- Avoid generic purchase-checklist advice unless the user asks for it or it is essential to the answer.
- Sound like a knowledgeable conversational assistant, not a database report.
- If there are many matches, summarize and help narrow intelligently rather than enumerating results.
- The application handles guided narrowing before this function is called, so when you are called, answer the buyer's latest need directly and concisely.
- Do not call a vehicle "best value", "the best", or objectively superior merely because it is cheap.
- Do not claim a specific listed vehicle is mechanically reliable, economical, safe, or in good condition.
- supplied_results is a representative candidate sample, not necessarily the full matching set.
- Prices in supplied_results can contain seller-entry/data-normalisation anomalies. Treat implausibly low prices cautiously;
  never build a recommendation around a suspicious price simply because it is the cheapest.
"""

    payload = {
        "language": language,
        "latest_message": message,
        "active_filters": filters,
        "soft_preferences": preferences,
        "total_count": count,
        "supplied_results": candidates,
        "important_note": (
            "supplied_results is a selected representative sample when total_count is larger "
            "than the number of supplied results. Raw listing URLs must never be printed."
        ),
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
        },
        timeout=45,
    )

    response.raise_for_status()

    response_payload = response.json()
    answer = extract_response_text(response_payload).strip()

    if not answer:
        raise ValueError("AI_ASSISTANT_EMPTY_RESPONSE")

    return answer


@app.route("/api/assistant", methods=["POST"])
def api_ai_buying_assistant():
    try:
        data = request.json or {}

        message = str(data.get("message", "")).strip()
        language = str(data.get("language", "TR")).upper()
        current_filters = data.get("current_filters") or {}
        current_preferences = data.get("current_preferences") or []

        if not message:
            return jsonify({
                "success": False,
                "error": "MESSAGE_REQUIRED"
            }), 400

        interpretation = interpret_market_query(
            message=message,
            current_filters=current_filters,
            language=language,
        )

        next_filters = apply_interpretation_to_filters(
            current_filters,
            interpretation,
        )

        next_preferences = merge_preferences(
            current_preferences,
            interpretation.get("preferences", []),
        )

        if (
            interpretation.get("needs_clarification")
            and interpretation.get("clarification_question")
        ):
            return jsonify({
                "success": True,
                "answer": interpretation["clarification_question"],
                "filters": next_filters,
                "preferences": next_preferences,
                "count": None,
                "returned": 0,
                "results": [],
                "interpretation": interpretation,
            })

        # Pull the full filtered result set for candidate selection. The public
        # response is still capped below so the frontend is not flooded with rows.
        search_result = market_search(
            budget=next_filters.get("budget"),
            min_budget=next_filters.get("min_budget"),
            brands=next_filters.get("brands"),
            exclude_brands=next_filters.get("exclude_brands"),
            models=next_filters.get("models"),
            exclude_models=next_filters.get("exclude_models"),
            categories=next_filters.get("categories"),
            exclude_categories=next_filters.get("exclude_categories"),
            locations=next_filters.get("locations"),
            exclude_locations=next_filters.get("exclude_locations"),
            companies=next_filters.get("companies"),
            exclude_companies=next_filters.get("exclude_companies"),
            transmissions=next_filters.get("transmissions"),
            colors=next_filters.get("colors"),
            min_year=next_filters.get("min_year"),
            max_year=next_filters.get("max_year"),
            min_km=next_filters.get("min_km"),
            max_km=next_filters.get("max_km"),
            limit=5000,
        )

        if not search_result.get("success"):
            return jsonify(search_result), 503

        # Guided buying flow: do not jump straight into recommendations while
        # the search is still extremely broad. Ask one useful question per turn.
        guide_question = guided_narrowing_question(
            filters=next_filters,
            preferences=next_preferences,
            count=search_result.get("count", 0),
            language=language,
        )

        if guide_question:
            public_results = (search_result.get("results") or [])[:100]
            return jsonify({
                "success": True,
                "answer": guide_question,
                "filters": next_filters,
                "preferences": next_preferences,
                "count": search_result.get("count", 0),
                "returned": len(public_results),
                "results": public_results,
                "interpretation": interpretation,
                "stage": "narrowing",
            })

        answer = generate_grounded_market_answer(
            message=message,
            language=language,
            filters=next_filters,
            preferences=next_preferences,
            search_result=search_result,
        )

        public_results = (search_result.get("results") or [])[:100]
        return jsonify({
            "success": True,
            "answer": answer,
            "filters": next_filters,
            "preferences": next_preferences,
            "count": search_result.get("count", 0),
            "returned": len(public_results),
            "results": public_results,
            "interpretation": interpretation,
            "stage": "recommendation",
        })

    except requests.HTTPError as e:
        status_code = (
            e.response.status_code
            if e.response is not None
            else None
        )

        response_text = (
            e.response.text[:2000]
            if e.response is not None
            else ""
        )

        print(
            "OPENAI ASSISTANT HTTP ERROR:",
            status_code,
            response_text,
            flush=True
        )

        return jsonify({
            "success": False,
            "error": "AI_ASSISTANT_FAILED",
            "openai_status": status_code,
            "openai_message": response_text
        }), 502

    except RuntimeError as e:
        print("AI ASSISTANT CONFIG ERROR:", e, flush=True)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 503

    except Exception as e:
        print("AI ASSISTANT FAILED:", repr(e), flush=True)

        return jsonify({
            "success": False,
            "error": "AI_ASSISTANT_FAILED"
        }), 500


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

            categories=data.get("categories"),
            exclude_categories=data.get("exclude_categories"),

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
# AI BUYING ASSISTANT - MARKET OPTIONS
# =========================================================

@app.route("/api/market_options", methods=["GET"])
def market_options():
    if not MARKET_READY or market_df is None or market_df.empty:
        return jsonify({
            "success": False,
            "error": "MARKET_DATA_NOT_READY"
        }), 503

    try:
        brands = sorted(
            market_df["Brand"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        models = sorted(
            market_df["Model"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        locations = sorted(
            market_df["Location"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        transmissions = sorted(
            market_df["Transmission"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        companies = sorted(
            market_df["Company"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        return jsonify({
            "success": True,
            "brands": brands,
            "models": models,
            "locations": locations,
            "transmissions": transmissions,
            "companies": companies
        })

    except Exception as e:
        print("MARKET OPTIONS FAILED:", e)

        return jsonify({
            "success": False,
            "error": "MARKET_OPTIONS_FAILED"
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