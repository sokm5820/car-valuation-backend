from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import requests
import io
import threading
import traceback
import time
import json
import math
import re
from datetime import datetime, timezone

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

# =========================================================
# AI BUYING ASSISTANT - BUYER INTELLIGENCE v1
# =========================================================
BUYER_MODEL_CSV_URL = (
    "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/"
    "buyer_model_intelligence.csv"
)
BUYER_CATEGORY_CSV_URL = (
    "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/"
    "buyer_category_intelligence.csv"
)

buyer_model_df = pd.DataFrame()
buyer_category_df = pd.DataFrame()
BUYER_INTELLIGENCE_READY = False

# Stable model-level buyer profiles keep DISCOVER deterministic and fast.
# These are generated offline by build_buyer_model_profiles_v1.py and committed
# beside the other production intelligence CSVs.
BUYER_MODEL_PROFILE_CSV_URL = (
    "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/"
    "buyer_model_profiles.csv"
)
model_profile_df = pd.DataFrame()
MODEL_PROFILE_READY = False
MODEL_PROFILE_LOOKUP = {}
ASSISTANT_PROFILE_VERSION = "1.0"


def _prepare_buyer_intelligence_frame(frame):
    frame = frame.copy()

    text_cols = [
        "VehicleType", "Brand", "Model", "CategoryDetail",
        "LiquidityEvidenceLevel", "PricePressureEvidenceLevel",
        "Buyer_LiquidityEvidenceConfidence",
        "Buyer_PricePressureEvidenceConfidence",
        "RecommendationGranularity",
    ]
    for col in text_cols:
        if col in frame.columns:
            frame[col] = (
                frame[col].fillna("").astype(str).str.strip()
            )

    numeric_cols = [
        "Year", "CurrentListings", "CurrentStartingPrice",
        "CurrentMedianPrice", "CurrentHighestPrice", "CurrentMedianKM",
        "GalleryListings", "PrivateListings",
        "Buyer_HistoricalDistinctListings",
        "Buyer_MedianObservedDaysToExit",
        "Buyer_Exit30EligibleListings",
        "Buyer_ObservedExitWithin30DaysRate",
        "Buyer_Exit60EligibleListings",
        "Buyer_ObservedExitWithin60DaysRate",
        "Buyer_Exit90EligibleListings",
        "Buyer_ObservedExitWithin90DaysRate",
        "Buyer_PricePressureEligibleListings",
        "Buyer_PriceReductionRate",
        "Buyer_MedianReductionPctAmongReduced",
        "Buyer_ListingVolumeRankWithinVehicleType",
        "Buyer_CategoryListingVolumeRankWithinModel",
        "Buyer_ModelListingVolumeRankWithinVehicleType",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def load_buyer_intelligence():
    global buyer_model_df, buyer_category_df, BUYER_INTELLIGENCE_READY

    try:
        model_r = requests.get(BUYER_MODEL_CSV_URL, timeout=20)
        model_r.raise_for_status()
        category_r = requests.get(BUYER_CATEGORY_CSV_URL, timeout=20)
        category_r.raise_for_status()

        new_model = pd.read_csv(
            io.StringIO(model_r.text), low_memory=False
        )
        new_category = pd.read_csv(
            io.StringIO(category_r.text), low_memory=False
        )

        required_model = {
            "VehicleType", "Brand", "Model", "Year",
            "CurrentListings", "CurrentStartingPrice",
            "Buyer_ObservedExitWithin60DaysRate",
            "Buyer_LiquidityEvidenceConfidence",
            "LiquidityEvidenceLevel",
        }
        required_category = {
            "VehicleType", "Brand", "Model", "CategoryDetail", "Year",
            "CurrentListings", "CurrentStartingPrice",
            "Buyer_ObservedExitWithin60DaysRate",
            "Buyer_LiquidityEvidenceConfidence",
            "LiquidityEvidenceLevel",
        }

        missing_model = required_model - set(new_model.columns)
        missing_category = required_category - set(new_category.columns)
        if missing_model or missing_category:
            raise ValueError(
                "Buyer Intelligence schema mismatch. "
                f"model missing={sorted(missing_model)}, "
                f"category missing={sorted(missing_category)}"
            )

        buyer_model_df = _prepare_buyer_intelligence_frame(new_model)
        buyer_category_df = _prepare_buyer_intelligence_frame(new_category)
        BUYER_INTELLIGENCE_READY = True

        print(
            "Buyer Intelligence loaded successfully: "
            f"{len(buyer_model_df)} model-year rows, "
            f"{len(buyer_category_df)} category-year rows"
        )

    except Exception as e:
        print("BUYER INTELLIGENCE LOAD FAILED:", e)

        # Keep the last successful intelligence snapshot alive.
        if (
            buyer_model_df is None or buyer_model_df.empty
            or buyer_category_df is None or buyer_category_df.empty
        ):
            buyer_model_df = pd.DataFrame()
            buyer_category_df = pd.DataFrame()
            BUYER_INTELLIGENCE_READY = False


def load_model_profiles():
    global model_profile_df, MODEL_PROFILE_READY, MODEL_PROFILE_LOOKUP

    try:
        r = requests.get(BUYER_MODEL_PROFILE_CSV_URL, timeout=15)
        r.raise_for_status()
        new_profiles = pd.read_csv(io.StringIO(r.text), low_memory=False).fillna("")
        required = {
            "Brand", "Model", "VehicleType", "BodyStyle", "SizeClass",
            "Economy", "Luxury", "Comfort", "Performance", "Practicality",
            "Family", "Commute", "Confidence"
        }
        missing = required - set(new_profiles.columns)
        if missing:
            raise ValueError(f"buyer_model_profiles.csv missing columns: {sorted(missing)}")

        for col in required:
            new_profiles[col] = new_profiles[col].fillna("").astype(str).str.strip()

        lookup = {}
        for row in new_profiles.to_dict("records"):
            brand = str(row.get("Brand") or "").strip()
            model = str(row.get("Model") or "").strip()
            if not brand or not model:
                continue
            lookup[(brand.casefold(), model.casefold())] = row

        model_profile_df = new_profiles
        MODEL_PROFILE_LOOKUP = lookup
        MODEL_PROFILE_READY = bool(lookup)
        print(f"Model Buyer Profiles loaded successfully: {len(lookup)} model families")
    except Exception as e:
        print("MODEL BUYER PROFILE LOAD FAILED:", e)
        # Preserve the last successful snapshot if one already exists.
        if not MODEL_PROFILE_LOOKUP:
            model_profile_df = pd.DataFrame()
            MODEL_PROFILE_READY = False


def _profile_level(value):
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}.get(str(value or "").strip().upper(), 0)


def _profile_matches_vehicle_type(profile, requested_type):
    requested = str(requested_type or "").strip().casefold()
    if not requested:
        return True
    vt = str(profile.get("VehicleType") or "").strip().upper()
    body = str(profile.get("BodyStyle") or "").strip().upper()
    size = str(profile.get("SizeClass") or "").strip().upper()

    if requested == "car":
        return vt == "CAR"
    if requested == "small_car":
        return vt == "CAR" and size in {"MICRO", "SMALL", "COMPACT"} and body not in {"SUV", "CROSSOVER", "PICKUP", "VAN", "MPV"}
    if requested in {"suv"}:
        return vt == "CAR" and body in {"SUV", "CROSSOVER"}
    if requested in {"crossover"}:
        return vt == "CAR" and body == "CROSSOVER"
    if requested in {"pickup", "pick-up"}:
        return vt == "PICKUP" or body == "PICKUP"
    if requested in {"motorcycle", "motosiklet"}:
        return vt in {"MOTORCYCLE", "SCOOTER"}
    if requested == "scooter":
        return vt == "SCOOTER" or body == "SCOOTER"
    return True


def _canonicalize_buyer_preferences(preferences):
    """Map multilingual/legacy preference labels to one internal taxonomy.

    Language is an input/output concern only. The market/profile engine should only
    ever see these canonical values, so EN/TR/RU requests behave identically.
    """
    aliases = {
        # vehicle profile
        "vehicle_type:car": "vehicle_type:car",
        "vehicle_type:automobile": "vehicle_type:car",
        "vehicle_type:otomobil": "vehicle_type:car",
        "vehicle_type:araba": "vehicle_type:car",
        "vehicle_type:автомобиль": "vehicle_type:car",
        "vehicle_type:машина": "vehicle_type:car",
        "vehicle_type:small_car": "vehicle_type:small_car",
        "vehicle_type:small car": "vehicle_type:small_car",
        "vehicle_type:küçük araç": "vehicle_type:small_car",
        "vehicle_type:kucuk arac": "vehicle_type:small_car",
        "vehicle_type:небольшая машина": "vehicle_type:small_car",
        "vehicle_type:небольшой автомобиль": "vehicle_type:small_car",
        "vehicle_type:suv": "vehicle_type:SUV",
        "vehicle_type:crossover": "vehicle_type:crossover",
        "vehicle_type:pickup": "vehicle_type:pickup",
        "vehicle_type:pick-up": "vehicle_type:pickup",
        "vehicle_type:motorcycle": "vehicle_type:motorcycle",
        "vehicle_type:motosiklet": "vehicle_type:motorcycle",
        "vehicle_type:мотоцикл": "vehicle_type:motorcycle",
        "vehicle_type:scooter": "vehicle_type:scooter",
        "vehicle_type:скутер": "vehicle_type:scooter",
        # buyer priorities
        "priority:economy": "priority:economy",
        "priority:economical": "priority:economy",
        "priority:ekonomik": "priority:economy",
        "priority:экономичная": "priority:economy",
        "priority:экономичный": "priority:economy",
        "priority:luxury": "priority:luxury",
        "priority:comfort": "priority:comfort",
        "priority:performance": "priority:performance",
        "priority:practicality": "priority:practicality",
        "use_case:family": "use_case:family",
        "use_case:commute": "use_case:commute",
    }
    out = []
    seen = set()
    for pref in preferences or []:
        raw = str(pref or "").strip()
        if not raw:
            continue
        canonical = aliases.get(raw.casefold(), raw)
        key = canonical.casefold()
        if key not in seen:
            out.append(canonical)
            seen.add(key)
    return out


def _profile_preference_score(profile, preferences):
    """Score buyer-profile fit without allowing soft traits to erase the market.

    Concrete vehicle classes (SUV/pickup/motorcycle/scooter) remain strict.
    Descriptors such as small-car, economy, comfort, practicality, family and
    commute are ranking preferences. This prevents a taxonomy/profile mismatch
    from turning valid hard-filtered inventory into an incorrect zero-result state.
    """
    preferences = _canonicalize_buyer_preferences(preferences)
    score = 0
    matched_soft = 0
    requested_soft = 0

    for pref in preferences or []:
        p = str(pref or "").strip().casefold()
        if p.startswith("vehicle_type:"):
            requested = p.split(":", 1)[1]
            # Explicit physical classes are genuine constraints. "small_car" is a
            # buyer profile/size preference and is therefore scored rather than fatal.
            if requested in {"car", "suv", "crossover", "pickup", "pick-up", "motorcycle", "motosiklet", "scooter"}:
                if not _profile_matches_vehicle_type(profile, requested):
                    return None
                score += 8
            elif requested == "small_car":
                requested_soft += 1
                if _profile_matches_vehicle_type(profile, requested):
                    matched_soft += 1
                    score += 8
                else:
                    score -= 5
        elif p == "priority:economy":
            requested_soft += 1
            level = _profile_level(profile.get("Economy"))
            if level >= 2:
                matched_soft += 1
                score += level * 4
            else:
                score -= 4
        elif p == "priority:luxury":
            requested_soft += 1
            level = _profile_level(profile.get("Luxury"))
            if level >= 2:
                matched_soft += 1
                score += level * 4
            else:
                score -= 4
        elif p == "priority:comfort":
            requested_soft += 1
            level = _profile_level(profile.get("Comfort"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "priority:performance":
            requested_soft += 1
            level = _profile_level(profile.get("Performance"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "priority:practicality":
            requested_soft += 1
            level = _profile_level(profile.get("Practicality"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "use_case:family":
            requested_soft += 1
            level = _profile_level(profile.get("Family"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "use_case:commute":
            requested_soft += 1
            level = _profile_level(profile.get("Commute"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3

    # At least one requested soft trait must be genuinely supported. This avoids
    # returning unrelated cars while still permitting graceful partial matches.
    if requested_soft and matched_soft == 0:
        return None

    confidence = str(profile.get("Confidence") or "").strip().upper()
    if confidence == "HIGH":
        score += 2
    elif confidence == "MEDIUM":
        score += 1
    return score


def _deterministic_profile_shortlist(model_market, preferences, max_models=30):
    if not MODEL_PROFILE_READY or not MODEL_PROFILE_LOOKUP:
        return []
    scored = []
    for summary in model_market:
        key = (str(summary.get("brand") or "").casefold(), str(summary.get("model") or "").casefold())
        profile = MODEL_PROFILE_LOOKUP.get(key)
        if not profile:
            continue
        score = _profile_preference_score(profile, preferences)
        if score is None:
            continue
        ranked_summary = dict(summary)
        ranked_summary["_profile_score"] = score
        scored.append((
            -score,
            -int(summary.get("newest_year") or 0),
            -min(int(summary.get("count") or 0), 100),
            str(summary.get("brand") or "").casefold(),
            str(summary.get("model") or "").casefold(),
            ranked_summary,
        ))
    scored.sort(key=lambda x: x[:-1])
    return [x[-1] for x in scored[:max_models]]


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
load_buyer_intelligence()
load_model_profiles()


# =========================================================
# MARKET DATA AUTO REFRESH EVERY 12 HOURS
# =========================================================

def refresh_market_data_loop():
    while True:
        time.sleep(12 * 60 * 60)

        print("Refreshing market CSV data from GitHub...")

        load_market_data()
        print("Refreshing Buyer Intelligence from GitHub...")
        load_buyer_intelligence()
        print("Refreshing Model Buyer Profiles from GitHub...")
        load_model_profiles()


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

def sanitize_conversation_history(history, max_messages=16):
    """Keep only a compact, safe recent chat context for interpretation/response quality."""
    if not isinstance(history, list):
        return []

    cleaned = []
    for item in history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        text = str(item.get("text") or item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        cleaned.append({
            "role": role,
            "text": text[:2500],
        })
    return cleaned


def _detect_message_language(text):
    text = str(text or "").strip()
    if not text:
        return None

    low = text.casefold()

    if re.search(r"[\u0400-\u04FF]", text):
        return "RU"

    if re.search(r"[çğıöşüÇĞİÖŞÜ]", text) or re.search(
        r"\b(istiyorum|olsun|bakıyorum|bütçe|araç|araba|motosiklet|hangisi|hepsine|"
        r"evet|hayır|yok|var mı|şart değil|istemiyorum|karşılaştır|göster)\b",
        low,
    ):
        return "TR"

    if re.search(
        r"\b(i|want|buy|suv|what|can|could|get|are|there|more|options|yes|no|all|them|"
        r"what about|show|compare|cheaper|newer|which|please|under|budget|available|"
        r"have|with|without|looking|find|tell|me|cars?|vehicles?)\b",
        low,
    ):
        return "EN"

    return None


def detect_conversation_language(message, requested_language="TR", conversation_history=None):
    """
    Follow the language the buyer is actually using.

    Short follow-ups such as "GLA", "yes" or a bare brand inherit the most recent
    detectable user language instead of snapping back to the UI default.
    """
    current = _detect_message_language(message)
    if current:
        return current

    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").lower() != "user":
            continue
        detected = _detect_message_language(item.get("text"))
        if detected:
            return detected

    requested = str(requested_language or "TR").upper()
    return requested if requested in {"TR", "EN", "RU"} else "TR"

def normalize_assistant_format(answer):
    """
    Enforce one consistent premium chat rhythm:
    intro paragraph
    model/listing lines with NO blank lines between them
    final paragraph

    The model is still responsible for wording; this only normalizes whitespace.
    """
    text = str(answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    raw = [line.strip() for line in text.split("\n")]
    nonempty = [line for line in raw if line]
    if len(nonempty) <= 2:
        return "\n\n".join(nonempty)

    def looks_like_option(line):
        # Vehicle/listing lines normally contain an em dash and a price/year fact.
        return "—" in line and (
            "£" in line
            or re.search(r"\b(19|20)\d{2}\b", line)
        )

    first_option = next((i for i, line in enumerate(nonempty) if looks_like_option(line)), None)
    if first_option is None:
        return "\n\n".join(nonempty)

    last_option = first_option
    while last_option + 1 < len(nonempty) and looks_like_option(nonempty[last_option + 1]):
        last_option += 1

    before = nonempty[:first_option]
    options = nonempty[first_option:last_option + 1]
    after = nonempty[last_option + 1:]

    parts = []
    if before:
        parts.append(" ".join(before))
    if options:
        parts.append("\n".join(options))
    if after:
        parts.append(" ".join(after))
    return "\n\n".join(parts)


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


def _normalize_vehicle_phrase(value):
    """
    Normalize user/market vehicle names for deterministic mention matching.
    Hyphens and punctuation are treated as spaces so forms such as
    "e-Power", "e power" and "E-POWER" can resolve to the same market value.
    """
    value = str(value or "").casefold()
    value = re.sub(r"[^a-z0-9çğıöşüа-яё]+", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def resolve_market_vehicle_mentions(message):
    """
    Resolve explicitly named vehicles against live Brand + Model + Category values.

    This version is intentionally linear-time. The previous implementation repeatedly
    filtered the market universe once for every model family, which could add ~20s to
    every assistant request on the production dataset.
    """
    if not MARKET_READY or market_df is None or market_df.empty:
        return []

    message_n = _normalize_vehicle_phrase(message)
    if not message_n:
        return []
    padded_message = f" {message_n} "

    universe = (
        market_df[["Brand", "Model", "Category"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
    )
    # A dirty-data row can occasionally have a brand name in the Model column
    # (for example Brand=Çekici, Model=Nissan). Never treat a model-only token as
    # a vehicle mention when that token is itself a real brand name.
    known_brand_norms = {
        _normalize_vehicle_phrase(x)
        for x in universe["Brand"].tolist()
        if str(x).strip()
    }

    # Build all lookup structures in one pass instead of rescanning the dataframe
    # for each Brand/Model candidate.
    families = {}
    model_to_brands = {}
    for row in universe.itertuples(index=False):
        brand = str(row.Brand).strip()
        model = str(row.Model).strip()
        category = str(row.Category).strip()
        if not brand or not model:
            continue

        brand_n = _normalize_vehicle_phrase(brand)
        model_n = _normalize_vehicle_phrase(model)
        key = (brand_n, model_n)
        family = families.setdefault(key, {
            "brand": brand,
            "model": model,
            "brand_n": brand_n,
            "model_n": model_n,
            "categories": {},
        })
        model_to_brands.setdefault(model_n, set()).add(brand_n)
        if category:
            category_n = _normalize_vehicle_phrase(category)
            if category_n:
                family["categories"][category_n] = category

    candidates = []
    for family in families.values():
        brand = family["brand"]
        model = family["model"]
        brand_n = family["brand_n"]
        model_n = family["model_n"]
        full_n = f"{brand_n} {model_n}".strip()

        match_strength = 0
        if full_n and f" {full_n} " in padded_message:
            match_strength = 3
        elif (
            model_n
            and len(model_n) >= 3
            and model_n not in known_brand_norms
            and f" {model_n} " in padded_message
            and len(model_to_brands.get(model_n, set())) == 1
        ):
            match_strength = 2

        if not match_strength:
            continue

        target = {
            "brand": brand,
            "model": model,
            "category": None,
            "_match_strength": match_strength,
            "_phrase_len": len(full_n),
        }

        category_matches = []
        for category_n, category in family["categories"].items():
            # Do not infer plain engine-size categories from incidental numbers.
            if not re.search(r"[a-zçğıöşüа-яё]", category_n, flags=re.IGNORECASE):
                continue
            model_category = f"{model_n} {category_n}".strip()
            full_category = f"{brand_n} {model_n} {category_n}".strip()
            if (
                (full_category and f" {full_category} " in padded_message)
                or (model_category and f" {model_category} " in padded_message)
            ):
                category_matches.append((len(category_n), category))

        if category_matches:
            category_matches.sort(reverse=True)
            target["category"] = category_matches[0][1]
            target["_match_strength"] = 4
            target["_phrase_len"] += category_matches[0][0]

        candidates.append(target)

    # If a longer model phrase contains a shorter model phrase from the same brand,
    # keep the most specific family. This protects compound model names.
    candidates.sort(key=lambda x: (-x["_match_strength"], -x["_phrase_len"]))
    resolved = []
    seen = set()
    for item in candidates:
        key = (_normalize_vehicle_phrase(item["brand"]), _normalize_vehicle_phrase(item["model"]))
        if key in seen:
            continue
        # Skip a shorter same-brand model fully contained in an already-selected model.
        model_n = key[1]
        brand_n = key[0]
        if any(
            _normalize_vehicle_phrase(x["brand"]) == brand_n
            and model_n != _normalize_vehicle_phrase(x["model"])
            and f" {model_n} " in f" {_normalize_vehicle_phrase(x['model'])} "
            for x in resolved
        ):
            continue
        seen.add(key)
        item.pop("_match_strength", None)
        item.pop("_phrase_len", None)
        resolved.append(item)

    return resolved

def _search_market_for_vehicle_targets(base_filters, targets):
    """
    Search each explicit vehicle target independently, then merge the results.

    This is essential for comparisons where each target can have its own category,
    e.g. Toyota Aqua versus Nissan Note e-Power. A single global Category filter
    would incorrectly apply e-Power to both models.
    """
    merged = []
    seen_links = set()
    total = 0

    neutral = dict(base_filters or {})
    for key in (
        "brands", "exclude_brands",
        "models", "exclude_models",
        "categories", "exclude_categories",
    ):
        neutral.pop(key, None)

    for target in targets or []:
        target_filters = dict(neutral)
        target_filters["brands"] = [target["brand"]]
        target_filters["models"] = [target["model"]]
        if target.get("category"):
            target_filters["categories"] = [target["category"]]

        result = market_search(
            budget=target_filters.get("budget"),
            min_budget=target_filters.get("min_budget"),
            brands=target_filters.get("brands"),
            exclude_brands=target_filters.get("exclude_brands"),
            models=target_filters.get("models"),
            exclude_models=target_filters.get("exclude_models"),
            categories=target_filters.get("categories"),
            exclude_categories=target_filters.get("exclude_categories"),
            locations=target_filters.get("locations"),
            exclude_locations=target_filters.get("exclude_locations"),
            companies=target_filters.get("companies"),
            exclude_companies=target_filters.get("exclude_companies"),
            transmissions=target_filters.get("transmissions"),
            colors=target_filters.get("colors"),
            min_year=target_filters.get("min_year"),
            max_year=target_filters.get("max_year"),
            min_km=target_filters.get("min_km"),
            max_km=target_filters.get("max_km"),
            limit=5000,
            max_limit=5000,
        )

        if not result.get("success"):
            return result

        total += int(result.get("count", 0) or 0)

        for item in result.get("results", []) or []:
            link = str(item.get("link") or "").strip()
            dedupe_key = link or json.dumps(item, ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen_links:
                continue
            seen_links.add(dedupe_key)
            merged.append(item)

    merged.sort(
        key=lambda x: (
            float(x.get("price") or 10**12),
            -(int(x.get("year") or 0)),
        )
    )

    return {
        "success": True,
        "count": len(merged),
        "returned": len(merged),
        "results": merged,
        "target_raw_count": total,
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
    Never trust model/client filter values directly.

    Numeric filters must remain scalar numbers and list filters must remain
    arrays of strings. Malformed generative output such as {"max": 18000}
    is rejected here so it can never reach market_search / float().
    """
    if not isinstance(raw_filters, dict):
        return {}

    numeric_fields = {
        "budget", "min_budget", "min_year", "max_year", "min_km", "max_km",
    }
    integer_fields = {"min_year", "max_year"}
    list_fields = {
        "brands", "exclude_brands", "models", "exclude_models",
        "categories", "exclude_categories", "locations", "exclude_locations",
        "companies", "exclude_companies", "transmissions", "colors",
    }

    clean = {}

    for key, value in raw_filters.items():
        if key not in AI_FILTER_KEYS:
            continue

        if value in [None, "", [], {}]:
            continue

        if key in numeric_fields:
            if isinstance(value, bool) or isinstance(value, (dict, list, tuple, set)):
                print(f"IGNORING MALFORMED NUMERIC FILTER {key}: {value!r}", flush=True)
                continue

            try:
                number = float(value)
            except (TypeError, ValueError):
                print(f"IGNORING NON-NUMERIC FILTER {key}: {value!r}", flush=True)
                continue

            if not math.isfinite(number):
                continue

            clean[key] = int(number) if key in integer_fields else number
            continue

        if key in list_fields:
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, list):
                values = value
            else:
                print(f"IGNORING MALFORMED LIST FILTER {key}: {value!r}", flush=True)
                continue

            normalized = []
            seen = set()
            for item in values:
                if isinstance(item, (dict, list, tuple, set)):
                    continue
                text = str(item or "").strip()
                key_text = text.casefold()
                if text and key_text not in seen:
                    normalized.append(text)
                    seen.add(key_text)

            if key == "transmissions" and normalized:
                # market_base.csv stores transmission values in the source-market
                # vocabulary. Normalize multilingual/user-facing variants to those
                # canonical stored values before deterministic filtering.
                transmission_aliases = {
                    "automatic": "Otomatik", "auto": "Otomatik", "otomatik": "Otomatik",
                    "автомат": "Otomatik", "акпп": "Otomatik",
                    "manual": "Düz", "manuel": "Düz", "düz": "Düz", "duz": "Düz",
                    "механика": "Düz", "мкпп": "Düz",
                    "semi automatic": "Yarı Otomatik", "semi-automatic": "Yarı Otomatik",
                    "yarı otomatik": "Yarı Otomatik", "yari otomatik": "Yarı Otomatik",
                }
                mapped = []
                mapped_seen = set()
                for value in normalized:
                    canonical = transmission_aliases.get(value.casefold(), value)
                    k = canonical.casefold()
                    if k not in mapped_seen:
                        mapped.append(canonical)
                        mapped_seen.add(k)
                normalized = mapped

            if normalized:
                clean[key] = normalized

    return clean


def _parse_human_number(token):
    token = str(token or "").strip().lower().replace(" ", "").rstrip(".,;:")
    if not token:
        return None
    multiplier = 1
    if token.endswith("k"):
        multiplier = 1000
        token = token[:-1]
    # 15,000 / 15.000 are thousands; 15.5 is decimal when small.
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif token.count(",") == 1:
        left, right = token.split(",")
        token = left + right if len(right) == 3 else left + "." + right
    elif token.count(".") == 1:
        left, right = token.split(".")
        token = left + right if len(right) == 3 and len(left) >= 1 else left + "." + right
    try:
        return float(token) * multiplier
    except (TypeError, ValueError):
        return None


def fast_common_interpretation(message, resolved_targets=None):
    """
    Deterministic multilingual fast path for common buyer requests.
    Supports English, Turkish and Russian without a live LLM round-trip.
    Ambiguous conversational turns still fall back to the LLM interpreter.
    """
    raw = str(message or "").strip()
    low = raw.casefold()
    if not raw:
        return None

    # Terse contextual replies and explicit corrections/negations are intentionally
    # left to the conversational interpreter because they depend heavily on history.
    terse = {
        "yes", "no", "evet", "hayır", "да", "нет",
        "more", "more?", "daha", "daha?", "ещё", "еще",
        "all", "all of them", "hepsi", "все", "все варианты",
    }
    if low in terse:
        return None
    if re.search(r"\b(don't|do not|without|exclude|forget|istemiyorum|olmasın|hariç|istemem|unut|без|исключи|не хочу|забудь)\b", low):
        return None

    targets = list(resolved_targets or [])
    filters = {}
    preferences = []
    seller_mode = None

    # Decision mode — deliberately multilingual and conservative.
    shop_words = re.search(
        r"\b(listings?|ads?|advert(?:s|isements?)?|for sale|show me actual|"
        r"ilan(?:lar|ları|lari)?|satılık|satilik|göster|goster|"
        r"объявлен(?:ие|ия|ий)|покажи|показать|в продаже|прода(?:же|ются))\b",
        low,
    )
    compare_words = re.search(
        r"\b(compare|comparison|versus|vs\.?|karşılaştır|karsilastir|kıyasla|kiyasla|"
        r"сравни(?:ть|те)?|сравнение|против)\b",
        low,
    )
    listing_superlative = re.search(
        r"\b(lowest[- ]?mileage|lowest km|cheapest|lowest[- ]?priced|newest|"
        r"en düşük kilometreli|en dusuk kilometreli|en az kilometreli|en ucuz|en yeni|"
        r"с минимальным пробегом|сам(?:ый|ая|ое) дешев\w*|сам(?:ый|ая|ое) нов\w*)\b",
        low,
    )
    if shop_words or listing_superlative:
        decision_mode = "SHOP"
    elif compare_words or len(targets) >= 2:
        decision_mode = "COMPARE"
    else:
        decision_mode = "DISCOVER"

    # Budget forms: £15,000; 15k GBP; 15000 pounds; 15 bin; 15 тыс.
    budget_match = re.search(r"£\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)", raw)
    if not budget_match:
        budget_match = re.search(r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*£", raw)
    if not budget_match:
        budget_match = re.search(
            r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*(?:gbp|pounds?|sterling|sterlin|sterlinlik|"
            r"фунт(?:ов|а)?|стерлинг(?:ов|а)?)\b",
            low,
        )
    if not budget_match:
        bin_match = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*bin\b", low)
        if bin_match:
            amount = _parse_human_number(bin_match.group(1))
            if amount is not None:
                filters["budget"] = amount * 1000
        else:
            ru_thousand = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*(?:тыс|тысяч)\b", low)
            if ru_thousand:
                amount = _parse_human_number(ru_thousand.group(1))
                if amount is not None:
                    filters["budget"] = amount * 1000
    else:
        amount = _parse_human_number(budget_match.group(1))
        if amount is not None:
            filters["budget"] = amount

    # Common currency-less shorthand: "15k budget", "around 15k max",
    # "under 15k". Guard it with price/budget language so mileage such as
    # "under 60k km" is never mistaken for a purchase budget.
    if "budget" not in filters:
        shorthand_budget = re.search(
            r"(?:(?:budget|price|spend|cost|around|max(?:imum)?|under|below|up to|"
            r"bütçe|butce|bütçem|butcem|fiyat|до|бюджет)\D{0,18})"
            r"([0-9]+(?:[.,][0-9]+)?)\s*[kK]\b",
            low,
        )
        if shorthand_budget:
            tail = low[shorthand_budget.end():shorthand_budget.end() + 12]
            if not re.match(r"\s*(?:km|kilomet|км)", tail):
                amount = _parse_human_number(shorthand_budget.group(1))
                if amount is not None:
                    filters["budget"] = amount * 1000

    if "budget" not in filters:
        shorthand_budget_after = re.search(
            r"\b([0-9]+(?:[.,][0-9]+)?)\s*[kK]\s*"
            r"(?:budget|max(?:imum)?|to spend|spend|price|"
            r"bütçe|butce|bütçem|butcem|бюджет)\b",
            low,
        )
        if shorthand_budget_after:
            amount = _parse_human_number(shorthand_budget_after.group(1))
            if amount is not None:
                filters["budget"] = amount * 1000

    # Common mileage restrictions, including both prefix and suffix forms.
    km_patterns = [
        r"(?:under|below|max(?:imum)?|less than|no more than|altında|en fazla|maksimum|до|максимум|не более)\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*(?:km|kilomet(?:er|re)?|км)",
        r"([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*(?:km|kilomet(?:er|re)?|км)\s*(?:altında|altinda|ve altı|ve alti|or less|or below|maximum|максимум|или меньше)",
    ]
    for pattern in km_patterns:
        km_match = re.search(pattern, low)
        if km_match:
            amount = _parse_human_number(km_match.group(1))
            if amount is not None:
                filters["max_km"] = amount
            break

    # When the user has already established that the number is mileage,
    # accept shorthand such as "not crazy mileage, like max 60k".
    if "max_km" not in filters:
        mileage_context = re.search(
            r"\b(?:mileage|kilomet(?:er|re)s?|kilometre|kilometer|пробег)\b",
            low,
        )
        if mileage_context:
            # Only inspect text AFTER the mileage cue. This prevents a budget
            # such as "under 15k" earlier in the sentence from being reused as
            # max_km in "under 15k and not crazy mileage, like max 60k".
            mileage_tail = low[mileage_context.end():]
            implied_km = re.search(
                r"\b(?:max(?:imum)?|under|below|less than|up to|"
                r"en fazla|altında|altinda|до|максимум|не более)\s*"
                r"([0-9]+(?:[.,][0-9]+)?\s*[kK])\b",
                mileage_tail,
            )
            if implied_km:
                amount = _parse_human_number(implied_km.group(1))
                if amount is not None:
                    filters["max_km"] = amount

    # Common minimum-year wording in EN/TR/RU.
    year_patterns = [
        r"\bс\s*((?:19|20)\d{2})\s*(?:года|г\.?|и новее)?\b",
        r"\b((?:19|20)\d{2})\s*(?:onwards|or newer|and newer|ve üzeri|ve uzeri|ve sonrası|ve sonrasi|и новее|или новее)\b",
        r"(?:minimum|min(?:imum)? year|from|since|en az|minimum yıl|min yıl|от|не старше)\s*((?:19|20)\d{2})",
        r"((?:19|20)\d{2})\s*(?:model ve üstü|model ve ustu|modelden yeni|года и новее)",
    ]
    for pattern in year_patterns:
        year_match = re.search(pattern, low)
        if year_match:
            filters["min_year"] = int(year_match.group(1))
            break

    # A bare model year attached to a vehicle-class noun means that exact model
    # year (e.g. "2026 SUVs", "2024 cars"). Do not reinterpret this as a
    # minimum-year request.
    if "min_year" not in filters and "max_year" not in filters:
        exact_year = re.search(
            r"\b((?:19|20)\d{2})\s*(?:model\s*)?(?:cars?|automobiles?|suvs?|crossovers?|pick-?ups?|motorcycles?|scooters?|"
            r"arabalar?|otomobiller?|suv|motosikletler?|скутеры?|мотоциклы?|автомобили?)\b",
            low,
        )
        if not exact_year:
            exact_year = re.search(
                r"\b(?:cars?|automobiles?|suvs?|crossovers?|pick-?ups?|motorcycles?|scooters?|"
                r"arabalar?|otomobiller?|suv|motosikletler?|скутеры?|мотоциклы?|автомобили?)\s*(?:model\s*)?((?:19|20)\d{2})\b",
                low,
            )
        if exact_year:
            filters["min_year"] = int(exact_year.group(1))
            filters["max_year"] = int(exact_year.group(1))

    # Transmission.
    if re.search(r"\b(automatic|otomatik|автомат(?:ическ\w*)?|акпп)\b", low):
        filters["transmissions"] = ["Automatic"]
    elif re.search(r"\b(manual|manuel|механик(?:а|ическая)?|мкпп)\b", low):
        filters["transmissions"] = ["Manual"]

    # Seller type.
    if re.search(r"\b(private sellers?|private cars?|individual sellers?|bireysel|özel satıcı(?:lar)?|ozel satici(?:lar)?|частн(?:ый|ого|ые) продав(?:ец|цы)|частник(?:и)?)\b", low):
        seller_mode = "individual"
    elif re.search(r"\b(dealers?|dealerships?|galler(?:y|ies)|galeri(?:ler)?(?:den)?|дилер(?:ы)?|автосалон(?:ы)?)\b", low):
        seller_mode = "gallery"

    # Buyer-oriented soft preferences. These map to the precomputed profile layer,
    # so they remain deterministic and fast in all three supported languages.
    if re.search(
        r"\b(economical|economic|fuel efficient|fuel-efficient|economy|cheap to run|"
        r"ekonomik|az yakan|tasarruflu|düşük tüketim|dusuk tuketim|"
        r"экономич\w*|экономн\w*|низкий расход)\b",
        low,
    ):
        preferences.append("priority:economy")

    if re.search(r"\b(reliable|reliability|most reliable|güvenilir|guvenilir|dayanıklı|dayanikli|sorunsuz|надёжн\w*|надежн\w*)\b", low):
        preferences.append("priority:reliability")

    if re.search(r"\b(luxury|premium|luxurious|lüks|luks|премиальн\w*|роскошн\w*)\b", low):
        preferences.append("priority:luxury")

    if re.search(r"\b(comfortable|comfort|konforlu|konfor|комфортн\w*|комфорт)\b", low):
        preferences.append("priority:comfort")

    if re.search(r"\b(sporty|performance|sportif|performans|спортивн\w*|динамичн\w*|производительн\w*)\b", low):
        preferences.append("priority:performance")

    if re.search(r"\b(practical|practicality|pratik|kullanışlı|kullanisli|практичн\w*)\b", low):
        preferences.append("priority:practicality")

    if re.search(r"\b(family car|family vehicle|for my family|aile arabası|aile arabasi|aile için|aile icin|семейн\w* автомобил\w*|для семьи)\b", low):
        preferences.append("use_case:family")

    if re.search(r"\b(commute|commuting|daily commute|işe gidip gel|ise gidip gel|günlük kullanım|gunluk kullanim|для поездок на работу|на каждый день|ежедневн\w*)\b", low):
        preferences.append("use_case:commute")

    # Vehicle/body type. Note the natural Turkish/Russian variants that were
    # previously missed (e.g. "küçük bir araç", "небольшую машину").
    if re.search(
        r"\b(small(?:\s+[a-z-]+){0,2}\s+(?:cars?|vehicles?)|city cars?|compact cars?|"
        r"küçük(?:\s+bir)?(?:\s+[a-zçğıöşü-]+){0,2}\s+(?:araba|otomobil|araç)|şehir arabası|sehir arabasi|kompakt araba|"
        r"маленьк\w*(?:\s+[а-яё-]+){0,2}\s+(?:машин\w*|автомобил\w*)|небольш\w*(?:\s+[а-яё-]+){0,2}\s+(?:машин\w*|автомобил\w*)|"
        r"компактн\w*\s+(?:машин\w*|автомобил\w*)|городск\w*\s+автомобил\w*)\b",
        low,
    ):
        preferences.append("vehicle_type:small_car")
    elif re.search(r"\b(cars?|automobiles?|araba(?:lar)?|otomobil(?:ler)?|машин\w*|автомобил\w*)\b", low):
        preferences.append("vehicle_type:car")
    elif re.search(r"\b(suvs?|кроссовер(?:ы)?|внедорожник(?:и)?)\b", low):
        preferences.append("vehicle_type:SUV")
    elif re.search(r"\b(crossovers?|crossover cars?)\b", low):
        preferences.append("vehicle_type:crossover")
    elif re.search(r"\b(pick-?ups?|pickups?|kamyonet(?:ler)?|пикап(?:ы)?)\b", low):
        preferences.append("vehicle_type:pickup")
    elif re.search(r"\b(motorcycles?|motosiklet(?:ler)?|мотоцикл(?:ы)?)\b", low):
        preferences.append("vehicle_type:motorcycle")
    elif re.search(r"\b(scooters?|skuters?|скутер(?:ы)?)\b", low):
        preferences.append("vehicle_type:scooter")

    # Listing-level ordering requests are deterministic state, not subjective
    # recommendations. They are applied only after every hard market filter.
    if re.search(r"\b(lowest[- ]?mileage|lowest km|en düşük kilometreli|en dusuk kilometreli|en az kilometreli|с минимальным пробегом)\b", low):
        preferences.append("listing_sort:lowest_km")
    elif re.search(r"\b(cheapest|lowest[- ]?priced|en ucuz|сам(?:ый|ая|ое) дешев\w*)\b", low):
        preferences.append("listing_sort:cheapest")
    elif re.search(r"\b(newest|en yeni|сам(?:ый|ая|ое) нов\w*)\b", low):
        preferences.append("listing_sort:newest")

    # Named single targets can safely be canonicalized here; multi-target COMPARE
    # is handled independently by _search_market_for_vehicle_targets.
    if len(targets) == 1:
        filters["brands"] = [targets[0]["brand"]]
        filters["models"] = [targets[0]["model"]]
        if targets[0].get("category"):
            filters["categories"] = [targets[0]["category"]]

    # Stable de-duplication while preserving preference order.
    preferences = list(dict.fromkeys(preferences))

    recognized = bool(filters or preferences or targets or compare_words or shop_words or seller_mode)
    if not recognized:
        return None

    return {
        "filters": sanitize_ai_filters(filters),
        "clear_filters": [],
        "seller_mode": seller_mode,
        "preferences": _canonicalize_buyer_preferences(preferences),
        "needs_clarification": False,
        "clarification_question": None,
        "decision_mode": decision_mode,
        "fast_path": True,
    }

def interpret_market_query(message, current_filters=None, language="TR", conversation_history=None):
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
- Set "decision_mode" to exactly one of DISCOVER, COMPARE or SHOP.
  DISCOVER = broad vehicle/model discovery, recommendations, narrowing, "what can I buy?", or "more options?".
  COMPARE = comparing two or more models/brands, evaluating a chosen model in depth, or questions about resale/liquidity/price-pressure/trade-offs for models under discussion.
  SHOP = the buyer explicitly asks to see/find/show actual individual listings/ads/vehicles for sale, or asks about a specific advertised vehicle.
- Do NOT use SHOP merely because the buyer wants to buy a vehicle; SHOP requires listing-level intent.
- A bare model name that deepens the conversation is COMPARE, not SHOP.
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
- resolved_vehicle_mentions is deterministic application evidence from the live market.
  When it contains a named vehicle, use its canonical Brand/Model spelling rather than
  inventing a compound model name. Example: a resolved target may be
  Brand=Nissan, Model=Note, Category=e-Power even if the buyer wrote "Nissan Note e-Power".
- For COMPARE turns with multiple resolved_vehicle_mentions, do NOT try to express
  per-vehicle categories as one global categories filter; the application searches
  those targets independently.
- Do not translate brand/model names.
- Do not turn subjective ideas such as reliable, sporty, economical,
  family-friendly, small, luxurious, or good value into unsupported hard
  filters. Put those concepts in "preferences".
- IMPORTANT: vehicle classes are represented as canonical vehicle_type tags in "preferences"
  because market_base Category is variant-level, not a trustworthy body-type field. The application
  enforces explicit physical classes (car/SUV/crossover/pick-up/motorcycle/scooter) as STRICT
  constraints using the validated model-profile VehicleType/BodyStyle layer. "small car" remains
  a softer size/profile preference. Never place vehicle classes into Category unless that exact
  value is explicitly confirmed as a real market Category value.
- Use short canonical preference tags whenever possible so they persist cleanly across turns:
  vehicle_type:car, vehicle_type:SUV, vehicle_type:crossover, vehicle_type:pickup, vehicle_type:small_car,
  vehicle_type:motorcycle, use_case:commute, use_case:family, priority:economy,
  priority:reliability, priority:performance, priority:luxury, priority:comfort,
  priority:practicality.
- Treat words such as economical/economic/ekonomik/fuel-efficient/az yakan as priority:economy.
- Treat luxury/premium/lüks as priority:luxury.
- Treat comfortable/konforlu as priority:comfort and practical/pratik as priority:practicality.
- Treat sporty/sportif/performance/performance-oriented/performans as priority:performance.
- When the latest message changes vehicle type (for example SUV -> motorcycle or SUV -> small car), return only the NEW vehicle_type tag; the application will replace the previous vehicle_type preference.
- When the user says a previous vehicle type is no longer required (for example "SUV olmasına gerek yok"), add any_vehicle_type unless they also specify a replacement vehicle type in the same message.
- If the user explicitly says they have no brand/model preference, add "any_brand_model".
- If they explicitly say they have no year or mileage restriction, add "any_year_km".
- If they explicitly say any vehicle type is fine, add "any_vehicle_type".
- "preferences" is for useful soft intent that the deterministic filters
  cannot represent yet.
- "needs_clarification" should only be true when the latest message itself is genuinely
  ambiguous (for example an unclear number/unit or an unclear brand/model reference).
- A broad request such as "recommend me a car" is NOT an ambiguity. Set
  needs_clarification=false and let the application's guided narrowing flow handle it.
- Do not use clarification_question merely to ask for budget, year, vehicle type, brand,
  model or mileage because the guided narrowing flow handles those choices.
- "clarification_question" should be short and in the user's language.
- Use recent_conversation ONLY to resolve terse follow-ups/corrections such as "yes", "all of them",
  "more?", "Mercedes?", "No Mercedes", "GLA", "cheaper" or "newer". Do not re-create old hard
  filters from prose history; current_filters is the authoritative persisted hard state.
- A short bare brand/model availability challenge such as "Mercedes?", "No Mercedes", "no BMW?",
  "Mercedes yok mu?" means "are there any?" unless the user clearly expresses exclusion intent.
  Treat it as an INCLUDE/query for that brand/model, preserving the other active criteria.
- Only exclude a brand when intent is explicit, e.g. "I don't want Mercedes", "exclude Mercedes",
  "without Mercedes", "Mercedes istemiyorum", "Mercedes olmasın", "Mercedes hariç".
- If the user corrects a misunderstanding ("No, I meant is there not Mercedes"), prefer the corrected
  availability intent and clear any contradictory brand exclusion introduced by the prior turn.
"""

    resolved_vehicle_mentions = resolve_market_vehicle_mentions(message)

    user_payload = {
        "language": language,
        "latest_message": message,
        "current_filters": current_filters,
        "recent_conversation": sanitize_conversation_history(conversation_history, max_messages=10),
        "market_context": market_context,
        "resolved_vehicle_mentions": resolved_vehicle_mentions,
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
            "decision_mode": {
                "type": "string",
                "enum": ["DISCOVER", "COMPARE", "SHOP"],
            },
        },
        "required": [
            "filters",
            "clear_filters",
            "seller_mode",
            "preferences",
            "needs_clarification",
            "clarification_question",
            "decision_mode",
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
            "reasoning": {"effort": "none"},
            "max_output_tokens": 700,
            "instructions": instructions,
            "input": json.dumps(user_payload, ensure_ascii=False),
        },
        timeout=(1.0, 4.0),
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

    decision_mode = str(interpreted.get("decision_mode") or "DISCOVER").upper()
    if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
        decision_mode = "DISCOVER"
    interpreted["decision_mode"] = decision_mode

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
    limit=20,
    max_limit=100
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
        wanted = set(normalize_list(transmissions))
        filtered = filtered[
            filtered["Transmission"]
            .fillna("")
            .astype(str)
            .map(normalize)
            .isin(wanted)
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

    try:
        max_limit = int(max_limit)
    except:
        max_limit = 100

    max_limit = max(1, min(max_limit, 5000))

    limit = max(
        1,
        min(limit, max_limit)
    )

    results_df = filtered.head(limit).copy()

    # -----------------------
    # JSON-SAFE RESULTS
    # -----------------------

    # Vectorized serialization is materially faster than iterrows() when the
    # assistant asks for thousands of internal candidates.
    raw_records = results_df[[
        "Brand", "Model", "Category", "Year", "Price", "KM",
        "Company", "Location", "Transmission", "Color", "Image", "Link"
    ]].to_dict(orient="records")

    results = [
        {
            "brand": row["Brand"],
            "model": row["Model"],
            "category": row["Category"],
            "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "price": float(row["Price"]) if pd.notna(row["Price"]) else None,
            "km": int(row["KM"]) if pd.notna(row["KM"]) else None,
            "company": row["Company"],
            "location": row["Location"],
            "transmission": row["Transmission"],
            "color": row["Color"],
            "image": row["Image"],
            "link": row["Link"],
        }
        for row in raw_records
    ]

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
    Keep soft preferences across turns, but treat vehicle type as a replaceable
    conversational state rather than an accumulating tag.
    """
    previous = [str(v).strip() for v in (previous_preferences or []) if str(v).strip()]
    incoming = [str(v).strip() for v in (new_preferences or []) if str(v).strip()]
    incoming_cf = [v.casefold() for v in incoming]

    incoming_vehicle_types = [
        v for v in incoming if v.casefold().startswith("vehicle_type:")
    ]
    incoming_any_vehicle = "any_vehicle_type" in incoming_cf
    incoming_listing_sorts = [
        v for v in incoming if v.casefold().startswith("listing_sort:")
    ]

    if incoming_listing_sorts:
        previous = [v for v in previous if not v.casefold().startswith("listing_sort:")]

    # A new explicit vehicle type replaces the old one. "Any vehicle type"
    # clears all old vehicle-type restrictions.
    if incoming_vehicle_types or incoming_any_vehicle:
        previous = [
            v for v in previous
            if not v.casefold().startswith("vehicle_type:")
            and v.casefold() != "any_vehicle_type"
        ]

    # If a concrete new type is supplied, do not retain any_vehicle_type.
    if incoming_vehicle_types:
        incoming = [v for v in incoming if v.casefold() != "any_vehicle_type"]

    merged = []
    seen = set()
    for value in previous + incoming:
        key = value.casefold()
        if key not in seen:
            merged.append(value)
            seen.add(key)

    return merged


def _preference_flags(preferences):
    prefs = {str(p).strip().casefold() for p in (preferences or []) if str(p).strip()}
    return {
        "has_vehicle_type": any(p.startswith("vehicle_type:") for p in prefs) or "any_vehicle_type" in prefs,
        "has_use_case": any(p.startswith("use_case:") for p in prefs),
        "any_brand_model": "any_brand_model" in prefs,
        "any_year_km": "any_year_km" in prefs,
    }


def _strict_vehicle_type_from_preferences(preferences):
    """Return the explicit physical vehicle class, if one is active."""
    for pref in reversed(_canonicalize_buyer_preferences(preferences)):
        p = str(pref or "").strip().casefold()
        if not p.startswith("vehicle_type:"):
            continue
        requested = p.split(":", 1)[1]
        if requested == "small_car":
            return "car"
        if requested in {"car", "suv", "crossover", "pickup", "pick-up", "motorcycle", "motosiklet", "scooter"}:
            return requested
    return None


def _apply_strict_vehicle_type_to_search_result(search_result, preferences):
    """Enforce explicit physical vehicle classes with the validated model profiles.

    This is deliberately separate from market_base Category: Category is variant-level
    and must never be repurposed/inferred as a body type.
    """
    requested = _strict_vehicle_type_from_preferences(preferences)
    if not requested or not search_result.get("success"):
        return search_result
    if not MODEL_PROFILE_READY or not MODEL_PROFILE_LOOKUP:
        # Fail closed for class-sensitive discovery rather than leaking boats/bikes
        # into a car request when the taxonomy layer is unavailable.
        result = dict(search_result)
        result["count"] = 0
        result["returned"] = 0
        result["results"] = []
        result["vehicle_type_filter_unavailable"] = True
        return result

    kept = []
    for item in search_result.get("results", []) or []:
        key = (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold())
        profile = MODEL_PROFILE_LOOKUP.get(key)
        if profile and _profile_matches_vehicle_type(profile, requested):
            kept.append(item)

    result = dict(search_result)
    result["count"] = len(kept)
    result["returned"] = len(kept)
    result["results"] = kept
    result["strict_vehicle_type"] = requested
    return result


def _asks_reliability_question(message):
    low = str(message or "").casefold()
    return bool(re.search(r"\b(reliable|reliability|most reliable|güvenilir|guvenilir|dayanıklı|dayanikli|sorunsuz|надёжн\w*|надежн\w*)\b", low))


def _reliability_scope_answer(language, filters):
    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None
    if language == "TR":
        scope = f" {budget_text} bütçeniz içindeki" if budget_text else ""
        return (
            "Güvenilirliği yalnızca Kuzey Kıbrıs ilan verilerinden güvenilir biçimde belirleyemem; "
            "bunun için uzun dönem güvenilirlik, arıza/servis ve kullanıcı verileri gibi dış kaynaklar gerekir. "
            f"Yine de{scope} seçenekleri güncel fiyat, yaş, kilometre ve gözlenen yeniden satış piyasası davranışına göre karşılaştırabilirim; "
            "dış güvenilirlik kanıtı olmadan bir modeli ‘en güvenilir’ diye etiketlemem."
        )
    if language == "RU":
        scope = f" в пределах бюджета {budget_text}" if budget_text else ""
        return (
            "Надёжность нельзя достоверно определить только по объявлениям Северного Кипра: для этого нужны внешние данные "
            "о долгосрочной надёжности, ремонтах/сервисе и опыте владельцев. "
            f"Я могу сравнить варианты{scope} по текущей цене, возрасту, пробегу и наблюдаемому поведению на рынке перепродажи, "
            "но не буду называть модель «самой надёжной» без таких внешних доказательств."
        )
    scope = f" within your {budget_text} budget" if budget_text else ""
    return (
        "Reliability isn't something I can determine reliably from North Cyprus listing data alone; "
        "it requires external evidence such as long-term reliability, repair/service and owner data. "
        f"I can still compare the options{scope} by current price, age, mileage and observed resale-market behaviour, "
        "but I won't label one model ‘most reliable’ without that external evidence."
    )


def _listing_sort_mode(preferences):
    for pref in reversed(preferences or []):
        p = str(pref or "").strip().casefold()
        if p == "listing_sort:lowest_km":
            return "lowest_km"
        if p == "listing_sort:cheapest":
            return "cheapest"
        if p == "listing_sort:newest":
            return "newest"
    return None


def guided_narrowing_question(filters, preferences, count, language="TR"):
    """Only guide when the request is genuinely too broad to be useful yet."""
    flags = _preference_flags(preferences)

    has_budget = filters.get("budget") not in [None, ""]
    has_year = bool(
        filters.get("min_year") not in [None, ""] or
        filters.get("max_year") not in [None, ""] or
        flags["any_year_km"]
    )
    has_vehicle_type = flags["has_vehicle_type"]
    has_brand_model = bool(
        filters.get("brands") or filters.get("models") or
        filters.get("exclude_brands") or filters.get("exclude_models") or
        flags["any_brand_model"]
    )
    has_other_hard_constraint = bool(
        filters.get("max_km") not in [None, ""] or
        filters.get("min_km") not in [None, ""] or
        filters.get("transmissions") or
        filters.get("locations") or
        filters.get("companies") or
        filters.get("exclude_companies")
    )

    # Once the buyer has provided two meaningful dimensions, start helping with
    # real market options instead of automatically asking another question.
    # A genuine buying preference (economy, luxury, practicality, family use, etc.)
    # counts as a dimension too: "£15k economical" is already useful enough.
    has_soft_priority = any(
        str(p).strip().casefold().startswith(("priority:", "use_case:"))
        for p in (preferences or [])
    )
    supplied = sum([
        bool(has_budget),
        bool(has_year),
        bool(has_vehicle_type),
        bool(has_brand_model),
        bool(has_other_hard_constraint),
        bool(has_soft_priority),
    ])
    if supplied >= 2:
        return None

    copy = {
        "TR": {
            "intro": "Size daha isabetli öneriler sunabilmem için aramanızı biraz daraltmanızı öneririm.",
            "budget": "maksimum bütçenizi",
            "year": "minimum model yılı beklentinizi",
            "type": "araç tipini (ör. SUV, pick-up, otomobil veya motosiklet)",
        },
        "EN": {
            "intro": "To give you more relevant recommendations, I'd suggest narrowing the search a little.",
            "budget": "your maximum budget",
            "year": "your minimum year requirement",
            "type": "vehicle type (e.g. SUV, pickup, car or motorcycle)",
        },
        "RU": {
            "intro": "Чтобы дать более точные рекомендации, я бы предложил немного сузить поиск.",
            "budget": "максимальный бюджет",
            "year": "минимальный год выпуска",
            "type": "тип транспорта (например SUV, пикап, автомобиль или мотоцикл)",
        },
    }
    t = copy.get(language, copy["TR"])

    missing = []
    if not has_budget:
        missing.append(t["budget"])
    if not has_year:
        missing.append(t["year"])
    if not has_vehicle_type:
        missing.append(t["type"])

    if not missing:
        return None

    if language == "TR":
        details = ", ".join(missing)
        details = details[:1].upper() + details[1:] if details else details
        return f'{t["intro"]} ' + details + " belirtebilirsiniz; bunlardan biri veya birkaçı yeterli olabilir."
    if language == "RU":
        return f'{t["intro"]} ' + ", ".join(missing) + ". Можно указать один или несколько из этих параметров."
    return f'{t["intro"]} You can add ' + ", ".join(missing) + "; one or more of these may be enough."


def _group_market_models(results, max_groups=350):
    """Aggregate real matching listings into factual brand/model market options."""
    grouped = {}

    for item in results or []:
        brand = str(item.get("brand") or "").strip()
        model = str(item.get("model") or "").strip()
        if not brand or not model:
            continue

        key = (brand.casefold(), model.casefold())
        bucket = grouped.setdefault(key, {
            "brand": brand,
            "model": model,
            "count": 0,
            "prices": [],
            "years": [],
            "year_prices": {},
            "kms": [],
            "transmissions": set(),
            "locations": set(),
        })
        bucket["count"] += 1

        parsed_price = None
        parsed_year = None

        try:
            if item.get("price") is not None:
                parsed_price = float(item["price"])
                bucket["prices"].append(parsed_price)
        except (TypeError, ValueError):
            parsed_price = None

        try:
            if item.get("year") is not None:
                parsed_year = int(item["year"])
                bucket["years"].append(parsed_year)
        except (TypeError, ValueError):
            parsed_year = None

        if parsed_price is not None and parsed_year is not None:
            bucket["year_prices"].setdefault(parsed_year, []).append(parsed_price)
        try:
            if item.get("km") is not None:
                bucket["kms"].append(int(item["km"]))
        except (TypeError, ValueError):
            pass

        transmission = str(item.get("transmission") or "").strip()
        location = str(item.get("location") or "").strip()
        if transmission:
            bucket["transmissions"].add(transmission)
        if location:
            bucket["locations"].add(location)

    summaries = []
    for bucket in grouped.values():
        prices = bucket.pop("prices")
        years = bucket.pop("years")
        year_prices = bucket.pop("year_prices")
        kms = bucket.pop("kms")
        bucket["transmissions"] = sorted(bucket["transmissions"])
        bucket["locations"] = sorted(bucket["locations"])
        bucket["starting_price"] = min(prices) if prices else None
        bucket["highest_price"] = max(prices) if prices else None
        bucket["oldest_year"] = min(years) if years else None
        bucket["newest_year"] = max(years) if years else None

        newest_year = bucket["newest_year"]
        newest_prices = year_prices.get(newest_year, []) if newest_year is not None else []
        bucket["newest_year_starting_price"] = min(newest_prices) if newest_prices else None
        bucket["newest_year_highest_price"] = max(newest_prices) if newest_prices else None
        bucket["newest_year_count"] = len(newest_prices)

        if prices and years and year_prices:
            overall_start = bucket["starting_price"]
            cheapest_years = [
                year for year, vals in year_prices.items()
                if vals and min(vals) == overall_start
            ]
            bucket["starting_price_year"] = max(cheapest_years) if cheapest_years else None
        else:
            bucket["starting_price_year"] = None

        bucket["lowest_km"] = min(kms) if kms else None
        bucket["highest_km"] = max(kms) if kms else None
        summaries.append(bucket)

    summaries.sort(
        key=lambda x: (
            -int(x.get("count") or 0),
            -(int(x.get("newest_year") or 0)),
            float(x.get("starting_price") or 10**12),
        )
    )
    return summaries[:max_groups]


_MODEL_QUALIFICATION_CACHE = {}
_MODEL_QUALIFICATION_CACHE_MAX = 200


def _qualification_cache_key(filters, preferences, model_market):
    # Includes the actual market families/aggregates, so a refreshed market naturally
    # produces a different key without coupling this cache to the valuation dataset.
    market_signature = tuple(
        (
            str(m.get("brand") or "").casefold(),
            str(m.get("model") or "").casefold(),
            int(m.get("count") or 0),
            int(m.get("newest_year") or 0),
            float(m.get("newest_year_starting_price") or 0),
            float(m.get("starting_price") or 0),
        )
        for m in model_market
    )
    relevant_prefs = tuple(sorted(
        str(p).strip().casefold()
        for p in (preferences or [])
        if str(p).strip().casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
    ))
    hard_signature = json.dumps(filters or {}, ensure_ascii=False, sort_keys=True, default=str)
    return (hard_signature, relevant_prefs, market_signature)


def shortlist_models_for_preferences(message, language, filters, preferences, results):
    """Qualify/rank real model families against soft intent.

    Production path is deterministic via buyer_model_profiles.csv. A bounded LLM
    fallback exists only while profiles are unavailable/incomplete and is never
    allowed to hold a live request for tens of seconds.
    """
    preferences = _canonicalize_buyer_preferences(preferences)
    relevant_preferences = [
        p for p in preferences
        if str(p).casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
    ]

    model_market = _group_market_models(results)
    if not model_market:
        return [], [], []
    if not relevant_preferences:
        return list(results or []), [], model_market

    deterministic = _deterministic_profile_shortlist(model_market, relevant_preferences, max_models=30)
    if deterministic:
        selected_keys = {(m["brand"].casefold(), m["model"].casefold()) for m in deterministic}
        qualified = [
            item for item in (results or [])
            if (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold()) in selected_keys
        ]
        reasons = [{"brand": m["brand"], "model": m["model"], "reason": "profile_match"} for m in deterministic]
        return qualified, reasons, deterministic

    # If the production profile catalogue is loaded but nothing matches, respect that
    # result rather than asking a live generative model to override stable taxonomy.
    if MODEL_PROFILE_READY:
        return [], [], []

    # Temporary resilience path for deployments before buyer_model_profiles.csv exists.
    # Keep the latency budget tight; if the external model is slow, return the hard-filtered
    # market immediately rather than making the product appear broken.
    cache_key = _qualification_cache_key(filters, preferences, model_market)
    cached = _MODEL_QUALIFICATION_CACHE.get(cache_key)
    if cached is not None:
        selected_keys, reasons, selected_summaries = cached
        selected_set = set(selected_keys)
        qualified = [item for item in (results or []) if (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold()) in selected_set]
        return qualified, list(reasons), list(selected_summaries)

    qualification_market = model_market[:120]
    instructions = """
Return JSON only: {"models":[{"brand":"...","model":"...","reason":"..."}]}.
Select only exact supplied model candidates matching the soft preferences.
Vehicle type is mandatory. Economy/luxury/comfort/performance/practicality/family/commute are broad model-level positioning only.
Never infer listing condition, reliability or value retention. Select up to 20.
"""
    payload = {
        "soft_preferences": relevant_preferences,
        "model_candidates": [{"brand": m["brand"], "model": m["model"]} for m in qualification_market],
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL, "reasoning": {"effort": "none"},
                "max_output_tokens": 450, "instructions": instructions,
                "input": json.dumps(payload, ensure_ascii=False),
            },
            timeout=(1.0, 3.0),
        )
        response.raise_for_status()
        parsed = json.loads(extract_response_text(response.json()).strip())
    except Exception as exc:
        print(f"MODEL_QUALIFICATION_FAST_FALLBACK: {exc}", flush=True)
        # Hard-filtered candidates are safer than a long timeout. The response remains
        # grounded; it is simply less preference-specific until profiles are deployed.
        return list(results or []), [], model_market[:20]

    market_by_key = {(m["brand"].casefold(), m["model"].casefold()): m for m in qualification_market}
    selected_keys, reasons, selected_summaries = [], [], []
    for item in parsed.get("models", []) if isinstance(parsed, dict) else []:
        if not isinstance(item, dict):
            continue
        brand, model = str(item.get("brand") or "").strip(), str(item.get("model") or "").strip()
        key = (brand.casefold(), model.casefold())
        if key in market_by_key and key not in selected_keys:
            selected_keys.append(key); selected_summaries.append(market_by_key[key])
            reasons.append({"brand": brand, "model": model, "reason": str(item.get("reason") or "").strip()})
        if len(selected_keys) >= 20:
            break

    if not selected_keys:
        return list(results or []), [], model_market[:20]
    selected_set = set(selected_keys)
    qualified = [item for item in (results or []) if (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold()) in selected_set]
    if len(_MODEL_QUALIFICATION_CACHE) >= _MODEL_QUALIFICATION_CACHE_MAX:
        _MODEL_QUALIFICATION_CACHE.pop(next(iter(_MODEL_QUALIFICATION_CACHE)))
    _MODEL_QUALIFICATION_CACHE[cache_key] = (tuple(selected_keys), tuple(reasons), tuple(selected_summaries))
    return qualified, reasons, selected_summaries


def _select_model_options(model_summaries, reasons, filters, max_options=8):
    """Build a diverse factual option set; AI will normally surface only 3-5."""
    if not model_summaries:
        return []

    reason_map = {
        (str(x.get("brand") or "").casefold(), str(x.get("model") or "").casefold()): x.get("reason", "")
        for x in (reasons or [])
    }

    enriched = []
    for summary in model_summaries:
        item = dict(summary)
        item["reason"] = reason_map.get(
            (item["brand"].casefold(), item["model"].casefold()),
            "",
        )
        enriched.append(item)

    # AI qualifies soft concepts; deterministic Python ordering decides what is surfaced.
    # This makes identical market/filter state stable and favours newer options rather than
    # allowing model ordering to vary from one generation call to another.
    if reasons:
        budget = filters.get("budget")
        try:
            ceiling = float(budget) if budget not in [None, ""] else None
        except (TypeError, ValueError):
            ceiling = None

        def qualified_rank(x):
            profile_score = int(x.get("_profile_score") or 0)
            newest = int(x.get("newest_year") or 0)
            # For budgeted discovery, compare the price of the newest reachable year,
            # not the cheapest old example in the model family.
            anchor_price = x.get("newest_year_starting_price")
            if anchor_price in [None, ""]:
                anchor_price = x.get("starting_price")
            try:
                anchor_price = float(anchor_price)
            except (TypeError, ValueError):
                anchor_price = 10**12
            budget_distance = abs(ceiling - anchor_price) if ceiling is not None else anchor_price
            return (-profile_score, -newest, budget_distance, -min(int(x.get("count") or 0), 100), x["brand"].casefold(), x["model"].casefold())

        enriched.sort(key=qualified_rank)
        return enriched[:max_options]

    # Without a soft preference, provide a useful spread: represented, newest, and near budget.
    chosen, seen = [], set()

    def add(items):
        for item in items:
            key = (item["brand"].casefold(), item["model"].casefold())
            if key not in seen:
                chosen.append(item)
                seen.add(key)
            if len(chosen) >= max_options:
                return

    add(sorted(enriched, key=lambda x: (-int(x.get("count") or 0), -(int(x.get("newest_year") or 0))))[:3])
    add(sorted(enriched, key=lambda x: (-(int(x.get("newest_year") or 0)), float(x.get("starting_price") or 10**12)))[:3])

    budget = filters.get("budget")
    if budget not in [None, ""]:
        try:
            ceiling = float(budget)
            add(sorted(
                enriched,
                key=lambda x: abs(ceiling - float(x.get("starting_price") or 0))
            )[:4])
        except (TypeError, ValueError):
            pass

    add(enriched)
    return chosen[:max_options]


def select_assistant_candidates(results, filters, max_candidates=3):
    """Choose factual listing rows only for explicit listing-level follow-up."""
    if not results:
        return []

    clean = list(results)
    budget = filters.get("budget")
    chosen, seen = [], set()

    def add(items):
        for item in items:
            identity = item.get("link") or (
                item.get("brand"), item.get("model"), item.get("year"), item.get("price")
            )
            if identity not in seen:
                chosen.append(item)
                seen.add(identity)
            if len(chosen) >= max_candidates:
                return

    if budget not in [None, ""]:
        try:
            ceiling = float(budget)
            add(sorted(clean, key=lambda x: abs(ceiling - float(x.get("price") or 0)))[:8])
        except (TypeError, ValueError):
            pass

    add(sorted(clean, key=lambda x: (-(int(x.get("year") or 0)), float(x.get("price") or 0)))[:8])
    add(sorted(
        [x for x in clean if x.get("km") is not None],
        key=lambda x: (int(x.get("km") or 0), -int(x.get("year") or 0)),
    )[:8])
    add(sorted(clean, key=lambda x: float(x.get("price") or 0)))
    return chosen[:max_candidates]



def _json_number(value, integer=False):
    if value is None or pd.isna(value):
        return None
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


def _vehicle_type_preference(preferences):
    for pref in preferences or []:
        value = str(pref or "").strip()
        if value.casefold().startswith("vehicle_type:"):
            return value.split(":", 1)[1].strip()
    return None


def _buyer_vehicle_type_matches(series, requested_type):
    """
    Deterministic only where Category Master gives us a defensible mapping.
    Ambiguous concepts such as small_car/crossover remain AI qualifications.
    """
    if not requested_type:
        return pd.Series(True, index=series.index)

    requested = str(requested_type).strip().casefold()
    normalized = series.fillna("").astype(str).str.strip().str.casefold()

    exact_map = {
        # SUV / pickup / crossover / small-car are model-profile/body-class
        # qualifications. They have already been enforced before enrichment,
        # so do not re-filter Buyer Intelligence using Category Master's broader
        # VehicleType field here.
        "motorcycle": {"motosiklet"},
        "motosiklet": {"motosiklet"},
        "atv": {"atv & utv"},
        "utv": {"atv & utv"},
        "classic": {"klasik araçlar"},
        "boat": {"deniz araçları"},
    }

    allowed = exact_map.get(requested)
    if not allowed:
        return pd.Series(True, index=series.index)

    return normalized.isin(allowed)


def _matching_buyer_model_rows(model_option, filters, preferences):
    if (
        not BUYER_INTELLIGENCE_READY
        or buyer_model_df is None
        or buyer_model_df.empty
    ):
        return pd.DataFrame()

    brand = str(model_option.get("brand") or "").strip().casefold()
    model = str(model_option.get("model") or "").strip().casefold()

    rows = buyer_model_df[
        (buyer_model_df["Brand"].astype(str).str.casefold() == brand)
        & (buyer_model_df["Model"].astype(str).str.casefold() == model)
    ].copy()

    if rows.empty:
        return rows

    min_year = filters.get("min_year")
    max_year = filters.get("max_year")
    budget = filters.get("budget")
    min_budget = filters.get("min_budget")

    if min_year not in [None, ""]:
        rows = rows[rows["Year"] >= int(min_year)]
    if max_year not in [None, ""]:
        rows = rows[rows["Year"] <= int(max_year)]
    if budget not in [None, ""]:
        rows = rows[
            rows["CurrentStartingPrice"].notna()
            & (rows["CurrentStartingPrice"] <= float(budget))
        ]
    if min_budget not in [None, ""]:
        # Keep a year if its market range can reach the buyer's floor.
        rows = rows[
            rows["CurrentHighestPrice"].notna()
            & (rows["CurrentHighestPrice"] >= float(min_budget))
        ]

    requested_type = _vehicle_type_preference(preferences)
    if requested_type and not rows.empty:
        rows = rows[
            _buyer_vehicle_type_matches(
                rows["VehicleType"], requested_type
            )
        ]

    return rows


def _matching_buyer_category_rows(model_option, filters, preferences, hard_results):
    if (
        not BUYER_INTELLIGENCE_READY
        or buyer_category_df is None
        or buyer_category_df.empty
    ):
        return pd.DataFrame()

    brand = str(model_option.get("brand") or "").strip()
    model = str(model_option.get("model") or "").strip()
    brand_cf = brand.casefold()
    model_cf = model.casefold()

    # Restrict variants to CategoryDetail strings actually represented in the
    # current hard-filtered listing result set for this model.
    observed_categories = {
        str(x.get("category") or "").strip().casefold()
        for x in (hard_results or [])
        if str(x.get("brand") or "").strip().casefold() == brand_cf
        and str(x.get("model") or "").strip().casefold() == model_cf
        and str(x.get("category") or "").strip()
    }

    rows = buyer_category_df[
        (buyer_category_df["Brand"].astype(str).str.casefold() == brand_cf)
        & (buyer_category_df["Model"].astype(str).str.casefold() == model_cf)
    ].copy()

    if rows.empty:
        return rows

    if observed_categories:
        detail_cf = rows["CategoryDetail"].astype(str).str.casefold()
        mask = detail_cf.map(
            lambda detail: any(
                detail and detail in observed
                for observed in observed_categories
            )
        )
        rows = rows[mask]

    min_year = filters.get("min_year")
    max_year = filters.get("max_year")
    budget = filters.get("budget")

    if min_year not in [None, ""]:
        rows = rows[rows["Year"] >= int(min_year)]
    if max_year not in [None, ""]:
        rows = rows[rows["Year"] <= int(max_year)]
    if budget not in [None, ""]:
        rows = rows[
            rows["CurrentStartingPrice"].notna()
            & (rows["CurrentStartingPrice"] <= float(budget))
        ]

    requested_type = _vehicle_type_preference(preferences)
    if requested_type and not rows.empty:
        rows = rows[
            _buyer_vehicle_type_matches(
                rows["VehicleType"], requested_type
            )
        ]

    return rows


def enrich_model_options_with_buyer_intelligence(
    model_options,
    filters,
    preferences,
    hard_results,
):
    """
    Attach proprietary Cyprus-market evidence to the real model options.

    This does NOT calculate a universal score. It exposes transparent current
    market, liquidity and asking-price-pressure evidence for the response model.
    """
    if not model_options:
        return []

    enriched = []

    for option in model_options:
        item = dict(option)
        model_rows = _matching_buyer_model_rows(
            item, filters, preferences
        )

        if not model_rows.empty:
            model_rows = model_rows.sort_values(
                "Year", ascending=False
            )
            newest = model_rows.iloc[0]

            # Historical evidence repeats across model-year rows, so take the
            # newest matching row as the carrier of those model-level fields.
            item["buyer_intelligence"] = {
                "available_years": sorted(
                    [
                        int(x) for x in model_rows["Year"].dropna().unique()
                    ],
                    reverse=True,
                ),
                "newest_affordable_year": _json_number(
                    model_rows["Year"].max(), integer=True
                ),
                "newest_affordable_year_starting_price": _json_number(
                    newest.get("CurrentStartingPrice")
                ),
                "newest_affordable_year_median_price": _json_number(
                    newest.get("CurrentMedianPrice")
                ),
                "newest_affordable_year_median_km": _json_number(
                    newest.get("CurrentMedianKM")
                ),
                "historical_distinct_listings": _json_number(
                    newest.get("Buyer_HistoricalDistinctListings"),
                    integer=True,
                ),
                "median_observed_days_to_exit": _json_number(
                    newest.get("Buyer_MedianObservedDaysToExit")
                ),
                "exit_30_rate": _json_number(
                    newest.get("Buyer_ObservedExitWithin30DaysRate")
                ),
                "exit_30_eligible": _json_number(
                    newest.get("Buyer_Exit30EligibleListings"),
                    integer=True,
                ),
                "exit_60_rate": _json_number(
                    newest.get("Buyer_ObservedExitWithin60DaysRate")
                ),
                "exit_60_eligible": _json_number(
                    newest.get("Buyer_Exit60EligibleListings"),
                    integer=True,
                ),
                "exit_90_rate": _json_number(
                    newest.get("Buyer_ObservedExitWithin90DaysRate")
                ),
                "exit_90_eligible": _json_number(
                    newest.get("Buyer_Exit90EligibleListings"),
                    integer=True,
                ),
                "liquidity_confidence": str(
                    newest.get("Buyer_LiquidityEvidenceConfidence") or ""
                ),
                "liquidity_evidence_level": str(
                    newest.get("LiquidityEvidenceLevel") or ""
                ),
                "price_reduction_rate": _json_number(
                    newest.get("Buyer_PriceReductionRate")
                ),
                "price_pressure_eligible": _json_number(
                    newest.get("Buyer_PricePressureEligibleListings"),
                    integer=True,
                ),
                "median_reduction_pct_when_reduced": _json_number(
                    newest.get("Buyer_MedianReductionPctAmongReduced")
                ),
                "price_pressure_confidence": str(
                    newest.get("Buyer_PricePressureEvidenceConfidence") or ""
                ),
                "price_pressure_evidence_level": str(
                    newest.get("PricePressureEvidenceLevel") or ""
                ),
            }
        else:
            item["buyer_intelligence"] = None

        category_rows = _matching_buyer_category_rows(
            item, filters, preferences, hard_results
        )
        variant_context = []
        if not category_rows.empty:
            # Prefer variants with the most current supply, then newest year.
            category_rows = category_rows.sort_values(
                ["CurrentListings", "Year"],
                ascending=[False, False],
            )
            seen = set()
            for _, row in category_rows.iterrows():
                detail = str(row.get("CategoryDetail") or "").strip()
                key = detail.casefold()
                if not detail or key in seen:
                    continue
                seen.add(key)
                variant_context.append({
                    "category_detail": detail,
                    "year": _json_number(row.get("Year"), integer=True),
                    "current_listings": _json_number(
                        row.get("CurrentListings"), integer=True
                    ),
                    "starting_price": _json_number(
                        row.get("CurrentStartingPrice")
                    ),
                    "median_price": _json_number(
                        row.get("CurrentMedianPrice")
                    ),
                    "median_km": _json_number(
                        row.get("CurrentMedianKM")
                    ),
                    "exit_60_rate": _json_number(
                        row.get("Buyer_ObservedExitWithin60DaysRate")
                    ),
                    "exit_60_eligible": _json_number(
                        row.get("Buyer_Exit60EligibleListings"),
                        integer=True,
                    ),
                    "liquidity_confidence": str(
                        row.get("Buyer_LiquidityEvidenceConfidence") or ""
                    ),
                    "liquidity_evidence_level": str(
                        row.get("LiquidityEvidenceLevel") or ""
                    ),
                    "price_reduction_rate": _json_number(
                        row.get("Buyer_PriceReductionRate")
                    ),
                    "price_pressure_evidence_level": str(
                        row.get("PricePressureEvidenceLevel") or ""
                    ),
                })
                if len(variant_context) >= 4:
                    break

        item["variant_intelligence"] = variant_context
        enriched.append(item)

    return enriched





def _confidence_rank(value):
    text = str(value or "").strip().casefold()
    if text in {"high", "yüksek", "yuksek", "высокая", "высокий"}:
        return 3
    if text in {"medium", "moderate", "orta", "средняя", "средний"}:
        return 2
    if text in {"low", "düşük", "dusuk", "низкая", "низкий"}:
        return 1
    return 0


def _rerank_discover_options_with_buyer_intelligence(model_options, filters):
    """
    Evidence-aware DISCOVER ordering.

    This is deliberately NOT a universal vehicle score. It is a lexicographic
    ordering of already-qualified market options:
      1) preserve buyer-fit/profile qualification as the primary signal;
      2) prefer stronger historical evidence;
      3) among similarly qualified/evidenced models, prefer stronger observed
         market-turnover signals and deeper historical evidence;
      4) retain current-market relevance through newest affordable year,
         budget proximity and active choice.

    Reliability and value retention are intentionally absent because those are
    not established by OtoDeğer's proprietary market data.
    """
    options = [dict(x) for x in (model_options or [])]
    if len(options) < 2:
        return options

    budget = filters.get("budget")
    try:
        ceiling = float(budget) if budget not in [None, ""] else None
    except (TypeError, ValueError):
        ceiling = None

    def finite_float(value, default):
        try:
            value = float(value)
            if pd.isna(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    def rank_key(item):
        bi = item.get("buyer_intelligence") or {}

        profile_score = int(item.get("_profile_score") or 0)

        confidence = _confidence_rank(
            bi.get("liquidity_confidence")
        )
        eligible_60 = int(bi.get("exit_60_eligible") or 0)
        exit_60 = finite_float(bi.get("exit_60_rate"), -1.0)
        median_days = finite_float(
            bi.get("median_observed_days_to_exit"), 10**9
        )
        historical_depth = int(
            bi.get("historical_distinct_listings") or 0
        )

        # Do not let tiny cohorts create a misleading liquidity advantage.
        # Below 10 mature 60-day observations, use the signal only after
        # stronger-evidence options have already ranked ahead.
        mature_liquidity = exit_60 if eligible_60 >= 10 else -1.0
        mature_days = median_days if eligible_60 >= 10 else 10**9

        newest = int(item.get("newest_year") or 0)
        active_count = int(item.get("count") or 0)

        anchor_price = item.get("newest_year_starting_price")
        if anchor_price in [None, ""]:
            anchor_price = item.get("starting_price")
        anchor_price = finite_float(anchor_price, 10**12)

        budget_distance = (
            abs(ceiling - anchor_price)
            if ceiling is not None else anchor_price
        )

        return (
            -profile_score,
            -confidence,
            -mature_liquidity,
            mature_days,
            -historical_depth,
            -newest,
            budget_distance,
            -min(active_count, 100),
            str(item.get("brand") or "").casefold(),
            str(item.get("model") or "").casefold(),
        )

    options.sort(key=rank_key)
    return options

def _format_gbp(value, language="EN"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if language == "TR":
        return "£" + f"{number:,.0f}".replace(",", ".")
    return "£" + f"{number:,.0f}"


def _fast_discover_answer(language, filters, model_options):
    """Render the factual discovery list locally so DISCOVER needs no final writing LLM."""
    options = list(model_options or [])[:12]
    if not options:
        return None

    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None

    if language == "TR":
        intro = (
            f"{budget_text} bütçeyle mevcut piyasada değerlendirebileceğiniz güçlü bir seçenek yelpazesi var."
            if budget_text else
            "Mevcut piyasada değerlendirebileceğiniz geniş bir seçenek yelpazesi var."
        )
    elif language == "RU":
        intro = (
            f"С бюджетом до {budget_text} на текущем рынке есть широкий выбор подходящих вариантов."
            if budget_text else
            "На текущем рынке есть широкий выбор подходящих вариантов."
        )
    else:
        intro = (
            f"With a {budget_text} ceiling, the current North Cyprus market gives you a broad range of relevant options."
            if budget_text else
            "The current North Cyprus market gives you a broad range of relevant options."
        )

    lines = []
    for item in options:
        name = f"{item.get('brand','')} {item.get('model','')}".strip()
        year = item.get("newest_year")
        year_price = _format_gbp(item.get("newest_year_starting_price"), language)
        count = int(item.get("count") or 0)
        if language == "TR":
            if year and year_price:
                lines.append(f"{name} — {year}'e kadar · {year} {year_price}'dan · bütçe içinde {count} ilan")
            else:
                lines.append(f"{name} — bütçe içinde {count} ilan")
        elif language == "RU":
            if year and year_price:
                lines.append(f"{name} — до {year} · {year} от {year_price} · {count} в рамках бюджета")
            else:
                lines.append(f"{name} — {count} в рамках бюджета")
        else:
            if year and year_price:
                lines.append(f"{name} — up to {year} · {year} from {year_price} · {count} within budget")
            else:
                lines.append(f"{name} — {count} matching listings")

    newest = max((int(x.get("newest_year") or 0) for x in options), default=0)
    newest_names = [
        f"{x.get('brand','')} {x.get('model','')}".strip()
        for x in options if int(x.get("newest_year") or 0) == newest
    ][:3]
    confidences = [
        str((x.get("buyer_intelligence") or {}).get("liquidity_confidence") or "").strip().upper()
        for x in options
    ]
    thin_evidence_group = bool(confidences) and all(
        c in {"LOW", "INSUFFICIENT", ""} for c in confidences
    )

    caveat = None
    if thin_evidence_group:
        if language == "TR":
            caveat = (
                "Bu grupta geçmiş piyasa verisi daha sınırlı; bu nedenle sıralamayı daha düşük güvenle "
                "değerlendirip tek tek ilanları yakından karşılaştırmak daha doğru olur."
            )
        elif language == "RU":
            caveat = (
                "По этой группе исторических рыночных данных меньше, поэтому к порядку рекомендаций "
                "стоит относиться с меньшей уверенностью и внимательнее сравнивать конкретные объявления."
            )
        else:
            caveat = (
                "Market-history evidence is thinner for this group, so I’d treat the ordering as "
                "lower-confidence and compare individual listings closely."
            )

    if language == "TR":
        closing = f"En yeni seçenekler {newest} model yılına kadar çıkıyor. İsterseniz buradan belirli modelleri karşılaştırabilir veya yıl/kilometre sınırı ekleyebilirsiniz."
    elif language == "RU":
        closing = f"Самые новые варианты доходят до {newest} года. Дальше можно сравнить конкретные модели или задать ограничение по году и пробегу."
    else:
        names = ", ".join(newest_names)
        closing = f"The newest options reach {newest}" + (f", including {names}" if names else "") + ". You can now compare specific models or narrow the search further."

    parts = [intro, "\n".join(lines)]
    if caveat:
        parts.append(caveat)
    parts.append(closing)
    return "\n\n".join(parts)


def _fast_compare_answer(message, language, filters, model_options):
    """Concise consumer comparison using only grounded market/Buyer Intelligence facts."""
    options = list(model_options or [])
    if len(options) < 2:
        return None

    targets = resolve_market_vehicle_mentions(message)
    target_labels = {}
    for t in targets:
        key = (str(t.get("brand") or "").casefold(), str(t.get("model") or "").casefold())
        label = f"{t.get('brand','')} {t.get('model','')}".strip()
        if t.get("category"):
            label += f" {t['category']}"
        target_labels[key] = label

    # Only compare explicitly named targets. This is also a defensive barrier
    # against malformed market rows ever surfacing as a third comparison vehicle.
    if target_labels:
        options = [
            o for o in options
            if (str(o.get("brand") or "").casefold(), str(o.get("model") or "").casefold()) in target_labels
        ]
    chosen = options[:4]
    if len(chosen) < 2:
        return None
    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None

    def label(o):
        key = (str(o.get("brand") or "").casefold(), str(o.get("model") or "").casefold())
        return target_labels.get(key) or f"{o.get('brand','')} {o.get('model','')}".strip()

    def km_text(v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return f"{n:,}" if language != "TR" else f"{n:,}".replace(",", ".")

    # Determine factual leaders.
    newest_winner = max(chosen, key=lambda o: (int(o.get("newest_year") or 0), int(o.get("count") or 0)))
    choice_winner = max(chosen, key=lambda o: int(o.get("count") or 0))

    liquidity_candidates=[]
    for o in chosen:
        bi=o.get("buyer_intelligence") or {}
        days=bi.get("median_observed_days_to_exit")
        rate=bi.get("exit_60_rate")
        if days is not None or rate is not None:
            liquidity_candidates.append((o, days, rate))
    liquidity_winner=None
    if liquidity_candidates:
        liquidity_winner=min(
            liquidity_candidates,
            key=lambda t: (float(t[1]) if t[1] is not None else 10**9, -(float(t[2]) if t[2] is not None else -1)),
        )[0]

    same_market_winner = label(choice_winner) == label(newest_winner)
    liquidity_label = label(liquidity_winner) if liquidity_winner is not None else None

    if language == "TR":
        prefix = f"{budget_text} bütçenizde " if budget_text else ""
        if same_market_winner:
            intro = prefix + f"{label(choice_winner)} hem daha fazla seçenek hem de daha yeni araçlara erişim sunuyor"
        else:
            intro = prefix + f"{label(choice_winner)} daha fazla seçenek sunarken {label(newest_winner)} daha yeni araçlara erişim sağlıyor"
        if liquidity_label and liquidity_label != label(choice_winner):
            intro += f"; {liquidity_label} ise tarihsel yeniden satış kolaylığı sinyalinde daha güçlü."
        else:
            intro += "."
    elif language == "RU":
        prefix = f"При бюджете {budget_text} " if budget_text else ""
        if same_market_winner:
            intro = prefix + f"{label(choice_winner)} предлагает и больший выбор, и доступ к более новым автомобилям"
        else:
            intro = prefix + f"у {label(choice_winner)} больше выбора, а {label(newest_winner)} даёт доступ к более новым автомобилям"
        if liquidity_label and liquidity_label != label(choice_winner):
            intro += f"; при этом исторический сигнал по лёгкости перепродажи сильнее у {liquidity_label}."
        else:
            intro += "."
    else:
        prefix = f"With your {budget_text} ceiling, " if budget_text else ""
        if same_market_winner:
            intro = prefix + f"{label(choice_winner)} offers considerably more choice and access to newer cars"
        else:
            intro = prefix + f"{label(choice_winner)} offers more choice, while {label(newest_winner)} gives you access to newer cars"
        if liquidity_label and liquidity_label != label(choice_winner):
            intro += f", while {liquidity_label} has the stronger historical resale-ease signal."
        else:
            intro += "."

    sections=[]
    for o in chosen:
        name=label(o)
        count=int(o.get("count") or 0)
        year=o.get("newest_year")
        price=_format_gbp(o.get("newest_year_starting_price"), language)
        lo=km_text(o.get("lowest_km")); hi=km_text(o.get("highest_km"))
        bi=o.get("buyer_intelligence") or {}
        days=bi.get("median_observed_days_to_exit")
        rate=bi.get("exit_60_rate")

        if language == "TR":
            facts=f"{count} eşleşme"
            if year and price: facts += f" · en yeni uygun yıl {year}, {price}'dan"
            if lo and hi: facts += f" · km aralığı {lo}–{hi}"
            resale=""
            if days is not None:
                resale=f" Tarihsel veride ilanların medyan gözlenen piyasa süresi yaklaşık {float(days):.0f} gündü."
            sections.append(f"{name}\n{facts}.{resale}")
        elif language == "RU":
            facts=f"{count} подходящих вариантов"
            if year and price: facts += f" · самый новый доступный год {year}, от {price}"
            if lo and hi: facts += f" · пробег {lo}–{hi} км"
            resale=""
            if days is not None:
                resale=f" Историческая медиана наблюдаемого присутствия объявления на рынке составляла около {float(days):.0f} дней."
            sections.append(f"{name}\n{facts}.{resale}")
        else:
            facts=f"{count} matches"
            if year and price: facts += f" · newest affordable {year} from {price}"
            if lo and hi: facts += f" · advertised mileage range {lo}–{hi} km"
            resale=""
            if days is not None:
                resale=f" Historically, its listings showed a median observed market presence of about {float(days):.0f} days."
            sections.append(f"{name}\n{facts}.{resale}")

    if language == "TR":
        close_parts=[]
        close_parts.append(f"Daha fazla seçenek ve daha yeni araç bulma açısından {label(choice_winner)} öne çıkıyor.")
        if liquidity_winner is not None:
            close_parts.append(f"Yeniden satılabilirlik için sahip olduğumuz piyasa sinyallerinde {label(liquidity_winner)} daha hızlı gözlenen devir gösteriyor.")
        close_parts.append("Değerini ne kadar koruyacağını bu verilerle güvenilir biçimde garanti edemeyiz. Piyasa devri, ilanların gözlemden çıkışını ölçer; doğrulanmış satış anlamına gelmez. Maksimum kilometrenizi söylerseniz karşılaştırmayı doğrudan o sınırın içindeki araçlara indirebilirim.")
        closing=" ".join(close_parts)
    elif language == "RU":
        close_parts=[f"По выбору и более новым машинам сильнее выглядит {label(choice_winner)}."]
        if liquidity_winner is not None:
            close_parts.append(f"По наблюдаемому рыночному обороту сигнал сильнее у {label(liquidity_winner)}.")
        close_parts.append("Надёжно гарантировать сохранение стоимости по этим данным нельзя. Рыночный оборот отражает исчезновение объявления из наблюдаемого рынка, а не подтверждённую продажу. Укажите максимальный пробег — и я сравню только варианты в этом диапазоне.")
        closing=" ".join(close_parts)
    else:
        close_parts=[f"For choice and access to newer cars, {label(choice_winner)} is stronger."]
        if liquidity_winner is not None:
            close_parts.append(f"For resale ease, the historical market signal is stronger for {label(liquidity_winner)}, based on faster observed turnover.")
        close_parts.append("The data cannot reliably guarantee which one will hold its value better. Market turnover reflects listing disappearance from the observed market, not confirmed sales. Give me your maximum mileage and I can compare only the cars that actually meet it.")
        closing=" ".join(close_parts)

    return intro + "\n\n" + "\n\n".join(sections) + "\n\n" + closing


def _localize_listing_value(value, field, language):
    """Localize common structured market values without altering seller names."""
    text = str(value or "").strip()
    if not text or language == "TR":
        return text
    key = _normalize_vehicle_phrase(text)
    if language == "EN":
        maps = {
            "transmission": {"otomatik": "Automatic", "manuel": "Manual"},
            "company": {"bireysel": "Individual seller"},
            "color": {
                "siyah": "Black", "beyaz": "White", "gumus": "Silver",
                "gri": "Grey", "fume": "Dark grey", "mavi": "Blue",
                "kirmizi": "Red", "yesil": "Green", "sari": "Yellow",
                "bej": "Beige", "kahverengi": "Brown", "turuncu": "Orange",
                "lacivert": "Navy", "bordo": "Burgundy",
                "mavi okyanus": "Ocean Blue", "mavi parlement": "Parliament Blue",
                "inci beyaz": "Pearl White", "metalik gri": "Metallic Grey",
                "koyu gri": "Dark Grey", "acik gri": "Light Grey"
            },
        }
        return maps.get(field, {}).get(key, text)
    if language == "RU":
        maps = {
            "transmission": {"otomatik": "автомат", "manuel": "механика"},
            "company": {"bireysel": "частный продавец"},
            "color": {
                "siyah": "чёрный", "beyaz": "белый", "gumus": "серебристый",
                "gri": "серый", "fume": "тёмно-серый", "mavi": "синий",
                "kirmizi": "красный", "yesil": "зелёный", "sari": "жёлтый"
            },
        }
        return maps.get(field, {}).get(key, text)
    return text


def _listing_mileage_anomaly(item):
    """Flag implausibly low advertised mileage without asserting the listing is wrong."""
    try:
        year = int(item.get("year"))
        km = float(item.get("km"))
    except (TypeError, ValueError):
        return False
    current_year = datetime.now(timezone.utc).year
    age = max(0, current_year - year)
    if km < 0:
        return True
    if age >= 2 and km < 500:
        return True
    if age >= 5 and km < 1500:
        return True
    return False


def _select_shop_representatives(results, max_candidates=3, sort_mode=None):
    """Pick a small useful SHOP set, or honor an explicit listing-level sort.

    All rows have already passed the deterministic active filters. Selection is
    presentation-only and never relaxes budget/model/year/KM/location/seller rules.
    """
    clean = list(results or [])
    if not clean:
        return []

    if sort_mode == "lowest_km":
        ranked = [x for x in clean if x.get("km") is not None]
        ranked.sort(key=lambda x: (int(x.get("km")), -int(x.get("year") or 0), float(x.get("price") or 10**12)))
        return ranked[:max_candidates]
    if sort_mode == "cheapest":
        ranked = sorted(clean, key=lambda x: (float(x.get("price") or 10**12), -int(x.get("year") or 0), int(x.get("km")) if x.get("km") is not None else 10**12))
        return ranked[:max_candidates]
    if sort_mode == "newest":
        ranked = sorted(clean, key=lambda x: (-int(x.get("year") or 0), int(x.get("km")) if x.get("km") is not None else 10**12, float(x.get("price") or 10**12)))
        return ranked[:max_candidates]

    chosen, seen = [], set()

    def identity(item):
        return item.get("link") or (
            item.get("brand"), item.get("model"), item.get("category"),
            item.get("year"), item.get("price"), item.get("km"), item.get("company")
        )

    def take(item):
        if item is None:
            return
        key = identity(item)
        if key not in seen:
            chosen.append(item)
            seen.add(key)

    # 1) Newest available example; among the newest year prefer lower mileage,
    # then lower asking price.
    newest = sorted(
        clean,
        key=lambda x: (
            -(int(x.get("year") or 0)),
            int(x.get("km")) if x.get("km") is not None else 10**12,
            float(x.get("price") or 10**12),
        ),
    )
    take(newest[0] if newest else None)

    # 2) Lowest-mileage remaining example.
    low_km_pool = [x for x in clean if x.get("km") is not None and not _listing_mileage_anomaly(x)]
    if not low_km_pool:
        low_km_pool = [x for x in clean if x.get("km") is not None]
    low_km = sorted(
        low_km_pool,
        key=lambda x: (int(x.get("km") or 0), -int(x.get("year") or 0), float(x.get("price") or 10**12)),
    )
    for item in low_km:
        if identity(item) not in seen:
            take(item)
            break

    # 3) Lower-priced remaining example. This is a factual price distinction,
    # not a claim that the listing is better value.
    lower_price = sorted(
        clean,
        key=lambda x: (float(x.get("price") or 10**12), -int(x.get("year") or 0), int(x.get("km")) if x.get("km") is not None else 10**12),
    )
    for item in lower_price:
        if identity(item) not in seen:
            take(item)
            break

    # Defensive fill for tiny/duplicate datasets.
    for item in newest:
        if len(chosen) >= max_candidates:
            break
        take(item)

    return chosen[:max_candidates]


def _fast_shop_answer(language, filters, search_result, listing_candidates, preferences=None):
    """Render listing-level results locally and always use progressive disclosure."""
    candidates = list(listing_candidates or [])[:3]
    if not candidates:
        return None
    total = int(search_result.get("count", 0) or 0)
    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None

    sort_mode = _listing_sort_mode(preferences)
    first = candidates[0]
    vehicle_name = f"{first.get('brand','')} {first.get('model','')}".strip()
    vehicle_name = re.sub(r"\\s+", " ", vehicle_name)

    if language == "TR":
        qualifier = {"lowest_km": "en düşük kilometreli", "cheapest": "en düşük fiyatlı", "newest": "en yeni"}.get(sort_mode, "güncel")
        intro = f"İşte" + (f" {budget_text} bütçeniz içinde" if budget_text else "") + f" {qualifier} {vehicle_name} ilanları:"
    elif language == "RU":
        qualifier = {"lowest_km": "с минимальным пробегом", "cheapest": "с самой низкой ценой", "newest": "самые новые"}.get(sort_mode, "актуальные")
        intro = f"Вот {qualifier} объявления {vehicle_name}" + (f" в рамках бюджета {budget_text}:" if budget_text else ":")
    else:
        qualifier = {"lowest_km": "lowest-mileage", "cheapest": "lowest-priced", "newest": "newest"}.get(sort_mode, "current")
        intro = f"Here are the {qualifier} {vehicle_name} listings" + (f" within your {budget_text} budget:" if budget_text else ":")

    lines=[]
    for x in candidates:
        name=f"{x.get('brand','')} {x.get('model','')}".strip()
        name=re.sub(r"\\s+", " ", name)
        year=x.get('year') or '—'
        price=_format_gbp(x.get('price'), language) or '—'
        details=[]
        if x.get('km') is not None:
            km=int(x['km'])
            km_txt=f"{km:,}" if language != 'TR' else f"{km:,}".replace(',', '.')
            if _listing_mileage_anomaly(x):
                if language == "TR":
                    details.append(f"ilan km: {km_txt} (doğrulayın)")
                elif language == "RU":
                    details.append(f"заявленный пробег: {km_txt} км (проверьте)")
                else:
                    details.append(f"advertised {km_txt} km (verify)")
            else:
                details.append(f"{km_txt} km")
        # Colour stays available as a filter, but is omitted from default replies.
        for key in ('transmission','company','location'):
            val = _localize_listing_value(x.get(key), key, language)
            if val:
                details.append(val)
        lines.append(f"{name}, {year} — {price}" + (" · " + " · ".join(details) if details else ""))

    if total > 10:
        if language == "TR":
            closing=f"Toplam {total} eşleşme var; yüzlerce ilan sıralamak yerine aramayı daraltmak daha faydalı olur. Maksimum kilometre, minimum model yılı, konum veya galeri/bireysel satıcı tercihinizi yazabilirsiniz."
        elif language == "RU":
            closing=f"Всего найдено {total} вариантов. Вместо длинного списка лучше сузить поиск: укажите максимальный пробег, минимальный год, район или дилер/частный продавец."
        else:
            closing=f"There are {total} matches, so a long list would not be very useful. Give me a maximum mileage, minimum year, location, or dealer/private-seller preference and I'll narrow it down."
    elif total > len(candidates):
        closing = {
            'TR': f"Toplam {total} eşleşme var. Daha fazlasını gösterebilir veya kilometre/yıl gibi bir kriterle daraltabilirim.",
            'RU': f"Всего найдено {total} вариантов. Можно показать ещё или сузить поиск по пробегу/году.",
        }.get(language, f"There are {total} matches. I can show more or narrow them by mileage, year or another preference.")
    else:
        closing = {
            'TR': f"Mevcut kriterlerinize uyan {total} ilan bunlar.",
            'RU': f"Это все {total} объявлений, соответствующих текущим критериям.",
        }.get(language, f"These are the {total} current listings matching your criteria.")

    return intro + "\n\n" + "\n".join(lines) + "\n\n" + closing


def generate_grounded_market_answer(message, language, filters, preferences, search_result, conversation_history=None, decision_mode="DISCOVER"):
    """Progressive-disclosure buying advice grounded in deterministic market data."""
    hard_count = int(search_result.get("count", 0) or 0)
    hard_results = search_result.get("results", []) or []

    if _asks_reliability_question(message):
        return _reliability_scope_answer(language, filters), hard_results, hard_count, []

    if hard_count == 0:
        fallback = {
            "TR": "Bu kriterlere uyan aktif ilan bulamadım. İsterseniz bütçe, yıl, kilometre veya diğer kriterlerden birini esnetebiliriz.",
            "EN": "I couldn't find an active listing matching those criteria. We can loosen the budget, year, mileage or another filter.",
            "RU": "Я не нашёл активных объявлений по этим критериям. Можно немного ослабить бюджет, год, пробег или другой фильтр.",
        }
        return fallback.get(language, fallback["TR"]), [], 0, []

    decision_mode = str(decision_mode or "DISCOVER").upper()
    if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
        decision_mode = "DISCOVER"

    # Model qualification is useful during DISCOVER because soft ideas such as
    # economical/small/luxury need model-level automotive knowledge. Once the buyer
    # explicitly moves to COMPARE or SHOP, the named market targets are already known.
    # Re-running that extra LLM step adds latency and can only narrow relevant evidence.
    if decision_mode == "DISCOVER":
        qualified_results, model_reasons, qualified_summaries = shortlist_models_for_preferences(
            message=message,
            language=language,
            filters=filters,
            preferences=preferences,
            results=hard_results,
        )
    else:
        qualified_results = list(hard_results)
        model_reasons = []
        qualified_summaries = _group_market_models(hard_results)

    has_soft_pref = any(
        str(p).casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
        for p in (preferences or [])
    )

    if decision_mode == "DISCOVER" and has_soft_pref and not qualified_results:
        # Soft buyer traits are advisory ranking signals. Never report an empty market
        # when the user's hard constraints actually have inventory. Fall back to that
        # inventory and explain/surface the closest options deterministically.
        qualified_results = list(hard_results)
        model_reasons = []
        qualified_summaries = _group_market_models(hard_results)

    advisory_results = qualified_results if qualified_results else hard_results
    advisory_count = len(advisory_results)
    model_summaries = qualified_summaries if qualified_summaries else _group_market_models(advisory_results)
    model_options = _select_model_options(model_summaries, model_reasons, filters, max_options=20)
    model_options = enrich_model_options_with_buyer_intelligence(
        model_options=model_options,
        filters=filters,
        preferences=preferences,
        hard_results=advisory_results,
    )

    if decision_mode == "DISCOVER":
        model_options = _rerank_discover_options_with_buyer_intelligence(
            model_options=model_options,
            filters=filters,
        )

    # Make buyer relevance explicit for the response model. `count`, `newest_year`
    # and the newest-year price above are calculated from the ACTIVE deterministic
    # search result, so when a budget exists they describe what that budget can buy.
    budget_ceiling = filters.get("budget")
    for option in model_options:
        option["active_filter_context"] = {
            "budget_ceiling": budget_ceiling,
            "matching_listing_count": option.get("count"),
            "newest_matching_year": option.get("newest_year"),
            "newest_matching_year_starting_price": option.get("newest_year_starting_price"),
            "lowest_matching_asking_price": option.get("starting_price"),
        }
    if decision_mode == "SHOP":
        listing_candidates = _select_shop_representatives(
            advisory_results, max_candidates=3, sort_mode=_listing_sort_mode(preferences)
        )
    else:
        listing_candidates = select_assistant_candidates(
            advisory_results, filters, max_candidates=3
        )

    # Fast paths: DISCOVER and SHOP are already fully grounded by deterministic data.
    # Avoid a second writing-model request; this removes one sequential network/LLM
    # round trip from the two most common customer journeys.
    if decision_mode == "DISCOVER":
        fast_answer = _fast_discover_answer(language, filters, model_options)
        if fast_answer:
            return fast_answer, advisory_results, advisory_count, model_options

    if decision_mode == "COMPARE":
        fast_answer = _fast_compare_answer(message, language, filters, model_options)
        if fast_answer:
            return fast_answer, advisory_results, advisory_count, model_options

    if decision_mode == "SHOP":
        fast_answer = _fast_shop_answer(language, filters, search_result, listing_candidates, preferences)
        if fast_answer:
            return fast_answer, advisory_results, advisory_count, model_options

    instructions = """
You are a premium, neutral vehicle-buying decision assistant for North Cyprus.
Think like a helpful expert conversation, not a search form and not a salesperson.

CORE BEHAVIOUR — PREMIUM DECISION ASSISTANT:
1. Decide whether the buyer has supplied enough information to make a useful answer.
   - Too broad: "I have £18k, what should I buy?" -> ask ONE high-value follow-up question.
   - Specific enough: "I have £18k and want an SUV" -> answer now with a substantial market overview.
   Never ask a follow-up merely to reduce an already-useful answer to an arbitrary 3 models.
2. When specific enough, surface a BROAD, useful set of model families from model_options.
   - Normally show 8-12 options when that many exist.
   - If the buyer explicitly asks for ALL options, show all supplied model_options (up to 20).
   - If the buyer asks "more?", use recent_conversation to avoid repeating models already shown and
     show additional supplied options. If none remain, say so plainly.
3. Answer the user's actual question FIRST. Follow-up questions are optional, not mandatory.
   Do not create permission loops ("shall I...?", "would you like...?") when the user has already
   asked for the action.
4. Short replies inherit conversational meaning from recent_conversation:
   - "yes" executes the action just offered.
   - "all of them" means all currently relevant options.
   - "Mercedes?" / "No Mercedes?" means check Mercedes within active criteria, not exclude it.
   - a model name such as "GLA" means deepen into that model's current market.
5. Progressive depth:
   broad insufficient request -> one clarifying question
   sufficient request -> broad model-family overview
   chosen brand/model -> useful model-market summary
   explicit listing request -> concrete listings
   specific listing question -> factual listing discussion
6. For a chosen model, prioritize facts that are relevant to the buyer's ACTIVE filters. If a budget
   ceiling exists, clearly distinguish what is available WITHIN that budget from any broader market
   context. Never present an out-of-budget year/price as though it satisfies the buyer's budget.
   Useful facts include matching count, newest matching year, cheapest listing IN THAT NEWEST YEAR,
   mileage range and transmissions. Broader market context is secondary and must be clearly labelled.
7. Individual listing mode is for explicit listing requests. Show up to 3 by default; show more/all
   only when the user explicitly asks. Never print raw URLs unless requested.

DECISION MODE — AUTHORITATIVE:
- decision_mode is application state, not a suggestion. Follow it.
- DISCOVER: help the buyer understand the model-family opportunity set. Do not drift into individual listings.
- COMPARE: directly evaluate the model(s) under discussion. Keep it CONSUMER-FIRST and concise.
  Prioritize, in this order where data supports it: within-budget availability, newest affordable year,
  whether the buyer's mileage threshold can be met, observed resale/liquidity ease, and only then
  cautious value-retention context. Use Buyer Intelligence behind the scenes; do NOT dump raw analytics.
  Normally give 2-4 short paragraphs/sections total. A consumer should not need to interpret exit-rate,
  price-pressure or confidence tables. Mention at most one or two supporting figures when they materially
  change the decision. Do not claim reliability or appearance from market data. Do not collapse the comparison into a universal score.
- SHOP: work at individual-listing level using listing_candidates. Default to 3 listings. Never dump dozens or hundreds of listings into chat, even if the buyer asks for all of them; show a small useful batch and prompt for maximum mileage, minimum year, location, seller type or another restriction when many matches remain. Keep every listing statement factual and grounded.
- Never change modes yourself. The application has already classified the turn.

NEUTRALITY:
- Never tell the buyer to buy a specific advertised vehicle.
- Never call a listing good, safe, reliable, problem-free, high-quality, best value or mechanically sound.
- Do not rank individual listings as "best".
- If asked which vehicle/model is definitely reliable, problem-free, safe, guaranteed, or which one
  they should definitely buy, DO NOT substitute different models based on general reputation and do
  not endorse one. State plainly that you cannot determine a definite purchase choice or guaranteed
  condition from these data. Then offer an objective comparison of ONLY the models already under
  discussion. Mention inspection/service history only as a sensible verification step, not as proof.
- Model-level positioning may be discussed cautiously when the buyer asks for comparison or a
  soft preference such as economical, premium/luxury, practical, sporty or family-oriented.
- During initial discovery, keep model lines factual: model name, newest matching year and starting
  asking price. Do not append sales-like adjectives such as balanced, quality, comfortable, best,
  low ownership cost, good choice or driving-focused unless the buyer explicitly asks to compare
  characteristics and the statement is clearly model-level general context.
- If the buyer says "economic/economical/ekonomik", help identify suitable real model families.
- If they say "luxury/premium/lüks", help identify premium-positioned real model families.
- Hard facts about the current market always override general automotive knowledge.

BUYER INTELLIGENCE:
- model_options may contain buyer_intelligence and variant_intelligence calculated from OtoDeğer's
  validated current + historical Cyprus-market datasets.
- Treat these fields as proprietary factual evidence, not as general automotive knowledge.
- Use historical evidence to improve recommendations when it is relevant to the buyer's decision.
- "exit_60_rate" means the observed share of a mature historical listing cohort that left the observed
  market within 60 days. It does NOT prove the vehicles were sold.
- "median_observed_days_to_exit" is observed listing duration before market exit; do not call it
  guaranteed "days to sell".
- "price_reduction_rate" is the share of eligible historical listings whose asking price was reduced.
  It is evidence of asking-price pressure, NOT depreciation or value retention.
- Respect liquidity_confidence / price_pressure_confidence. Avoid strong conclusions from LOW or
  INSUFFICIENT evidence. Mention limited evidence when it materially affects a comparison.
- liquidity_evidence_level / price_pressure_evidence_level tells you whether CATEGORY or MODEL
  evidence is being used. Category evidence is more specific; MODEL means the category sample was
  too thin and the system deliberately fell back upward.
- Never convert listing volume into a claim of popularity.
- Never invent a universal resale score, reliability score or recommendation score.
- variant_intelligence may be used to distinguish variants within the same model when the data
  supports it. Do not invent a variant that is absent from variant_intelligence.
- Prefer useful relative conclusions ("historically faster observed turnover", "more current supply",
  "less asking-price reduction pressure") backed by supplied evidence over dumping statistics.
- When the buyer asks about resale/liquidity, use the historical evidence directly and describe it as
  observed market behaviour, not a guarantee of future resale.
- When historical intelligence is unavailable, simply omit that claim rather than filling it with
  general knowledge.

GROUNDING:
- model_options and listing_candidates come from the live deterministic market search.
- Never invent a model, price, year, mileage, seller, location, transmission or count.
- Never mention a model that is not in model_options during discovery.
- "starting_price" is the lowest asking price across matching listings for that model.
- "starting_price_year" is the model year attached to that cheapest asking-price level.
- "newest_year" is the newest matching model year.
- "newest_year_starting_price" is the cheapest asking price specifically among listings of newest_year.
- `active_filter_context` repeats the buyer-relevant facts from the deterministic active search.
- If active_hard_filters contains a budget, DISCOVER is about WHAT THAT BUDGET BUYS. Do NOT mention
  the model's low-end/overall `starting_price` unless the buyer explicitly asks for the cheapest/low-end
  market. A £2,900 old example is not useful merely because the buyer can spend £15,000.
- With a budget, preferred concise DISCOVER line in English:
  Mazda CX-3 — up to 2021 · 2021 from £X · N within budget
  or natural equivalent. The year and price MUST belong together.
- Without a budget, an overall starting price may be used when useful.
- In COMPARE with a budget, lead with each model's within-budget matching count and newest affordable
  year/price. Broader/out-of-budget inventory can be mentioned only as clearly labelled market context.
- Equivalent natural phrasing should be used in Turkish/Russian.
- If a requested vehicle type is present, all surfaced models must satisfy it.

MONEY:
- Every supplied market price is GBP.
- Always format as £18.000 / £17.500 style in Turkish, and appropriate thousands formatting in English/Russian.
- Never output $, USD, EUR, €, TL or another currency for supplied market prices.

LANGUAGE — IMPORTANT:
- Reply in the language of the user's latest message. Do not keep replying in Turkish merely because
  earlier UI state or conversation content was Turkish.
- If the latest message is English, answer in natural English. If Turkish, answer in natural Turkish.
  If Russian, answer in natural Russian.

FORMAT — IMPORTANT:
- Clean, premium chat formatting. No markdown bullets, numbering, asterisks or bold markers.
- For an option overview use exactly:
  one concise intro paragraph
  BLANK LINE
  one model per line, with NO blank lines between model lines
  BLANK LINE
  one concise synthesis/next-step paragraph only when it adds value
- Keep terminology stable inside one conversation. When a budget exists, prefer concise
  buyer-relevant wording such as "up to 2024 · 2024 from £13,500" rather than "overall from".
- Do not hide useful options merely to make the answer shorter.
- Do not explain the system or database to the buyer.
"""

    payload = {
        "language": language,
        "decision_mode": decision_mode,
        "latest_message": message,
        "recent_conversation": sanitize_conversation_history(conversation_history),
        "active_hard_filters": filters,
        "soft_preferences": preferences,
        "hard_filter_count": hard_count,
        "preference_qualified_count": advisory_count if qualified_results else None,
        "buyer_intelligence_ready": BUYER_INTELLIGENCE_READY,
        "model_options": model_options,
        "listing_candidates": listing_candidates,
        "listing_display_limit": 3 if decision_mode == "SHOP" else 0,
        "instruction_note": (
            "Follow decision_mode exactly. In DISCOVER, surface useful model options. In COMPARE, directly "
            "compare/evaluate the model(s) under discussion using grounded market and Buyer Intelligence evidence. "
            "In SHOP, use individual listing facts; default to three unless more/all was explicitly requested. "
            "All prices are GBP."
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
            "reasoning": {"effort": "none"},
            "max_output_tokens": 900,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
        },
        timeout=(1.0, 4.0),
    )
    response.raise_for_status()

    answer = extract_response_text(response.json()).strip()
    if not answer:
        raise ValueError("AI_ASSISTANT_EMPTY_RESPONSE")

    answer = answer.replace("$", "£").replace(" USD", " GBP").replace("USD ", "GBP ")
    answer = normalize_assistant_format(answer)
    return answer, advisory_results, advisory_count, model_options



def _recover_recent_compare_targets(message, conversation_history):
    """
    Recover the most recent explicit multi-model comparison only for a genuine
    constraint-only continuation. This preserves COMPARE state across turns such
    as "Only consider cars below 80,000 km" without carrying stale targets into
    unrelated new searches.
    """
    low = str(message or "").strip().casefold()
    if not low or not conversation_history:
        return []

    continuation_cue = re.search(
        r"\b(?:only consider|consider only|only include|within|under|below|"
        r"maximum|max\b|up to|from\s+(?:19|20)\d{2}|"
        r"automatic only|manual only|"
        r"sadece|yalnızca|yalnizca|altında|altinda|"
        r"только|до\s+[0-9]|не более)\b",
        low,
    )
    if not continuation_cue:
        return []

    # Clear scope changes should start a new search rather than revive old models.
    scope_switch = re.search(
        r"\b(?:make it|instead|actually i want|i want|"
        r"suvs?|crossovers?|pick-?ups?|motorcycles?|scooters?|"
        r"small cars?|small vehicles?|"
        r"motosiklet(?:ler)?|мотоцикл(?:ы)?|скутер(?:ы)?)\b",
        low,
    )
    if scope_switch:
        return []

    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").casefold() != "user":
            continue
        content = str(item.get("text") or item.get("content") or "")
        content_low = content.casefold()
        if not re.search(
            r"\b(?:compare|comparison|versus|vs\.?|karşılaştır|karsilastir|kıyasla|kiyasla|"
            r"сравни(?:ть|те)?|сравнение|против)\b",
            content_low,
        ):
            continue
        targets = resolve_market_vehicle_mentions(content)
        if len(targets) >= 2:
            return targets

    return []


def _recover_recent_recommendation_target(message, conversation_history):
    """
    Resolve contextual references such as "your first recommendation" or
    "the first option" to the first vehicle named in the most recent assistant
    recommendation. This is intentionally narrow so stale models are not carried
    into unrelated turns.
    """
    low = str(message or "").strip().casefold()
    if not low or not conversation_history:
        return []

    first_ref = re.search(
        r"\b(?:first recommendation|first option|first one|top recommendation|"
        r"ilk öneri(?:niz|n)?|ilk seçenek|ilk secenek|"
        r"первая рекомендация|первый вариант)\b",
        low,
    )
    listing_ref = re.search(
        r"\b(?:listings?|ads?|show me actual|for sale|"
        r"ilan(?:lar|ları|lari)?|göster|goster|"
        r"объявлен(?:ие|ия|ий)|покажи|показать)\b",
        low,
    )
    if not (first_ref and listing_ref):
        return []

    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").casefold() != "assistant":
            continue
        content = str(item.get("text") or item.get("content") or "")
        targets = resolve_market_vehicle_mentions(content)
        if targets:
            return [targets[0]]

    return []


def _looks_like_unknown_explicit_vehicle_shop_request(message, resolved_targets, current_filters, conversation_history):
    """
    Detect a fresh SHOP request that appears to name a specific vehicle, but that
    name cannot be resolved against the live Brand+Model universe.

    This prevents e.g. "Show me listings for Zorblax Hypercar 9000" from silently
    falling back to the entire market. Broad requests such as "show me SUV listings"
    and contextual follow-ups are deliberately excluded.
    """
    if resolved_targets:
        return False
    if current_filters or conversation_history:
        return False

    raw = str(message or "").strip()
    if not raw:
        return False

    patterns = [
        r"(?i)\b(?:listings?|ads?|adverts?|advertisements?)\s+(?:for|of)\s+(.+?)\s*[?.!]*$",
        r"(?i)\bshow\s+me\s+(?:actual\s+)?(.+?)\s+(?:listings?|ads?|adverts?|advertisements?)\s*[?.!]*$",
    ]
    candidate = None
    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            candidate = m.group(1).strip(" \t\r\n.,!?")
            break

    if not candidate:
        return False

    candidate_n = _normalize_vehicle_phrase(candidate)
    generic = {
        "car", "cars", "vehicle", "vehicles", "suv", "suvs", "crossover", "crossovers",
        "motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes",
        "pickup", "pickups", "pick up", "pick ups", "4x4", "4x4s",
        "family car", "family cars", "small car", "small cars", "city car", "city cars",
        "economical car", "economical cars", "automatic car", "automatic cars",
    }
    if candidate_n in generic:
        return False

    # If the phrase contains a real market brand, it can still be a legitimate
    # broad brand request even when no particular model was named.
    if MARKET_READY and market_df is not None and not market_df.empty:
        known_brands = {
            _normalize_vehicle_phrase(x)
            for x in market_df["Brand"].dropna().astype(str).unique().tolist()
            if str(x).strip()
        }
        padded = f" {candidate_n} "
        if any(f" {brand_n} " in padded for brand_n in known_brands if brand_n):
            return False

    # Treat it as an attempted specific name only when it looks name-like:
    # multiple title-cased words and/or a model-number token.
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё0-9-]+", candidate)
    alpha_words = [w for w in words if re.search(r"[A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё]", w)]
    titleish = sum(1 for w in alpha_words if w[:1].isupper())
    has_number = any(re.search(r"\d", w) for w in words)

    return (titleish >= 2 and len(alpha_words) >= 2) or (titleish >= 1 and has_number)


@app.route("/api/assistant", methods=["POST"])
def api_ai_buying_assistant():
    request_started = time.perf_counter()
    try:
        data = request.json or {}

        message = str(data.get("message", "")).strip()
        conversation_history = sanitize_conversation_history(
            data.get("conversation_history") or []
        )
        requested_language = str(data.get("language", "TR")).upper()
        language = detect_conversation_language(
            message,
            requested_language=requested_language,
            conversation_history=conversation_history,
        )
        current_filters = sanitize_ai_filters(
            data.get("current_filters") or {}
        )
        current_preferences = data.get("current_preferences") or []

        if not message:
            return jsonify({
                "success": False,
                "error": "MESSAGE_REQUIRED"
            }), 400

        resolve_started = time.perf_counter()
        resolved_targets = resolve_market_vehicle_mentions(message)
        if not resolved_targets:
            resolved_targets = _recover_recent_compare_targets(
                message,
                conversation_history,
            )
        if not resolved_targets:
            resolved_targets = _recover_recent_recommendation_target(
                message,
                conversation_history,
            )
        resolve_seconds = time.perf_counter() - resolve_started

        interpret_started = time.perf_counter()
        interpretation = fast_common_interpretation(
            message=message,
            resolved_targets=resolved_targets,
        )
        if interpretation is None:
            try:
                interpretation = interpret_market_query(
                    message=message,
                    current_filters=current_filters,
                    language=language,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                print(f"INTERPRETER_DEGRADED_FALLBACK: {exc}", flush=True)
                question = {
                    "TR": "Bu değişikliği net uygulayabilmem için bütçe, model, yıl, kilometre veya satıcı tercihinizi biraz daha açık yazar mısınız?",
                    "RU": "Чтобы точно применить изменение, уточните бюджет, модель, год, пробег или тип продавца.",
                    "EN": "To apply that accurately, please rephrase it with the budget, model, year, mileage or seller preference you want to change.",
                }.get(language, "To apply that accurately, please rephrase the restriction you want to change.")
                interpretation = {
                    "filters": {}, "clear_filters": [], "seller_mode": None, "preferences": [],
                    "needs_clarification": True, "clarification_question": question,
                    "decision_mode": "COMPARE" if resolved_targets else "DISCOVER",
                    "fast_path": False, "degraded": True,
                }
        interpret_seconds = time.perf_counter() - interpret_started

        decision_mode = str(
            interpretation.get("decision_mode") or "DISCOVER"
        ).upper()
        if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
            decision_mode = "DISCOVER"

        if (
            decision_mode == "SHOP"
            and _looks_like_unknown_explicit_vehicle_shop_request(
                message,
                resolved_targets,
                current_filters,
                conversation_history,
            )
        ):
            unknown_answer = {
                "TR": "Bu araç adını güncel Kıbrıs piyasa verilerimde eşleştiremedim. Marka ve modeli kontrol edip tekrar yazar mısınız?",
                "RU": "Я не смог сопоставить это название автомобиля с актуальными данными рынка Кипра. Проверьте, пожалуйста, марку и модель.",
                "EN": "I couldn't match that vehicle name to the current Cyprus market data. Please check the make and model and try again.",
            }.get(language, "I couldn't match that vehicle name to the current Cyprus market data. Please check the make and model and try again.")

            return jsonify({
                "success": True,
                "answer": unknown_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": 0,
                "returned": 0,
                "results": [],
                "model_options": [],
                "interpretation": interpretation,
                "decision_mode": "SHOP",
                "stage": "clarification",
                "resolved_vehicle_targets": [],
            })

        next_filters = apply_interpretation_to_filters(
            current_filters,
            interpretation,
        )

        # Deterministic explicit-constraint clearing.
        # Conversational phrases such as "mileage doesn't matter anymore" are
        # unambiguous state changes and should not depend on the LLM returning
        # clear_filters correctly.
        low_message_for_clear = message.casefold()

        clear_mileage = bool(
            re.search(
                r"\b(?:mileage|kilomet(?:er|re)s?|km)\s+"
                r"(?:doesn['’]?t|does not|doesnt)\s+matter(?:\s+anymore)?\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:any|whatever)\s+(?:mileage|kilomet(?:er|re)s?|km)\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:forget|remove|clear|ignore)\s+(?:the\s+)?"
                r"(?:mileage|kilomet(?:er|re)s?|km)(?:\s+(?:limit|restriction))?\b",
                low_message_for_clear,
            )
            or re.search(
                r"\bkilometre\s+(?:önemli\s+değil|onemli\s+degil)\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:пробег\s+не\s+важен|любой\s+пробег)\b",
                low_message_for_clear,
            )
        )

        if clear_mileage:
            next_filters.pop("min_km", None)
            next_filters.pop("max_km", None)

        next_preferences = merge_preferences(
            current_preferences,
            interpretation.get("preferences", []),
        )

        # Deterministic scope-reset guard for explicit physical-class changes.
        # The interpreter remains responsible for soft use-cases, but phrases such
        # as "forget small cars, I want an SUV" must not preserve the old class or
        # an old budget simply because the conversational interpreter missed part
        # of a compound correction.
        low_message = message.casefold()
        explicit_scope_reset = bool(re.search(
            r"\b(?:forget|actually|instead|rather|unut|aslında|aslinda|yerine|"
            r"забудь|на самом деле|вместо)\b",
            low_message,
        ))

        if explicit_scope_reset:
            explicit_type = None
            if re.search(r"\b(?:suv|4x4|crossover)\b", low_message):
                explicit_type = "vehicle_type:SUV"
            elif re.search(r"\b(?:motorcycle|motorbike|motosiklet|мотоцикл)\b", low_message):
                explicit_type = "vehicle_type:motorcycle"
            elif re.search(r"\b(?:pickup|pick-up|pick up)\b", low_message):
                explicit_type = "vehicle_type:pickup"

            if explicit_type:
                next_preferences = [
                    p for p in next_preferences
                    if not str(p).casefold().startswith("vehicle_type:")
                    and str(p).casefold() != "any_vehicle_type"
                ]
                next_preferences.append(explicit_type)

                # "Forget <old vehicle class>, I want <new class>" is an explicit
                # replacement of that earlier recommendation brief. Soft buyer-fit
                # preferences attached to the abandoned brief must not silently
                # survive unless the user states them again in the correction turn.
                #
                # Keep this deliberately narrower than generic "actually make it
                # an SUV": that wording can legitimately retain preferences such
                # as economical/family-friendly.
                explicit_forget_old_class = bool(re.search(
                    r"\b(?:forget|unut|забудь)\b.*\b"
                    r"(?:small\s+cars?|city\s+cars?|cars?|suvs?|crossovers?|"
                    r"motorcycles?|motorbikes?|motosiklet(?:ler)?|мотоцикл(?:ы|ов)?|"
                    r"pick(?:-|\s*)ups?)\b",
                    low_message,
                ))

                if explicit_forget_old_class:
                    incoming_pref_keys = {
                        str(p).casefold()
                        for p in (interpretation.get("preferences", []) or [])
                        if str(p).strip()
                        and not str(p).casefold().startswith("vehicle_type:")
                        and str(p).casefold() != "any_vehicle_type"
                    }

                    next_preferences = [
                        p for p in next_preferences
                        if str(p).casefold().startswith("vehicle_type:")
                        or str(p).casefold() == "any_vehicle_type"
                        or str(p).casefold() in incoming_pref_keys
                    ]

            # Preserve the already-validated budget parser as the source of truth,
            # but ensure a replacement budget in a compound scope-change sentence
            # is applied even if the conversational interpreter omitted it.
            # Do not call fast_common_interpretation() here: explicit correction
            # turns are intentionally excluded from that fast path. Extract only
            # the replacement budget deterministically from the correction text.
            reset_budget = None

            budget_match = re.search(
                r"£\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)",
                message,
            )
            if not budget_match:
                budget_match = re.search(
                    r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*£",
                    message,
                )
            if not budget_match:
                budget_match = re.search(
                    r"\b([0-9]+(?:[.,][0-9]+)?)\s*[kK]\b",
                    low_message,
                )

            if budget_match:
                reset_budget = _parse_human_number(budget_match.group(1))
                if (
                    reset_budget is not None
                    and re.search(r"[kK]\s*$", budget_match.group(1).strip())
                ):
                    reset_budget *= 1000

            if reset_budget is not None:
                next_filters["budget"] = reset_budget

        # Canonicalize explicitly named single vehicles. This fixes natural compound
        # names such as "Nissan Note e-Power" without hard-coding any vehicle.
        if len(resolved_targets) == 1 and decision_mode in {"COMPARE", "SHOP"}:
            target = resolved_targets[0]
            next_filters["brands"] = [target["brand"]]
            next_filters["models"] = [target["model"]]
            if target.get("category"):
                next_filters["categories"] = [target["category"]]
            else:
                next_filters.pop("categories", None)

            next_filters.pop("exclude_brands", None)
            next_filters.pop("exclude_models", None)
            next_filters.pop("exclude_categories", None)

        if interpretation.get("degraded") and interpretation.get("clarification_question"):
            return jsonify({
                "success": True,
                "answer": interpretation["clarification_question"],
                "filters": next_filters,
                "preferences": next_preferences,
                "count": None,
                "returned": 0,
                "results": [],
                "interpretation": interpretation,
                "decision_mode": decision_mode,
                "stage": "clarification",
            })

        if (
            interpretation.get("needs_clarification")
            and interpretation.get("clarification_question")
            and (interpretation.get("filters") or interpretation.get("clear_filters"))
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
                "decision_mode": decision_mode,
            })

        # Pull the full filtered result set for candidate selection.
        #
        # Explicit multi-vehicle comparisons are searched target-by-target so each
        # named vehicle can carry its own category/variant. This avoids the invalid
        # cross-product created by global filters such as:
        #   models=[Aqua, Note], categories=[e-Power]
        # which would wrongly require Aqua itself to be e-Power.
        search_started = time.perf_counter()
        if decision_mode in {"COMPARE", "SHOP"} and len(resolved_targets) >= 2:
            compare_base_filters = dict(next_filters)
            for key in (
                "brands", "exclude_brands",
                "models", "exclude_models",
                "categories", "exclude_categories",
            ):
                compare_base_filters.pop(key, None)

            next_filters = compare_base_filters
            search_result = _search_market_for_vehicle_targets(
                base_filters=compare_base_filters,
                targets=resolved_targets,
            )
        else:
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
                max_limit=5000,
            )

        search_seconds = time.perf_counter() - search_started

        if not search_result.get("success"):
            return jsonify(search_result), 503

        # Explicit physical vehicle classes are authoritative. Apply them after
        # the normal market filters using the validated model-profile taxonomy,
        # never by guessing from listing titles or CategoryDetail.
        search_result = _apply_strict_vehicle_type_to_search_result(
            search_result, next_preferences
        )

        # Guided buying flow: broad searches get a compact group of useful
        # narrowing dimensions; only later do we offer secondary refinements.
        guide_question = None
        if decision_mode == "DISCOVER" and not _asks_reliability_question(message):
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
                "decision_mode": "DISCOVER",
                "stage": "narrowing",
            })

        answer_started = time.perf_counter()
        answer, advisory_results, advisory_count, model_options = generate_grounded_market_answer(
            message=message,
            language=language,
            filters=next_filters,
            preferences=next_preferences,
            search_result=search_result,
            conversation_history=conversation_history,
            decision_mode=decision_mode,
        )
        answer_seconds = time.perf_counter() - answer_started
        total_seconds = time.perf_counter() - request_started
        print(
            f"ASSISTANT_TIMING mode={decision_mode} "
            f"resolve={resolve_seconds:.2f}s interpret={interpret_seconds:.2f}s "
            f"search={search_seconds:.2f}s answer={answer_seconds:.2f}s "
            f"fast_interpret={bool(interpretation.get('fast_path'))} total={total_seconds:.2f}s",
            flush=True,
        )
        if total_seconds > 5.0:
            print(
                f"ASSISTANT_SLO_WARN mode={decision_mode} total={total_seconds:.2f}s "
                f"profile_ready={MODEL_PROFILE_READY}",
                flush=True,
            )

        public_results = (advisory_results or search_result.get("results") or [])[:100]
        return jsonify({
            "success": True,
            "answer": answer,
            "filters": next_filters,
            "preferences": next_preferences,
            "count": search_result.get("count", 0),
            "advisory_count": advisory_count,
            "model_options": model_options,
            "returned": len(public_results),
            "results": public_results,
            "interpretation": interpretation,
            "resolved_vehicle_targets": resolved_targets,
            "decision_mode": decision_mode,
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
        traceback.print_exc()

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
        "rows": len(market_df),
        "buyer_intelligence_ready": BUYER_INTELLIGENCE_READY,
        "buyer_model_rows": len(buyer_model_df),
        "buyer_category_rows": len(buyer_category_df),
        "model_profiles_ready": MODEL_PROFILE_READY,
        "model_profile_rows": len(MODEL_PROFILE_LOOKUP),
        "model_profile_coverage": (
            round(len(MODEL_PROFILE_LOOKUP) / max(1, len(market_df[["Brand", "Model"]].drop_duplicates())), 4)
            if MARKET_READY and market_df is not None and not market_df.empty else 0
        ),
        "assistant_profile_version": ASSISTANT_PROFILE_VERSION,
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