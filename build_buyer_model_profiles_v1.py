import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROFILE_VERSION = "1.0"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

PROFILE_COLUMNS = [
    "Brand", "Model", "VehicleType", "BodyStyle", "SizeClass",
    "Economy", "Luxury", "Comfort", "Performance", "Practicality",
    "Family", "Commute", "Confidence", "Source", "ProfileVersion", "LastUpdatedUTC",
]

TRAIT_VALUES = {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}
VEHICLE_TYPES = {"CAR", "MOTORCYCLE", "SCOOTER", "PICKUP", "VAN", "ATV_UTV", "OTHER", "UNKNOWN"}
BODY_STYLES = {
    "HATCHBACK", "SEDAN", "SUV", "CROSSOVER", "PICKUP", "MPV", "WAGON",
    "COUPE", "CONVERTIBLE", "VAN", "MOTORCYCLE", "SCOOTER", "ATV_UTV", "OTHER", "UNKNOWN"
}
SIZE_CLASSES = {"MICRO", "SMALL", "COMPACT", "MIDSIZE", "LARGE", "UNKNOWN"}
CONFIDENCE_VALUES = {"HIGH", "MEDIUM", "LOW"}


def norm(v):
    return str(v or "").strip()


def extract_response_text(payload):
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    return ""


def canonical_row(raw, allowed_pairs):
    if not isinstance(raw, dict):
        return None
    brand, model = norm(raw.get("brand")), norm(raw.get("model"))
    key = (brand.casefold(), model.casefold())
    if key not in allowed_pairs:
        return None
    brand, model = allowed_pairs[key]

    def enum(field, allowed, default):
        value = norm(raw.get(field)).upper().replace("-", "_").replace(" ", "_")
        return value if value in allowed else default

    return {
        "Brand": brand,
        "Model": model,
        "VehicleType": enum("vehicle_type", VEHICLE_TYPES, "UNKNOWN"),
        "BodyStyle": enum("body_style", BODY_STYLES, "UNKNOWN"),
        "SizeClass": enum("size_class", SIZE_CLASSES, "UNKNOWN"),
        "Economy": enum("economy", TRAIT_VALUES, "UNKNOWN"),
        "Luxury": enum("luxury", TRAIT_VALUES, "UNKNOWN"),
        "Comfort": enum("comfort", TRAIT_VALUES, "UNKNOWN"),
        "Performance": enum("performance", TRAIT_VALUES, "UNKNOWN"),
        "Practicality": enum("practicality", TRAIT_VALUES, "UNKNOWN"),
        "Family": enum("family", TRAIT_VALUES, "UNKNOWN"),
        "Commute": enum("commute", TRAIT_VALUES, "UNKNOWN"),
        "Confidence": enum("confidence", CONFIDENCE_VALUES, "LOW"),
        "Source": "MODEL_PROFILE_AI",
        "ProfileVersion": PROFILE_VERSION,
        "LastUpdatedUTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def classify_batch(api_key, model_name, batch):
    instructions = """
You are building a stable model-level buyer-profile catalogue for a North Cyprus vehicle buying assistant.
Classify ONLY the supplied exact Brand + Model families using broad, well-known model positioning. This is NOT listing-level advice.
Do not infer mechanical condition, safety, reliability, resale value, or future depreciation.
Do not use title inference. Structured category_examples are context only and may be ignored if ambiguous.

Return JSON only with this shape:
{"profiles":[{"brand":"...","model":"...","vehicle_type":"CAR|MOTORCYCLE|SCOOTER|PICKUP|VAN|ATV_UTV|OTHER|UNKNOWN","body_style":"HATCHBACK|SEDAN|SUV|CROSSOVER|PICKUP|MPV|WAGON|COUPE|CONVERTIBLE|VAN|MOTORCYCLE|SCOOTER|ATV_UTV|OTHER|UNKNOWN","size_class":"MICRO|SMALL|COMPACT|MIDSIZE|LARGE|UNKNOWN","economy":"HIGH|MEDIUM|LOW|UNKNOWN","luxury":"HIGH|MEDIUM|LOW|UNKNOWN","comfort":"HIGH|MEDIUM|LOW|UNKNOWN","performance":"HIGH|MEDIUM|LOW|UNKNOWN","practicality":"HIGH|MEDIUM|LOW|UNKNOWN","family":"HIGH|MEDIUM|LOW|UNKNOWN","commute":"HIGH|MEDIUM|LOW|UNKNOWN","confidence":"HIGH|MEDIUM|LOW"}]}

Guidance:
- Economy means broadly economical to operate/use for the model family, not a guaranteed fuel-consumption figure.
- Luxury means premium/luxury positioning.
- Comfort, performance, practicality, family and commute are broad model-positioning traits.
- SMALL/MICRO/COMPACT car size is what the live assistant will use for requests such as "small car".
- If a model name is generic/ambiguous (for example "Other"/"Diğer") or you are not confident what it is, prefer UNKNOWN and LOW confidence.
- Preserve supplied Brand and Model spelling exactly.
- Return one profile for every supplied item and no extra models.
"""
    payload = {
        "models": batch,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "reasoning": {"effort": "none"},
            "max_output_tokens": 5000,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    text = extract_response_text(response.json()).strip()
    if not text:
        raise RuntimeError("Empty profile-classification response")
    parsed = json.loads(text)
    profiles = parsed.get("profiles", []) if isinstance(parsed, dict) else []
    if not isinstance(profiles, list):
        raise RuntimeError("Invalid profile-classification response")
    return profiles


def save_profiles(path, rows):
    df = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(["Brand", "Model"], keep="last")
        df = df.sort_values(["Brand", "Model"], key=lambda s: s.astype(str).str.casefold())
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Build stable model-level buyer profiles for OtoDeğer Assistant")
    parser.add_argument("--market", default="market_base.csv")
    parser.add_argument("--output", default="buyer_model_profiles.csv")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--refresh-low-confidence", action="store_true")
    parser.add_argument("--git-push", action="store_true", help="Commit and push the profile CSV when it changed")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Set it in this terminal/session and run again.", file=sys.stderr)
        return 2

    market_path = Path(args.market).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not market_path.exists():
        print(f"Market file not found: {market_path}", file=sys.stderr)
        return 2

    market = pd.read_csv(market_path, low_memory=False)
    required = {"Brand", "Model", "Category"}
    missing = required - set(market.columns)
    if missing:
        print(f"market_base.csv missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    market["Brand"] = market["Brand"].fillna("").astype(str).str.strip()
    market["Model"] = market["Model"].fillna("").astype(str).str.strip()
    market["Category"] = market["Category"].fillna("").astype(str).str.strip()
    market = market[(market["Brand"] != "") & (market["Model"] != "")]

    families = market[["Brand", "Model"]].drop_duplicates().copy()
    category_map = (
        market[market["Category"] != ""]
        .groupby(["Brand", "Model"])["Category"]
        .apply(lambda s: list(dict.fromkeys(s.astype(str).tolist()))[:6])
        .to_dict()
    )

    existing_rows = []
    existing_by_key = {}
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path, low_memory=False).fillna("")
            for row in existing.to_dict("records"):
                key = (norm(row.get("Brand")).casefold(), norm(row.get("Model")).casefold())
                if key[0] and key[1]:
                    existing_by_key[key] = row
        except Exception as exc:
            print(f"Existing profile file could not be read; rebuilding safely: {exc}")

    pending = []
    for row in families.to_dict("records"):
        brand, model = norm(row["Brand"]), norm(row["Model"])
        key = (brand.casefold(), model.casefold())
        existing = existing_by_key.get(key)
        if existing and not (args.refresh_low_confidence and norm(existing.get("Confidence")).upper() == "LOW"):
            continue
        pending.append({
            "brand": brand,
            "model": model,
            "category_examples": category_map.get((brand, model), []),
        })

    print(f"Market model families: {len(families):,}")
    print(f"Already profiled: {len(existing_by_key):,}")
    print(f"To classify: {len(pending):,}")

    all_rows = list(existing_by_key.values())
    all_by_key = {(norm(r.get("Brand")).casefold(), norm(r.get("Model")).casefold()): r for r in all_rows}

    batch_size = max(5, min(50, int(args.batch_size)))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        allowed = {(x["brand"].casefold(), x["model"].casefold()): (x["brand"], x["model"]) for x in batch}
        raw_profiles = None
        error = None
        for attempt in range(3):
            try:
                raw_profiles = classify_batch(api_key, args.model, batch)
                break
            except Exception as exc:
                error = exc
                time.sleep(1.5 * (attempt + 1))
        if raw_profiles is None:
            print(f"Batch {start // batch_size + 1} failed after retries: {error}", file=sys.stderr)
            # Save completed work before stopping.
            save_profiles(output_path, list(all_by_key.values()))
            return 1

        accepted = 0
        for raw in raw_profiles:
            row = canonical_row(raw, allowed)
            if row is None:
                continue
            key = (row["Brand"].casefold(), row["Model"].casefold())
            all_by_key[key] = row
            accepted += 1

        # If the model omitted any family, keep a safe UNKNOWN profile rather than silently losing coverage.
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for key, (brand, model) in allowed.items():
            if key not in all_by_key:
                all_by_key[key] = {
                    "Brand": brand, "Model": model, "VehicleType": "UNKNOWN", "BodyStyle": "UNKNOWN", "SizeClass": "UNKNOWN",
                    "Economy": "UNKNOWN", "Luxury": "UNKNOWN", "Comfort": "UNKNOWN", "Performance": "UNKNOWN",
                    "Practicality": "UNKNOWN", "Family": "UNKNOWN", "Commute": "UNKNOWN", "Confidence": "LOW",
                    "Source": "MODEL_PROFILE_AI_MISSING_FALLBACK", "ProfileVersion": PROFILE_VERSION, "LastUpdatedUTC": now,
                }

        save_profiles(output_path, list(all_by_key.values()))
        done = min(start + len(batch), len(pending))
        print(f"Classified {done:,}/{len(pending):,} pending families (accepted {accepted}/{len(batch)})")

    # Remove stale families that no longer exist from the production view while preserving deterministic ordering.
    live_keys = {(r["Brand"].casefold(), r["Model"].casefold()) for r in families.to_dict("records")}
    final_rows = [r for k, r in all_by_key.items() if k in live_keys]
    save_profiles(output_path, final_rows)

    print(f"Wrote {len(final_rows):,} model profiles to: {output_path}")

    if args.git_push:
        repo = output_path.parent
        try:
            subprocess.run(["git", "-C", str(repo), "add", output_path.name], check=True)
            diff = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"])
            if diff.returncode == 0:
                print("No profile changes to commit.")
            else:
                msg = f"Update buyer model profiles {datetime.now().strftime('%Y-%m-%d')}"
                subprocess.run(["git", "-C", str(repo), "commit", "-m", msg], check=True)
                subprocess.run(["git", "-C", str(repo), "push", "origin", "main"], check=True)
                print("Profile update pushed to origin/main.")
        except Exception as exc:
            print(f"Git publish failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
