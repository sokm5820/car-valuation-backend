import argparse
import ast
from pathlib import Path
import pandas as pd

ALLOWED = {
    "VehicleType": {"CAR","MOTORCYCLE","SCOOTER","PICKUP","VAN","ATV_UTV","OTHER","UNKNOWN"},
    "BodyStyle": {"HATCHBACK","SEDAN","SUV","CROSSOVER","PICKUP","MPV","WAGON","COUPE","CONVERTIBLE","VAN","MOTORCYCLE","SCOOTER","ATV_UTV","OTHER","UNKNOWN"},
    "SizeClass": {"MICRO","SMALL","COMPACT","MIDSIZE","LARGE","UNKNOWN"},
    "Economy": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Luxury": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Comfort": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Performance": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Practicality": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Family": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Commute": {"HIGH","MEDIUM","LOW","UNKNOWN"},
    "Confidence": {"HIGH","MEDIUM","LOW"},
}


def keyset(df):
    return set(zip(df["Brand"].fillna("").astype(str).str.strip().str.casefold(), df["Model"].fillna("").astype(str).str.strip().str.casefold()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--app", default="app.py")
    ap.add_argument("--market", default="market_base.csv")
    ap.add_argument("--profiles", default="buyer_model_profiles.csv")
    args=ap.parse_args()

    failures=[]; warnings=[]
    app=Path(args.app); marketp=Path(args.market); profilep=Path(args.profiles)
    for p in (app, marketp, profilep):
        if not p.exists(): failures.append(f"missing file: {p}")
    if failures:
        print("FAIL"); [print(" -",x) for x in failures]; return 1

    try:
        ast.parse(app.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"app.py does not compile: {exc}")

    market=pd.read_csv(marketp, low_memory=False)
    profiles=pd.read_csv(profilep, low_memory=False).fillna("")
    req_market={"Brand","Model","Year","Price","KM","Link"}
    req_profiles={"Brand","Model"}|set(ALLOWED)
    if req_market-set(market.columns): failures.append(f"market missing columns: {sorted(req_market-set(market.columns))}")
    if req_profiles-set(profiles.columns): failures.append(f"profiles missing columns: {sorted(req_profiles-set(profiles.columns))}")
    if failures:
        print("FAIL"); [print(" -",x) for x in failures]; return 1

    market_keys=keyset(market)
    profile_keys=keyset(profiles)
    dup=profiles.duplicated(["Brand","Model"], keep=False).sum()
    if dup: failures.append(f"duplicate profile rows: {dup}")
    stale=profile_keys-market_keys
    if stale: warnings.append(f"profile rows not in live market: {len(stale)}")
    coverage=len(profile_keys & market_keys)/max(1,len(market_keys))
    if coverage < .95: failures.append(f"profile coverage too low: {coverage:.1%}")

    for col, allowed in ALLOWED.items():
        vals=set(profiles[col].astype(str).str.strip().str.upper())
        invalid=vals-allowed
        if invalid: failures.append(f"invalid {col} values: {sorted(invalid)[:10]}")

    unknown_profile=(
        (profiles["VehicleType"].astype(str).str.upper()=="UNKNOWN") |
        (profiles["Confidence"].astype(str).str.upper()=="LOW")
    ).mean()
    if unknown_profile > .20:
        warnings.append(f"high unknown/low-confidence profile share: {unknown_profile:.1%}")

    if market["Link"].fillna("").astype(str).str.strip().duplicated().sum():
        warnings.append("market_base.csv contains duplicate Link values")
    if pd.to_numeric(market["Price"], errors="coerce").isna().mean() > .01:
        failures.append("more than 1% of market prices are non-numeric")

    print(f"Market model families: {len(market_keys):,}")
    print(f"Profile families: {len(profile_keys):,}")
    print(f"Profile coverage: {coverage:.1%}")
    print(f"Unknown/low-confidence share: {unknown_profile:.1%}")
    if warnings:
        print("WARNINGS")
        [print(" -",x) for x in warnings]
    if failures:
        print("FAIL")
        [print(" -",x) for x in failures]
        return 1
    print("PASS - production assistant inputs validated")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
