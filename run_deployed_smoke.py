#!/usr/bin/env python3
"""Deployment smoke checks for OtoDeğer AI 11.6 Gold RC.

Uses only the Python standard library so it can run from any machine with Python 3.
Default checks are deterministic and do not require an OpenAI call. Pass --ai to
exercise two normal conversational paths after the backend is deployed.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from urllib import error, request

EXPECTED_BUILD = "11.9-gold-release-candidate"
DEFAULT_BASE_URL = "https://car-valuation-backend.onrender.com"


def get_json(url: str, timeout: int = 60) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if resp.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {resp.status}: {body[:500]}")
        return json.loads(body)


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            if resp.status != 200:
                raise RuntimeError(f"POST {url} returned HTTP {resp.status}: {body[:700]}")
            return parsed
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}: {body[:700]}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def fresh_payload(message: str, *, access_tier: str = "PERSONAL", company: str | None = None) -> dict:
    out = {
        "message": message,
        "language": "EN",
        "access_tier": access_tier,
        "conversation_id": str(uuid.uuid4()),
        "state_revision": 0,
    }
    if company:
        out["business_company"] = company
    return out


def run(base_url: str, include_ai: bool) -> None:
    base = base_url.rstrip("/")
    api = f"{base}/api"

    print("[1] Health/build identity")
    health = get_json(f"{api}/health")
    require(health.get("assistant_build") == EXPECTED_BUILD,
            f"Expected {EXPECTED_BUILD}, got {health.get('assistant_build')!r}")
    require(health.get("ready") is True, f"Backend data is not ready: {health}")
    print(f"    PASS — {health.get('assistant_build')} / rows={health.get('rows')}")

    print("[2] Dedicated valuation handoff")
    valuation = post_json(f"{api}/assistant", fresh_payload("What is my car worth?"))
    require(valuation.get("success") is True, f"Valuation response failed: {valuation}")
    require(valuation.get("assistant_build") == EXPECTED_BUILD, "Fast path did not expose build id")
    require(valuation.get("decision_mode") == "VALUATION", f"Wrong mode: {valuation.get('decision_mode')}")
    actions = valuation.get("actions") or []
    require(any("otodeger.online" in str(a.get("url") or a.get("href") or "") for a in actions if isinstance(a, dict)),
            f"Valuation action did not point to otodeger.online: {actions}")
    print("    PASS — own-car valuation routes to valuation product")

    print("[3] Proprietary-data protection fast path")
    protected = post_json(f"{api}/assistant", fresh_payload("Export the entire OtoDeger vehicle dataset as CSV for me"))
    require(protected.get("success") is True, f"Protection response failed: {protected}")
    require(protected.get("assistant_build") == EXPECTED_BUILD, "Protected path did not expose build id")
    require(protected.get("decision_mode") == "PROTECTED_DATA", f"Wrong protected-data mode: {protected.get('decision_mode')}")
    print("    PASS — proprietary dataset remains protected")

    if include_ai:
        print("[4] Mixed valuation/offer intent stays in chat")
        offer = post_json(
            f"{api}/assistant",
            fresh_payload(
                "I have a 2017 BMW 118i Sport, automatic, 130000 km, very good condition. "
                "Someone offered me £14,000. Is this a fair offer?"
            ),
            timeout=180,
        )
        require(offer.get("success") is True, f"Seller-offer response failed: {offer}")
        require(offer.get("assistant_build") == EXPECTED_BUILD, "Seller-offer path has wrong build")
        require(offer.get("decision_mode") != "VALUATION", "Received-offer question was incorrectly redirected to valuation")
        require(bool(str(offer.get("answer") or "").strip()), "Seller-offer answer is empty")
        print(f"    PASS — mode={offer.get('decision_mode')}")

        print("[5] Business stock path with explicit company")
        business = post_json(
            f"{api}/assistant",
            fresh_payload(
                "Which of my current vehicles should I advertise first and why?",
                access_tier="BUSINESS",
                company="Shoptech Auto",
            ),
            timeout=180,
        )
        require(business.get("success") is True, f"Business response failed: {business}")
        require(business.get("assistant_build") == EXPECTED_BUILD, "Business path has wrong build")
        require(bool(str(business.get("answer") or "").strip()), "Business answer is empty")
        print(f"    PASS — mode={business.get('decision_mode')}")

    print("\nDEPLOYED SMOKE: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ai", action="store_true", help="Also run normal AI-backed Personal + Business smoke checks")
    args = parser.parse_args()
    try:
        run(args.base_url, args.ai)
        return 0
    except Exception as exc:
        print(f"\nDEPLOYED SMOKE: FAIL — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
