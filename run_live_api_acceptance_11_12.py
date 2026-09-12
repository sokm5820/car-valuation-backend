#!/usr/bin/env python3
"""Live API structural + quality acceptance for OtoDeğer AI 11.12 Gold RC.

This is intentionally separate from the manual 12-point quality scoring plan. It
exercises the deployed /api/assistant endpoint with realistic Personal/Business
flows, preserves server state_revision/conversation_id, asserts routing/state
contracts, and writes every response to JSON + Markdown for review.

It makes real assistant calls. Run only after deterministic deployment smoke passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib import error, request

EXPECTED_BUILD = "11.12-gold-release-candidate"
DEFAULT_BASE_URL = "https://car-valuation-backend.onrender.com"


@dataclass
class Session:
    base_url: str
    access_tier: str = "PERSONAL"
    company: Optional[str] = None
    language: str = "EN"
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state_revision: int = 0
    transcript: List[dict] = field(default_factory=list)

    def send(self, message: str, timeout: int = 180) -> dict:
        payload = {
            "message": message,
            "language": self.language,
            "access_tier": self.access_tier,
            "conversation_id": self.conversation_id,
            "state_revision": self.state_revision,
        }
        if self.company:
            payload["business_company"] = self.company
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/api/assistant",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                out = json.loads(raw)
                status = resp.status
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                out = json.loads(raw)
            except Exception:
                out = {"raw": raw}
            status = exc.code
        self.transcript.append({"message": message, "status": status, "response": out})
        if status == 200 and isinstance(out.get("state_revision"), int):
            self.state_revision = out["state_revision"]
        if out.get("conversation_id"):
            self.conversation_id = str(out["conversation_id"])
        return out


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def common(resp: dict) -> None:
    require(resp.get("success") is True, f"request failed: {resp}")
    require(resp.get("assistant_build") == EXPECTED_BUILD, f"wrong build: {resp.get('assistant_build')!r}")
    ans = str(resp.get("answer") or "")
    require(ans.strip(), "empty answer")
    forbidden = ["ASSISTANT_INTERPRETATION_FAILED", "Traceback", "InvalidTurnPlan", "ContractError"]
    require(not any(x in ans for x in forbidden), f"internal error leaked in answer: {ans[:500]}")


def action_is(resp: dict, *allowed: str) -> None:
    common(resp)
    require(resp.get("action") in allowed, f"expected action in {allowed}, got {resp.get('action')!r}")


def case_p1(base: str) -> List[Session]:
    s = Session(base)
    r = s.send("How much should I sell my 2017 BMW 118i Sport for?")
    require(r.get("success") is True, f"valuation failed: {r}")
    require(r.get("assistant_build") == EXPECTED_BUILD, "valuation fast path wrong build")
    require(r.get("decision_mode") == "VALUATION", f"pure valuation wrong mode: {r.get('decision_mode')}")
    return [s]


def case_p2(base: str) -> List[Session]:
    s = Session(base)
    r = s.send("I have a 2017 BMW 118i Sport, 130,000 km, automatic, great condition. Someone offered me £14,000. Is that good?")
    action_is(r, "EVALUATE_SALE")
    return [s]


def case_p3(base: str) -> List[Session]:
    s = Session(base)
    r = s.send("I have £15,000. I want a small family car that's good on fuel. What should I look at?")
    action_is(r, "SEARCH_VEHICLES")
    require(r.get("decision") != "NO_MATCHES", f"broad family-car discovery falsely returned no matches: {r.get('answer')}")
    require(len(r.get("model_options") or []) > 0, "broad family-car discovery returned no model options")
    r = s.send("Actually my budget is £17,000.")
    common(r)
    state = r.get("assistant_state") or {}
    constraints = state.get("constraints") or {}
    require(float(constraints.get("budget_max") or 0) == 17000, f"budget did not refine to 17000: {constraints}")
    return [s]


def case_p4(base: str) -> List[Session]:
    s = Session(base)
    r = s.send("I like the Suzuki Swift. My budget is £15,000 and I want a small fuel-efficient family car.")
    action_is(r, "SEARCH_VEHICLES")
    require(r.get("decision") != "NO_MATCHES", f"selected Swift falsely returned no matches: {r.get('answer')}")
    r = s.send("What about 2024 instead?")
    common(r)
    require(r.get("decision") != "NO_MATCHES", f"2024 Swift falsely returned no matches: {r.get('answer')}")
    constraints = (r.get("assistant_state") or {}).get("constraints") or {}
    models = [str(x).casefold() for x in (constraints.get("models") or [])]
    require(any("swift" in x for x in models), f"Swift was lost after year refinement: {constraints}")
    require(2024 in [constraints.get("min_year"), constraints.get("max_year")], f"2024 not preserved as year: {constraints}")
    return [s]


def case_p5(base: str) -> List[Session]:
    s = Session(base)
    r = s.send("I have £20,000 for an SUV. Compare a Honda Vezel and Toyota C-HR and tell me which you would choose.")
    action_is(r, "COMPARE_VEHICLES")
    require(r.get("decision") in {"PREFER", "COMPARE_OPTIONS", "INFORM"}, f"unexpected comparison decision: {r.get('decision')}")
    return [s]


def case_p6(base: str) -> List[Session]:
    s = Session(base)
    common(s.send("Show me 2024 Suzuki Swift listings."))
    r = s.send("White only, max 60,000 km, automatic, gallery sellers only.")
    action_is(r, "SHOW_LISTINGS")
    constraints = (r.get("assistant_state") or {}).get("constraints") or {}
    require(any("swift" in str(x).casefold() for x in (constraints.get("models") or [])), f"Swift lost: {constraints}")
    require(float(constraints.get("max_km") or 0) == 60000, f"max km lost: {constraints}")
    trans = str(constraints.get("transmission") or "").casefold()
    require("automatic" in trans or "otomatik" in trans, f"automatic lost: {constraints}")
    return [s]


def case_p7(base: str) -> List[Session]:
    s = Session(base)
    first = s.send("Is £5,000 a good price for a 2007 Nissan March 1.2L?")
    action_is(first, "EVALUATE_PURCHASE")
    require(first.get("decision") != "INSUFFICIENT_EVIDENCE", f"1.2L March lost usable comparable evidence: {first.get('answer')}")
    require("comparable" in str(first.get("answer") or "").casefold(), "1.2L March answer did not explain comparable evidence")
    r = s.send("What about 1.0?")
    action_is(r, "EVALUATE_PURCHASE")
    require(r.get("decision") != "INSUFFICIENT_EVIDENCE", f"1.0L March failed to use available thin evidence: {r.get('answer')}")
    return [s]


def business(base: str) -> Session:
    return Session(base, access_tier="BUSINESS", company="Shoptech Auto")


def case_b1(base: str) -> List[Session]:
    s = business(base)
    r = s.send("A customer wants to trade in a 2020 Toyota Yaris 1.5 automatic with 55,000 km. What should I offer?")
    action_is(r, "EVALUATE_TRADE_IN")
    return [s]


def case_b2(base: str) -> List[Session]:
    s = business(base)
    action_is(s.send("What are good vehicles for my gallery to stock right now?"), "RECOMMEND_ACQUISITIONS")
    r = s.send("Keep it under £15,000 acquisition cost and focus on small automatics.")
    action_is(r, "RECOMMEND_ACQUISITIONS")
    require(r.get("decision") != "INSUFFICIENT_EVIDENCE", f"constrained acquisition falsely returned insufficient evidence: {r.get('answer')}")
    require(len(r.get("business_options") or []) > 0, "constrained acquisition returned no business options")
    return [s]


def case_b3(base: str) -> List[Session]:
    s = business(base)
    r = s.send("Which of my vehicles are overpriced, and what should I price them at?")
    action_is(r, "ANALYZE_STOCK_PRICES")
    return [s]


def case_b4(base: str) -> List[Session]:
    s = business(base)
    r = s.send("Which of my cars have been sitting too long, and what should I do with them?")
    action_is(r, "ANALYZE_AGING_STOCK")
    return [s]


def case_b5(base: str) -> List[Session]:
    s = business(base)
    r = s.send("I can promote one of my cars this weekend. Which one should I advertise?")
    action_is(r, "RECOMMEND_AD_CANDIDATE")
    return [s]


def case_b6(base: str) -> List[Session]:
    s = business(base)
    action_is(s.send("What happened at my gallery last month?"), "ANALYZE_BUSINESS_PERIOD")
    r = s.send("Which segments were strongest for us? Break it down by price band and fuel.")
    common(r)
    return [s]


def case_b7(base: str) -> List[Session]:
    s = business(base)
    r = s.send("Sales feel slow. Is the whole market slow, or is it just us?")
    action_is(r, "ANALYZE_MARKET")
    ans=str(r.get("answer") or "").casefold()
    require("listing" in ans or "advert" in ans, "dealer-vs-market answer did not clarify listing-activity scope")
    require("not confirmed sales" in ans or "not confirmed sale" in ans, "dealer-vs-market answer did not distinguish listing exits from sales")
    return [s]


CASES: Dict[str, Callable[[str], List[Session]]] = {
    "P1": case_p1, "P2": case_p2, "P3": case_p3, "P4": case_p4,
    "P5": case_p5, "P6": case_p6, "P7": case_p7,
    "B1": case_b1, "B2": case_b2, "B3": case_b3, "B4": case_b4,
    "B5": case_b5, "B6": case_b6, "B7": case_b7,
}

CASE_CALLS = {
    "P1": 1, "P2": 1, "P3": 2, "P4": 2, "P5": 1, "P6": 2, "P7": 2,
    "B1": 1, "B2": 2, "B3": 1, "B4": 1, "B5": 1, "B6": 2, "B7": 1,
}


def write_report(path: Path, base: str, results: List[dict]) -> None:
    lines = ["# OtoDeğer 11.12 — Live API Acceptance", "", f"Backend: `{base}`", f"Expected build: `{EXPECTED_BUILD}`", ""]
    for r in results:
        lines += [f"## {r['case']} — {r['status']}", ""]
        if r.get("error"):
            lines += [f"**Error:** {r['error']}", ""]
        for session in r.get("sessions", []):
            for turn in session.get("transcript", []):
                resp = turn.get("response") or {}
                lines += [f"> **User:** {turn.get('message')}", "", f"**Action:** `{resp.get('action')}`  **Decision:** `{resp.get('decision')}`  **Mode:** `{resp.get('decision_mode')}`", "", str(resp.get("answer") or ""), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--cases", default="all", help="all or comma-separated case ids, e.g. P1,P2,B5")
    ap.add_argument("--confirm-ai-calls", required=True, help="Must be YES; this harness makes real assistant calls")
    ap.add_argument("--output-prefix", default="live_api_acceptance_11_11")
    ap.add_argument("--pause", type=float, default=0.6, help="Seconds between capability cases")
    args = ap.parse_args()
    if args.confirm_ai_calls != "YES":
        print("Refusing to run: pass --confirm-ai-calls YES", file=sys.stderr)
        return 2
    selected = list(CASES) if args.cases.casefold() == "all" else [x.strip().upper() for x in args.cases.split(",") if x.strip()]
    unknown = [x for x in selected if x not in CASES]
    if unknown:
        print(f"Unknown case(s): {unknown}", file=sys.stderr)
        return 2

    estimated_calls = sum(CASE_CALLS.get(x, 1) for x in selected)
    print(f"Selected {len(selected)} capability gates / approximately {estimated_calls} assistant requests.")
    if estimated_calls >= 20:
        print("Owner-only full acceptance is at the default 20 requests/10 min limit. If deterministic smoke or other tests were run recently, temporarily set DISABLE_ASSISTANT_RATE_LIMITS=true during this private acceptance run, then remove/set it false before public launch.")

    results = []
    failures = 0
    for case_id in selected:
        sessions = []
        print(f"[{case_id}] running...", flush=True)
        try:
            sessions = CASES[case_id](args.base_url)
            serial = [{"conversation_id": s.conversation_id, "transcript": s.transcript} for s in sessions]
            results.append({"case": case_id, "status": "PASS", "sessions": serial})
            print(f"[{case_id}] PASS", flush=True)
        except Exception as exc:
            failures += 1
            sessions = locals().get("sessions", []) if isinstance(locals().get("sessions", []), list) else []
            serial = [{"conversation_id": s.conversation_id, "transcript": s.transcript} for s in sessions if isinstance(s, Session)]
            results.append({"case": case_id, "status": "FAIL", "error": str(exc), "sessions": serial})
            print(f"[{case_id}] FAIL — {exc}", flush=True)
        time.sleep(max(0.0, args.pause))

    prefix = Path(args.output_prefix)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps({"base_url": args.base_url, "expected_build": EXPECTED_BUILD, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, args.base_url, results)
    print(f"\nLive API acceptance: {len(selected)-failures}/{len(selected)} structural gates PASS")
    print(f"Saved: {json_path} and {md_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
