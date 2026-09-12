"""Currency normalisation for OtoDeğer.

The market database remains GBP-native.  User-entered TRY/TL, USD and EUR
amounts are converted to GBP before the Gold decision engine sees the turn.
The original amount plus source date/rate is returned as metadata so the reply
can disclose the conversion instead of silently changing the user's budget.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from otodeger_v11_contract import parse_number, ContractError


SUPPORTED = {"GBP", "TRY", "USD", "EUR"}
SYMBOL_TO_CODE = {"£": "GBP", "₺": "TRY", "$": "USD", "€": "EUR"}
WORD_TO_CODE = {
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "sterlin": "GBP",
    "try": "TRY", "tl": "TRY", "lira": "TRY", "liras": "TRY", "lirası": "TRY", "lirasi": "TRY",
    "usd": "USD", "dollar": "USD", "dollars": "USD", "dolar": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "avro": "EUR",
}

# Conservative numeric shape: only activated when directly attached to a known
# currency symbol/word, so years/mileage are never interpreted as money.
_NUMBER = r"\d(?:[\d\s\u00a0\u202f.,'’]*\d)?(?:\s*(?:k|bin|thousand|m|mn|million|milyon|млн|тыс\.?))?"
_PREFIX_RE = re.compile(rf"(?P<currency>[£₺$€])\s*(?P<amount>{_NUMBER})", re.I)
_SUFFIX_RE = re.compile(
    rf"(?P<amount>{_NUMBER})\s*(?P<currency>GBP|pounds?|sterling|sterlin|TRY|TL|lirası|lirasi|liras?|USD|dollars?|dolar|EUR|euros?|euro|avro)\b",
    re.I,
)
_MULTIPLIER_RE = re.compile(r"\s*(?P<suffix>k|bin|thousand|m|mn|million|milyon|млн|тыс\.?)\s*$", re.I)


class FXUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FXConversion:
    original_text: str
    original_amount: float
    original_currency: str
    gbp_amount: float
    rate_to_gbp: float
    rate_date: str
    source: str

    def public_payload(self) -> Dict[str, Any]:
        return asdict(self)


def _currency_code(raw: str) -> Optional[str]:
    raw = str(raw or "").strip()
    if raw in SYMBOL_TO_CODE:
        return SYMBOL_TO_CODE[raw]
    return WORD_TO_CODE.get(raw.casefold())


def _parse_amount(raw: str) -> float:
    text = str(raw or "").strip()
    multiplier = 1.0
    match = _MULTIPLIER_RE.search(text)
    if match:
        suffix = match.group("suffix").casefold().rstrip(".")
        multiplier = 1_000_000.0 if suffix in {"m", "mn", "million", "milyon", "млн"} else 1000.0
        text = text[:match.start()].strip()
    # Contract parser already handles Turkish/English grouping conventions.
    amount = float(parse_number(text)) * multiplier
    if amount <= 0 or amount > 1_000_000_000:
        raise ContractError("Currency amount outside supported range")
    return amount


class FXRateService:
    def __init__(self):
        self.base_url = str(os.environ.get("FX_API_BASE_URL") or "https://api.frankfurter.dev/v2/rate").rstrip("/")
        self.timeout = max(1.0, min(float(os.environ.get("FX_API_TIMEOUT_SECONDS", "5")), 15.0))
        self.cache_seconds = max(60, min(int(os.environ.get("FX_CACHE_SECONDS", "3600")), 86400))
        self._cache: Dict[Tuple[str, str], Tuple[float, str, float]] = {}
        self._lock = threading.RLock()

    def rate(self, base: str, quote: str = "GBP") -> Tuple[float, str]:
        base = str(base or "").upper()
        quote = str(quote or "").upper()
        if base not in SUPPORTED or quote not in SUPPORTED:
            raise FXUnavailable("Unsupported currency")
        if base == quote:
            return 1.0, time.strftime("%Y-%m-%d", time.gmtime())

        key = (base, quote)
        now = time.time()
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[2] > now:
                return cached[0], cached[1]

        url = f"{self.base_url}/{base}/{quote}"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            rate = float(payload.get("rate"))
            date = str(payload.get("date") or "").strip() or time.strftime("%Y-%m-%d", time.gmtime())
            if not (0 < rate < 1_000_000):
                raise ValueError("Invalid FX rate")
        except Exception as exc:
            # Do not silently use an old/stale rate: the user explicitly asked us
            # to handle budgets with up-to-date conversion.
            raise FXUnavailable(f"Could not retrieve {base}/{quote} rate") from exc

        with self._lock:
            self._cache[key] = (rate, date, now + self.cache_seconds)
        return rate, date


_SERVICE: Optional[FXRateService] = None
_SERVICE_LOCK = threading.Lock()


def get_fx_service() -> FXRateService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = FXRateService()
        return _SERVICE


def reset_fx_service_for_tests() -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = None


def _find_mentions(message: str):
    candidates = []
    for pattern in (_PREFIX_RE, _SUFFIX_RE):
        for match in pattern.finditer(message):
            code = _currency_code(match.group("currency"))
            if not code:
                continue
            try:
                amount = _parse_amount(match.group("amount"))
            except (ContractError, ValueError, TypeError):
                continue
            candidates.append((match.start(), match.end(), match.group(0), amount, code))
    # Remove overlaps, preferring the first/longest match.
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected = []
    last_end = -1
    for item in candidates:
        if item[0] < last_end:
            continue
        selected.append(item)
        last_end = item[1]
    return selected


def normalize_message_currency(message: str, service: Optional[FXRateService] = None) -> Tuple[str, List[FXConversion]]:
    text = str(message or "")
    mentions = _find_mentions(text)
    if not mentions:
        return text, []
    service = service or get_fx_service()
    replacements = []
    conversions: List[FXConversion] = []
    for start, end, original, amount, currency in mentions:
        if currency == "GBP":
            continue
        rate, rate_date = service.rate(currency, "GBP")
        gbp = amount * rate
        # Search/decision amounts are whole pounds; retain exact rate metadata.
        rounded = float(round(gbp))
        replacement = f"£{int(rounded):,}"
        replacements.append((start, end, replacement))
        conversions.append(FXConversion(
            original_text=original,
            original_amount=float(amount),
            original_currency=currency,
            gbp_amount=rounded,
            rate_to_gbp=float(rate),
            rate_date=rate_date,
            source="Frankfurter",
        ))

    if not replacements:
        return text, []
    result = text
    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]
    return result, conversions


def conversion_note(conversions: List[FXConversion], language: str) -> str:
    if not conversions:
        return ""
    c = conversions[0]
    original = c.original_text.strip()
    gbp = f"£{int(round(c.gbp_amount)):,}"
    date = c.rate_date
    lang = str(language or "TR").upper()
    if lang == "RU":
        return f"Я пересчитал {original} примерно в **{gbp}** по последнему доступному курсу ({date}); все рыночные цены ниже указаны в GBP."
    if lang == "EN":
        return f"I've treated {original} as approximately **{gbp}** using the latest available exchange rate ({date}); all market prices below are in GBP."
    return f"{original} bütçenizi son mevcut döviz kuruyla ({date}) yaklaşık **{gbp}** olarak değerlendirdim; aşağıdaki tüm piyasa fiyatları GBP cinsindendir."


__all__ = [
    "FXUnavailable",
    "FXConversion",
    "FXRateService",
    "get_fx_service",
    "reset_fx_service_for_tests",
    "normalize_message_currency",
    "conversion_note",
]
