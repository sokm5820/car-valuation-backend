"""Assistant-only boundary contracts. No network, Flask or valuation dependencies.

Number convention: a single separator followed by exactly three digits denotes
thousands (25.000 == 25,000). A k/bin/thousand suffix makes decimal notation
explicit (25,5k == 25,500). Malformed grouping is rejected rather than guessed.
"""
import math
import re
import unicodedata
from collections.abc import Mapping


class ContractError(ValueError):
    pass


SUFFIX = r"(?:k|bin|thousand|тыс\.?|тысяч(?:а|и)?)"
NUMBER = rf"\d+(?:(?:[.,'’]|[ \u00a0\u202f])\d+)*(?:\s*{SUFFIX})?"
NUMBER_RE = re.compile(rf"(?<![\w.,])(?P<number>{NUMBER})(?!\w|[.,]\d)", re.I)


def parse_number(value):
    if isinstance(value, bool) or value is None:
        raise ContractError("A number is required")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = unicodedata.normalize('NFKC', value).strip().casefold()
        text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
        text = re.sub(r"^(?:£|gbp)\s*", '', text)
        text = re.sub(r"\s*(?:gbp|pounds?|sterlin(?:e)?|фунт(?:ов|а|ы)?)$", '', text).strip()
        suffix = re.search(rf"\s*{SUFFIX}$", text, re.I)
        multiplier = 1000 if suffix else 1
        if suffix:
            text = text[:suffix.start()].strip()
        sign = -1 if text.startswith('-') else 1
        if text.startswith(('+', '-')):
            text = text[1:]
        # Space/apostrophe groups must be complete groups of three.
        if re.search(r"[ '’]", text):
            if not re.fullmatch(r"\d{1,3}(?:[ '’]\d{3})+(?:[.,]\d{1,2})?", text):
                raise ContractError("Invalid digit grouping")
            text = re.sub(r"[ '’]", '', text)
        if ',' in text and '.' in text:
            decimal = ',' if text.rfind(',') > text.rfind('.') else '.'
            group = '.' if decimal == ',' else ','
            whole, fraction = text.rsplit(decimal, 1)
            if not re.fullmatch(rf"\d{{1,3}}(?:{re.escape(group)}\d{{3}})+", whole) or not re.fullmatch(r"\d{1,2}", fraction):
                raise ContractError("Invalid number separators")
            text = whole.replace(group, '') + '.' + fraction
        elif ',' in text or '.' in text:
            separator = ',' if ',' in text else '.'
            parts = text.split(separator)
            if multiplier == 1 and 1 <= len(parts[0]) <= 3 and all(len(p) == 3 and p.isdigit() for p in parts[1:]):
                text = ''.join(parts)
            elif len(parts) == 2 and all(p.isdigit() for p in parts) and 1 <= len(parts[1]) <= 2:
                text = '.'.join(parts)
            else:
                raise ContractError("Invalid number separators")
        if not re.fullmatch(r"\d+(?:\.\d+)?", text):
            raise ContractError("Invalid number")
        result = sign * float(text) * multiplier
    else:
        raise ContractError("Expected a number or numeric text")
    if not math.isfinite(result):
        raise ContractError("Number must be finite")
    return result


def clarification(value):
    """Normalize legacy strings and structured clarifications into one shape."""
    if value is None or value == '':
        return None
    if isinstance(value, str):
        return {'question': value.strip()[:1000], 'field': None, 'options': []}
    if not isinstance(value, Mapping):
        raise ContractError('Clarification must be text or an object')
    question = value.get('question')
    field = value.get('field')
    options = value.get('options')
    if question is not None and not isinstance(question, str):
        raise ContractError('Clarification question must be text')
    if field is not None and not isinstance(field, str):
        raise ContractError('Clarification field must be text')
    if options is None:
        options = []
    if not isinstance(options, list) or any(not isinstance(x, str) for x in options):
        raise ContractError('Clarification options must be a list of text')
    return {'question': (question or '').strip()[:1000], 'field': field,
            'options': list(dict.fromkeys(x.strip()[:160] for x in options if x.strip()))[:30]}
