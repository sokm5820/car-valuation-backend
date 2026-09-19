"""Stripe Checkout + Clerk-bound, Redis-backed sandbox billing for the AI assistant.

IMPORTANT: This is a TEST integration. Before taking live payments, complete tax,
refund, account recovery, durable billing storage, and server-side report gating.
Stripe credentials and price IDs are supplied only through server environment vars.
"""
from __future__ import annotations

import hashlib
import os
import time
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request

billing = Blueprint("billing", __name__)

PLAN_CONFIG = {
    "personal": ("STRIPE_PRICE_PERSONAL", "payment", 24 * 3600, "PERSONAL"),
    "personal_plus": ("STRIPE_PRICE_PERSONAL_PLUS", "payment", 30 * 24 * 3600, "PERSONAL_PLUS"),
    "business_monthly": ("STRIPE_PRICE_BUSINESS_MONTHLY", "subscription", 0, "BUSINESS"),
    "business_annual": ("STRIPE_PRICE_BUSINESS_ANNUAL", "subscription", 0, "BUSINESS"),
}
SCOPE = "assistant:stripe:v1"


def _k(*parts):
    return ":".join((SCOPE, *[str(p) for p in parts]))


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:32]


def _stripe():
    import stripe
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY is not configured")
    stripe.api_key = key
    return stripe


def _redis():
    from otodeger_access_control import get_access_manager
    manager = get_access_manager()
    if not manager.has_shared_backend:
        raise RuntimeError("REDIS_URL is required for Stripe entitlements")
    return manager.backend.redis


def _auth():
    # Reuse the application's existing server-verified Clerk authentication.
    from otodeger_access_control import get_access_manager
    manager = get_access_manager()
    claims = manager._verify_clerk_claims(request)
    if not claims or not claims.get("sub"):
        raise PermissionError("Please sign in before purchasing access")
    return manager, claims


def _pass_key(user_id, tier):
    return _k("passes", _hash(user_id), tier)


def _subscription_key(sub_id):
    return _k("subscription", sub_id)


def _price_for(plan):
    return os.environ.get(PLAN_CONFIG[plan][0], "").strip()


def _period_end(sub):
    end = sub.get("current_period_end")
    items = sub.get("items", {}).get("data", [])
    if not end and items:
        end = items[0].get("current_period_end")
    return int(end or 0)


def _is_business_price(price_id):
    return price_id in {_price_for("business_monthly"), _price_for("business_annual")}


def read_paid_access(redis_client, user_id, clerk_org_id=""):
    """Authoritative account snapshot from signed, server-owned billing records."""
    now = int(time.time())
    def pass_until(tier):
        scores = redis_client.zrevrange(_pass_key(user_id, tier), 0, 0, withscores=True)
        return int(scores[0][1]) if scores and scores[0][1] > now else 0

    result = {
        "personal_until": pass_until("PERSONAL"),
        "personal_plus_until": pass_until("PERSONAL_PLUS"),
        "business_until": 0,
        "business_scope": "",
        "business_subscription_id": "",
        "business_customer_id": "",
    }
    # An active, paid gallery organisation takes priority over an independent plan.
    scopes = [("org", clerk_org_id)] if clerk_org_id else []
    scopes.append(("user", user_id))
    for scope, subject in scopes:
        for sid in redis_client.smembers(_k("subscriptions", scope, _hash(subject))):
            record = redis_client.hgetall(_subscription_key(sid))
            until = int(record.get("paid_until") or 0)
            if record.get("status") not in {"active", "past_due"} or until <= now:
                continue
            if record.get("scope") != scope or record.get("subject") != subject:
                continue
            if not _is_business_price(record.get("price_id")):
                continue
            if until > result["business_until"] or (scope == "org" and result["business_scope"] != "org"):
                result.update({
                    "business_until": until,
                    "business_scope": scope,
                    "business_subscription_id": sid,
                    "business_customer_id": record.get("customer_id", ""),
                })
        if result["business_scope"] == "org":
            break
    return result


def _grant_pass(session):
    """Idempotent, atomic grant: a webhook replay cannot extend the same purchase."""
    if session.get("payment_status") != "paid" or session.get("mode") != "payment":
        return
    meta = session.get("metadata") or {}
    plan = meta.get("plan")
    if plan not in ("personal", "personal_plus"):
        return
    user_id = meta.get("clerk_user_id", "")
    if not user_id or session.get("client_reference_id") != user_id:
        raise ValueError("Checkout identity mismatch")
    price_id = _price_for(plan)
    if not price_id:
        raise RuntimeError("Stripe price is not configured")
    # The session was created by this backend with a server-selected price.
    # Independently retrieve line items before granting access to guard against
    # accidental Stripe Dashboard/manual sessions carrying similar metadata.
    stripe = _stripe()
    lines = stripe.checkout.Session.list_line_items(session["id"], limit=2).get("data", [])
    if len(lines) != 1 or lines[0].get("price", {}).get("id") != price_id or int(lines[0].get("quantity") or 0) != 1:
        raise ValueError("Checkout price mismatch")
    ttl, tier = PLAN_CONFIG[plan][2], PLAN_CONFIG[plan][3]
    receipt = _k("receipt", session["id"])
    expiry = int(time.time()) + ttl
    redis_client = _redis()
    # Atomic transaction; both receipt and entitlement are written together.
    script = """
    if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
    redis.call('ZADD', KEYS[2], ARGV[1], ARGV[2])
    redis.call('HSET', KEYS[1], 'user_id', ARGV[3], 'tier', ARGV[4],
       'payment_intent', ARGV[5], 'status', 'paid', 'expires_at', ARGV[1])
    if ARGV[5] ~= '' then redis.call('SET', KEYS[3], ARGV[2]) end
    return 1
    """
    redis_client.eval(
        script, 3, receipt, _pass_key(user_id, tier), _k("intent", session.get("payment_intent") or "none"),
        expiry, session["id"], user_id, tier, session.get("payment_intent") or "",
    )


def _sync_subscription(subscription, paid=False):
    meta = subscription.get("metadata") or {}
    scope, subject, purchaser = meta.get("billing_scope"), meta.get("billing_subject"), meta.get("clerk_user_id")
    if scope not in {"org", "user"} or not subject or not purchaser:
        return
    items = subscription.get("items", {}).get("data", [])
    if len(items) != 1 or not _is_business_price(items[0].get("price", {}).get("id")):
        return
    sid = subscription["id"]
    client = _redis()
    old = client.hgetall(_subscription_key(sid))
    previous_until = int(old.get("paid_until") or 0)
    # On renewals only a paid invoice may extend the entitlement period.
    paid_until = max(previous_until, _period_end(subscription)) if paid else previous_until
    record = {
        "scope": scope, "subject": subject, "purchaser": purchaser,
        "price_id": items[0]["price"]["id"],
        "customer_id": subscription.get("customer") or "",
        "status": subscription.get("status") or "unknown",
        "paid_until": str(paid_until),
    }
    pipe = client.pipeline(transaction=True)
    pipe.hset(_subscription_key(sid), mapping=record)
    pipe.sadd(_k("subscriptions", scope, _hash(subject)), sid)
    pipe.execute()


def _fulfill_checkout(session):
    if session.get("payment_status") != "paid":
        return
    if session.get("mode") == "payment":
        _grant_pass(session)
    elif session.get("mode") == "subscription" and session.get("subscription"):
        stripe = _stripe()
        sub = stripe.Subscription.retrieve(session["subscription"])
        _sync_subscription(sub, paid=True)


def _error(message, status=400):
    return jsonify({"success": False, "error": str(message)}), status


@billing.post("/api/billing/checkout")
def start_checkout():
    try:
        manager, claims = _auth()
        if not manager.has_shared_backend:
            return _error("Shared entitlement storage is not configured", 503)
        data = request.get_json(silent=True) or {}
        plan = str(data.get("plan") or "")
        if plan not in PLAN_CONFIG:
            return _error("Unknown plan")
        price = _price_for(plan)
        if not price:
            return _error("Stripe price ID is missing", 503)
        stripe = _stripe()
        user_id = str(claims["sub"])
        org_id = str(claims.get("org_id") or "")
        paid = read_paid_access(_redis(), user_id, org_id)
        if plan == "personal" and (paid["personal_until"] or paid["personal_plus_until"] or paid["business_until"]):
            return _error("You already have active access", 409)
        if plan == "personal_plus" and (paid["personal_plus_until"] or paid["business_until"]):
            return _error("You already have active access", 409)

        kwargs = {}
        if plan.startswith("business_"):
            scope = str(data.get("business_scope") or "independent")
            if scope == "gallery":
                role = str(claims.get("org_role") or "").lower()
                if not org_id or org_id not in manager.business_orgs or role not in {"admin", "owner", "org:admin"}:
                    return _error("Gallery access requires an approved Clerk organisation and an admin account. Contact support before buying.", 409)
                subject, billing_scope = org_id, "org"
            elif scope == "independent":
                subject, billing_scope = user_id, "user"
            else:
                return _error("Unknown Business account type")
            if paid["business_until"]:
                return _error("You already have an active Business subscription. Use Manage subscription to change it.", 409)
            metadata = {"clerk_user_id": user_id, "billing_scope": billing_scope, "billing_subject": subject, "plan": plan}
            kwargs["subscription_data"] = {"metadata": metadata}
            mode = "subscription"
        else:
            metadata = {"clerk_user_id": user_id, "plan": plan}
            mode = "payment"

        # Set APP_FRONTEND_URL to the actual AI assistant origin at deployment.
        frontend = os.environ.get("APP_FRONTEND_URL", "http://localhost:5173").rstrip("/")
        parsed = urlparse(frontend)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            return _error("APP_FRONTEND_URL is invalid", 503)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            return _error("A deployed checkout return URL must use HTTPS", 503)
        if mode == "payment":
            kwargs["customer_creation"] = "always"
        session = stripe.checkout.Session.create(
            mode=mode, line_items=[{"price": price, "quantity": 1}],
            client_reference_id=user_id, metadata=metadata,
            success_url=f"{frontend}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{frontend}/?checkout=cancelled",
            payment_method_types=["card"],
            **kwargs,
        )
        return jsonify({"success": True, "url": session.url})
    except PermissionError as exc:
        return _error(exc, 401)
    except Exception as exc:
        print("STRIPE CHECKOUT ERROR:", repr(exc), flush=True)
        return _error("Checkout is temporarily unavailable", 503)


@billing.get("/api/billing/checkout-status")
def checkout_status():
    """Authenticated recovery path, not just an untrusted browser redirect."""
    try:
        _, claims = _auth()
        sid = str(request.args.get("session_id") or "")
        if not sid.startswith(("cs_test_", "cs_live_")) or len(sid) > 255:
            return _error("Invalid checkout session")
        session = _stripe().checkout.Session.retrieve(sid)
        if session.get("client_reference_id") != claims["sub"]:
            return _error("This checkout belongs to another account", 403)
        if session.get("payment_status") == "paid":
            _fulfill_checkout(session)
        return jsonify({"success": True, "payment_status": session.get("payment_status")})
    except PermissionError as exc:
        return _error(exc, 401)
    except Exception as exc:
        print("STRIPE CHECKOUT STATUS ERROR:", repr(exc), flush=True)
        return _error("Could not verify this checkout", 503)


@billing.post("/api/billing/portal")
def open_portal():
    try:
        _, claims = _auth()
        user_id = claims["sub"]
        paid = read_paid_access(_redis(), user_id, str(claims.get("org_id") or ""))
        customer_id = paid["business_customer_id"]
        if not customer_id:
            return _error("No active Business subscription was found", 404)
        if paid["business_scope"] == "org" and str(claims.get("org_role") or "").lower() not in {"admin", "owner", "org:admin"}:
            return _error("Only an organisation admin can manage billing", 403)
        frontend = os.environ.get("APP_FRONTEND_URL", "http://localhost:5173").rstrip("/")
        session = _stripe().billing_portal.Session.create(customer=customer_id, return_url=frontend)
        return jsonify({"success": True, "url": session.url})
    except PermissionError as exc:
        return _error(exc, 401)
    except Exception as exc:
        print("STRIPE PORTAL ERROR:", repr(exc), flush=True)
        return _error("Subscription management is temporarily unavailable", 503)


@billing.post("/api/billing/webhook")
def stripe_webhook():
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not secret:
        return _error("Webhook signing is not configured", 503)
    try:
        event = _stripe().Webhook.construct_event(
            request.get_data(cache=False), request.headers.get("Stripe-Signature", ""), secret,
        )
    except Exception as exc:
        # Never process unauthenticated or malformed events.
        print("STRIPE WEBHOOK VERIFICATION ERROR:", repr(exc), flush=True)
        return _error("Invalid webhook signature", 400)
    try:
        kind, obj = event["type"], event["data"]["object"]
        if kind in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
            _fulfill_checkout(obj)
        elif kind == "invoice.paid":
            parent = obj.get("parent") or {}
            sub_id = obj.get("subscription") or (parent.get("subscription_details") or {}).get("subscription")
            if sub_id:
                _sync_subscription(_stripe().Subscription.retrieve(sub_id), paid=True)
        elif kind in {"customer.subscription.updated", "customer.subscription.deleted"}:
            # Webhook delivery may be out of order: use Stripe's current state.
            _sync_subscription(_stripe().Subscription.retrieve(obj["id"]), paid=False)
        elif kind == "charge.refunded":
            # A *fully* refunded Personal pass no longer grants access.
            if int(obj.get("amount_refunded") or 0) >= int(obj.get("amount") or 1):
                intent = obj.get("payment_intent") or ""
                client = _redis()
                receipt_id = client.get(_k("intent", intent)) if intent else None
                if receipt_id:
                    receipt = client.hgetall(_k("receipt", receipt_id))
                    if receipt.get("user_id") and receipt.get("tier"):
                        pipe = client.pipeline(transaction=True)
                        pipe.zrem(_pass_key(receipt["user_id"], receipt["tier"]), receipt_id)
                        pipe.hset(_k("receipt", receipt_id), "status", "refunded")
                        pipe.execute()
        return jsonify({"received": True})
    except Exception as exc:
        # Stripe retries 5xx events; never acknowledge failed entitlement writes.
        print("STRIPE WEBHOOK PROCESSING ERROR:", repr(exc), flush=True)
        return _error("Webhook processing failed", 503)
