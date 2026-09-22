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

from flask import Blueprint, jsonify, request, current_app

billing = Blueprint("billing", __name__)

PLAN_CONFIG = {
    "personal": ("STRIPE_PRICE_PERSONAL", "payment", 24 * 3600, "PERSONAL"),
    "personal_plus": ("STRIPE_PRICE_PERSONAL_PLUS", "payment", 30 * 24 * 3600, "PERSONAL_PLUS"),
    "business_monthly": ("STRIPE_PRICE_BUSINESS_MONTHLY", "subscription", 0, "BUSINESS"),
    "business_annual": ("STRIPE_PRICE_BUSINESS_ANNUAL", "subscription", 0, "BUSINESS"),
    "business_independent_monthly": ("STRIPE_PRICE_INDEPENDENT_MONTHLY", "subscription", 0, "BUSINESS"),
    "business_independent_annual": ("STRIPE_PRICE_INDEPENDENT_ANNUAL", "subscription", 0, "BUSINESS"),
}
# Gallery prices are NOT interchangeable with Independent prices.
GALLERY_PLANS = frozenset({"business_monthly", "business_annual"})
INDEPENDENT_PLANS = frozenset({"business_independent_monthly", "business_independent_annual"})
EXPECTED_BUSINESS_PRICES = {
    "business_monthly": (6499 * 100, "month"),
    "business_annual": (49999 * 100, "year"),
    "business_independent_monthly": (3999 * 100, "month"),
    "business_independent_annual": (34999 * 100, "year"),
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
    return bool(price_id) and price_id in { _price_for(plan) for plan in EXPECTED_BUSINESS_PRICES }


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
            # A paid gallery-labelled subscription is displayed before any
            # overlapping Independent subscription for this same user.
            prefer_gallery = (scope == "user" and bool(record.get("gallery_name"))
                              and result["business_scope"] != "org"
                              and not result.get("business_gallery_name"))
            if (until > result["business_until"] or prefer_gallery
                    or (scope == "org" and result["business_scope"] != "org")) and not (result.get("business_gallery_name") and scope == "user" and not record.get("gallery_name")):
                result.update({
                    "business_until": until,
                    "business_scope": scope,
                    "business_subscription_id": sid,
                    "business_customer_id": record.get("customer_id", ""),
                    "business_gallery_name": record.get("gallery_name", ""),
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
    plan = meta.get("plan")
    actual_price_id = items[0].get("price", {}).get("id")
    # Do not grant a Gallery subscription for an Independent price or vice versa.
    if plan not in EXPECTED_BUSINESS_PRICES or _price_for(plan) != actual_price_id:
        return
    if plan in INDEPENDENT_PLANS and (scope != "user" or meta.get("gallery_name")):
        return
    if plan in GALLERY_PLANS and scope == "user" and not meta.get("gallery_name"):
        return
    sid = subscription["id"]
    client = _redis()
    # Gallery intelligence is sold for a single publicly listed gallery, not
    # restricted to its owner. The validated gallery selected at checkout is
    # bound to the subscription; no manual approval or saved-card setup is used.
    if plan in GALLERY_PLANS:
        # Preserve existing organisation subscriptions. New direct-checkout
        # Gallery subscriptions are always user-scoped to one selected gallery.
        if scope == "user" and (subject != purchaser or not meta.get("gallery_name")):
            return
        if scope == "org" and meta.get("gallery_name"):
            return

    # When the purchaser disables Auto-Renew, schedule the subscription to end
    # at the end of its PAID term. Stripe is authoritative for billing; a client
    # toggle alone must never promise cancellation without this server update.
    if meta.get("auto_renew", "true") == "false" and not subscription.get("cancel_at_period_end"):
        subscription = _stripe().Subscription.modify(sid, cancel_at_period_end=True)
    old = client.hgetall(_subscription_key(sid))
    previous_until = int(old.get("paid_until") or 0)
    # On renewals only a paid invoice may extend the entitlement period.
    paid_until = max(previous_until, _period_end(subscription)) if paid else previous_until
    record = {
        "scope": scope, "subject": subject, "purchaser": purchaser,
        "price_id": items[0]["price"]["id"],
        "customer_id": subscription.get("customer") or "",
        "gallery_name": str(meta.get("gallery_name") or ""),
        "gallery_request_id": str(meta.get("gallery_request_id") or ""),
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


def _frontend_url():
    frontend = os.environ.get("APP_FRONTEND_URL", "http://localhost:5173").rstrip("/")
    parsed = urlparse(frontend)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise RuntimeError("APP_FRONTEND_URL is invalid")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise RuntimeError("A deployed checkout return URL must use HTTPS")
    return frontend


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
            if "auto_renew" in data and type(data["auto_renew"]) is not bool:
                return _error("Invalid Auto-Renew selection")
            auto_renew = data.get("auto_renew", True) is True
            scope = str(data.get("business_scope") or "")
            if (scope == "gallery" and plan not in GALLERY_PLANS) or (scope == "independent" and plan not in INDEPENDENT_PLANS):
                return _error("The selected Business type does not match its subscription price", 400)
            if scope not in {"gallery", "independent"}:
                return _error("Please select a Business account type", 400)
            # Verify the exact Stripe product amount and recurring interval before
            # presenting ANY Business checkout, never trust a client-supplied price.
            actual_price = stripe.Price.retrieve(price)
            expected_amount, expected_interval = EXPECTED_BUSINESS_PRICES[plan]
            if (actual_price.get("currency") != "try" or actual_price.get("unit_amount") != expected_amount
                    or (actual_price.get("recurring") or {}).get("interval") != expected_interval):
                return _error("Business subscription pricing is not configured correctly; no payment has been requested", 503)
            if scope == "gallery":
                selected_gallery = str(data.get("gallery_name") or "").strip()
                if not selected_gallery or len(selected_gallery) > 160:
                    return _error("Please select a gallery before continuing")
                gallery_view = current_app.view_functions.get("api_business_gallery_options")
                if gallery_view is None:
                    return _error("Gallery directory is unavailable", 503)
                gallery_response = gallery_view()
                gallery_catalogue = gallery_response.get_json(silent=True) or {}
                if not gallery_catalogue.get("success"):
                    return _error("Gallery directory is temporarily unavailable", 503)
                canonical = next((name for name in gallery_catalogue.get("galleries", []) if name == selected_gallery), "")
                if not canonical:
                    return _error("Please choose a gallery from the available options", 400)
                # Direct paid checkout, identical to Independent: no setup-only
                # session, phone form, manual approval, or deferred charge.
                # The gallery is fixed in the server-owned subscription record.
                selected_gallery = canonical
                subject, billing_scope = user_id, "user"
            elif scope == "independent":
                if data.get("gallery_name"):
                    return _error("Independent subscriptions cannot select a gallery", 400)
                subject, billing_scope = user_id, "user"
            else:
                return _error("Unknown Business account type")
            if paid["business_until"]:
                return _error("You already have an active Business subscription. Use Manage subscription to change it.", 409)
            metadata = {"clerk_user_id": user_id, "billing_scope": billing_scope,
                        "billing_subject": subject, "plan": plan, "auto_renew": str(auto_renew).lower()}
            if scope == "gallery":
                metadata["gallery_name"] = selected_gallery
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
        return jsonify({"success": True, "url": session.url, "auto_renew": auto_renew if plan.startswith("business_") else None})
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
        if session.get("mode") == "setup":
            # Retired manual-review flow; no new setup sessions are created.
            return _error("The old gallery checkout is no longer supported. Please start a new purchase.", 410)
        recorded = False
        plan = str((session.get("metadata") or {}).get("plan") or "")
        if session.get("payment_status") == "paid":
            _fulfill_checkout(session)
            paid = read_paid_access(_redis(), str(claims["sub"]), str(claims.get("org_id") or ""))
            if plan == "personal":
                recorded = bool(paid["personal_until"])
            elif plan == "personal_plus":
                recorded = bool(paid["personal_plus_until"])
            elif plan in EXPECTED_BUSINESS_PRICES:
                recorded = bool(paid["business_until"] and
                                paid["business_subscription_id"] == session.get("subscription"))
        return jsonify({"success": True, "payment_status": session.get("payment_status"),
                        "plan": plan, "entitlement_recorded": recorded})
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
                subscription = _stripe().Subscription.retrieve(sub_id)
                _sync_subscription(subscription, paid=True)
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
