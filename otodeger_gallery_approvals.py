"""OtoDost gallery review, bound to Clerk users and server-owned Stripe setup sessions.

Sandbox-first: Redis holds pending and approved identities. A production launch
requires durable, backed-up storage and review of subscription recovery procedures.
Never treat user-typed phone numbers or WhatsApp URLs as proof of authority.
"""
from __future__ import annotations

import os
import re
import secrets
import time
from urllib.parse import quote

from flask import Blueprint, jsonify, request

gallery_approvals = Blueprint("gallery_approvals", __name__)
PREFIX = "otodost:gallery:review:v1"


def _key(*parts):
    return ":".join((PREFIX, *[str(p) for p in parts]))


def _stripe_from_gallery():
    from otodeger_stripe_billing import _stripe
    return _stripe()


def mark_gallery_invoice_paid(subscription):
    metadata = subscription.get("metadata") or {}
    rid = str(metadata.get("gallery_request_id") or "")
    if not rid:
        return
    from otodeger_stripe_billing import _redis
    redis = _redis()
    row = _row(redis, rid)
    if row and row.get("user_id") == metadata.get("clerk_user_id") and row.get("gallery_name") == metadata.get("gallery_name") and approved_gallery(redis, row["user_id"]) == row["gallery_name"] and row.get("status") not in {"denied", "pending", "awaiting_contact", "awaiting_setup"}:
        redis.hset(_key("request", rid), mapping={"status": "paid_active", "payment_status": "paid"})
        redis.zrem(_key("pending"), rid)


def _services():
    from otodeger_stripe_billing import _auth, _redis, _stripe
    manager, claims = _auth()
    if not manager.has_shared_backend:
        raise RuntimeError("Shared storage unavailable")
    return manager, claims, _redis(), _stripe()


def _admin(claims):
    # Not a browser role or caller-supplied email: only server-verified Clerk sub.
    allowed = {part.strip() for part in os.environ.get("OTODOST_GALLERY_ADMIN_USER_IDS", "").split(",") if part.strip()}
    if not allowed or str(claims.get("sub") or "") not in allowed:
        raise PermissionError("Gallery approval administrator only")


def _row(redis, request_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{16,80}", str(request_id or "")):
        return {}
    row = redis.hgetall(_key("request", request_id))
    return row if row and row.get("request_id") == request_id else {}


def _public(row, admin=False):
    fields = ("request_id", "gallery_name", "plan", "status", "contact_method", "created_at", "payment_status")
    if admin:
        fields += ("user_id", "name_claimed", "phone_claimed", "checkout_session", "approved_by", "verified_phone", "review_note", "subscription_id", "price_id")
    return {k: row.get(k, "") for k in fields}


def approved_gallery(redis, user_id):
    row = redis.hgetall(_key("approved", user_id))
    return row.get("gallery_name", "") if row.get("status") == "approved" and row.get("user_id") == user_id else ""


def new_setup_request(redis, *, user_id, gallery_name, plan, session, price_id, auto_renew=True):
    """Store server-selected plan, exact gallery, and Checkout Session identity."""
    request_id = secrets.token_urlsafe(20)
    row = {
        "request_id": request_id, "user_id": user_id, "gallery_name": gallery_name,
        "plan": plan, "price_id": price_id, "checkout_session": session.id,
        "auto_renew": "true" if auto_renew else "false", "status": "awaiting_setup",
        "contact_method": "", "name_claimed": "", "phone_claimed": "", "created_at": str(int(time.time())),
        "payment_status": "not_charged", "subscription_id": "",
    }
    pipe = redis.pipeline(transaction=True)
    pipe.hset(_key("request", request_id), mapping=row)
    pipe.set(_key("checkout", session.id), request_id)
    pipe.set(_key("latest", user_id), request_id)
    pipe.execute()
    return request_id


def complete_setup(redis, stripe, session, user_id):
    """A setup-only Checkout can never be mistaken for a paid subscription."""
    if session.get("mode") != "setup" or session.get("status") != "complete":
        raise ValueError("Payment details have not been saved")
    if session.get("client_reference_id") != user_id:
        raise PermissionError("Checkout account mismatch")
    request_id = redis.get(_key("checkout", session["id"]))
    row = _row(redis, request_id)
    if not row or row.get("user_id") != user_id or row.get("checkout_session") != session["id"]:
        raise PermissionError("Unknown gallery verification request")
    setup = stripe.SetupIntent.retrieve(session.get("setup_intent"))
    if setup.get("status") != "succeeded" or not setup.get("payment_method") or not setup.get("customer"):
        raise ValueError("Payment method setup is not complete")
    if session.get("customer") != setup.get("customer"):
        raise PermissionError("Saved payment method owner mismatch")
    if row["status"] == "awaiting_setup":
        redis.hset(_key("request", request_id), mapping={
            "status": "awaiting_contact", "stripe_customer": str(setup["customer"]),
            "stripe_payment_method": str(setup["payment_method"]),
            "setup_intent": str(setup["id"]),
        })
    return _row(redis, request_id)


@gallery_approvals.get("/api/gallery/my-request")
def own_request():
    try:
        _, claims, redis, _ = _services()
        rid = redis.get(_key("latest", claims["sub"]))
        row = _row(redis, rid)
        if not row or row.get("user_id") != claims["sub"]:
            return jsonify({"success": True, "request": None, "approved_gallery": approved_gallery(redis, claims["sub"])})
        view = _public(row)
        if row.get("status") in {"payment_action_required", "approved_payment_pending"} and row.get("subscription_id"):
            try:
                stripe = _stripe_from_gallery()
                sub = stripe.Subscription.retrieve(row["subscription_id"], expand=["latest_invoice"])
                invoice = sub.get("latest_invoice") or {}
                if isinstance(invoice, dict) and invoice.get("status") != "paid":
                    url = invoice.get("hosted_invoice_url")
                    if url and str(url).startswith("https://invoice.stripe.com/"):
                        view["complete_payment_url"] = url
            except Exception:
                pass
        return jsonify({"success": True, "request": view, "approved_gallery": approved_gallery(redis, claims["sub"])})
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 401
    except Exception:
        return jsonify({"error": "Verification request unavailable"}), 503


@gallery_approvals.post("/api/gallery/contact")
def submit_contact():
    try:
        _, claims, redis, _ = _services()
        data = request.get_json(silent=True) or {}
        rid = str(data.get("request_id") or "")
        row = _row(redis, rid)
        if not row or row.get("user_id") != claims["sub"]:
            return jsonify({"error": "Request not found"}), 404
        method = str(data.get("method") or "").lower()
        name = str(data.get("name") or "").strip()
        phone = str(data.get("phone") or "").strip()
        if method != "phone" or len(name) < 2 or len(name) > 120 or re.search(r"[\x00-\x1f\x7f]", name) or not re.fullmatch(r"\+?[0-9\s()\-]{7,26}", phone):
            return jsonify({"error": "Enter your full name and a valid phone number"}), 400
        if row.get("status") == "pending":
            return jsonify({"success": True, "request": _public(row)})
        if row.get("status") != "awaiting_contact":
            return jsonify({"error": "This request cannot accept contact details"}), 409
        # A claimed number is NEVER authenticated by this endpoint. The admin
        # must independently compare the actual WhatsApp sender or call back a
        # number obtained from the gallery's public profile.
        redis.hset(_key("request", rid), mapping={
            "contact_method": method, "name_claimed": name, "phone_claimed": phone, "status": "pending",
            "submitted_at": str(int(time.time())),
        })
        redis.zadd(_key("pending"), {rid: int(time.time())})
        return jsonify({"success": True, "request": _public(_row(redis, rid))})
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 401
    except Exception:
        return jsonify({"error": "Could not submit verification request"}), 503


@gallery_approvals.get("/api/gallery/admin/requests")
def admin_requests():
    try:
        _, claims, redis, _ = _services()
        _admin(claims)
        request_id = str(request.args.get("request_id") or "")
        if request_id:
            row = _row(redis, request_id)
            return jsonify({"success": True, "requests": [_public(row, admin=True)] if row else []})
        ids = redis.zrevrange(_key("pending"), 0, 99)
        return jsonify({"success": True, "requests": [_public(row, admin=True) for rid in ids if (row := _row(redis, rid)) and row.get("status") in {"pending", "processing", "payment_action_required"}]})
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except Exception:
        return jsonify({"error": "Admin queue unavailable"}), 503


@gallery_approvals.post("/api/gallery/admin/decision")
def admin_decision():
    try:
        _, claims, redis, stripe = _services()
        _admin(claims)
        data = request.get_json(silent=True) or {}
        rid = str(data.get("request_id") or "")
        action = str(data.get("action") or "")
        row = _row(redis, rid)
        if not row or row.get("status") not in {"pending", "processing", "payment_action_required"}:
            return jsonify({"error": "Request is not pending"}), 409
        if action == "deny":
            if row["status"] != "pending":
                return jsonify({"error": "Payment already attempted; review in Stripe before taking action"}), 409
            redis.hset(_key("request", rid), mapping={"status": "denied", "reviewed_by": claims["sub"], "reviewed_at": str(int(time.time()))})
            redis.zrem(_key("pending"), rid)
            return jsonify({"success": True, "status": "denied", "charged": False})
        if action != "approve":
            return jsonify({"error": "Unknown review action"}), 400
        # A reviewer must perform an independent authority check. WhatsApp
        # messages and customer-supplied phone strings never auto-authorise.
        independently_checked = data.get("independently_checked") is True
        observed_phone = str(data.get("verified_phone") or "").strip()
        if not independently_checked or (observed_phone and not re.fullmatch(r"\+?[0-9\s()\-]{7,26}", observed_phone)):
            return jsonify({"error": "You must independently verify gallery authority before approving"}), 400
        user_id, company = row["user_id"], row["gallery_name"]
        previous = approved_gallery(redis, user_id)
        if previous and previous != company:
            return jsonify({"error": "This account is already authorised for a different gallery"}), 409
        # Fail closed if the customer is already paying for business access.
        from otodeger_stripe_billing import read_paid_access, _price_for
        existing = read_paid_access(redis, user_id)
        if existing["business_until"] and existing["business_scope"] == "user" and not row.get("subscription_id"):
            return jsonify({"error": "Customer already has an active Business subscription; review billing before approving"}), 409
        if _price_for(row["plan"]) != row.get("price_id"):
            return jsonify({"error": "Stripe price configuration changed since the customer consented. Obtain fresh consent before approval."}), 409
        if row["status"] == "pending":
            # Redis watch protects against an accidental double-click by two admins.
            key = _key("request", rid)
            with redis.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(key)
                        if pipe.hget(key, "status") != "pending":
                            pipe.unwatch()
                            return jsonify({"error": "This request is already being processed; refresh"}), 409
                        pipe.multi()
                        pipe.hset(key, mapping={"status": "processing", "approved_by": claims["sub"],
                                                "verified_phone": observed_phone, "reviewed_at": str(int(time.time()))})
                        pipe.hset(_key("approved", user_id), mapping={"user_id": user_id,
                                  "gallery_name": company, "status": "approved", "approved_by": claims["sub"],
                                  "verified_phone": observed_phone, "request_id": rid})
                        pipe.execute()
                        break
                    except __import__("redis").WatchError:
                        continue
        elif row.get("approved_by") != claims["sub"]:
            return jsonify({"error": "Request is being processed by a different administrator"}), 409
        # Stripe idempotency prevents duplicate subscriptions/charges if the
        # response fails after the external subscription call succeeds.
        row = _row(redis, rid)
        if row.get("subscription_id"):
            sub = stripe.Subscription.retrieve(row["subscription_id"])
        else:
            sub = stripe.Subscription.create(
                customer=row["stripe_customer"], items=[{"price": row["price_id"]}],
                default_payment_method=row["stripe_payment_method"],
                collection_method="charge_automatically", payment_behavior="allow_incomplete",
                cancel_at_period_end=row.get("auto_renew", "true") == "false",
                metadata={"clerk_user_id": user_id, "billing_scope": "user", "billing_subject": user_id,
                          "plan": row["plan"], "gallery_name": company, "gallery_request_id": rid,
                          "auto_renew": row.get("auto_renew", "true")},
                expand=["latest_invoice"], idempotency_key=f"otodost-gallery-approve-{rid}",
            )
            redis.hset(_key("request", rid), "subscription_id", sub.id)
        from otodeger_stripe_billing import _sync_subscription
        if sub.get("status") == "active":
            _sync_subscription(sub, paid=False)  # invoice.paid webhook determines paid_until
        status = "approved_payment_pending" if sub.get("status") == "active" else "payment_action_required"
        latest = _row(redis, rid)
        if latest.get("status") == "paid_active":
            status = "paid_active"
        else:
            redis.hset(_key("request", rid), mapping={"status": status, "payment_status": str(sub.get("status") or "pending")})
        redis.zrem(_key("pending"), rid)
        return jsonify({"success": True, "status": status, "charged": status == "paid_active",
                        "message": "Access activates only after the first invoice is confirmed paid."})
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except Exception as exc:
        print("GALLERY APPROVAL ERROR:", repr(exc), flush=True)
        # Processing status deliberately remains locked for manual Stripe review;
        # do not offer Deny once there is uncertainty about a charge attempt.
        return jsonify({"error": "Approval requires review; check the Stripe subscription before retrying"}), 503
