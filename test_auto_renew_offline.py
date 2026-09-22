"""Offline regression checks for the optional Business renewal toggle.
Does not contact Stripe or Clerk. Run after test_gallery_approvals_offline.py.
"""
import importlib
import os
import types

# Reuse the established fake Flask, Stripe and Redis verification fixtures.
base = importlib.import_module('test_gallery_approvals_offline')
billing, gallery, redis, stripe = base.billing, base.gallery, base.redis, base.stripe

# A setup-only gallery request persists the selected renewal flag server-side.
rid = gallery.new_setup_request(
    redis, user_id='user_auto_off', gallery_name='ABC Cars',
    plan='business_monthly', session=base.Obj(id='cs_auto_off'),
    price_id='price_test', auto_renew=False,
)
row = gallery._row(redis, rid)
assert row['auto_renew'] == 'false'

# After independent verification, approval must create a NON-renewing subscription.
redis.hset(gallery._key('request', rid), mapping={
    'status': 'pending', 'stripe_customer': 'cus_auto', 'stripe_payment_method': 'pm_auto',
})
redis.zadd(gallery._key('pending'), {rid: 1})
base.claims['sub'] = 'admin_A'
base.fake_flask.request.get_json = lambda silent=True: {
    'request_id': rid, 'action': 'approve', 'independently_checked': True,
    'verified_phone': '+905551112233',
}
previous = len(base.subscriptions)
response = gallery.admin_decision()
assert response['success'] and len(base.subscriptions) == previous + 1, response
kwargs = base.subscriptions[-1]
assert kwargs['cancel_at_period_end'] is True
assert kwargs['metadata']['auto_renew'] == 'false'

# For already-verified gallery + independent direct Checkout, Stripe's actual
# subscription gets an end-of-term cancellation before the entitlement is synced.
modified = []
def fake_modify(sid, **kwargs):
    modified.append((sid, kwargs))
    return {
        'id': sid,
        'metadata': {'clerk_user_id': 'user_auto_off', 'billing_scope': 'user',
                     'billing_subject': 'user_auto_off', 'auto_renew': 'false'},
        'items': {'data': [{'price': {'id': 'price_test'}, 'current_period_end': 1_800_000_000}]},
        'customer': 'cus_auto', 'status': 'active', 'cancel_at_period_end': True,
    }
stripe.Subscription.modify = fake_modify
redis.sadd = lambda key, member: redis.sets.setdefault(key, set()).add(member)
billing._sync_subscription({
    'id': 'sub_auto_off_independent',
    'metadata': {'clerk_user_id': 'user_auto_off', 'billing_scope': 'user',
                 'billing_subject': 'user_auto_off', 'auto_renew': 'false'},
    'items': {'data': [{'price': {'id': 'price_test'}, 'current_period_end': 1_800_000_000}]},
    'status': 'active', 'customer': 'cus_auto', 'cancel_at_period_end': False,
}, paid=False)
assert modified == [('sub_auto_off_independent', {'cancel_at_period_end': True})], modified
assert redis.hgetall(billing._subscription_key('sub_auto_off_independent'))['paid_until'] == '0'
print('PASS: opt-out persisted through gallery approval and independent billing; no unpaid entitlement granted')

# Checkout must echo the chosen setting, so a newer frontend refuses to open
# an old, incompatible billing server that might ignore Auto-Renew OFF.
base.claims['sub'] = 'independent_auto_off'
created = []
stripe.checkout.Session.create = lambda **kwargs: (created.append(kwargs) or base.Obj(id='cs_independent',url='https://checkout.stripe.com/independent'))
base.fake_flask.request.get_json = lambda silent=True: {
    'plan': 'business_monthly', 'business_scope': 'independent',
    'auto_renew': False,
}
response = billing.start_checkout()
assert response['success'] and response['auto_renew'] is False, response
assert created[-1]['subscription_data']['metadata']['auto_renew'] == 'false'
print('PASS: direct Business checkout echoes opt-out and persists it in Stripe subscription metadata')
