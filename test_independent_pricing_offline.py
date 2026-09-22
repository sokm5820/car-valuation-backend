"""Offline price, checkout and entitlement boundary checks. No network or charges.
Run python test_independent_pricing_offline.py
"""
import importlib
import os
import time

# Existing mocked Flask/Stripe/Clerk/Redis fixture; does NOT talk to hosted apps.
base = importlib.import_module('test_gallery_approvals_offline')
billing, gallery, redis, stripe = base.billing, base.gallery, base.redis, base.stripe
os.environ['STRIPE_PRICE_INDEPENDENT_MONTHLY'] = 'price_independent_month'
os.environ['STRIPE_PRICE_INDEPENDENT_ANNUAL'] = 'price_independent_year'
os.environ['STRIPE_PRICE_BUSINESS_ANNUAL'] = 'price_gallery_year'
prices = {
    'price_test': (649900, 'month'),
    'price_gallery_year': (4999900, 'year'),
    'price_independent_month': (349900, 'month'),
    'price_independent_year': (2999900, 'year'),
}
stripe.Price.retrieve = lambda pid: base.Obj(currency='try', unit_amount=prices[pid][0], recurring={'interval': prices[pid][1]})
base.claims['sub'] = 'independent_test_user'
created = []
stripe.checkout.Session.create = lambda **kw: (created.append(kw) or base.Obj(id='cs_test_independent',url='https://checkout.stripe.com/test'))

def buy(plan, scope='independent', gallery_name=''):
    base.fake_flask.request.get_json = lambda silent=True: {
        'plan': plan, 'business_scope': scope, 'gallery_name': gallery_name,
        'auto_renew': True, 'gallery_billing_consent': True,
    }
    return billing.start_checkout()

for plan, price in [('business_independent_monthly', 'price_independent_month'), ('business_independent_annual', 'price_independent_year')]:
    result = buy(plan)
    assert result['success'] and result['url'].startswith('https://'), result
    c = created[-1]
    assert c['mode'] == 'subscription' and c['line_items'] == [{'price': price, 'quantity': 1}]
    assert c['subscription_data']['metadata']['plan'] == plan
    assert 'gallery_name' not in c['subscription_data']['metadata']
assert not gallery.approved_gallery(redis, 'independent_test_user')
print('PASS: both Independent amounts map to their own direct subscription checkout, no gallery approval')

for plan, scope in [('business_monthly','independent'), ('business_annual','independent'),
                    ('business_independent_monthly','gallery'), ('business_independent_annual','gallery')]:
    n = len(created)
    result = buy(plan, scope, 'ABC Cars' if scope == 'gallery' else '')
    assert result[1] == 400 and len(created) == n, (plan, result)
print('PASS: forged gallery/independent plan combinations rejected before Stripe session')

n = len(created)
result = buy('business_independent_monthly', 'independent', 'ABC Cars')
assert result[1] == 400 and len(created) == n
prices['price_independent_month'] = (649900, 'month') # mistake in Stripe dashboard
result = buy('business_independent_monthly')
assert result[1] == 503 and len(created) == n
prices['price_independent_month'] = (349900, 'month')
print('PASS: Independent checkout cannot target gallery and fails closed on wrong Stripe amount')

# Even if someone tampers with subscription metadata, the low-price plan
# must never acquire a gallery-labelled subscription record.
redis.sadd = lambda key, value: redis.sets.setdefault(key, set()).add(value)
stripe.Subscription.modify = lambda sid, **kw: None
sub = {
    'id': 'sub_test_forged', 'metadata': {'clerk_user_id': 'independent_test_user', 'billing_scope': 'user',
              'billing_subject': 'independent_test_user', 'gallery_name': 'ABC Cars',
              'plan': 'business_independent_monthly'},
    'items': {'data': [{'price': {'id': 'price_independent_month'},'current_period_end':int(time.time())+2592000}]},
    'status':'active', 'customer':'cus_independent', 'cancel_at_period_end':False,
}
billing._sync_subscription(sub, paid=True)
assert not redis.hgetall(billing._subscription_key(sub['id']))
sub['id'] = 'sub_test_valid'
sub['metadata'].pop('gallery_name')
billing._sync_subscription(sub, paid=True)
record = redis.hgetall(billing._subscription_key(sub['id']))
assert record.get('gallery_name') == '' and record.get('price_id') == 'price_independent_month'
# Test the real billing reader rather than the mocked read_paid_access fixture.
real_read = importlib.reload(billing).read_paid_access
paid = real_read(redis, 'independent_test_user')
assert paid['business_scope'] == 'user' and paid['business_gallery_name'] == '' and paid['business_until'] > time.time()
assert gallery.approved_gallery(redis, 'independent_test_user') == ''
print('PASS: paid Independent entitlement is active but cannot be labelled as or approved for a gallery')
