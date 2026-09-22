"""Offline gate tests. Run: python test_gallery_approvals_offline.py

Uses fake Flask/Stripe/Redis objects; does NOT call Stripe, Clerk, Render, or
change any customer record. This is not a substitute for Stripe sandbox tests.
"""
import importlib
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

os.environ['STRIPE_PRICE_BUSINESS_MONTHLY']='price_test'
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

fake_flask = types.ModuleType("flask")
class Blueprint:
    def __init__(self, *args, **kwargs): pass
    def post(self, _): return lambda fn: fn
    def get(self, _): return lambda fn: fn
fake_flask.Blueprint = Blueprint
fake_flask.jsonify = lambda obj: obj
fake_flask.request = types.SimpleNamespace(get_json=lambda silent=True: {}, args={})
class GalleryDirectory:
    def get_json(self, silent=True): return {"success": True, "galleries": ["ABC Cars"]}
fake_flask.current_app = types.SimpleNamespace(view_functions={"api_business_gallery_options":lambda:GalleryDirectory()})
sys.modules["flask"] = fake_flask

class FakeRedis:
    def __init__(self): self.hashes={};self.strings={};self.zsets={};self.sets={}
    def hgetall(self,k): return dict(self.hashes.get(k,{}))
    def hget(self,k,f): return self.hgetall(k).get(f)
    def hset(self,k,*args,**kwargs):
        row=self.hashes.setdefault(k,{})
        if args: row[str(args[0])]=str(args[1])
        row.update({str(key):str(value) for key,value in kwargs.get('mapping',{}).items()})
    def get(self,k):return self.strings.get(k)
    def set(self,k,v): self.strings[k]=v
    def zadd(self,k,rows):self.zsets.setdefault(k,{}).update(rows)
    def zrem(self,k,member):self.zsets.setdefault(k,{}).pop(member,None)
    def zrevrange(self,k,a,b,withscores=False):
        items=sorted(self.zsets.get(k,{}).items(),key=lambda v:v[1],reverse=True)[a:b+1]
        return items if withscores else [x for x,y in items]
    def smembers(self,k):return self.sets.get(k,set())
    def pipeline(self,*args,**kwargs):return self
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def watch(self,k):pass
    def multi(self):pass
    def unwatch(self):pass
    def execute(self):pass

redis=FakeRedis()
manager=types.SimpleNamespace(has_shared_backend=True,business_orgs={})
claims={"sub":"user_a"}
subscriptions=[]
class Obj(dict):
    __getattr__=dict.get
class FakeStripe:
    api_key=""
    Price=types.SimpleNamespace(retrieve=lambda _id:Obj(currency='try',unit_amount=649900,recurring={'interval':'month'}))
    Customer=types.SimpleNamespace(create=lambda **kwargs:Obj(id='cus_A'))
    def setup_session_create(self,**kwargs):
        return Obj(id='cs_test_saved',url='https://checkout.stripe.com/setup')
    checkout=types.SimpleNamespace()
    def retrieve_setup(self,value): return Obj(id='seti_A',status='succeeded',payment_method='pm_A',customer='cus_A')
    SetupIntent=types.SimpleNamespace()
    def create_subscription(self,**kwargs):
        subscriptions.append(kwargs)
        return Obj(id='sub_A',status='incomplete',latest_invoice={'hosted_invoice_url':'https://invoice.stripe.com/i/test'})
    Subscription=types.SimpleNamespace()
stripe=FakeStripe()
stripe.checkout.Session=types.SimpleNamespace(create=lambda **kwargs:stripe.setup_session_create(**kwargs))
stripe.SetupIntent.retrieve=lambda value:stripe.retrieve_setup(value)
stripe.Subscription.create=lambda **kwargs:stripe.create_subscription(**kwargs)
stripe.Subscription.retrieve=lambda _id,**kw:Obj(id='sub_A',status='incomplete',latest_invoice={})

fake_access=types.ModuleType("otodeger_access_control")
fake_access.get_access_manager=lambda:manager
sys.modules["otodeger_access_control"]=fake_access
billing=importlib.import_module("otodeger_stripe_billing")
gallery=importlib.import_module("otodeger_gallery_approvals")
billing._auth=lambda: (manager,claims)
billing._redis=lambda: redis
billing._stripe=lambda: stripe
billing.read_paid_access=lambda *args, **kw:{"personal_until":0,"personal_plus_until":0,"business_until":0,"business_scope":"","business_subscription_id":""}

# No Stripe subscription is created when the customer initiates gallery checkout.
fake_flask.request.get_json=lambda silent=True: {"plan":"business_monthly","business_scope":"gallery","gallery_name":"ABC Cars","gallery_billing_consent":False}
r=billing.start_checkout()
assert r[1]==400 and not subscriptions, "Checkout must require express future-billing consent"
fake_flask.request.get_json=lambda silent=True: {"plan":"business_monthly","business_scope":"gallery","gallery_name":"ABC Cars","gallery_billing_consent":True}
r=billing.start_checkout()
assert r["flow"]=="gallery_setup" and not subscriptions
rid=redis.get(gallery._key("checkout","cs_test_saved"))
row=gallery._row(redis,rid)
assert row["status"]=="awaiting_setup" and row["user_id"]=="user_a" and not gallery.approved_gallery(redis,'user_a')

# A setup redirect from someone else's Clerk account must not expose the record.
session=Obj(id='cs_test_saved',mode='setup',status='complete',client_reference_id='user_a',setup_intent='seti_A',customer='cus_A')
try: gallery.complete_setup(redis,stripe,session,'user_b'); assert False
except PermissionError: pass
row=gallery.complete_setup(redis,stripe,session,'user_a')
assert row['status']=='awaiting_contact'

# The applicant must provide both a name and phone before review can begin.
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'method':'phone','name':'','phone':'+905551112233'}
r=gallery.submit_contact()
assert r[1] == 400 and gallery._row(redis,rid)['status'] == 'awaiting_contact'
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'method':'phone','name':'Gallery Applicant','phone':'bad'}
r=gallery.submit_contact()
assert r[1] == 400 and gallery._row(redis,rid)['status'] == 'awaiting_contact'

# Applicant-supplied details NEVER verify an account or initiate a payment.
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'method':'phone','name':'Gallery Applicant','phone':'+905551112233'}
r=gallery.submit_contact()
assert r['success'] and gallery._row(redis,rid)['status']=='pending'
assert gallery._row(redis,rid)['name_claimed']=='Gallery Applicant'
assert gallery._public(gallery._row(redis,rid),admin=True)['name_claimed']=='Gallery Applicant'
assert 'name_claimed' not in gallery._public(gallery._row(redis,rid)), 'Do not expose the applicant name via customer status endpoint'
assert not gallery.approved_gallery(redis,'user_a') and not subscriptions

# No customer may approve themselves, regardless of their browser payload.
os.environ['OTODOST_GALLERY_ADMIN_USER_IDS']='admin_A'
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'action':'approve','independently_checked':True,'verified_phone':'+905551112233'}
r=gallery.admin_decision()
assert r[1]==403 and not subscriptions

# Genuine admin must explicitly attest to independent authority verification.
claims['sub']='admin_A'
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'action':'approve','independently_checked':False,'verified_phone':'+905551112233'}
r=gallery.admin_decision()
assert r[1]==400 and not subscriptions

# Denied requests are never charged and cannot be approved afterwards.
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'action':'deny'}
r=gallery.admin_decision()
assert r['status']=='denied' and r['charged'] is False and not subscriptions
fake_flask.request.get_json=lambda silent=True: {'request_id':rid,'action':'approve','independently_checked':True,'verified_phone':'+905551112233'}
r=gallery.admin_decision()
assert r[1]==409 and not subscriptions

# A separate, manually checked request may attempt payment ONCE after approval.
claims['sub']='user_a'
row=gallery.new_setup_request(redis,user_id='user_a',gallery_name='ABC Cars',plan='business_monthly',session=Obj(id='cs_test_saved_2'),price_id='price_test')
redis.hset(gallery._key('request',row),mapping={'status':'pending','stripe_customer':'cus_A','stripe_payment_method':'pm_A'})
redis.zadd(gallery._key('pending'),{row:1})
claims['sub']='admin_A'
fake_flask.request.get_json=lambda silent=True: {'request_id':row,'action':'approve','independently_checked':True,'verified_phone':'+905551112233'}
r=gallery.admin_decision()
assert r['success'] and len(subscriptions)==1 and gallery.approved_gallery(redis,'user_a')=='ABC Cars'
assert r['charged'] is False and r['status']=='payment_action_required', 'Payment is not guaranteed until Stripe confirms first invoice'
r=gallery.admin_decision()
assert len(subscriptions)==1,'Cannot double-subscribe via repeated approval'
print('PASS: gallery contact name/phone validation, admin visibility and existing no-charge-on-denial checks')
