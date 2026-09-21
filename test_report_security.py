"""Offline authorization regression checks for the guided-report patch.

Run: python test_report_security.py
Does not call Stripe, Clerk, GitHub, Redis or the live Render backend.
"""
import importlib.util
import os
import sys
import time
import types
import unittest
from unittest.mock import patch

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location('otodeger_access_control', os.path.join(HERE, 'otodeger_access_control.py'))
access = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = access
spec.loader.exec_module(access)

class FakePipeline:
    def __init__(self, redis): self.redis, self.ops = redis, []
    def exists(self, key): self.ops.append(('exists', key)); return self
    def hgetall(self, key): self.ops.append(('hgetall', key)); return self
    def execute(self): return [getattr(self.redis, op)(key) for op, key in self.ops]

class FakeRedis:
    def __init__(self): self.strings, self.hashes = {}, {}
    def exists(self, key): return key in self.strings or key in self.hashes
    def hgetall(self, key): return dict(self.hashes.get(key, {}))
    def pipeline(self, transaction=False): return FakePipeline(self)
    def eval(self, script, count, used_key, active_key, scope, expires, seconds):
        if self.exists(used_key): return 0
        self.strings[used_key] = '1'
        self.hashes[active_key] = {'scope_hash': scope, 'expires_at': str(expires)}
        return 1

class ReportAccessTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {'COMMERCIAL_SECURITY_MODE':'legacy', 'STRIPE_ENFORCE_ENTITLEMENTS':'false'}, clear=False):
            self.m = access.CommercialAccessManager()
        self.m.mode = 'enforced'
        self.m.stripe_enforce_entitlements = True
        self.m.backend.redis = FakeRedis() if hasattr(self.m.backend,'redis') else None
        if self.m.backend.redis is None: self.m.backend.redis = FakeRedis()
        self.m.has_shared_backend = True
        self.ctx = access.AccessContext(authenticated=True,user_id='clerk_one',tier='PERSONAL', device_id='device', source='clerk_session')
        self.other = access.AccessContext(authenticated=True,user_id='clerk_other',tier='PERSONAL', device_id='device', source='clerk_session')
        self.scope = {'task':'VALUE','answers':{'year':2020, 'brand':'BMW', 'model':'118i'}}
        self.paid_users = set()
        def read_paid(redis, user_id, org_id=''):
            return {'personal_until': int(time.time())+1000 if user_id in self.paid_users else 0,
                    'personal_plus_until':0,'business_until':0}
        self.mod = patch.dict(sys.modules,{'otodeger_stripe_billing': types.SimpleNamespace(read_paid_access=read_paid)})
        self.mod.start()
    def tearDown(self): self.mod.stop()
    def check_access(self, context=None, ctx=None, **kwargs):
        return self.m.require_guided_report_access(context or self.ctx, ctx if ctx is not None else self.scope,**kwargs)
    def test_unpaid_denied_without_trial(self):
        with self.assertRaises(access.AccessControlError): self.check_access()
        with self.assertRaises(access.BusinessAccessRequired): self.check_access(business=True)
    def test_one_report_trial_allows_same_selection_not_other(self):
        res=self.m.start_personal_report_trial(self.ctx, self.scope)
        self.assertTrue(res['success'])
        self.check_access(actual_task='VALUE',actual_answers=self.scope['answers'])
        self.assertFalse(self.m.personal_trial_status(self.ctx)['personal_trial_available'])
        with self.assertRaises(access.AccessControlError): self.check_access(ctx={'task':'VALUE','answers':{'year':2021}},actual_task='VALUE',actual_answers={'year':2021})
        with self.assertRaises(access.AccessControlError): self.check_access(actual_task='VALUE', actual_answers={'year':2021})
        with self.assertRaises(access.AccessControlError): self.check_access(context=self.other)
        with self.assertRaises(access.AccessControlError): self.m.start_personal_report_trial(self.ctx,self.scope)
    def test_expired_trial_denied(self):
        self.m.start_personal_report_trial(self.ctx, self.scope)
        key=self.m._trial_keys(self.ctx.user_id)[1]
        self.m.backend.redis.hashes[key]['expires_at']=str(int(time.time())-1)
        with self.assertRaises(access.AccessControlError): self.check_access()
    def test_paid_pass_authorized_for_any_personal_report(self):
        self.paid_users.add(self.ctx.user_id)
        self.check_access(ctx=None)
        self.check_access(ctx={'task':'NEXT_VEHICLE', 'answers':{}}, actual_task='NEXT_VEHICLE',actual_answers={'any':'payload'})
        self.assertTrue(self.m.start_personal_report_trial(self.ctx,self.scope)['already_paid'])
    def test_anon_and_fake_source_not_allowed(self):
        anonymous=access.AccessContext(authenticated=False,user_id='anon',tier='ANONYMOUS',device_id='d')
        with self.assertRaises(access.AuthenticationRequired): self.check_access(context=anonymous)
        fake=access.AccessContext(authenticated=True,user_id='spoof',tier='PERSONAL',device_id='d',source='legacy_client')
        with self.assertRaises(access.AuthenticationRequired): self.m.start_personal_report_trial(fake,self.scope)
    def test_scoped_report_uses_stable_key_order(self):
        self.m.start_personal_report_trial(self.ctx,self.scope)
        self.check_access(ctx={'answers':{'model':'118i','brand':'BMW','year':2020},'task':'VALUE'})

if __name__=='__main__': unittest.main(verbosity=2)
