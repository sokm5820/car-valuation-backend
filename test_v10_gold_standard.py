import os, copy, unittest
import pandas as pd

os.environ['V10_STATE_BACKEND']='memory'
os.environ['V10_REQUIRE_REDIS']='false'

import otodeger_v10_state as state_mod
import otodeger_v10_agent as agent
from otodeger_v10_state import Action, Job, Transition


def fake_market_search(budget=None,min_budget=None,brands=None,exclude_brands=None,models=None,exclude_models=None,categories=None,exclude_categories=None,locations=None,exclude_locations=None,companies=None,exclude_companies=None,transmissions=None,colors=None,min_year=None,max_year=None,min_km=None,max_km=None,limit=20,max_limit=100,analysis_mode=False):
    rows=[
        {'brand':'BMW','model':'X1','category':'SUV','year':2019,'price':15000.0,'km':140000,'company':'A','location':'Girne','transmission':'Otomatik','color':'Black','image':'','link':'https://x/x1-2019'},
        {'brand':'BMW','model':'X1','category':'SUV','year':2017,'price':14750.0,'km':105000,'company':'B','location':'Lefkosa','transmission':'Otomatik','color':'White','image':'','link':'https://x/x1-2017a'},
        {'brand':'BMW','model':'X1','category':'SUV','year':2017,'price':14900.0,'km':90000,'company':'C','location':'Girne','transmission':'Otomatik','color':'Grey','image':'','link':'https://x/x1-2017b'},
        {'brand':'Mercedes-Benz','model':'GLA','category':'SUV','year':2017,'price':14000.0,'km':85000,'company':'D','location':'Girne','transmission':'Otomatik','color':'Black','image':'','link':'https://x/gla-2017a'},
        {'brand':'Mercedes-Benz','model':'GLA','category':'SUV','year':2015,'price':14500.0,'km':193000,'company':'E','location':'Lefkosa','transmission':'Otomatik','color':'White','image':'','link':'https://x/gla-2015'},
        {'brand':'Ford','model':'EcoSport','category':'SUV','year':2022,'price':14300.0,'km':30000,'company':'F','location':'Girne','transmission':'Otomatik','color':'Blue','image':'','link':'https://x/eco'},
        {'brand':'Honda','model':'Activa 110','category':'Motorcycle','year':2025,'price':3000.0,'km':1000,'company':'G','location':'Girne','transmission':'Otomatik','color':'Red','image':'','link':'https://x/activa'},
    ]
    def cf(v): return str(v or '').casefold()
    out=[]
    for r in rows:
        if budget is not None and r['price']>float(budget): continue
        if min_budget is not None and r['price']<float(min_budget): continue
        if brands and cf(r['brand']) not in {cf(x) for x in brands}: continue
        if models and not any(cf(x) in cf(r['model']) for x in models): continue
        if categories and not any(cf(x) in cf(r['category']) for x in categories): continue
        if locations and not any(cf(x) in cf(r['location']) for x in locations): continue
        if transmissions and cf(r['transmission']) not in {cf(x) for x in transmissions}: continue
        if min_year is not None and r['year']<int(min_year): continue
        if max_year is not None and r['year']>int(max_year): continue
        if min_km is not None and (r['km'] is None or r['km']<float(min_km)): continue
        if max_km is not None and (r['km'] is None or r['km']>float(max_km)): continue
        out.append(copy.deepcopy(r))
    out=sorted(out,key=lambda x:(x['price'],-x['year']))
    return {'success':True,'count':len(out),'returned':len(out[:limit]),'results':out if analysis_mode else out[:limit]}


def fake_search_targets(base_filters,targets):
    merged=[]
    for t in targets:
        r=fake_market_search(budget=base_filters.get('budget'),min_budget=base_filters.get('min_budget'),brands=[t['brand']],models=[t['model']],min_year=base_filters.get('min_year'),max_year=base_filters.get('max_year'),min_km=base_filters.get('min_km'),max_km=base_filters.get('max_km'),limit=5000,max_limit=5000,analysis_mode=True)
        merged.extend(r['results'])
    seen=set(); clean=[]
    for x in merged:
        if x['link'] in seen: continue
        seen.add(x['link']); clean.append(x)
    return {'success':True,'count':len(clean),'returned':len(clean),'results':clean}


def fake_resolve(msg):
    low=msg.casefold(); out=[]
    if 'bmw x1' in low: out.append({'brand':'BMW','model':'X1'})
    if 'mercedes' in low and 'gla' in low: out.append({'brand':'Mercedes-Benz','model':'GLA'})
    return out


def fake_brands(msg):
    low=msg.casefold(); out=[]
    if 'bmw' in low: out.append('BMW')
    if 'mercedes' in low: out.append('Mercedes-Benz')
    return out

MODEL_PROFILE_LOOKUP={
    ('honda','activa 110'):{'VehicleType':'MOTORCYCLE'},
    ('bmw','x1'):{'VehicleType':'CAR','BodyStyle':'SUV'},
    ('mercedes-benz','gla'):{'VehicleType':'CAR','BodyStyle':'SUV'},
    ('ford','ecosport'):{'VehicleType':'CAR','BodyStyle':'SUV'},
}

BASE_HOST={
    'market_search':fake_market_search,
    '_search_market_for_vehicle_targets':fake_search_targets,
    'resolve_market_vehicle_mentions':fake_resolve,
    '_extract_explicit_market_brands':fake_brands,
    '_attach_explicit_years_to_vehicle_targets':lambda m,t:t,
    'MODEL_PROFILE_LOOKUP':MODEL_PROFILE_LOOKUP,
    'OPENAI_MODEL':'fake',
    'ASSISTANT_MAX_MESSAGE_CHARS':1200,
}

class V10GoldTests(unittest.TestCase):
    def setUp(self):
        state_mod._state_service=None
        self.plans=[]
        self.orig_semantic=agent._semantic_plan
        self.orig_render=agent._render_answer
        agent._render_answer=lambda message,language,state,action,evidence,decision,host: agent._fallback_answer(language,state,decision,evidence)

    def tearDown(self):
        agent._semantic_plan=self.orig_semantic
        agent._render_answer=self.orig_render
        state_mod._state_service=None

    def install_plans(self, plans):
        self.plans=list(plans)
        def planner(message,language,state,host):
            # Let deterministic offered-action/reference logic run first.
            fast=agent._deterministic_followup_plan(message,state)
            if fast: return fast
            if not self.plans: raise AssertionError('No semantic plan queued')
            return self.plans.pop(0)
        agent._semantic_plan=planner

    def call(self,msg,cid='journey',rev=None,**extra):
        data={'message':msg,'language':'EN','conversation_id':cid,'access_tier':'PERSONAL'}
        if rev is not None: data['state_revision']=rev
        data.update(extra)
        payload,status=agent.handle_v10_request(data,BASE_HOST)
        self.assertEqual(status,200,payload)
        return payload

    def test_full_bmw_mercedes_context_and_links_journey(self):
        self.install_plans([
            {'transition':'START_NEW_GOAL','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','goal_summary':'Find a car within £15,000','constraints_delta':{'budget_max':15000}},
            {'transition':'REFINE','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','constraints_delta':{'vehicle_type':'SUV'}},
            {'transition':'REFINE','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','constraints_delta':{'brands':['BMW','Mercedes-Benz']}},
        ])
        a=self.call('I have a 15.000 GBP budget, what can I buy?')
        b=self.call('SUV',rev=a['state_revision'])
        c=self.call('I want only BMW or Mercedes',rev=b['state_revision'])
        self.assertEqual(c['assistant_state']['constraints']['budget_max'],15000)
        self.assertEqual(c['assistant_state']['constraints']['vehicle_type'],'SUV')
        self.assertEqual(c['assistant_state']['constraints']['brands'],['BMW','Mercedes-Benz'])
        d=self.call('Yes, compare them',rev=c['state_revision'])
        self.assertEqual(d['decision_mode'],'COMPARE')
        names={(x.get('brand'),x.get('model')) for x in d['business_options']}
        self.assertIn(('BMW','X1'),names)
        self.assertIn(('Mercedes-Benz','GLA'),names)
        e=self.call('Okay, can I see the links?',rev=d['state_revision'])
        self.assertEqual(e['decision_mode'],'SHOP')
        links={x['link'] for x in e['results']}
        self.assertIn('https://x/x1-2019',links)
        self.assertIn('https://x/gla-2017a',links)
        self.assertNotIn('https://x/eco',links)

    def test_default_discovery_excludes_motorcycle(self):
        st=state_mod.new_state('x')
        st['constraints']={'budget_max':18000}
        rows=fake_market_search(budget=18000,analysis_mode=True,limit=5000,max_limit=5000)['results']
        out=agent._normal_vehicle_default_filter(rows,st,BASE_HOST)
        self.assertFalse(any(x['model']=='Activa 110' for x in out))

    def test_budget_correction_preserves_suv_and_brand(self):
        self.install_plans([
            {'transition':'START_NEW_GOAL','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','constraints_delta':{'budget_max':15000,'vehicle_type':'SUV','brands':['BMW']}},
            {'transition':'CORRECT','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','constraints_delta':{'budget_max':18000}},
        ])
        a=self.call('£15k SUV BMW',cid='corr')
        b=self.call('Actually make that £18k',cid='corr',rev=a['state_revision'])
        c=b['assistant_state']['constraints']
        self.assertEqual(c['budget_max'],18000); self.assertEqual(c['vehicle_type'],'SUV'); self.assertEqual(c['brands'],['BMW'])

    def test_new_goal_clears_stale_vehicle_context(self):
        self.install_plans([
            {'transition':'START_NEW_GOAL','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','constraints_delta':{'budget_max':15000,'brands':['BMW']}},
            {'transition':'START_NEW_GOAL','action':'EVALUATE_SALE','job':'EVALUATE_SALE','goal_summary':'Decide whether to sell my Aqua','constraints_delta':{'offer_price':11000},'objects':[{'type':'OWNED_VEHICLE','alias':'mine','payload':{'brand':'Toyota','model':'Aqua','year':2020}}],'target_ids':['mine']},
        ])
        a=self.call('BMW under 15k',cid='newgoal')
        b=self.call('Forget that. Someone offered £11k for my 2020 Toyota Aqua',cid='newgoal',rev=a['state_revision'])
        self.assertEqual(b['job'],'EVALUATE_SALE')
        self.assertNotIn('brands',b['assistant_state']['constraints'])
        self.assertEqual(b['assistant_state']['constraints']['offer_price'],11000)

    def test_purchase_evaluation_is_current_asking_market_grounded(self):
        st=state_mod.new_state('eval')
        oid=state_mod.register_object(st,'MODEL',{'brand':'BMW','model':'X1','year':2019,'price':15000,'km':140000})
        st['focus']={'object_ids':[oid]}; st['constraints']={'budget_max':15000,'asking_price':15000}
        ev=agent._evaluate_price_subject(st,[oid],BASE_HOST,'purchase_evaluation')
        self.assertEqual(ev['status'],'ok')
        self.assertIn('asking prices',ev['basis'])
        self.assertGreaterEqual(ev['comparables_count'],3)

    def test_business_period_aggregates_events_not_llm_math(self):
        activity=pd.DataFrame([
            {'Date':pd.Timestamp('2026-08-01'),'Company':'Dealer A','OpeningObservedStockCount':40,'ClosingObservedStockCount':42,'NetObservedStockChange':2,'NewlyObservedListings':3,'ObservedMarketExits':1,'AskingPriceReductions':1,'AskingPriceIncreases':0},
            {'Date':pd.Timestamp('2026-08-31'),'Company':'Dealer A','OpeningObservedStockCount':46,'ClosingObservedStockCount':47,'NetObservedStockChange':1,'NewlyObservedListings':9,'ObservedMarketExits':6,'AskingPriceReductions':7,'AskingPriceIncreases':2},
        ])
        host=dict(BASE_HOST); host['business_activity_df']=activity
        st=state_mod.new_state('biz',audience='BUSINESS'); st['constraints']={'company':'Dealer A','period_start':'2026-08-01','period_end':'2026-08-31'}
        ev=agent._business_period_tool(st,host)
        self.assertEqual(ev['status'],'ok')
        self.assertEqual(ev['summary']['opening_observed_stock'],40)
        self.assertEqual(ev['summary']['closing_observed_stock'],47)
        self.assertEqual(ev['summary']['net_observed_stock_change'],7)
        self.assertEqual(ev['summary']['newly_observed_listings'],12)
        self.assertEqual(ev['summary']['observed_market_exits'],7)
        self.assertEqual(ev['summary']['asking_price_reductions'],8)

    def test_raw_url_renderer_is_rejected(self):
        ok,reason=agent._evidence_validate('See https://example.com',{'kind':'listings','listings':[{'link':'x'}]})
        self.assertFalse(ok); self.assertEqual(reason,'RAW_URL_IN_ANSWER')

    def test_false_absence_claim_is_rejected(self):
        ok,reason=agent._evidence_validate('There are no listings available.',{'kind':'listings','listings':[{'link':'x'}]})
        self.assertFalse(ok); self.assertEqual(reason,'FALSE_ABSENCE_CLAIM')

    def test_stale_revision_returns_409(self):
        self.install_plans([{'transition':'START_NEW_GOAL','action':'SEARCH_VEHICLES','job':'FIND_A_CAR','constraints_delta':{'budget_max':15000}}])
        a=self.call('15k car',cid='stale')
        payload,status=agent.handle_v10_request({'message':'SUV','language':'EN','conversation_id':'stale','state_revision':1,'access_tier':'PERSONAL'},BASE_HOST)
        self.assertEqual(status,409); self.assertEqual(payload['error'],'STATE_STALE')

if __name__=='__main__': unittest.main(verbosity=2)
