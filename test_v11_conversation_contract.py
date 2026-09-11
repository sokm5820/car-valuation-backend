import os
os.environ.setdefault('V10_STATE_BACKEND','memory')
os.environ.setdefault('V10_REQUIRE_REDIS','false')

import inspect
import otodeger_v10_agent as a
from otodeger_v10_state import Job


def state(**constraints):
    return {
        'language':'EN','audience':'PERSONAL','job':Job.FIND_A_CAR.value,
        'constraints':constraints,'preferences':{},'objects':{},'shortlist':[],
        'focus':{'object_ids':[]},'offered_action':None,'awaiting':None,
    }


def test_version_and_taxonomy():
    assert a.V10_VERSION == '11.0-conversation-contract'
    assert Job.ANALYZE_BUSINESS.value == 'ANALYZE_BUSINESS'


def test_guided_bounds_are_not_arbitrary():
    s=state(budget_max=15000, vehicle_type='SUV')
    p=a._deterministic_followup_plan('Set minimum year',s)
    assert p['action']=='ASK_CLARIFICATION' and p['awaiting']['field']=='min_year'
    p=a._deterministic_followup_plan('Set mileage limit',s)
    assert p['action']=='ASK_CLARIFICATION' and p['awaiting']['field']=='max_km'


def test_listing_rows_are_rich_and_up_to_ten():
    s=state(currency='GBP')
    ev={'kind':'listings','listings':[]}
    for i in range(12):
        ev['listings'].append({'link':f'https://example/{i}','year':2020,'brand':'BMW','model':'X1','category':'18i',
            'price':15000,'km':90000,'transmission':'Automatic','fuel':'Petrol','location':'Girne','company':'Dealer'})
    actions, offered=a._actions_from_evidence(s,'SHOW_LISTINGS',ev,'EN')
    assert len(actions)==10 and offered is None
    assert 'Petrol' in actions[0]['label'] and 'Dealer' in actions[0]['label']


def test_listing_fallback_reports_ten():
    ev={'kind':'listings','count':20,'listings':[{} for _ in range(12)]}
    ans=a._fallback_answer('EN',state(),{'verdict':'INFORM'},ev)
    assert '10 current listings' in ans and '20 matches' in ans


def test_comparison_cohort_consistency():
    ev={'kind':'comparison','vehicles':[
        {'brand':'BMW','model':'X1','newest_year':2019,'newest_year_starting_price':15000,'newest_year_median_price':15000,'newest_year_highest_price':15000,'newest_year_count':1,'count':10},
        {'brand':'Mercedes-Benz','model':'GLA','newest_year':2017,'newest_year_starting_price':14000,'newest_year_median_price':14000,'newest_year_highest_price':14000,'newest_year_count':1,'count':2},
    ]}
    ans=a._fallback_answer('EN',state(),{'verdict':'COMPARE_OPTIONS'},ev)
    assert '2019' in ans and '£15,000' in ans and '10 total' in ans
    assert 'median' not in ans.lower()


def test_no_nationality_prompt():
    ev={'kind':'vehicle_search','models':[{'brand':'BMW','model':'X1'}], 'alternative_models':['Nissan Qashqai']}
    sug=a._suggestions_from_context(state(budget_max=15000,vehicle_type='SUV'),ev,'EN')
    joined=' '.join(sug).lower()
    assert 'german' not in joined and 'japanese' not in joined


def test_budget_and_mileage_are_separate():
    s=state(budget_max=15000,vehicle_type='SUV')
    parsed={'constraints_delta':{}}
    out=a._authoritative_numeric_constraints('Under 100,000 km',parsed,s,{})
    assert out['constraints_delta'].get('max_km')==100000
    assert out['constraints_delta'].get('budget_max') in (None,15000)


def test_evidence_language_rejects_confirmed_sale_claim():
    # Self-contained release check: evidence instructions must preserve the
    # distinction between observed listing exits and verified transactions.
    renderer_source=inspect.getsource(a._render_answer).lower()
    search_source=inspect.getsource(a._search_vehicle_models).lower()
    combined=renderer_source + '\n' + search_source
    assert 'observed market exit is not a confirmed sale' in combined
    assert 'never turn observed listing removals into confirmed sales' in combined


if __name__ == '__main__':
    tests=[v for k,v in globals().copy().items() if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('PASS',t.__name__)
    print(f'{len(tests)}/{len(tests)} passed')
