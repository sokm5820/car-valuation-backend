import json
import pandas as pd
import otodeger_v10_agent as agent
import otodeger_v10_state as state_mod


def test_requested_price_band_and_fuel_are_preserved_and_rendered():
    rows=[]
    fuels=['Petrol','Diesel','Hybrid','Electric']
    prices=[9000,18000,30000,50000]
    for i in range(40):
        rows.append({
            'Company':'Shoptech Auto' if i < 20 else f'Gallery {i}',
            'Brand':f'Brand{i}', 'Model':f'Model{i}',
            'CurrentAskingPrice':prices[i % 4], 'Fuel':fuels[i % 4],
        })
    stock=pd.DataFrame(rows)
    activity=pd.DataFrame([
        {
            'Date':'2026-08-01','Company':'Shoptech Auto',
            'OpeningObservedStockCount':20,'ClosingObservedStockCount':20,
            'NewlyObservedListings':4,'ObservedMarketExits':4,
            'AskingPriceReductions':0,'AskingPriceIncreases':0,
            'NewListingsJson':json.dumps([
                {'Brand':'Brand0','Model':'Model0','EventPrice':9000,'Fuel':'Petrol'},
                {'Brand':'Brand1','Model':'Model1','EventPrice':18000,'Fuel':'Diesel'},
                {'Brand':'Brand2','Model':'Model2','EventPrice':30000,'Fuel':'Hybrid'},
                {'Brand':'Brand3','Model':'Model3','EventPrice':50000,'Fuel':'Electric'},
            ]),
            'ObservedExitListingsJson':json.dumps([
                {'Brand':'Brand0','Model':'Model0','EventPrice':9000,'Fuel':'Petrol'},
                {'Brand':'Brand1','Model':'Model1','EventPrice':18000,'Fuel':'Diesel'},
                {'Brand':'Brand2','Model':'Model2','EventPrice':30000,'Fuel':'Hybrid'},
                {'Brand':'Brand3','Model':'Model3','EventPrice':50000,'Fuel':'Electric'},
            ]),
            'PriceReductionListingsJson':'[]','PriceIncreaseListingsJson':'[]',
        }
    ])
    host={
        'business_activity_df':activity,
        'business_stock_df':stock,
        'business_market_df':pd.DataFrame(),
        'business_company_df':pd.DataFrame(),
        'MODEL_PROFILE_LOOKUP':{},
    }
    st=state_mod.new_state('b6-11-14',audience='BUSINESS')
    st['language']='EN'
    st['constraints']={'company':'Shoptech Auto','period_start':'2026-08-01','period_end':'2026-08-31'}
    msg='Which segments were strongest for us? Break it down by price band and fuel.'
    ev=agent._business_period_tool(st,host,msg)
    dims={x.get('dimension') for x in ev.get('segments',[])}
    assert ev.get('requested_segment_dimensions') == ['PRICE_BAND','FUEL']
    assert 'PRICE_BAND' in dims
    assert 'FUEL' in dims
    ans=agent._fallback_answer('EN',st,{},ev)
    low=ans.casefold()
    assert 'by price band' in low
    assert 'by fuel' in low
    assert 'not confirmed sales' in low
    assert 'no verified price-band' not in low


def test_generic_business_period_keeps_existing_bounded_behavior():
    assert agent._requested_business_segment_dimensions('What happened at my gallery last month?') == []
    assert agent._requested_business_segment_dimensions('Break it down by price band and fuel.') == ['PRICE_BAND','FUEL']
