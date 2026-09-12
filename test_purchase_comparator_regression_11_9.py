from __future__ import annotations
import statistics
import pandas as pd
import otodeger_v10_agent as agent


def test_specific_vehicle_year_does_not_erase_nearby_comparators():
    market = pd.read_csv('/mnt/data/otodeger_gold/real_data/market_base.csv', low_memory=False)

    calls = []
    def market_search(**kwargs):
        calls.append(dict(kwargs))
        w = market.copy()
        if kwargs.get('brands'):
            wanted={str(x).casefold() for x in kwargs['brands']}
            w=w[w['Brand'].fillna('').astype(str).str.casefold().isin(wanted)]
        if kwargs.get('models'):
            wanted=[str(x).casefold() for x in kwargs['models']]
            mask=pd.Series(False,index=w.index)
            for val in wanted:
                mask |= w['Model'].fillna('').astype(str).str.casefold().str.contains(val, regex=False)
            w=w[mask]
        if kwargs.get('min_year') is not None:
            w=w[w['Year'] >= int(kwargs['min_year'])]
        if kwargs.get('max_year') is not None:
            w=w[w['Year'] <= int(kwargs['max_year'])]
        rows=[]
        for _,r in w.head(int(kwargs.get('max_limit') or 5000)).iterrows():
            rows.append({
                'brand':r['Brand'], 'model':r['Model'], 'category':r['Category'],
                'year':int(r['Year']), 'price':float(r['Price']),
                'km':None if pd.isna(r['KM']) else int(r['KM']),
                'transmission':r['Transmission'],
            })
        return {'results': rows}

    state={'constraints': {'min_year':2007, 'max_year':2007, 'category':'1.2L'}}
    target={'brand':'Nissan','model':'March','category':'1.2L','year':2007}
    rows=agent._comparable_rows(state,target,{'market_search':market_search})

    assert calls, 'market_search was not called'
    assert calls[0].get('min_year') is None and calls[0].get('max_year') is None
    assert len(rows) == 5
    assert {r['category'] for r in rows} == {'1.2'}
    assert statistics.median([r['price'] for r in rows]) == 4100.0
