"""Build OtoDeğer V10 dealer-period activity intelligence.

Input:  kktcarabam_daily.csv (historical daily listing observations)
Output: business_company_activity_daily.csv

Semantics are intentionally evidence-safe:
- "newly observed" = first date a listing appears in our observations.
- "observed exit" = first subsequent market snapshot after its last observation,
  excluding listings still present on the dataset's latest date.
- price changes are changes in observed asking price, not transaction prices.
- stock counts are observed advertised stock, not physical/dealer-owned inventory.

The builder is deterministic and can be added to the existing daily pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np


def _pick(cols, names):
    by_cf={str(c).strip().casefold():c for c in cols}
    for name in names:
        c=by_cf.get(name.casefold())
        if c is not None: return c
    return None


def _json_list(values, limit=250):
    clean=[]; seen=set()
    for v in values:
        s=str(v or '').strip()
        if not s or s in seen: continue
        seen.add(s); clean.append(s)
        if len(clean)>=limit: break
    return json.dumps(clean,ensure_ascii=False,separators=(',',':'))




def _fuel_from_link(link):
    text=str(link or "").strip().casefold()
    patterns=[
        (r"(?:^|-)mild-hibrit-benzin(?:-|$)", "Mild Hybrid Petrol"),
        (r"(?:^|-)mild-hibrit-dizel(?:-|$)", "Mild Hybrid Diesel"),
        (r"(?:^|-)plug-in-hibrit(?:-|$)", "Plug-in Hybrid"),
        (r"(?:^|-)elektrik(?:-|$)", "Electric"),
        (r"(?:^|-)hibrit(?:-|$)", "Hybrid"),
        (r"(?:^|-)dizel(?:-|$)", "Diesel"),
        (r"(?:^|-)benzin(?:-|$)", "Petrol"),
    ]
    for pattern,label in patterns:
        if re.search(pattern,text,re.IGNORECASE):
            return label
    return ""


def _split_category(value):
    return [x.strip() for x in str(value or "").split("/") if x and x.strip()]


def _best_category_map(frame, category_col):
    if not category_col:
        return {}
    work=frame[["_listing_id",category_col]].copy()
    work[category_col]=work[category_col].fillna("").astype(str).str.strip()
    work=work[work[category_col]!=""]
    if work.empty:
        return {}
    work["_parts"]=work[category_col].map(lambda x: len(_split_category(x)))
    work["_length"]=work[category_col].str.len()
    work=(work.sort_values(["_listing_id","_parts","_length"],ascending=[True,False,False])
              .drop_duplicates("_listing_id",keep="first"))
    return dict(zip(work["_listing_id"],work[category_col]))


def _brand_model_from_category(value):
    parts=_split_category(value)
    if len(parts)>=3:
        return parts[1],parts[2]
    return "",""


def _normalise_raw_price(series, source_name):
    numeric=pd.to_numeric(series,errors="coerce")
    valid=numeric.dropna()
    if valid.empty:
        return numeric
    # kktcarabam_daily.csv stores GBP prices in thousands (11 => £11,000),
    # while prebuilt intelligence files use full GBP units. Detect the raw
    # source conservatively so this builder also remains compatible with a
    # future full-unit input.
    q95=float(valid.quantile(0.95))
    raw_price_field=str(source_name or "").strip().casefold()=="price"
    factor=1000.0 if raw_price_field and q95<=1000 else 1.0
    return numeric*factor

def _safe_json_records(records, limit=250):
    clean=[]
    for r in records[:limit]:
        item={}
        for k,v in r.items():
            if pd.isna(v):
                continue
            if isinstance(v,(pd.Timestamp,)):
                v=v.strftime('%Y-%m-%d')
            elif hasattr(v,'item'):
                try: v=v.item()
                except Exception: pass
            item[str(k)]=v
        clean.append(item)
    return json.dumps(clean,ensure_ascii=False,separators=(',',':'))


def build(input_csv: str, output_csv: str) -> pd.DataFrame:
    raw=pd.read_csv(input_csv,low_memory=False)
    date_col=_pick(raw.columns,['DATE','Date','date','ScrapeDate','scrape_date'])
    company_col=_pick(raw.columns,['Company','company','Seller','Dealer'])
    link_col=_pick(raw.columns,['Link','link','URL','Url','url'])
    price_col=_pick(raw.columns,['Price','price','CurrentAskingPrice'])
    currency_col=_pick(raw.columns,['Currency','currency'])
    brand_col=_pick(raw.columns,['Brand','brand'])
    model_col=_pick(raw.columns,['Model','model'])
    year_col=_pick(raw.columns,['Year','year'])
    category_col=_pick(raw.columns,['Category','category','CategoryDetail'])
    fuel_col=_pick(raw.columns,['Fuel','fuel','Yakıt','Yakit','FuelType'])
    if not date_col or not company_col:
        raise ValueError('Input must contain date and Company columns')

    df=raw.copy()
    if currency_col:
        currency=df[currency_col].fillna('').astype(str).str.strip().str.upper()
        df=df[currency.eq('GBP')].copy()
    df['_date'] = pd.to_datetime(
        df[date_col],
        errors='coerce',
        dayfirst=True,
    ).dt.normalize()
    df['_company']=df[company_col].fillna('').astype(str).str.strip()
    df=df[df['_date'].notna() & (df['_company']!='')].copy()
    if df.empty: raise ValueError('No usable historical observations')

    # Stable listing identity. Link is preferred; deterministic composite only when missing.
    if link_col:
        link=df[link_col].fillna('').astype(str).str.strip()
    else:
        link=pd.Series('',index=df.index)
    composite=(
        df['_company'].astype(str)+'|'+
        (df[brand_col].fillna('').astype(str) if brand_col else '')+'|'+
        (df[model_col].fillna('').astype(str) if model_col else '')+'|'+
        (df[year_col].fillna('').astype(str) if year_col else '')+'|'+
        (df[category_col].fillna('').astype(str) if category_col else '')
    )
    df['_listing_id']=np.where(link.str.len()>0,link,composite)
    df['_price']=_normalise_raw_price(df[price_col],price_col) if price_col else np.nan

    # One observation per listing/date; last duplicate wins deterministically by file order.
    df['_row_order']=np.arange(len(df))
    daily=(df.sort_values('_row_order')
             .drop_duplicates(['_listing_id','_date'],keep='last')
             .sort_values(['_listing_id','_date']))

    dates=sorted(pd.Timestamp(x) for x in daily['_date'].dropna().unique())
    date_index={d:i for i,d in enumerate(dates)}
    latest=dates[-1]

    best_category=_best_category_map(daily,category_col)
    grouped=daily.groupby('_listing_id',sort=False)
    agg_spec={
        'Company':('_company','first'),
        'FirstObservedDate':('_date','min'),
        'LastObservedDate':('_date','max'),
    }
    if brand_col: agg_spec['Brand']=(brand_col,'first')
    if model_col: agg_spec['Model']=(model_col,'first')
    if year_col: agg_spec['Year']=(year_col,'first')
    if category_col: agg_spec['Category']=(category_col,'first')
    if fuel_col: agg_spec['Fuel']=(fuel_col,'first')
    if price_col:
        agg_spec['FirstObservedPrice']=('_price','first')
        agg_spec['LastObservedPrice']=('_price','last')
    if link_col: agg_spec['Link']=(link_col,'first')
    summary=grouped.agg(**agg_spec).reset_index()
    if category_col:
        summary['Category']=summary['_listing_id'].map(best_category).fillna(summary.get('Category',''))
    if not brand_col or not model_col:
        parsed=summary.get('Category',pd.Series('',index=summary.index)).map(_brand_model_from_category)
        if not brand_col:
            summary['Brand']=parsed.map(lambda x:x[0])
        if not model_col:
            summary['Model']=parsed.map(lambda x:x[1])
    if not fuel_col:
        summary['Fuel']=summary.get('Link',pd.Series('',index=summary.index)).map(_fuel_from_link)
    next_date={dates[i]:dates[i+1] for i in range(len(dates)-1)}
    summary['ExitEventDate']=summary['LastObservedDate'].map(next_date)
    summary.loc[summary['LastObservedDate']>=latest,'ExitEventDate']=pd.NaT

    # Price-change events from consecutive observed asking prices.
    daily['_prev_price']=daily.groupby('_listing_id')['_price'].shift(1)
    daily['_price_changed']=daily['_price'].notna() & daily['_prev_price'].notna() & (daily['_price']!=daily['_prev_price'])
    daily['_price_reduced']=daily['_price_changed'] & (daily['_price']<daily['_prev_price'])
    daily['_price_increased']=daily['_price_changed'] & (daily['_price']>daily['_prev_price'])

    companies=sorted(summary['Company'].dropna().astype(str).unique())
    n_dates=len(dates)

    # Precompute observed-stock counts with interval difference arrays. This is
    # O(listings + companies*dates), rather than rescanning every company's
    # listings for every date.
    stock_arrays={}
    for company,listings in summary.groupby('Company',sort=False):
        diff=np.zeros(n_dates+1,dtype=np.int64)
        for first,last in zip(listings['FirstObservedDate'],listings['LastObservedDate']):
            si=date_index.get(pd.Timestamp(first)); ei=date_index.get(pd.Timestamp(last))
            if si is None or ei is None: continue
            diff[si]+=1
            if ei+1<n_dates: diff[ei+1]-=1
        closing=np.cumsum(diff[:n_dates])
        opening=np.empty(n_dates,dtype=np.int64)
        opening[0]=0
        if n_dates>1: opening[1:]=closing[:-1]
        stock_arrays[str(company)]=(opening,closing)

    display_cols=[c for c in ['_listing_id','Brand','Model','Year','Category','Fuel','Link'] if c in summary.columns]
    new_cols=display_cols + (['FirstObservedPrice'] if 'FirstObservedPrice' in summary.columns else [])
    exit_cols=display_cols + (['LastObservedPrice'] if 'LastObservedPrice' in summary.columns else [])

    def _summary_event_maps(date_col, cols, price_name):
        rec_map={}; ids_map={}
        valid=summary[summary[date_col].notna()]
        for (company,date),g in valid.groupby(['Company',date_col],sort=False):
            key=(str(company),pd.Timestamp(date))
            ids_map[key]=g['_listing_id'].tolist()
            rec_map[key]=(g[cols]
                .rename(columns={'_listing_id':'listing_id',price_name:'EventPrice'})
                .to_dict('records'))
        return ids_map,rec_map

    new_ids_map,new_records_map=_summary_event_maps('FirstObservedDate',new_cols,'FirstObservedPrice')
    exit_ids_map,exit_records_map=_summary_event_maps('ExitEventDate',exit_cols,'LastObservedPrice')

    # Enrich the relatively small set of asking-price change events once, then
    # group them into O(1) lookup maps for the output loop.
    price_events=daily[daily['_price_changed']].copy()
    enrich_cols=[c for c in ['_listing_id','Brand','Model','Year','Category','Fuel','Link'] if c in summary.columns]
    if not price_events.empty and enrich_cols:
        enrich=summary[enrich_cols].drop_duplicates('_listing_id',keep='last')
        price_events=price_events.merge(enrich,on='_listing_id',how='left',suffixes=('','_Summary'))

    def _price_event_maps(mask_col):
        ids_map={}; rec_map={}
        if price_events.empty: return ids_map,rec_map
        subset=price_events[price_events[mask_col]].copy()
        if subset.empty: return ids_map,rec_map
        for (company,date),g in subset.groupby(['_company','_date'],sort=False):
            key=(str(company),pd.Timestamp(date))
            ids_map[key]=g['_listing_id'].tolist()
            records=[]
            for er in g.to_dict('records'):
                rec={'listing_id':er.get('_listing_id'),'old_price':er.get('_prev_price'),'new_price':er.get('_price')}
                for c in ['Brand','Model','Year','Category','Fuel','Link']:
                    v=er.get(c)
                    if v is None or (isinstance(v,float) and np.isnan(v)):
                        v=er.get(c+'_Summary')
                    if v is not None and not (isinstance(v,float) and np.isnan(v)):
                        rec[c]=v
                if not rec.get('Fuel'):
                    rec['Fuel']=_fuel_from_link(rec.get('Link'))
                if (not rec.get('Brand') or not rec.get('Model')) and rec.get('Category'):
                    b,m=_brand_model_from_category(rec.get('Category'))
                    rec['Brand']=rec.get('Brand') or b
                    rec['Model']=rec.get('Model') or m
                records.append(rec)
            rec_map[key]=records
        return ids_map,rec_map

    reduced_ids_map,reduction_records_map=_price_event_maps('_price_reduced')
    increased_ids_map,increase_records_map=_price_event_maps('_price_increased')

    rows=[]
    for company in companies:
        opening,closing=stock_arrays.get(company,(np.zeros(n_dates,dtype=np.int64),np.zeros(n_dates,dtype=np.int64)))
        for i,date in enumerate(dates):
            key=(company,date)
            new_ids=new_ids_map.get(key,[])
            exit_ids=exit_ids_map.get(key,[])
            reduced_ids=reduced_ids_map.get(key,[])
            increased_ids=increased_ids_map.get(key,[])
            rows.append({
                'Date':date.strftime('%Y-%m-%d'),
                'Company':company,
                'OpeningObservedStockCount':int(opening[i]),
                'ClosingObservedStockCount':int(closing[i]),
                'NetObservedStockChange':int(closing[i]-opening[i]),
                'NewlyObservedListings':len(new_ids),
                'ObservedMarketExits':len(exit_ids),
                'AskingPriceReductions':len(reduced_ids),
                'AskingPriceIncreases':len(increased_ids),
                'NewListingIds':_json_list(new_ids),
                'ObservedExitListingIds':_json_list(exit_ids),
                'PriceReductionListingIds':_json_list(reduced_ids),
                'PriceIncreaseListingIds':_json_list(increased_ids),
                'NewListingsJson':_safe_json_records(new_records_map.get(key,[])),
                'ObservedExitListingsJson':_safe_json_records(exit_records_map.get(key,[])),
                'PriceReductionListingsJson':_safe_json_records(reduction_records_map.get(key,[])),
                'PriceIncreaseListingsJson':_safe_json_records(increase_records_map.get(key,[])),
                'ObservedStockDefinition':'Listings observed advertised in our dataset; not confirmed physical inventory',
                'ObservedExitDefinition':'First subsequent market snapshot after a listing was last observed; not a confirmed sale',
            })
    out=pd.DataFrame(rows)
    out.to_csv(output_csv,index=False)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input',nargs='?',default='kktcarabam_daily.csv')
    ap.add_argument('output',nargs='?',default='business_company_activity_daily.csv')
    args=ap.parse_args()
    out=build(args.input,args.output)
    print(f'Wrote {len(out):,} company-day rows to {args.output}')
    print(f'Companies: {out.Company.nunique():,}; dates: {out.Date.min()} -> {out.Date.max()}')

if __name__=='__main__': main()
