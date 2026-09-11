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
    brand_col=_pick(raw.columns,['Brand','brand'])
    model_col=_pick(raw.columns,['Model','model'])
    year_col=_pick(raw.columns,['Year','year'])
    category_col=_pick(raw.columns,['Category','category','CategoryDetail'])
    if not date_col or not company_col:
        raise ValueError('Input must contain date and Company columns')

    df=raw.copy()
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
    df['_price']=pd.to_numeric(df[price_col],errors='coerce') if price_col else np.nan

    # One observation per listing/date; last duplicate wins deterministically by file order.
    df['_row_order']=np.arange(len(df))
    daily=(df.sort_values('_row_order')
             .drop_duplicates(['_listing_id','_date'],keep='last')
             .sort_values(['_listing_id','_date']))

    dates=sorted(pd.Timestamp(x) for x in daily['_date'].dropna().unique())
    date_index={d:i for i,d in enumerate(dates)}
    latest=dates[-1]

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
    if link_col: agg_spec['Link']=(link_col,'first')
    summary=grouped.agg(**agg_spec).reset_index()
    summary['ExitEventDate']=pd.NaT
    for idx,row in summary.iterrows():
        last=pd.Timestamp(row['LastObservedDate'])
        if last>=latest: continue
        pos=date_index.get(last)
        if pos is not None and pos+1<len(dates):
            summary.at[idx,'ExitEventDate']=dates[pos+1]

    # Price-change events from consecutive observed asking prices.
    daily['_prev_price']=daily.groupby('_listing_id')['_price'].shift(1)
    daily['_price_changed']=daily['_price'].notna() & daily['_prev_price'].notna() & (daily['_price']!=daily['_prev_price'])
    daily['_price_reduced']=daily['_price_changed'] & (daily['_price']<daily['_prev_price'])
    daily['_price_increased']=daily['_price_changed'] & (daily['_price']>daily['_prev_price'])

    companies=sorted(summary['Company'].dropna().astype(str).unique())
    rows=[]
    summary_by_company={c:g.copy() for c,g in summary.groupby('Company')}
    price_events=daily[daily['_price_changed']].copy()
    price_by_company={(c,d):g for (c,d),g in price_events.groupby(['_company','_date'])}

    for company in companies:
        listings=summary_by_company[company]
        for i,date in enumerate(dates):
            opening_date=dates[i-1] if i>0 else None
            # Conservative observed-advertised stock interval between first and last observation.
            closing_mask=(listings['FirstObservedDate']<=date) & (listings['LastObservedDate']>=date)
            closing_ids=listings.loc[closing_mask,'_listing_id'].tolist()
            if opening_date is None:
                opening_ids=[]
            else:
                opening_mask=(listings['FirstObservedDate']<=opening_date) & (listings['LastObservedDate']>=opening_date)
                opening_ids=listings.loc[opening_mask,'_listing_id'].tolist()
            new_ids=listings.loc[listings['FirstObservedDate']==date,'_listing_id'].tolist()
            exit_ids=listings.loc[listings['ExitEventDate']==date,'_listing_id'].tolist()
            pe=price_by_company.get((company,date))
            reduced_ids=pe.loc[pe['_price_reduced'],'_listing_id'].tolist() if pe is not None else []
            increased_ids=pe.loc[pe['_price_increased'],'_listing_id'].tolist() if pe is not None else []

            display_cols=[c for c in ['_listing_id','Brand','Model','Year','Category','Link'] if c in listings.columns]
            new_records=listings.loc[listings['FirstObservedDate']==date,display_cols].rename(columns={'_listing_id':'listing_id'}).to_dict('records')
            exit_records=listings.loc[listings['ExitEventDate']==date,display_cols].rename(columns={'_listing_id':'listing_id'}).to_dict('records')
            reduction_records=[]; increase_records=[]
            if pe is not None:
                for _,er in pe.iterrows():
                    rec={'listing_id':er.get('_listing_id'),'old_price':er.get('_prev_price'),'new_price':er.get('_price')}
                    if brand_col: rec['Brand']=er.get(brand_col)
                    if model_col: rec['Model']=er.get(model_col)
                    if year_col: rec['Year']=er.get(year_col)
                    if link_col: rec['Link']=er.get(link_col)
                    if bool(er.get('_price_reduced')): reduction_records.append(rec)
                    if bool(er.get('_price_increased')): increase_records.append(rec)
            rows.append({
                'Date':date.strftime('%Y-%m-%d'),
                'Company':company,
                'OpeningObservedStockCount':len(opening_ids),
                'ClosingObservedStockCount':len(closing_ids),
                'NetObservedStockChange':len(closing_ids)-len(opening_ids),
                'NewlyObservedListings':len(new_ids),
                'ObservedMarketExits':len(exit_ids),
                'AskingPriceReductions':len(reduced_ids),
                'AskingPriceIncreases':len(increased_ids),
                'NewListingIds':_json_list(new_ids),
                'ObservedExitListingIds':_json_list(exit_ids),
                'PriceReductionListingIds':_json_list(reduced_ids),
                'PriceIncreaseListingIds':_json_list(increased_ids),
                'NewListingsJson':_safe_json_records(new_records),
                'ObservedExitListingsJson':_safe_json_records(exit_records),
                'PriceReductionListingsJson':_safe_json_records(reduction_records),
                'PriceIncreaseListingsJson':_safe_json_records(increase_records),
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
