"""Pure feature preparation copied from original pilot; no fitting or file writes."""
import numpy as np
import pandas as pd

def prepare(d):
    num=['accommodates','bedrooms','beds','bathroom_count','bathroom_shared','host_days','host_response_rate','host_acceptance_rate','host_listings_count','minimum_nights','maximum_nights','number_of_reviews','reviews_per_month','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','quote_stay_nights','quote_lead_days','scrape_day']
    cat=['property_type','room_type','host_response_time','host_is_superhost','host_has_profile_pic','host_identity_verified','instant_bookable','quote_month','quote_weekday']
    X=pd.DataFrame(index=d.index)
    for v in num+cat:
     if v in d:X[v]=d[v]
    X['bathroom_count']=pd.to_numeric(d.bathrooms_text.str.extract(r'(\d+(?:\.\d+)?)',expand=False),errors='coerce')
    X.loc[d.bathrooms_text.str.contains('half-bath',case=False,na=False),'bathroom_count']=.5
    X['bathroom_shared']=d.bathrooms_text.str.contains('shared',case=False,na=False).astype(float)
    X.loc[d.bathrooms_text.isna(),'bathroom_shared']=np.nan
    scrape=pd.to_datetime(d.last_scraped,errors='coerce');ci=pd.to_datetime(d.price_quote_checkin_date,errors='coerce');co=pd.to_datetime(d.price_quote_checkout_date,errors='coerce')
    X['host_days']=(12*pd.to_numeric(d.hosts_time_as_host_years,errors='coerce')+pd.to_numeric(d.hosts_time_as_host_months,errors='coerce'))*365.25/12
    X['quote_stay_nights']=(co-ci).dt.days;X['quote_lead_days']=(ci-scrape).dt.days;X['scrape_day']=(scrape-pd.Timestamp('2026-09-17')).dt.days
    X['quote_month']=ci.dt.month;X['quote_weekday']=ci.dt.dayofweek
    for v in ['host_response_rate','host_acceptance_rate']:X[v]=pd.to_numeric(X[v].astype(str).str.rstrip('%'),errors='coerce')/100
    for v in num:X[v]=pd.to_numeric(X[v],errors='coerce')
    for v in cat:X[v]=X[v].fillna('missing').astype(str)
    # Semantic amenity families fixed without seeing targets. One source column, not 31 raw columns.
    patterns={
    'wifi':r'wifi','kitchen':r'kitchen','washer':r'washer','dryer':r'(?<!hair )dryer','dishwasher':r'dishwasher','heating':r'heating','air_conditioning':r'air conditioning','dedicated_workspace':r'dedicated workspace','parking':r'parking','elevator':r'elevator','private_outdoor':r'private patio|private balcony|private backyard|private garden','pool':r'pool','hot_tub':r'hot tub','gym':r'gym','breakfast':r'breakfast','tv':r'television|\btv\b|hdtv','self_checkin':r'self check-in','smoke_alarm':r'smoke alarm','carbon_monoxide_alarm':r'carbon monoxide alarm','fire_extinguisher':r'fire extinguisher','first_aid':r'first aid kit','crib':r'crib|travel cot|travel crib','high_chair':r'high chair','pets_allowed':r'pets allowed','luggage_dropoff':r'luggage dropoff','coffee_maker':r'coffee maker','oven':r'oven','refrigerator':r'refrigerator','bathtub':r'bathtub','hair_dryer':r'hair dryer','long_stays':r'long term stays allowed'}
    for name,pattern in patterns.items():
     v='amenity_'+name;X[v]=d.amenities.fillna('').str.contains(pattern,case=False,regex=True).astype(float);num.append(v)
    # Haversine distance in kilometres to the prespecified centre.
    a=np.radians(pd.to_numeric(d.latitude));b=np.radians(pd.to_numeric(d.longitude));a0,b0=np.radians([51.5074,-.1278])
    X['distance_km']=2*6371*np.arcsin(np.sqrt(np.clip(np.sin((a-a0)/2)**2+np.cos(a)*np.cos(a0)*np.sin((b-b0)/2)**2,0,1)))
    X['latitude']=d.latitude;X['longitude']=d.longitude;X['neighbourhood_cleansed']=d.neighbourhood_cleansed.fillna('missing').astype(str)
    return X, num, cat
