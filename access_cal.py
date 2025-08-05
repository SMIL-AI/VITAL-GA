import pandas as pd
import geopandas as gpd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
for year in range(2019, 2024):
    parquet_file_path = f'/combined_GA_weekly_patterns_{year}.parquet'
    df = pd.read_parquet(parquet_file_path)
    print(f'{year}',df.shape)


    naics_code_62 = df[df['naics_code'].astype(str).str.startswith('62')]
    hospital = naics_code_62[naics_code_62['naics_code'] == 622110]
    pcp_naics_codes = [621111]
    pcp = naics_code_62[naics_code_62['naics_code'].isin(pcp_naics_codes)]
    UR_naics_codes = [621493]
    UR = naics_code_62[naics_code_62['naics_code'].isin(UR_naics_codes)]
    emergency_cat_codes = ['Emergency Medicine']
    emergent_cat_hos =  hospital[hospital['category_tags'].isin(emergency_cat_codes)]
    print(emergent_cat_hos.shape)
    pcp_1 =pcp[pcp['category_tags'].isin(['Family Doctor', 'Internal Medicine'])]
    pcp_2 =pcp[pcp['category_tags'].isin(['Pediatricians'])]



    # Define datasets
    datasets = {
        'Emergency Room Visits': emergent_cat_hos,
        'Adult Primary Care': pcp_1,
        'Pediatric Primary Care': pcp_2,
        'Urgent Care Visits': UR,   
    }
    for title, df in datasets.items():
        print(title, df.shape)

    # Load the GeoJSON file once
    geojson_path = "tl_2019_13_bg.shp"
    geo_data = gpd.read_file(geojson_path)
    geo_data['GEOID'] = geo_data['GEOID'].astype(str).str.zfill(12)
    geo_data = geo_data.to_crs(epsg=3857).copy()
    # Calculate centroids and extract coordinates
    geo_data['centroid_lat'] = geo_data.geometry.centroid.y
    geo_data['centroid_lon'] = geo_data.geometry.centroid.x

    dem_data = {}
    dem_data[year] = pd.read_csv(f'dempgraphic_DCA/{year}.csv')
    dem_data[year].columns = dem_data[year].columns.str.strip()
    pop_column = 'Total Population' if year == 2019 else 'weighted_Total Population'
    dem_data[year] = dem_data[year].dropna(subset=[pop_column])
    cbg_column = 'CBG_ID' if year == 2019 else 'GEOID_BLKGRP_10'
    dem_data[year][cbg_column] = dem_data[year][cbg_column].astype(str).str.zfill(12)

    output_dir = '/acess_cal'
    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset and year
    for title, df in datasets.items():
            # Create a list to store parsed data
            visitor_home_data = []
            
            for idx, row in df.iterrows():
                if pd.notnull(row['visitor_home_cbgs']):
                    try:
                        cbg_dict = json.loads(row['visitor_home_cbgs'])  # Parse JSON
                        for cbg, count in cbg_dict.items():
                            if cbg in set(geo_data['GEOID']):
                                visitor_home_data.append({
                                    'cbg': cbg, 
                                    'visit_count': count, 
                                    'poi_latitude': row['latitude'],  # Access row values correctly
                                    'poi_longitude': row['longitude']
                            })
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON at index {idx}: {row['visitor_home_cbgs']}")
            visitor_home_data = pd.DataFrame(visitor_home_data)
            visitor_home_data = gpd.GeoDataFrame(
                visitor_home_data, 
                geometry=gpd.points_from_xy(visitor_home_data['poi_longitude'], visitor_home_data['poi_latitude']), 
                crs="EPSG:4326"
            )

            # Transform to EPSG:3857 for distance calculations
            visitor_home_data = visitor_home_data.to_crs(epsg=3857)

            visitor_home_df = visitor_home_data
            visitor_home_df['cbg'] = visitor_home_df['cbg'].astype(str).str.zfill(12)
            visitor_home_df['poi_x'] = visitor_home_df.geometry.x  # Extract projected coordinates
            visitor_home_df['poi_y'] = visitor_home_df.geometry.y
            visitor_home_df = visitor_home_df.merge(geo_data, 
                                        left_on='cbg', 
                                        right_on='GEOID', 
                                        how='left')
            visitor_home_df['neg_bi_dis_visit'] = (
                np.exp(-0.00005 * np.sqrt(
                    (visitor_home_df['poi_x'] - visitor_home_df['centroid_lon'])**2 + 
                    (visitor_home_df['poi_y'] - visitor_home_df['centroid_lat'])**2
                )) * visitor_home_df['visit_count']
            )
            
            visitor_home_df['visit_count'] = visitor_home_df['visit_count'].fillna(0)
            
            aggregated_neg_bi_dis_visit = visitor_home_df.groupby('cbg', as_index=False)['neg_bi_dis_visit'].sum()
            aggregated_neg_bi_dis_visit['cbg'] = aggregated_neg_bi_dis_visit['cbg'].astype(str).str.zfill(12)   
            
            dem_data[year][cbg_column] = dem_data[year][cbg_column].astype(str).str.replace('.0', '', regex=False).str.zfill(12)
            aggregated_neg_bi_dis_visit = aggregated_neg_bi_dis_visit.merge(dem_data[year], 
                                        left_on='cbg', 
                                        right_on= cbg_column, 
                                        how='inner')
            pop_column = 'Total Population' if year == 2019 else 'weighted_Total Population'
            aggregated_neg_bi_dis_visit['accessibility'] = aggregated_neg_bi_dis_visit['neg_bi_dis_visit'] / aggregated_neg_bi_dis_visit[pop_column]
            
            save_path = os.path.join(output_dir, f'{year}_{title}_access.csv')
            aggregated_neg_bi_dis_visit.to_csv(save_path, index=False)
            print(f'{year}_{title}_access.csv saved')


