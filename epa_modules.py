import pandas as pd
import numpy as np
import requests
import json
import geopandas as gpd
from brix import Handler, Indicator
import pandana
from scipy.optimize import linear_sum_assignment
import osmnx
import datetime

import sys
sys.path.append('../')
sys.path.append('../../')
from Proximization import proximity
from JobMap import jobmap
import toolbox

def get_overlap_props(gdf1, gdf2):
    gdf1['copy_index']=gdf1.index
    all_intersect=gpd.overlay(gdf2.to_crs(gdf1.crs), gdf1, 'intersection')
    all_intersect['intersect_area']=all_intersect.geometry.area
    zone_totals=all_intersect.groupby('copy_index').sum().rename(columns={'intersect_area': 'total_area'})
    all_intersect=all_intersect.merge(zone_totals[['total_area']],
        how='left', left_on='copy_index', right_index=True)
    all_intersect['prop_area']=all_intersect['intersect_area']/all_intersect['total_area']
    return all_intersect

def get_replica_net():
    net=pd.read_csv('data/replica_epa_net.csv')
    net=net.loc[net['lanes']>0]
    net[['stableEdgeId','startLat','startLon','endLat','endLon']]
    return net

def get_traffic(trips_df, included_edge_ids):
    expanded=[]
    for ind, row in trips_df.iterrows():
        for link in row['network_link_ids']:
            if int(link) in included_edge_ids:
                expanded.append({'link': int(link), 'industry': row['industry'], 
                                 'hour': int(row['start_local_hour']), 
                                 'mode': row['mode'],
                                 'soc': row['soc'],
#                                 'travel_purpose': row['travel_purpose'],
                                'Purpose': row['Purpose'],
                                'Commuter Type': row['Commuter Type'],
                                'pcu': row['pcu']})
    expanded_df=pd.DataFrame.from_dict(expanded)
    base_traffic_by_ctype=expanded_df.groupby(
        ['Commuter Type', 'link'])['pcu'].sum()
#     base_traffic_total=base_traffic_by_ctype.groupby('link')['pcu'].sum()
    return base_traffic_by_ctype

def return_jobs_commuting_metrics(expanded_geogriddata_df, 
                                 jobtrans, job_map, 
                                 pop, pcu_miles_pp_by_commute_status_tour_type, 
                                  pcu_miles_tour_type, n_by_commute_type,
                                 third_place_trip_factor):
    naics_to_N=expanded_geogriddata_df[[col for col in expanded_geogriddata_df.columns if (
        ('naics' in col)&('sqm' not in col))]].sum().to_dict()
    print(naics_to_N)
    naics_dict={k.split('_')[1]: naics_to_N[k] for k in naics_to_N}
    print(naics_dict)
    soc_dict=job_map.get_employees_by_soc(naics_dict, as_dict=True)
    job_list=job_map.get_job_list(soc_dict)
    jobs_df=pd.DataFrame(job_list)
    print('Number of jobs created: {}'.format(len(jobs_df)))
    if len(jobs_df)>3000:
        sample_rate=3000/len(jobs_df) # if more than 10k jobs, sample back to down to 10k
    else:
        sample_rate=1
    jobs_df=jobs_df.sample(frac=sample_rate).reset_index(drop=True)
    candidates=pop.loc[pop['is_resident']].sample(frac=sample_rate).reset_index(drop=True)
    
    if len(jobs_df)>0:
        print('Creating SOC mat jobs')
        soc_mat_jobs=jobtrans.soc_list_to_matrix(jobs_df['soc'].values)

        print('Creating SOC mat candidates')
        soc_mat_candidates=jobtrans.soc_list_to_matrix(candidates['soc'].values)

        salary_decrease=jobtrans.get_salary_decrease_mat(
            candidates['individual_income'].astype(float), jobs_df['salary'].astype(float))

        print('Attempting to match {} jobs from {} local candidates'.format(
            len(jobs_df), len(candidates)))

        transitions=jobtrans.match_jobs(
            from_soc_mat=soc_mat_candidates,  to_soc_mat=soc_mat_jobs, 
            salary_decrease=salary_decrease)
        print('\t Matched {} jobs locally'.format(len(transitions)))
    
        transitions=transitions.set_index('col_ind', drop=True)

        jobs_assignment=jobs_df.merge(transitions,
                       left_index=True, right_index=True, how='left').rename(
            columns={'soc': 'soc_new', 'salary': 'salary_new'
        })
    
        jobs_assignment=jobs_assignment.dropna(subset=['row_ind']).copy()
        jobs_assignment['row_ind']=jobs_assignment['row_ind'].astype(int)

        jobs_assignment=jobs_assignment.merge(
            candidates, left_on='row_ind', right_index=True,
            how='left')
    else:
        jobs_assignment=pd.DataFrame({
            'Commuter Type': [None], 'salary_new': [1], 'individual_income': [1]})
              
    print('Calculating Metrics')
    
    local_jobs_score=min(1, 2*(len(jobs_assignment)/len(candidates)))
    print('Local jobs score : {}'.format(local_jobs_score))
    
    # scale back up the jobs numbers
    n_jobs_filled_local=len(jobs_assignment)/sample_rate   
    jobs_not_filled_local=(len(jobs_df)/sample_rate)-n_jobs_filled_local
    
    new_housing_capacity=expanded_geogriddata_df.loc[
        expanded_geogriddata_df['interactive'], 'res_total'].sum()
    new_employee_housing_capacity=(new_housing_capacity/1.7)
    
    new_in_commuters=max(0, jobs_not_filled_local-new_employee_housing_capacity)
    new_lw=min(jobs_not_filled_local, new_housing_capacity)
    new_out_commuters=max(0, new_employee_housing_capacity-new_lw)

    n_cout_to_lw = (jobs_assignment['Commuter Type']=='Out Commuter').sum()/sample_rate

    updated_n_by_commute_type = n_by_commute_type.copy()
    updated_n_by_commute_type['Live-Work']+=(n_cout_to_lw + new_lw)
    updated_n_by_commute_type['Out Commuter']+=  (new_out_commuters-n_cout_to_lw)
    updated_n_by_commute_type['In Commuter']+= new_in_commuters

    # traffic['lanes']=traffic['lanes'].clip(lower=1)
    # traffic['pcu']=0
    # traffic['base_traffic']=traffic['pcu']/traffic['lanes']
    traffic['new_pcu']=traffic['pcu'].copy()
    traffic['new_pcu']+=(n_cout_to_lw + new_lw)*traffic['Live-Work_pcu_pp']
    traffic['new_pcu']+=(new_out_commuters-n_cout_to_lw)*traffic['Out Commuter_pcu_pp']
    traffic['new_pcu']+=(new_in_commuters)*traffic['In Commuter_pcu_pp']
    traffic['new_traffic']=traffic['new_pcu']/traffic['capacity']
    # traffic_max=traffic['new_traffic'].max()
    # traffic_max=1400
    # print('Max: {}'.format(traffic_max))
    # traffic['norm_traffic']=(traffic['new_traffic']/traffic_max).clip(upper=1)
    traffic['norm_traffic']=traffic['new_traffic'].clip(upper=1)
    traffic_output=traffic[['norm_traffic', 'lanes', 'geometry']].__geo_interface__

    print(updated_n_by_commute_type)

    updated_pcu_miles_tour_type={}
    for purpose in ['Home<->Work Trips', 'Third Place Trips']:
        updated_pcu_miles_tour_type[purpose]=0
        for ctype in ['Live-Work', 'Out Commuter', 'In Commuter']:
            updated_pcu_miles_tour_type[purpose]+=(
                pcu_miles_pp_by_commute_status_tour_type[ctype, purpose]*updated_n_by_commute_type[ctype])
    print(updated_pcu_miles_tour_type)
            
    updated_pcu_miles_tour_type['Third Place Trips']*=third_place_trip_factor

    commuting_score=min(1, 
                        (pcu_miles_tour_type['Home<->Work Trips']/
                         updated_pcu_miles_tour_type['Home<->Work Trips'])*0.5)
    print('Commuting score: {}'.format(commuting_score))
    third_place_trip_score=min(1, 
                        (pcu_miles_tour_type['Third Place Trips']/
                         updated_pcu_miles_tour_type['Third Place Trips'])*0.5)
    print('Third Place Trip score: {}'.format(third_place_trip_score))

    salary_delta=jobs_assignment['salary_new'].astype(float)-jobs_assignment['individual_income'].astype(float)
    avg_salary_delta=salary_delta.mean()

    # TODO: incorrect: redo this
    avg_salary_delta_pop=(1/sample_rate)*salary_delta.sum()/pop['is_resident'].sum()
    prop_change_salary_pop=1+(avg_salary_delta_pop/mean_salary_base)
    local_salary_score=min(prop_change_salary_pop*0.5, 1)
    
    inds=[{'name': 'Local Jobs', 'value':local_jobs_score, 'ref_value': 0, 'description': 'Amount of new job creation'},
         {'name': 'Local Salaries', 'value':local_salary_score, 'ref_value': 0.5, 'description': 'Average change in salary for locals'},
         {'name': 'Commute Trip Score', 'value':commuting_score, 'ref_value': 0.5, 'description': 'Reduction in miles driven for commuting between home and work'},
         {'name': 'Third Place Trip Score', 'value':third_place_trip_score, 'ref_value': 0.5, 'description': 'Reduction in miles driven for non-work purposes'},
         {'name': 'Workforce', 'value':(local_jobs_score+local_salary_score)/2, 'ref_value': 0.5, 'viz_type': 'bar', 'description':'Aggregate indicator of workforce impacts'},
         {'name': 'Mobility', 'value':(commuting_score+third_place_trip_score)/2, 'ref_value': 0.5, 'viz_type': 'bar', 'description':'Aggregate indicator of mobility impacts'}]    
    return inds, traffic_output

def tour_map(x):
    if x=='COMMUTE':
        return 'Home<->Work Trips'
    elif x=='OTHER_HOME_BASED':
        return 'Third Place Trips'
    elif x=='WORK_BASED':
        return 'Third Place Trips'

def get_pop_and_trip_miles(keep_cols):
    # TODO: this is all computed already in the pre-cook traffic script- just load the data
    # some series saved to csv and loaded from csv- need to take care of this
    with open('data/replica_epa_pop_soc.json') as f:
        trips = json.load(f)
    trips_df=pd.DataFrame.from_dict(trips)[keep_cols]
    trips_df=trips_df.dropna(subset=['soc', 'BLOCKGROUP', 'BLOCKGROUP_work']).copy()
    for col in ['BLOCKGROUP', 'BLOCKGROUP_work']:
        trips_df[col]=trips_df[col].astype(int)
        
    pcu_by_mode = {'PRIVATE_AUTO': 1, 'CARPOOL': 1/3, 'ON_DEMAND_AUTO': 1}
    
    trips_df['pcu']=0
    for mode in pcu_by_mode:
        trips_df.loc[trips_df['mode']==mode, 'pcu']=pcu_by_mode[mode]
    trips_df['pcu_miles']=trips_df['pcu']*trips_df['distance_miles']
    
    trips_df.loc[trips_df['BLOCKGROUP'].isin(epa_geoids), 'Commuter Type']='Out Commuter'
    trips_df.loc[trips_df['BLOCKGROUP_work'].isin(epa_geoids), 'Commuter Type']='In Commuter'
    trips_df.loc[((trips_df['BLOCKGROUP_work'].isin(epa_geoids))&
                  (trips_df['BLOCKGROUP'].isin(epa_geoids))), 'Commuter Type']='Live-Work'
    trips_df['Purpose']=trips_df['tour_type'].apply(tour_map)
    
    n_by_commute_type=trips_df.groupby('Commuter Type')['person_id'].nunique()
    pcu_miles_commute_status_tour_type=trips_df.groupby(['Commuter Type', 'Purpose'])['pcu_miles'].sum()
    pcu_miles_tour_type=trips_df.groupby('Purpose')['pcu_miles'].sum()
    denom = [n_by_commute_type[i[0]] for i in pcu_miles_commute_status_tour_type.index]
    pcu_miles_pp_by_commute_status_tour_type=pcu_miles_commute_status_tour_type/denom
    
    pop=trips_df[['person_id', 'BLOCKGROUP', 'BLOCKGROUP_work', 'BLOCKGROUP_school',
       'individual_income', 'industry', 'soc', 'Commuter Type']].drop_duplicates(subset=['person_id'])
    
    # TODO: this is all redundant?
    # net is only loaded to get the list of edge ids (which I think is also available in the traffic.geojson)
    # traffic_pp_by_ctype is not used- it's now already merged into the traffic.geojson
    net=get_replica_net() 
    traffic_by_ctype=get_traffic(trips_df, set(net['stableEdgeId']))
    denom=[n_by_commute_type[i[0]] for i in traffic_by_ctype.index]
    traffic_pp_by_ctype= traffic_by_ctype/denom
    
    return pop, pcu_miles_pp_by_commute_status_tour_type, pcu_miles_tour_type, n_by_commute_type, net, traffic_pp_by_ctype

class Comp_Indicators_EPA(proximity.Proximity_Indicator):
    def return_indicator(self, geogrid_data):
        start_time=datetime.datetime.now()
        geogrid_data_gdf=toolbox.expand_geogrid_data(geogrid_data)
        
        is_interactive=geogrid_data_gdf['interactive'].astype(bool)
        #TODO: parks in person capacity units
        for a in self.sqm_pp_major:
            area_col=self.target_settings[a]['column']+'_area'
            if area_col in geogrid_data_gdf.columns:
                geogrid_data_gdf.loc[is_interactive, self.target_settings[a]['column']
                    ]=geogrid_data_gdf.loc[is_interactive, area_col]/self.sqm_pp_major[a]
        geogrid_data_gdf.loc[is_interactive, 'emp_capacity'
            ]=self.employed_ratio*geogrid_data_gdf.loc[is_interactive, 'emp_total']

        # Concatenate the baseline static data with the updated geogrid data
        updated_places=pd.concat([self.static_places, geogrid_data_gdf.loc[is_interactive]]).fillna(0)

        geo_output, final_scores=proximity.calculate_access(
            updated_places, self.target_settings, self.reachable)

        heatmap_cols=[t+'_access' for t in self.target_list]+['geometry']
        geo_heatmap=geo_output.loc[geo_output['impact_area'], heatmap_cols]
        geo_heatmap.columns=self.target_list+['geometry']
        geo_heatmap.geometry=geo_heatmap.geometry.centroid
        cs_heatmap=self.get_cs_heatmap(geo_heatmap, self.target_list)
        result=[{'name': '{} Prox'.format(t).title(),
                 'description': 'Average proximity to supply of {}, normalised by demand'.format(t),
                 'value': final_scores[t],
                 'ref_value': self.baseline_scores[t]} for t in self.target_list]

        lw_symm=min(final_scores['Housing'], final_scores['Jobs'])

        result.append({'name': 'LW-Symmetry',
            'description': 'How well-balanced are the availabilies of jobs and housing',
            'value': lw_symm,
            'ref_value': self.baseline_scores['LW_Symmetry'],
            'viz_type': 'bar'})
        
        # assume half of miles driven for third places can be avoided by meeting demand locally
        # for this half, reduction in miles proportional to reduction in unmet demand
        # eg. defiency goes from 40% to 20%-> reduced by half-> miles driven *=0.75
        ratio_retail_defficiency=(1-final_scores['Retail'])/(1-self.baseline_scores['Retail'])
        third_place_trip_factor=0.5 + 0.5*ratio_retail_defficiency
        print('Third place trip factor: {}'.format(third_place_trip_factor))
        
        jobs_commuting_metrics, traffic_output=return_jobs_commuting_metrics(geogrid_data_gdf, 
                                 jobtrans, job_map, 
                                 pop, pcu_miles_pp_by_commute_status_tour_type, 
                                  pcu_miles_tour_type, n_by_commute_type,
                                 third_place_trip_factor=third_place_trip_factor)
        result.extend(jobs_commuting_metrics)

        table_name='epa'
        headers = {'Content-Type': 'application/json'}
        url='https://cityio.media.mit.edu/api/table/{}'.format(table_name)
        r=requests.post(url+'/traffic', json.dumps(traffic_output), headers=headers)
        print('Traffic output: {}'.format(r))
        
        print('Time taken: {}'.format(datetime.datetime.now()-start_time))  
        return {'heatmap':cs_heatmap,'numeric':result}


BLS_data_loc='data/BLS/oesm21all/all_data_M_2021.xlsx'

# Prepare the Proximity Indicator

nodes_gdf=gpd.read_file('data/nodes.geojson').set_index('osmid')
edges_gdf=gpd.read_file('data/edges.geojson')
bgs_local=gpd.read_file('data/blocks_local.geojson')
bgs_local['GEOID']=bgs_local['geoid']
bgs_local=bgs_local.set_index('geoid', drop=False)

geogrid_gdf=gpd.read_file('data/geogrid.geojson').set_index('id')

amenities_major=['Open Space', 'Retail']
sqm_pp_major={'Open Space': 20, 'Retail': 2.3}

retail_sqm_p_emp=50
fb_sqm_p_emp=10
bgs_local['Retail_sqm']=retail_sqm_p_emp*bgs_local['emp_naics_44-45']+fb_sqm_p_emp*bgs_local['emp_naics_72']
bgs_local['Retail']=bgs_local['Retail_sqm']/sqm_pp_major['Retail']


osm_geoms=osmnx.geometries.geometries_from_place('East Palo Alto, CA', tags={
    'leisure': True})

amenities=osm_geoms.loc[~(osm_geoms['leisure'].isnull()), ['leisure', 'geometry']].reset_index()
amenities['area']=amenities.to_crs(toolbox.get_crs(amenities)).geometry.area
amenities['Amenity']=amenities['area']/sqm_pp_major['Open Space']

## Prepare the full static places data
geogrid_gdf['cell_id']=geogrid_gdf.index.copy()
active_cells=geogrid_gdf.loc[~(geogrid_gdf['name']=='None')]

all_intersect=get_overlap_props(bgs_local.to_crs(toolbox.get_crs(bgs_local)),
                         active_cells.to_crs(toolbox.get_crs(geogrid_gdf)))
target_cols=['emp_total', 'res_total', 'Retail']
for col in target_cols:
    all_intersect[col]*=all_intersect['prop_area']
result=all_intersect.groupby('cell_id')[target_cols].sum()
geogrid_gdf=geogrid_gdf.merge(result, left_index=True, right_on='cell_id', how='left')
geogrid_gdf[target_cols]=geogrid_gdf[target_cols].fillna(0)

static_grid_cells=geogrid_gdf.loc[geogrid_gdf['interactive']==False]

overlap_ids=all_intersect['GEOID'].unique()
external_zones=bgs_local.loc[~(bgs_local['GEOID'].isin(overlap_ids))]
external_zones.plot()

external_zones['impact_area']=False
amenities['impact_area']=False
static_grid_cells['impact_area']=True

static_places=pd.concat(
    [static_grid_cells, amenities, external_zones]).reset_index()


target_settings={'Housing': {'column': 'res_total', 'demand_source': ['emp_total']},
                'Jobs': {'column': 'emp_capacity', 'demand_source': ['res_total']},
                'Open Space': {'column': 'Amenity', 'demand_source': ['res_total']},
                'Retail': {'column': 'Retail', 'demand_source': ['res_total']}}

pdna_net=pandana.Network(nodes_gdf["x"], nodes_gdf["y"],
                    edges_gdf["u"], edges_gdf["v"],
                    edges_gdf[["length"]])

# Prepare the jobs and commuting indicator
soc_code_map=jobmap.get_soc_code_map('data/BLS/soc_2018_definitions.xlsx')
state_code='CA'
bls_data=pd.read_excel('data/BLS/oesm21all/all_data_M_2021.xlsx')
job_map=jobmap.JobMap(state_code='CA', bls_data=bls_data)

traffic=gpd.read_file('data/traffic2.geojson')
with open('data/epa_geoids.json') as f:
    epa_geoids=json.load(f)
keep_cols=['person_id','BLOCKGROUP', 'BLOCKGROUP_work', 'BLOCKGROUP_school',
          'individual_income', 'mode', 'network_link_ids', 'start_local_hour',
          'industry', 'soc', 'distance_miles', 'travel_purpose','tour_type']
pop, pcu_miles_pp_by_commute_status_tour_type, pcu_miles_tour_type, n_by_commute_type, net, traffic_pp_by_ctype=get_pop_and_trip_miles(keep_cols)
is_resident=pop['Commuter Type'].isin(['Live-Work', 'Out Commuter'])
pop['is_resident']=is_resident
mean_salary_base=pop.loc[is_resident,
    'individual_income'].astype(float).mean()
jobtrans=jobmap.JobTransitioner()

max_dist=80*17

all_indicators=Comp_Indicators_EPA(
    static_places, geogrid_gdf, max_dist, 
    target_settings, pdna_net, nodes_gdf,sqm_pp_major,
    employed_ratio=1.7,
    reachable=None)
all_indicators.baseline_scores['LW_Symmetry']=min(
    all_indicators.baseline_scores['Housing'], all_indicators.baseline_scores['Jobs'])

H=Handler('epa')
H.add_indicator(all_indicators)
H.listen()


