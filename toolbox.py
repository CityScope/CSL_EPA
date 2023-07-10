import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

def expand_geogrid_data(geogrid_data):
    geogrid_data_gdf=geogrid_data.as_df()
    
    cell_area=(geogrid_data.get_geogrid_props()['header']['cellSize'])**2
    geogrid_data_gdf['height']=[h[1] if isinstance(h, list) else h for h in geogrid_data_gdf['height']]       
    geogrid_data_gdf['area']=cell_area
    is_interactive=geogrid_data_gdf['interactive'].astype(bool) # works for 'Web'
    geogrid_data_gdf['heatmap_area']=True
    # geogrid_data_gdf.loc[geogrid_data_gdf.index.isin([191, 207, 222, 223, 237, 238, 239, 253, 254, 255]), 'heatmap_area']=False
    geogrid_data_gdf['impact_area']=False
    geogrid_data_gdf.loc[geogrid_data_gdf['interactive'], 'impact_area']=True
    
    type_def=geogrid_data.get_type_info()
    
    # Add columns to each row (grid cell) for the sqm devoted to each :
    # CS_type, NAICS, LBCS and amenity
    present_types=[n for n in geogrid_data_gdf.loc[is_interactive, 'name'].unique() if 
                   ((n is not None) and (n!='None'))]
    for type_name in present_types:
        ind_this_type=((is_interactive)&(geogrid_data_gdf['name']==type_name))
        geogrid_data_gdf.loc[ind_this_type, '{}_area'.format(type_name)]=cell_area*geogrid_data_gdf.loc[ind_this_type,'height']
        for attr in ['sqmpp_res', 'sqmpp_emp']:
            if attr in type_def[type_name]:
                geogrid_data_gdf.loc[ind_this_type, attr]=type_def[type_name][attr]
        for attr in ['NAICS', 'LBCS', 'amenities']:
            if attr in type_def[type_name]:
                if type_def[type_name][attr] is not None:
                    for code in type_def[type_name][attr]:
                        col_name='sqm_{}_{}'.format(attr.lower(), code)
                        if col_name not in geogrid_data_gdf.columns:
                            geogrid_data_gdf[col_name]=0
                        code_prop=type_def[type_name][attr][code]
                        geogrid_data_gdf[col_name]+=(geogrid_data_gdf['{}_area'.format(type_name)]).fillna(0)*code_prop

    res_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_lbcs_1')]
    emp_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_naics_')]
    res_cols, emp_cols=[], []
    amenity_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_amenities_')]
    for col in res_sqm_cols:
        res_col_name=col.split('sqm_')[1]
        geogrid_data_gdf[res_col_name]=geogrid_data_gdf[col]/geogrid_data_gdf['sqmpp_res']
        res_cols.append(res_col_name)
    for col in emp_sqm_cols:
        emp_col_name=col.split('sqm_')[1]
        geogrid_data_gdf[emp_col_name]=geogrid_data_gdf[col]/geogrid_data_gdf['sqmpp_emp']
        emp_cols.append(emp_col_name)

    amenity_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_amenities_')]
    geogrid_data_gdf['amenity_total_sqm'] =geogrid_data_gdf[amenity_sqm_cols].sum(axis=1)    

    geogrid_data_gdf['res_total']=geogrid_data_gdf[res_cols].sum(axis=1)
    geogrid_data_gdf['emp_total']=geogrid_data_gdf[emp_cols].sum(axis=1)
    print('{} new residents'.format(geogrid_data_gdf['res_total'].sum()))
    print('{} new employees'.format(geogrid_data_gdf['emp_total'].sum()))
    return geogrid_data_gdf


def get_overlap_by_threshold(geom_1, geom_2, threshold=0.5, plot=True):
    """
    function for subsetting a polygon collection based on overlap with another polygon collection
    geom_1 and geom_2 should be geopandas GeoDataFrames
    geom_1 is the geometry to be subsetted
    subsetting is based on overlap with geom_2
    a zone in geom_1 will be included in the output if its area of overlap with geom_2 is greater than the threshold
    """
    geom_1['copy_index']=geom_1.index
    geom_1['zone_area']=geom_1.geometry.area
    all_intersect=gpd.overlay(geom_2.to_crs(geom_1.crs), geom_1, 'intersection')
    all_intersect['intersect_area']=all_intersect.geometry.area
    all_intersect=all_intersect[[col for col in all_intersect.columns if not col=='zone_area']]
    all_intersect=all_intersect.merge(geom_1[['copy_index', 'zone_area']],
        how='left', left_on='copy_index', right_on='copy_index')
    all_intersect['prop_area']=all_intersect['intersect_area']/all_intersect['zone_area']
    valid_intersect=all_intersect.loc[all_intersect['prop_area']>threshold]
    final_zone_ids=list(valid_intersect['copy_index'])
    if plot:
        fig, ax = plt.subplots(1, figsize=(10,10))
        geom_1.loc[final_zone_ids].plot(facecolor="none", 
            edgecolor='blue', ax=ax)
        geom_2.to_crs(geom_1.crs).plot(facecolor="none", 
            edgecolor='red', ax=ax)
    return final_zone_ids

def get_crs(gdf):
    avg_lng=gdf.unary_union.centroid.x
    utm_zone = int(np.floor((avg_lng + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return utm_crs

def overlay_gdf_properties_on_grid(grid_gdf, data_gdf, prop_column):
    data_over_grid=gpd.overlay(data_gdf, grid_gdf, 'intersection')
    data_over_grid['area']=data_over_grid.to_crs(get_crs(data_over_grid)).geometry.area
    largest_data_over_grid=data_over_grid.sort_values('area', ascending=False).drop_duplicates(subset=['id'])
    map_id_property={row['id']: row[prop_column] for ind, row in largest_data_over_grid.iterrows()}
    return map_id_property

def hex_to_rgba(value, alpha=255):
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]+[alpha]

standard_colors={'Residential Single': [[254,196,79,255], [254,227,145,255], [255,247,188,255]],
 'Residential Multi': [[205,133,63,255], [139,69,19,255], [128,0,0,255]],
 'Office': [[251,106,74,255], [222,45,38,255], [165,15,21, 255]],
 'Retail': [[253,208,162,255], [253,174,107,255], [253,141,60,255]],
 'Industrial': [[106,81,163,255], [84,39,143,255]],
 'Utility': [[189,189,189,255], [115,115,115,255], [37,37,37,255]],
 'Institutional': [[123,204,196,255], [78,179,211,255], [43,140,190,255], [8,104,172,255]],
 'Open Space': [[161,217,155,255], [0,109,44,255]], 
 'Public': [[201,148,199, 255], [231,41,138, 255]]}