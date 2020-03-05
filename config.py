test_mode = True
resource_class_path = 'onshore_wind_resource_classes.csv'
supply_curve_path = '/shared-projects/rev/projects/reeds_jan2020/wind/reeds_wind_sc.csv'
filter_cols = {'offshore':[0]} #set to None if the full dataframe is used
profile_h5_path = '/shared-projects/rev/projects/reeds_jan2020/wind/reeds_wind_rep_profiles_2012.h5'
profile_h5_dset = 'rep_profiles_0'
profile_id_col = 'sc_gid'
profile_weight_col = 'capacity'
timeslice_path = 'timeslices.csv'
output_prefix = 'windons'

#More consistent config
test_filters = {'model_region':[1,10,100]}
output_dir = 'output/'
bin_group_cols = ['model_region','class']
bin_param = 'trans_cap_cost'
bin_num = 5
bin_method = 'equal_capacity' #'kmeans', 'equal_capacity'

