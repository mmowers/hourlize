import datetime

test_mode = True
class_path = 'onshore_wind_resource_classes.csv'
sc_path = '../runs 2020-02-27/wind/reeds_wind_sc.csv'
filter_cols = {'offshore':[0]} #set to None if the full dataframe is used
profile_path = '../runs 2020-02-27/wind/reeds_wind_rep_profiles_2012.h5'
profile_dset = 'rep_profiles_0' #'rep_profiles_0', 'cf_profile'
profile_id_col = 'sc_gid'
profile_weight_col = 'capacity'
timeslice_path = 'timeslices.csv'
out_prefix = 'windons'

#More consistent config
out_dir = 'output_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + '/'
test_filters = {'model_region':[1,100]}
reg_col = 'model_region'
bin_group_cols = ['region','class']
bin_col = 'trans_cap_cost'
bin_num = 5
bin_method = 'equal_capacity' #'kmeans', 'equal_capacity'
rep_profile_method = 'rmse' #'rmse','ave'
to_local = True
to_1am = True
