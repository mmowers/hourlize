import datetime

test_mode = True

sc_path = '../runs 2020-02-27/pv_rural/outputs_sc.csv'
profile_path = '../runs 2020-02-27/pv_rural/outputs_gen_2012.h5'
class_path = 'resource_classes.csv'
timeslice_path = 'timeslices.csv'
filter_cols = {} #{'offshore':[0]} for onshore wind, {} if the full dataframe is used
profile_dset = 'cf_profile' #'rep_profiles_0' for wind, 'cf_profile' for pv
profile_id_col = 'gen_gids' #'sc_gid' for wind, 'gen_gids' for pv
profile_weight_col = 'gid_counts' #'capacity' for wind, 'gid_counts' for pv
out_prefix = 'upv' #'windons', 'windoff', 'upv', 'dupv'

#More consistent config
out_dir = 'output/' #for an added datetime string use 'output_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + '/'
test_filters = {'model_region':[1,100]}
reg_col = 'model_region'
bin_group_cols = ['region','class']
bin_col = 'trans_cap_cost'
bin_num = 5
bin_method = 'equal_capacity' #'kmeans', 'equal_capacity'
rep_profile_method = 'rmse' #'rmse','ave'
cfmean_type = 'rep' #'rep', 'ave'
to_local = True
to_1am = True
