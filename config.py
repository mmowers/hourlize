import datetime

#RESOURCE CONFIG
sc_path = '../runs 2020-02-27/pv_rural/outputs_sc.csv'
profile_path = '../runs 2020-02-27/pv_rural/outputs_gen_2012.h5'
class_path = 'resource_classes.csv'
filter_cols = {} #{'offshore':[0]} for onshore wind, {} if the full dataframe is used
profile_dset = 'cf_profile' #'rep_profiles_0' for wind, 'cf_profile' for pv
profile_id_col = 'gen_gids' #'sc_gid' for wind, 'gen_gids' for pv
profile_weight_col = 'gid_counts' #'capacity' for wind, 'gid_counts' for pv
out_prefix = 'upv' #'windons', 'windoff', 'upv', 'dupv'
#More consistent config
reg_col = 'model_region'
bin_group_cols = ['region','class']
bin_col = 'trans_cap_cost'
bin_num = 5
bin_method = 'equal_cap_cut' #'kmeans', 'equal_cap_man', 'equal_cap_cut'
rep_profile_method = 'rmse' #'rmse','ave'
cfmean_type = 'rep' #'rep', 'ave'
driver = 'H5FD_CORE' #'H5FD_CORE', None. H5FD_CORE will load the h5 into memory for better perforamnce, but None must be used for low-memory machines.
gather_method = 'smart' # 'list', 'slice', 'smart'. This setting will take a slice of profile ids from the min to max, rather than using a list of ids, for improved performance when ids are close together for each group.

#SHARED CONFIG
test_mode = True #This limits the regions considered to those listed below.
test_filters = {'model_region':[1,100]}
timeslice_path = 'timeslices.csv'
to_local = True
start_1am = True
out_dir = out_prefix + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + '/'
