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
out_dir = out_prefix + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + '/'

#LOAD CONFIG
#Note that calcs assume UTC, although the current load source is in eastern time (delete this after updating)
load_source = '//nrelqnap02/ReEDS/PLEXOS_ReEDS_Load/' #This is where the load inputs live
calibrate_path = 'load_inputs/EIA_2010loadbystate.csv' #Enter string to calibrate path or False to leave uncalibrated
ba_frac_path = 'load_inputs/load_participation_factors_st_to_ba.csv' #These are fractions of state load in each ba, unused if calibrate_path is False
hierarchy_path = 'load_inputs/hierarchy.csv' #These are fractions of state load in each ba, unused if calibrate_path is False
ba_timezone_path = 'ba_timezone.csv' #Should this be used for resource too, rather than site timezone?
select_year = 2012 #This is the year used for calibration and load outputs, although the profile outputs may still be multiyear (see multiyear)
multiyear = False #If True, the profile outputs will be multiyear, and if False, they will be single year using select_year

#SHARED CONFIG
test_mode = True #This limits the regions considered to those listed below.
test_filters = {'model_region':[1,100]}
timeslice_path = 'timeslices.csv'
to_local = True #False means keep the outputs in UTC. True means convert to local time of the respective region
truncate_leaps = True #Truncate leap years
