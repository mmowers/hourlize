testmode = True
resource_class_path = 'onshore_wind_resource_classes.csv'
supply_curve_path = '/shared-projects/rev/projects/reeds_jan2020/wind/reeds_wind_sc.csv'
filter_cols = {'offshore':[0]} #set to None if the full dataframe is used
output_prefix = 'windons'

#More consistent config
testmode_reg = [1,10,100]
output_dir = 'output/'
bin_group_cols = ['model_region','class']
bin_param = 'trans_cap_cost'
bin_num = 5
bin_method = 'kmeans'

