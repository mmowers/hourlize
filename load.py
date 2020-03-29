import pandas as pd
import numpy as np
import sklearn.cluster as sc
import pdb
import datetime
import os
import sys
import shutil
import tables
import json
import config as cf
import logging

#Setup logger
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.info('Load Logger setup!')

def setup(this_dir_path, out_dir, timeslice_path, calibrate_path, ba_frac_path):
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #Create output directory, creating backup if one already exists.
    if os.path.exists(out_dir):
        os.rename(out_dir, os.path.dirname(out_dir) + '-archive-'+time)
    os.makedirs(out_dir)

    #Add output file for logger
    fh = logging.FileHandler(out_dir + 'log.txt', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #Copy inputs to outputs
    shutil.copy2(this_dir_path + 'load.py', out_dir)
    shutil.copy2(this_dir_path + 'config.py', out_dir)
    shutil.copy2(timeslice_path, out_dir)
    shutil.copy2(calibrate_path, out_dir)
    shutil.copy2(ba_frac_path, out_dir)

def calc_outputs(load_source):
      df_EI = pd.read_csv(load_source + 'EI_pca_load.csv', low_memory=False, index_col='time', parse_dates=True)
      df_WI = pd.read_csv(load_source + 'WI_pca_load.csv', low_memory=False, index_col='time', parse_dates=True)
      df_ERCOT = pd.read_csv(load_source + 'ERCOT_pca_load.csv', low_memory=False, index_col='time', parse_dates=True)
      df_hr_load = pd.concat([df_EI, df_WI, df_ERCOT], axis=1)
      return df_hr_load

def save_outputs(df_hr_load, out_dir):
    logger.info('Saving outputs...')
    startTime = datetime.datetime.now()
    df_hr_load.to_csv(out_dir + 'full_hourly_load_input.csv')
    logger.info('Done saving outputs: '+ str(datetime.datetime.now() - startTime))

if __name__== '__main__':
    startTime = datetime.datetime.now()
    this_dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    out_dir = this_dir_path + 'out/' + cf.out_dir
    setup(this_dir_path, out_dir, cf.timeslice_path, cf.calibrate_path, cf.ba_frac_path)
    df_hr_load = calc_outputs(cf.load_source)
    save_outputs(df_hr_load, out_dir)
    logger.info('All done! total time: '+ str(datetime.datetime.now() - startTime))
