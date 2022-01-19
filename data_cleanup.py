# %% other imports

import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from pathlib import Path  # see https://docs.python.org/3/library/pathlib.html#basic-use

import collections

from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    Button,
    VBox,
    HBox,
    interactive,
    interactive_output,
)
import ipywidgets as widgets


import h5py

import math
import scipy

import pandas as pd

# pip install lmfit

from lmfit import Model
import warnings

# everything for deconvolution method

# Garbage Collector - use it like gc.collect() from https://stackoverflow.com/a/61193594
import gc

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

from scipy import fftpack

from scipy.optimize import curve_fit
from scipy.optimize import brenth
from scipy.optimize import minimize_scalar
import scipy.optimize as optimize
from scipy import ndimage

from IPython.display import display, clear_output

import os.path

# import pickle as pl

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


# %% Defining paths and loading files

"""# Mount drive and define paths"""


import_from_google_drive_in_colab = False
if import_from_google_drive_in_colab == True:
  # use data stored in own google drive location
  from google.colab import drive

  drive.mount('/content/gdrive', force_remount=True)
  data_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/')
  useful_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/useful/')
  bgsubtracted_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/bgsubtracted/')
  print(useful_dir)
  scratch_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/')
  #prebgsubtracted_dir
  #bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')

import_from_local_google_drive = True
if import_from_local_google_drive == True:
    data_dir = Path("g:/My Drive/PhD/coherence/data/")
    useful_dir = Path("g:/My Drive/PhD/coherence/data/useful/")
    bgsubtracted_dir = Path("g:/My Drive/PhD/coherence/data/bgsubtracted/")
    print(useful_dir)
    scratch_dir = Path("g:/My Drive/PhD/coherence/data/scratch_cc/")
    # prebgsubtracted_dir
    # bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')


"""# Dataset definitions for backgrounds and images"""

dataset_bg_args_py_file = str(Path.joinpath(data_dir, "dataset_bg_args.py"))
exec(open(dataset_bg_args_py_file).read())
dataset_image_args_py_file = str(Path.joinpath(data_dir, "dataset_image_args.py"))
exec(open(dataset_image_args_py_file).read())

"""# Load dph settings and combinations"""

dph_settings_py_file = str(Path.joinpath(data_dir, "dph_settings.py"))

# Commented out IPython magic to ensure Python compatibility.
# %run -i $dph_settings_py # see https://stackoverflow.com/a/14411126 and http://ipython.org/ipython-doc/dev/interactive/magics.html#magic-run
# see also https://stackoverflow.com/questions/4383571/importing-files-from-different-folder to import as a module,
# requires however that it is located in a folder with an empty __init__.py
exec(open(dph_settings_py_file).read())

# import sys
# sys.path.append('g:\\My Drive\\PhD\\coherence\data\\dph_settings_package\\')
# from dph_settings_package import dph_settings_module


dph_settings_widget_layout = widgets.Layout(width="100%")
dph_settings_widget = widgets.Dropdown(options=dph_settings, layout=dph_settings_widget_layout)
# settings_widget.observe(update_settings, names='value')
# display(dph_settings_widget)

dph_settings_bgsubtracted = list(bgsubtracted_dir.glob("*.h5"))

dph_settings_bgsubtracted_widget_layout = widgets.Layout(width="100%")
dph_settings_bgsubtracted_widget = widgets.Dropdown(
    options=dph_settings_bgsubtracted,
    layout=dph_settings_bgsubtracted_widget_layout,
    value=dph_settings_bgsubtracted[3],  # workaround, because some hdf5 files have no proper timestamp yet
)
# settings_widget.observe(update_settings, names='value')

# just hdf5_filename_bg_subtracted so we can use it to search in the dataframe
# dph_settings_bgsubtracted_widget.value.name

# how to get the hdf5_filename ?

# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
#     hdf5_file_useful_name = hdf5_file["/hdf5_file_useful_name"][0]
#     print(hdf5_file_useful_name)


"""# Load dataframes from csv"""


load_df_all = True # this takes around 5 min! ?
# load_df_all = False # this takes about 15 min!

if load_df_all == False: 

    # Create DataFrame of all available datasets (dataset_args or dataset_image_args)

    names_df_image_args = ['begin', 'end', 'pinholes', 'threshold', 'checked', 'aperture5_mm', 'aperture7_mm', 'filter']
    df_image_args = pd.DataFrame(columns=names_df_image_args)
    for key in dataset_image_args.keys():
        df_image_args_key = pd.DataFrame.from_records(dataset_image_args[key],columns=names_df_image_args, index=[key]*len(dataset_image_args[key]))
        df_image_args = df_image_args.append(df_image_args_key)
    df_image_args.index.name = 'hdf5_file_name'
    df_image_args.reset_index(inplace=True)

    # add separation_um and orientation based on pinhole"
    def get_sep_and_orient(pinholes):
        pinholes = pinholes[0:2]
        choices = {'1a': (50, 'vertical'), '1b': (707, 'vertical'),'1c': (50, 'horizontal'), '1d': (707, 'horizontal'),
                   '2a': (107, 'vertical'), '2b': (890, 'vertical'), '2c': (107, 'horizontal'), '2d': (890, 'horizontal'),
                   '3a': (215, 'vertical'), '3b': (1047, 'vertical'), '3c': (215, 'horizontal'), '3d': (1047, 'horizontal'),
                   '4a': (322, 'vertical'), '4b': (1335, 'vertical'), '4c': (322, 'horizontal'), '4d': (1335, 'horizontal'),
                   '5a': (445, 'vertical'), '5b': (1570, 'vertical'), '5c': (445, 'horizontal'), '5d': (1570, 'horizontal')}
        (sep, orient) = choices.get(pinholes,(np.nan,'bg'))
        return sep, orient

    df_image_args['separation_um'] = df_image_args['pinholes'].apply(lambda x: get_sep_and_orient(x)[0])
    df_image_args['orientation'] = df_image_args['pinholes'].apply(lambda x: get_sep_and_orient(x)[1])

    # Cast from object to int64, otherwise we cannot merge later
    df_image_args['begin']=df_image_args['begin'].astype('int64')
    df_image_args['end']=df_image_args['end'].astype('int64')

    # save to csv
    df_image_args.to_csv(scratch_dir+'image_args.csv')
    #df_image_args = pd.read_csv('/home/wodzinsk/PycharmProjects/coherence/'+'image_args.csv', index_col=0)

    # Create DataFrames for each dataset for each shot

    names_df_dataset_timestamp = ['timestamp_t_sec','timestamp_t_microsec','timestamp_pulse_id']
    names_df_dataset_daq_parameter=['mean wavelength',
                    'mean photon energy',
                    'SASE03 gap',
                    'SASE04 gap',
                    'SASE05 gap',
                    'SASE06 gap',
                    'SASE07 gap',
                    'SASE08 gap',
                    'SASE09 gap',
                    'SASE10 gap',
                    'SASE11 gap',
                    'SASE12 gap',
                    'SASE13 gap',
                    'SASE14 gap',
                    'set wavelength',
                    'average energy hall',
                    'average energy tunnel',
                    'beam position hall horizontal average',
                    'beam position hall vertical average',
                    'beam position tunnel horizontal average',
                    'beam position tunnel vertical average',
                    'beam position hall horizontal pulse resolved',
                    'beam position hall vertical pulse resolved',
                    'beam position tunnel horizontal pulse resolved',
                    'beam position tunnel vertical pulse resolved',
                    'energy aux hall',
                    'energy aux tunnel',
                    'energy hall',
                    'energy tunnel',
                    'GMD position horizontal hall',
                    'GMD position horizontal tunnel',
                    'GMD position vertical hall',
                    'GMD position vertical tunnel'
                    ]

    names_df_all = list(df_image_args.columns) + names_df_dataset_timestamp + names_df_dataset_daq_parameter 
    df_all = pd.DataFrame(columns = names_df_all)


    from datetime import datetime

    from pathlib import Path

    root_directory = Path(useful_dir)
    size_total_GB = sum(f.stat().st_size for f in root_directory.glob('*useful.h5') if f.is_file())/(1024*1024*1024)

    #count first the files
    n_files = 0
    i = 0
    n_filenames = len(list(df_image_args.hdf5_file_name.unique()))
    for filename in list(df_image_args.hdf5_file_name.unique()):
      j = 0
      for dataset_begin in list(df_image_args[df_image_args.hdf5_file_name == filename]['begin']):
        n_files = n_files+1

    i = 0
    k = 0
    time_taken = 0
    time_left = (n_files - k) * time_taken
    size_sum_GB = 0

    
    n_filenames = len(list(df_image_args.hdf5_file_name.unique()))
    start = datetime.now()
    for filename in list(df_image_args.hdf5_file_name.unique()):

      j = 0
      for dataset_begin in list(df_image_args[df_image_args.hdf5_file_name == filename]['begin']):
      #dataset_begin_index = 0
        
        
        
        

        df_selection = df_image_args[(df_image_args.hdf5_file_name==filename) & (df_image_args.begin==dataset_begin)]
        dataset_args = (df_selection.begin.item(),
                        df_selection.end.item(),
                        df_selection.pinholes.item(),
                        df_selection.threshold.item(),
                        df_selection.checked.item(),
                        df_selection.aperture5_mm.item(),
                        df_selection.aperture7_mm.item(),
                        df_selection['filter'].item())
        size_GB = os.path.getsize(useful_dir + filename + '_' + str(df_selection.begin.item())+ 'to' + str(df_selection.end.item()) + '_useful.h5')/(1024*1024*1024)
        size_sum_GB = size_sum_GB + size_GB
        size_left_GB = size_total_GB - size_sum_GB

        
        (pixis_dataset, pixis_avg, pixis_std, pinholes_dataset, pinholes_avg, timestamp_dataset, sep, orient, daq_parameter_dataset, aperture5_mm, aperture7_mm, filter_used) = get_images(filename,dataset_args)
        
        #pinholes_time_arr = timestamp_dataset[pinholes_event_id]
        #pinholes_t_sec = pinholes_time_arr[0]
        #pinholes_t_microsec = pinholes_time_arr[1]
        #pinholes_pulse_id = pinholes_time_arr[2]

        df_dataset_timestamp = pd.DataFrame(timestamp_dataset, columns=names_df_dataset_timestamp)

        df_dataset_daq_parameter = pd.DataFrame(dict(zip(names_df_dataset_daq_parameter,np.squeeze(daq_parameter_dataset))), columns=names_df_dataset_daq_parameter)

        df_dataset = pd.concat([df_dataset_timestamp, df_dataset_daq_parameter], axis=1)

        df_dataset['hdf5_file_name'] = filename
        df_dataset['begin']=dataset_begin

        df_dataset_merged = pd.merge(df_image_args,df_dataset)

        df_all = df_all.append(df_dataset_merged)

        j = j + 1
        k = k + 1
        end = datetime.now()
        time_taken = end - start 
        print('time taken to process file: ' + str(time_taken))
        rate =  time_taken/size_sum_GB  # size needs to be in MB, otherwise the numbers get too small
        print('rate:' + str(rate) + 'per GB')
        print('size left / GB: ' + str(size_left_GB))
        time_left = size_left_GB * rate
        print('progress=' + str(round(size_sum_GB/size_total_GB*100,1)) + '% ' + 'time left=' + str(time_left) +  ' file ' + str(i+1) + ' of ' + str(n_filenames) + ' subfile ' + str(j+1) + ' of ' + str(len(df_image_args[df_image_args.hdf5_file_name == filename])) +  ' ' + filename + '_' + str(df_selection.begin.item())+ 'to' + str(df_selection.end.item()) + '_useful.h5' )

      i = i + 1


    df_all.to_csv(scratch_dir+'df_all.csv')

else:
    # dataframe of extracted from all available useful hdf5 files
    df_all = pd.read_csv(Path.joinpath(scratch_dir, "df_all.csv"), index_col=0)
# maybe rename to df_hdf5_files? and then use df instead of df0?
df_all["imageid"] = df_all.index

# dataframe based on the dph_settings dictionary inside dph_settings.py

# del df_settings

hdf5_file_name = []
hdf5_file_name_background = []
setting_wavelength_nm = []
setting_energy_uJ = []
setting_undulators = []
KAOS = []
separation_um = []
pinholes = []
background = []

for idx in range(len(dph_settings.keys())):
    hdf5_file_name.append(dph_settings[list(dph_settings.keys())[idx]][2])
    hdf5_file_name_background.append(dph_settings[list(dph_settings.keys())[idx]][0])
    setting_wavelength_nm.append(float(list(dph_settings.keys())[idx].split()[1][:-2]))
    setting_energy_uJ.append(int(list(dph_settings.keys())[idx].split()[2][:-2]))
    setting_undulators.append(int(list(dph_settings.keys())[idx].split()[3][:-4]))
    KAOS.append(list(dph_settings.keys())[idx].split()[4][5:])
    separation_um.append(int(list(dph_settings.keys())[idx].split()[5][:-2]))
    pinholes.append((dph_settings[list(dph_settings.keys())[idx]][3][2]))
    background.append((dph_settings[list(dph_settings.keys())[idx]][1][2]))

df_settings = pd.DataFrame(
    {
        "hdf5_file_name": hdf5_file_name,
        "hdf5_file_name_background": hdf5_file_name_background,
        "setting_wavelength_nm": setting_wavelength_nm,
        "setting_energy_uJ": setting_energy_uJ,
        "setting_undulators": setting_undulators,
        "KAOS": KAOS,
        "separation_um": separation_um,
        "pinholes": pinholes,
        "background": background,
    }
)
# df_settings

# merge dataframe of hdf5files with dataframe of settings
df0 = []
df0 = pd.merge(df_all, df_settings)
df0["timestamp_pulse_id"] = df0["timestamp_pulse_id"].astype("int64")
# store this instead of df_all?

# definition of fits header columns
# needed in case we want to add new columns?
fits_header_list1 = [
    "bgfactor",
    "pixis_rotation",
    "pixis_centerx_px",
    "pixis_centery_px",
    "pinholes_centerx_px",
    "pinholes_centery_px",
    "pixis_profile_centerx_px_fit",
    "pixis_profile_centery_px_fit",
    "pinholes_cm_x_px",
    "pinholes_cm_y_px",
    "wavelength_nm_fit",
    "gamma_fit",
    "sigma_F_gamma_um_opt",
    "xi_um",
]
fits_header_list2 = [
    "shiftx_um_fit",
    "w1_um_fit",
    "w2_um_fit",
    "I_Airy1_fit",
    "I_Airy2_fit",
    "x1_um_fit",
    "x2_um_fit",
    "d_um_at_detector",
    "xi_x_um",
    "xi_y_um",
]
fits_header_list3 = [
    "pixis_image_minus_bg_rot_cropped_counts",
    "phcam_scalex_um_per_px",
    "phcam_scaley_um_per_px",
    "phap_diam_um",
    "phap_xc_px",
    "phap_yc_px",
    "phap_width_px",
    "phap_height_px",
    "pinholes_bg_avg_phi",
    "pinholes_bg_avg_xc_um",
    "pinholes_bg_avg_yc_um",
    "pinholes_bg_avg_sx_um",
    "pinholes_bg_avg_sy_um",
    "xi_x_um_fit",
    "zeta_x",
    "zeta_x_fit",
]
fits_header_list4 = ["xi_y_um_fit", "zeta_y", "zeta_y_fit"]
fits_header_list = fits_header_list1 + fits_header_list2 + fits_header_list3


# fits_header_list1 already exists in saved csv, only adding fits_header_list2, only initiate when
initiate_df_fits = True
# if initiate_df_fits == True:
# df0 = df0.reindex(columns = df0.columns.tolist() + fits_header_list2)
# df_fits = df0[['timestamp_pulse_id'] + fits_header_list]

# load saved df_fits from csv
load_df_fits_csv = True
if load_df_fits_csv == True:
    df_fits = pd.read_csv(Path.joinpath(scratch_dir, "df_fits_v2.csv"), index_col=0)
    df_fits_clean = df_fits[df_fits["pixis_rotation"].notna()].drop_duplicates()
    df_fits = df_fits_clean


df0 = pd.merge(df0, df_fits, on="timestamp_pulse_id", how="outer")







# %% 
def get_images(hdf5_file_name, dataset_args):
    
    begin = dataset_args[0]
    end = dataset_args[1]
    ph = dataset_args[2]
    thr = dataset_args[3]
    chk = dataset_args[4]
    
    aperture5_mm = dataset_args[5]
    aperture7_mm = dataset_args[6]
    filter_used = dataset_args[7]
    
    slice_beginning = begin
    slice_ending = end 
    pinholes = ph[0:2]
    
    choices = {'1a': (50, 'vertical'), '1b': (707, 'vertical'),'1c': (50, 'horizontal'), '1d': (707, 'horizontal'),
               '2a': (107, 'vertical'), '2b': (890, 'vertical'), '2c': (107, 'horizontal'), '2d': (890, 'horizontal'),
               '3a': (215, 'vertical'), '3b': (1047, 'vertical'), '3c': (215, 'horizontal'), '3d': (1047, 'horizontal'),
               '4a': (322, 'vertical'), '4b': (1335, 'vertical'), '4c': (322, 'horizontal'), '4d': (1335, 'horizontal'),
               '5a': (445, 'vertical'), '5b': (1570, 'vertical'), '5c': (445, 'horizontal'), '5d': (1570, 'horizontal')}
    (sep, orient) = choices.get(pinholes,(np.nan,'bg'))

    useful_hdf5_file_name = hdf5_file_name + '_' + str(slice_beginning) + 'to' + str(slice_ending) + '_useful.h5'

    if os.path.exists(useful_dir + useful_hdf5_file_name):
      #print('file' + useful_dir + useful_hdf5_file_name + 'does exist')
      (pixis_dataset0, pinholes_dataset0, timestamp_dataset, daq_parameter_dataset) = get_datasets(useful_dir, useful_hdf5_file_name,None,None)
    else:
      print('file' + useful_dir + useful_hdf5_file_name + 'does not exist')

    pixis_dataset = np.rot90(pixis_dataset0,axes=(1,2))
    pixis_avg = np.average(pixis_dataset, axis=0)
    pixis_std = np.std(pixis_dataset, axis=0)
    pixis_std = ndimage.gaussian_filter(pixis_std,sigma=5,order=0)
    
    pinholes_dataset = np.rot90(pinholes_dataset0,axes=(1,2))
    pinholes_avg = np.average(pinholes_dataset, axis=0)
    
    return (pixis_dataset, pixis_avg, pixis_std, pinholes_dataset, pinholes_avg, timestamp_dataset, sep, orient, daq_parameter_dataset, aperture5_mm, aperture7_mm, filter_used)


def normalize(inputarray):
    normalized_array = inputarray / np.max(inputarray)
    return(normalized_array)

def get_imageids_with_bgs(beamposition_horizontal_interval):

    imageid_sequence = []
    for imageid in df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('beam position hall horizontal pulse resolved')['imageid']:
        beamposition_horizontal_image = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])]['beam position hall horizontal pulse resolved'][imageid]
        matching_bg_indices = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_bg_widget.value) & (df_all['pinholes'] == dataset_bg_args_widget.value[2]) & (df_all['beam position hall horizontal pulse resolved'] > beamposition_horizontal_image - beamposition_horizontal_interval/2 ) & (df_all['beam position hall horizontal pulse resolved'] < beamposition_horizontal_image + beamposition_horizontal_interval/2 ) ]['beam position hall horizontal pulse resolved'].index
        if matching_bg_indices.empty == False:
            imageid_sequence.append(imageid)
    
    return imageid_sequence


def get_pixis_profiles(imageid,
                       use_pixis_avg,
                       bgfactor,
                       avg_width,
                       pixis_rotation,
                       pixis_center_autofind,
                       pixis_centerx_px,
                       pixis_centery_px,
                      crop_px):

    # using global datasets ... because of the way interactive works ... 'fixed(dataset)'
    # make a wrapper for this function, where datasets are global...s
    global pixis_dataset
    #gloabl pixis_bg_std
    global pixis_bg_avg
    global pixis_image_norm
    global pinholes_bg_avg
#     global matching_bg_indices

    # Choosing image
    #imageid = 1
    
    #imageid = 4
    
    
   
    pixis_centerx_px = int(pixis_centerx_px)
    pixis_centery_px = int(pixis_centery_px)


    if use_pixis_avg == True:
        imageid = -1
    
    if imageid == -1:
        pixis_image = pixis_avg
    else:
        pixis_image = pixis_dataset[imageid]
        beamposition_horizontal_interval = beamposition_horizontal_interval_widget.value
        beamposition_horizontal_image = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])]['beam position hall horizontal pulse resolved'][imageid]
        matching_bg_indices = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_bg_widget.value) & (df_all['pinholes'] == dataset_bg_args_widget.value[2]) & (df_all['beam position hall horizontal pulse resolved'] > beamposition_horizontal_image - beamposition_horizontal_interval/2 ) & (df_all['beam position hall horizontal pulse resolved'] < beamposition_horizontal_image + beamposition_horizontal_interval/2 ) ]['beam position hall horizontal pulse resolved'].index
        
        #beamposition_horizontal_interval = beamposition_horizontal_interval_widget.value
        #beamposition_horizontal_image = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])]['beam position hall horizontal pulse resolved'][imageid]
        #matching_bg_indices = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_bg_widget.value) & (df_all['pinholes'] == dataset_bg_args_widget.value[2]) & (df_all['beam position hall horizontal pulse resolved'] > beamposition_horizontal_image - beamposition_horizontal_interval/2 ) & (df_all['beam position hall horizontal pulse resolved'] < beamposition_horizontal_image + beamposition_horizontal_interval/2 ) ]['beam position hall horizontal pulse resolved'].index
        if matching_bg_indices.empty:
            print('no matching bg for that energy!!')
            pixis_bg_avg = pixis_image
        else:
            pixis_bg_avg = np.mean(pixis_bg_dataset[matching_bg_indices], axis=0)
            pinholes_bg_avg = np.mean(pinholes_bg_dataset[matching_bg_indices], axis=0)
            if df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)]['orientation'].iloc[0] == 'vertical':
                pinholes_bg_avg=ndimage.rotate(pinholes_bg_avg, 90)
            pinholes_bg_avg = np.fliplr(pinholes_bg_avg) # fliplr to match the microscope pictures
            energy_bg_mean = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_bg_widget.value) & (df_all['pinholes'] == dataset_bg_args_widget.value[2])]['energy hall'][matching_bg_indices].mean(axis=0)
            energy_bg_std = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_bg_widget.value) & (df_all['pinholes'] == dataset_bg_args_widget.value[2])]['energy hall'][matching_bg_indices].std(axis=0)
        
    pixis_image_minus_bg = np.subtract(pixis_image,bgfactor*pixis_bg_avg)
    if orient == 'vertical':
        pixis_image_minus_bg = ndimage.rotate(pixis_image_minus_bg, 90)

   # correct rotation and normalize, show where maximum is
    #pixis_rotation=-1
    crop = int(pixis_image.shape[0]*np.abs(np.sin(np.pi/180*pixis_rotation)))
    pixis_image_minus_bg_rot = ndimage.rotate(pixis_image_minus_bg, pixis_rotation, reshape=True)
    pixis_image_minus_bg_rot_cropped = pixis_image_minus_bg_rot[crop:-crop,crop+20:-crop]
    pixis_image_minus_bg_rot_cropped_counts = np.sum(pixis_image_minus_bg_rot_cropped)
    
    pixis_image_norm = pixis_image_minus_bg_rot_cropped
    pixis_image_norm = pixis_image_norm[0:min(pixis_image_norm.shape),0:min(pixis_image_norm.shape)]
    pixis_image_norm = normalize(pixis_image_norm)
    
    #np.where(pixis_image_norm < 0)
      
    if crop_px > 0:
        pixis_image_norm = pixis_image_norm[crop_px:-crop_px,crop_px:-crop_px]
        pixis_image_norm = normalize(pixis_image_norm)
    
    set_negative_to_zero = True
    if set_negative_to_zero == True:
        pixis_image_norm[np.where(pixis_image_norm < 0)] = 0
    
    pixis_bg_std_rot = ndimage.rotate(pixis_bg_std, pixis_rotation, reshape=True)
    pixis_bg_std_rot_cropped = pixis_bg_std_rot[crop:-crop,crop+20:-crop]
    pixis_bg_std_norm = pixis_bg_std_rot_cropped / np.max(pixis_image_minus_bg_rot_cropped)
    pixis_bg_std_norm = pixis_bg_std_norm[0:min(pixis_bg_std_norm.shape),0:min(pixis_bg_std_norm.shape)]
    
    if crop_px > 0:
        pixis_bg_std_norm = pixis_bg_std_norm[crop_px:-crop_px,crop_px:-crop_px]

    pixis_cts = np.sum(pixis_image_minus_bg_rot_cropped)

    if pixis_center_autofind == True:
        pixis_centerx_px = np.where(pixis_image_norm==1)[1][0]
        pixis_centery_px = np.where(pixis_image_norm==1)[0][0]
    else:
        pixis_centerx_px = int(pixis_centerx_px)
        pixis_centery_px = int(pixis_centery_px)
        
        

    pixis_profile = pixis_image_norm[ int(pixis_centery_px),:] # lineout at pixis_centery_px
    pixis_profile_avg = pixis_image_norm[ int(pixis_centery_px)-int(avg_width/2): int(pixis_centery_px)+int(avg_width/2),:]
    pixis_profile_avg = np.average(pixis_profile_avg,axis=0)
    pixis_profile_avg = pixis_profile_avg / np.max(pixis_profile_avg)  # why was this commented out?
    
    pixis_profile_avg_centerx_px = np.where(pixis_profile_avg==1)[0]
    
    pixis_profile_alongy = pixis_image_norm[ :, int(pixis_centerx_px)] # lineout at pixis_centerx_px along y
    
    n = pixis_profile_avg.size # number of sampling point  # number of pixels
    dx = 13*1e-6 # sampling # pixel size
    xdata = list(range(n))
    ydata = pixis_profile_alongy
    
    pixis_yshift_px = pixis_centery_px
    p0 = (pixis_yshift_px, 400)
    popt_gauss, pcov_gaussian = curve_fit(lambda x, m, w: gaussianbeam(x, 1, m ,w, 0), xdata, ydata, p0)
    pixis_beamwidth_px = popt_gauss[1]  # this is 2 sigma!
    
    if pixis_center_autofind == True:
#         pixis_centerx_px = np.where(pixis_image_norm==1)[1][0]
#         pixis_centery_px = np.where(pixis_image_norm==1)[0][0]
        pixis_centery_px = popt_gauss[0]
        
    else:
        pixis_centerx_px = pixis_centerx_px
        pixis_centery_px = pixis_centery_px
    
    pixis_profile = pixis_image_norm[ int(pixis_centery_px),:] # lineout at pixis_centery_px
    pixis_profile_avg = pixis_image_norm[ int(pixis_centery_px)-int(avg_width/2): int(pixis_centery_px)+int(avg_width/2),:]
    pixis_profile_avg = np.average(pixis_profile_avg,axis=0)
    pixis_profile_avg = pixis_profile_avg / np.max(pixis_profile_avg)  # why was this commented out?

    
    
    pixis_bg_std_avg = pixis_bg_std_norm[ int(pixis_centery_px)-int(avg_width/2): int(pixis_centery_px)+int(avg_width/2),:]
    pixis_bg_std_avg = np.average(pixis_bg_std_avg,axis=0)
    pixis_bg_std_avg = pixis_bg_std_avg / np.max(pixis_profile_avg)  # why was this commented out?

    (pixis_cm_y_px, pixis_cm_x_px) = ndimage.measurements.center_of_mass(pixis_image_norm)

    return pixis_image_norm, pixis_bg_std_norm, pixis_bg_std_avg, pixis_profile, pixis_profile_avg, pixis_profile_avg_centerx_px, pixis_centerx_px, pixis_centery_px, pixis_profile_alongy, pixis_cts, pixis_cm_x_px, pixis_cm_y_px, pinholes_bg_avg, pixis_image_minus_bg_rot_cropped_counts

def get_pinholes_profiles(imageid,bgfactor,pinholes_rotation,centerx_px,centery_px,avg_width_alongy,avg_width_alongx):

    # using global datasets ... because of the way interactive works ... 'fixed(dataset)'
    # make a wrapper for this function, where datasets are global...s
    #global pixis_dataset
    #global pixis_bg_avg

    # Choosing image
    if imageid == -1:
        pinholes_image = np.average(pinholes_dataset, axis=0)
    else:
        pinholes_image = pinholes_dataset[imageid]
    pinholes_image_minus_bg = pinholes_image
    if orient == 'vertical':
        pinholes_image_minus_bg = ndimage.rotate(pinholes_image_minus_bg, 90)
        
    

   # correct rotation and normalize, show where maximum is

    pinholes_image_minus_bg_rot = ndimage.rotate(pinholes_image_minus_bg, pinholes_rotation)
    pinholes_image_norm = pinholes_image_minus_bg_rot / np.max(pinholes_image_minus_bg_rot)
    
    pinholes_image_norm = np.fliplr(pinholes_image_norm)

    pinholes_cts = np.sum(pinholes_image_minus_bg_rot)

#     pixis_centerx = np.where(pinholes_image_norm==1)[1][0]
#     pixis_centery = np.where(pinholes_image_norm==1)[0][0]

    pinholes_profile_alongx = pinholes_image_norm[centery_px,:]
    pinholes_profile_alongx_avg = pinholes_image_norm[centery_px-int(avg_width_alongy/2):centery_px+int(avg_width_alongy/2),:]
    pinholes_profile_alongx_avg = np.average(pinholes_profile_alongx_avg,axis=0)
    pinholes_profile_alongx_avg = pinholes_profile_alongx_avg / np.max(pinholes_profile_alongx_avg)

    pinholes_profile_alongy = pinholes_image_norm[centerx_px,:]
    pinholes_profile_alongy_avg = pinholes_image_norm[centerx_px-int(avg_width_alongx/2):centerx_px+int(avg_width_alongx/2),:]
    pinholes_profile_alongy_avg = np.average(pinholes_profile_alongy_avg,axis=0)
    pinholes_profile_alongy_avg = pinholes_profile_alongy_avg / np.max(pinholes_profile_alongy_avg)

    (pinholes_cm_y_px, pinholes_cm_x_px) = ndimage.measurements.center_of_mass(pinholes_image_norm)


    #shiftx_px = pixis_centerx - pixis_profile_avg.size/2
    #shiftx_um = shiftx_px * 13
    #widget_shiftx_um.value = shiftx_um

    return pinholes_image_norm, pinholes_profile_alongx, pinholes_profile_alongx_avg, pinholes_profile_alongy, pinholes_profile_alongx_avg, pinholes_cts, pinholes_cm_x_px, pinholes_cm_y_px





aperture5_mm_bg_widget = widgets.FloatText(disabled=True, description='aperture5_mm_bg')
aperture7_mm_bg_widget = widgets.FloatText(disabled=True, description='aperture7_mm_bg')
aperture5_mm_image_widget = widgets.FloatText(disabled=True, description='aperture5_mm_image')
aperture7_mm_image_widget = widgets.FloatText(disabled=True, description='aperture7_mm_image')
filter_bg_widget = widgets.Text(disabled=True, description='filter_bg')
filter_image_widget = widgets.Text(disabled=True, description='filter_image')


def select_dataset_bg_args(hdf5_file_name):
    dataset_bg_args_widget.options = dataset_bg_args[hdf5_file_name]

layout = widgets.Layout(width='80%')
hdf5_file_name_bg_widget = widgets.Dropdown(options=dataset_bg_args.keys(),
                                            layout=layout)
dataset_bg_args_widget = widgets.Dropdown(options=dataset_bg_args[hdf5_file_name_bg_widget.value],
                                         layout=layout)



select_dataset_bg_args_interactive = interactive(select_dataset_bg_args,
                                                    hdf5_file_name=hdf5_file_name_bg_widget)

get_bg_interactive = interactive(get_images,
                                    hdf5_file_name=hdf5_file_name_bg_widget,
                                    dataset_args=dataset_bg_args_widget)



#display(select_dataset_bg_args_interactive)
display(get_bg_interactive)

(pixis_bg_dataset, pixis_bg_avg, pixis_bg_std, pinholes_bg_dataset, pinholes_bg_avg, timestamp_bg_dataset, sep_bg, orient_bg, daq_parameter_bg_dataset, aperture5_mm_bg_widget.value, aperture7_mm_bg_widget.value, filter_bg_widget.value) = get_bg_interactive.result

def select_first_bg_args_on_hdf5_change(change):
    dataset_bg_args_widget.value = dataset_bg_args[hdf5_file_name_bg_widget.value][0]

hdf5_file_name_bg_widget.observe(select_first_bg_args_on_hdf5_change, names='value')

def bind_pixis_bg_result_to_global_variables(change):
    global pixis_bg_dataset
    global pixis_bg_avg
    global pixis_bg_std
    global pinholes_bg_dataset
    global pinholes_bg_avg
    global timestamp_bg_dataset
    global sep_bg
    global orient_bg
    global daq_parameter_bg_dataset
    
    #plot_data_and_simulation_interactive_output.clear_output()
    get_bg_interactive.update()

    (pixis_bg_dataset, pixis_bg_avg, pixis_bg_std, pinholes_bg_dataset, pinholes_bg_avg, timestamp_bg_dataset, sep_bg, orient_bg, daq_parameter_bg_dataset, aperture5_mm_bg_widget.value, aperture7_mm_bg_widget.value, filter_bg_widget.value) = get_bg_interactive.result
    #Int_LightPipes, Intlinex_LightPipes, Intliney_LightPipes = run_doublepinholes_LightPipes_interactive.result

    imageid_widget.options = get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)
        
    #imageid_widget.value = get_imageids_with_bgs(energy_interval_widget.value)[0]
    #number_of_images_widget.value = len(df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].index)
    number_of_images_widget.value = len(get_imageids_with_bgs(beamposition_horizontal_interval_widget.value))
#     imageid_widget.options = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid'].tolist()
#     imageid_widget.value = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid'].tolist()[0]
    #plot_data_and_simulation_interactive.update()
    #plot_data_and_simulation_interactive_output
    #plot_data_and_simulation_interactive_input.update()

dataset_bg_args_widget.observe(bind_pixis_bg_result_to_global_variables, names='value')



# getting the image dataset

def select_dataset_image_args(hdf5_file_name):
    dataset_image_args_widget.options = dataset_image_args[hdf5_file_name]

hdf5_file_name_image_widget = widgets.Dropdown(options=dataset_image_args.keys(),
                                              layout=layout)
dataset_image_args_widget = widgets.Dropdown(options=dataset_image_args[hdf5_file_name_image_widget.value],
                                            layout=layout)


get_images_interactive = interactive(get_images,
                                    hdf5_file_name=hdf5_file_name_image_widget,
                                    dataset_args=dataset_image_args_widget)

select_dataset_image_args_interactive = interactive(select_dataset_image_args,
                                                    hdf5_file_name=hdf5_file_name_image_widget)

#display(select_dataset_image_args_interactive)
display(get_images_interactive)
#get_image_interactive.update()

(pixis_dataset, pixis_avg, pixis_std, pinholes_dataset, pinholes_avg, timestamp_dataset, sep, orient, daq_parameter_image_dataset, aperture5_mm_image_widget.value, aperture7_mm_image_widget.value, filter_image_widget.value) = get_images_interactive.result

def select_first_image_args_on_hdf5_change(change):
    dataset_image_args_widget.value = dataset_image_args[hdf5_file_name_image_widget.value][0]

hdf5_file_name_image_widget.observe(select_first_image_args_on_hdf5_change, names='value')


loadsettings_widget = widgets.IntProgress(
    value=0,
    min=0,
    max=10,
    step=1,
    description='Loading settings:',
    bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
    orientation='horizontal'
)

# def get_imageids_with_bgs(energy_interval):

#     imageid_sequence = []
#     for imageid in df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid']:
#         energy_image = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])]['energy hall'][imageid]
#         matching_bg_indices = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_bg_widget.value) & (df_all['pinholes'] == dataset_bg_args_widget.value[2]) & (df_all['energy hall'] > energy_image - energy_interval/2 ) & (df_all['energy hall'] < energy_image + energy_interval/2 ) ]['energy hall'].index
#         if matching_bg_indices.empty == False:
#             imageid_sequence.append(imageid)
        
#     return imageid_sequence



def bind_pixis_result_to_global_variables(change):
    global pixis_dataset
    global pixis_avg
    global pixis_std
    global pinholes_dataset
    global pinholes_avg
    global timestamp_dataset
    global sep
    global orient
    global daq_parameter_image_dataset
    
    loadsettings_widget.bar_style='info'
    loadsettings_widget.value=5
    
    #plot_data_and_simulation_interactive_output.clear_output()
    get_images_interactive.update()
    
    

    (pixis_dataset, pixis_avg, pixis_std, pinholes_dataset, pinholes_avg, timestamp_dataset, sep, orient, daq_parameter_image_dataset, aperture5_mm_image_widget.value, aperture7_mm_image_widget.value, filter_image_widget.value) = get_images_interactive.result
    #imageid_widget.options=df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid'].tolist()
    
    #imageid_widget.value=df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid'].tolist()[0]
    #imageid_widget.value = -1
    #imageid_widget.max = len(pixis_dataset)-1
    
    imageid_widget.options = get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)
    
    imageid_widget.value = get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)[0]
    #number_of_images_widget.value = len(df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].index)
    number_of_images_widget.value = len(get_imageids_with_bgs(beamposition_horizontal_interval_widget.value))
    d_um_widget.value = sep
    if daq_parameter_image_dataset[0][imageid_widget.value] > 0:
        _lambda_widget.value=daq_parameter_image_dataset[0][imageid_widget.value]
    else:
        _lambda_widget.value=daq_parameter_image_dataset[14][imageid_widget.value]

    if np.isfinite(df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_rotation'].iloc[0]) == True:
        pixis_rotation_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_rotation'].iloc[0]
     
    if np.isfinite(df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_centerx_px'].iloc[0]) == True:
        pixis_centerx_px_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_centerx_px'].iloc[0]
    
    if np.isfinite(df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_centery_px'].iloc[0]) == True:
        pixis_centery_px_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_centery_px'].iloc[0]
                                                                                                           
    if np.isfinite(df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pinholes_centerx_px'].iloc[0]) == True:
        pinholes_centerx_px_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pinholes_centerx_px'].iloc[0]
    
    if np.isfinite(df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pinholes_centery_px'].iloc[0]) == True:
        pinholes_centery_px_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pinholes_centery_px'].iloc[0]
    
        loadsettings_widget.value=10
        #run_plot_data_and_simulation_widget.value = True
    else:
        loadsettings_widget.bar_style='warning'


dataset_image_args_widget.observe(bind_pixis_result_to_global_variables, names='value')


# imageid_widget = widgets.BoundedIntText(
#     value=0,
#     step=1,
#     min=-1,
#     max=len(pixis_dataset)-1,
#     description='Image ID',
#     disabled=False,
#     #continuous_update=True
# )

energy_interval_widget = widgets.FloatText(
    value=0.5,
    step=0.1,
    description='energy_interval',
    disabled=False
)

beamposition_horizontal_interval_widget = widgets.FloatText(
    value=1000,
    step=1,
    description='beamposition_horizontal_interval',
    disabled=False
)

def update_imageid_widget(change):
    imageid_widget.options = get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)
    imageid_widget.value = get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)[0]
    number_of_images_widget.value = len(get_imageids_with_bgs(beamposition_horizontal_interval_widget.value))

beamposition_horizontal_interval_widget.observe(update_imageid_widget, names='value')

# imageid_widget = widgets.Dropdown(
#     options=df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid'].tolist(),
#     value=df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].sort_values('energy hall')['imageid'].tolist()[0],
#     description='imaged (sorted by energy):',
#     disabled=False,
# )

imageid_widget = widgets.Dropdown(
    options=get_imageids_with_bgs(beamposition_horizontal_interval_widget.value),
    value=get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)[0],
    description='imaged (sorted by beamposition_horizontal):',
    disabled=False,
)


use_pixis_avg_widget = widgets.Checkbox(
    value=False,
    description='use_pixis_avg_widget',
    disabled=False
)

pixis_rotation_widget = widgets.FloatText(
    value=-1.3,
    step=0.1,
    description='pixis rotation',
    disabled=False
)




pinholes_rotation_widget = widgets.FloatText(
    value=0,
    step=0.1,
    description='pinholes rotation',
    disabled=False
)

number_of_images_widget = widgets.IntText(
    value=len(pixis_dataset)-1,
    placeholder='Type something',
    description='max. Image ID:',
    disabled=True
)

pixis_bgfactor_widget = widgets.FloatText(
    value=1,
    step=0.01,
    description='pixis bgfactor',
    disabled=False
)

pinholes_bgfactor_widget = widgets.FloatText(
    value=0,
    step=0.01,
    description='pinholes bgfactor',
    disabled=False
)

pixis_avg_width_widget = widgets.FloatText(
    value=10,
    step=1,
    description='pixis avg_width',
    disabled=False
)

pinholes_avg_width_widget = widgets.FloatText(
    value=200,
    step=1,
    description='pinholes avg_width',
    disabled=False
)


#get_pixis_profiles_interactive = interactive(get_pixis_profiles,
                                      # imageid=imageid_widget,
                                      # bgfactor=widget_bgfactor,
                                      # avg_width=widget_avg_width)
#display(get_pixis_profiles_interactive)
#get_pixis_profiles_interactive.update()
#(pixis_image_norm, pixis_profile, pixis_profile_avg) = get_pixis_profiles_interactive.result

#centerx = np.where(pixis_profile_avg==1)[0][0]
#shiftx_px = centerx - pixis_profile_avg.size/2
#shiftx_um = shiftx_px * 13


_lambda_widget = widgets.FloatText(
    value=daq_parameter_image_dataset[0][imageid_widget.value],
    step=0.1,
    description='lambda/nm:',
    disabled=False
)


def update__lambda_widget(change):
    run_plot_data_and_simulation_widget.value = False
    if daq_parameter_image_dataset[0][imageid_widget.value] > 0:  # or daq_parameter_image_dataset[0][imageid_widget.value] == NaN):
        _lambda_widget.value=daq_parameter_image_dataset[0][imageid_widget.value]
    else:
        _lambda_widget.value=daq_parameter_image_dataset[14][imageid_widget.value]
    

# imageid_widget.observe(update__lambda_widget, names='value')

w1_widget = widgets.FloatText(
    value=10,
    step=0.1,
    description='ph1 width/um:',
    disabled=False
)

w2_widget = widgets.FloatText(
    value=10,
    step=0.1,
    description='ph2 width/um:',
    disabled=False
)

I_w1_widget = widgets.FloatText(
    value=1,
    step=0.1,
    description='I_w1:',
    disabled=False
)

I_w2_widget = widgets.FloatText(
    value=1,
    step=0.1,
    description='I_w2:',
    disabled=False
)

z_widget = widgets.FloatText(
    value=5781,
    step=1,
    description='Distance z/mm:',
    disabled=False
)

I_Airy1_widget = widgets.FloatText(
    value=0,
    step=0.1,
    description='I_Airy1:',
    disabled=False
)

I_Airy2_widget = widgets.FloatText(
    value=0,
    step=0.1,
    description='I_Airy2:',
    disabled=False
)

sep_factor_widget = widgets.FloatText(
    value=1,
    step=0.1,
    description='sep_factor:',
    disabled=False
)

gamma_widget = widgets.FloatText(
    value=1,
    step=0.1,
    description='Coherence Degree \gamma:',
    disabled=False
)

d_um_widget = widgets.FloatText(
    value=sep,
    step=0.1,
    description='Slit Separation/um',
    disabled=False
)

shiftx_um_adjust_widget = widgets.FloatText(
    value=0,
    step=1,
    description='shift x adjust / um',
    disabled=False
)

widget_shiftx_um_auto = widgets.FloatText(
    value=0,
    step=1,
    description='shift x auto / um',
    disabled=False
)

pinholes_centerx_px_widget = widgets.IntText(value=500,description='pinholes_centerx_px',disabled=False)
pinholes_centery_px_widget = widgets.IntText(value=500,description='pinholes_centery_px',disabled=False)
pinholes_avg_width_alongy_widget = widgets.IntText(value=100,description='pinholes_avg_width_alongy',disabled=False)
pinholes_avg_width_alongx_widget = widgets.IntText(value=100,description='pinholes_avg_width_alongx',disabled=False)

xlim_lower_widget = widgets.BoundedIntText(
    value=0,
    step=100,
    min=0,
    max=1047,
    description='xlim lower',
    disabled=False
)

xlim_upper_widget = widgets.BoundedIntText(
    value=1047,
    step=100,
    min=0,
    max=1047,
    description='xlim lower',
    disabled=False
)


pixis_centerx_px_widget = widgets.FloatText(
    value=523,
    step=1,
    description='centerx_px',
    disabled=False
)

pixis_centery_px_widget = widgets.IntText(
    value=523,
    step=1,
    description='centery_px',
    disabled=False
)

zoomwidth_widget = widgets.IntText(
    value=100,
    step=2,
    description='zoomwidth/px',
    disabled=False
)

zoomborderleft_widget = widgets.IntText(
    value=0,
    step=1,
    description='zoomborderleft/px',
    disabled=False
)

zoomborderright_widget = widgets.IntText(
    value=0,
    step=1,
    description='zoomborderright/px',
    disabled=False
)

#get_simulation_interactive = interactive(get_simulation,
                                    #     _lambda_nm=_lambda_widget,
                                    #     w_um=widget_w,
                                    #     z_mm=z_widget,
                                    #     gamma=gamma_widget,
                                    #     d_um=d_widget,
                                    #     shiftx_um=widget_shiftx_um)
#display(get_simulation_interactive)
#get_simulation_interactive.update()
#(theta, I_D_0, I_D_normalized) = get_simulation_interactive.result

display(number_of_images_widget)

pixis_centerx_px_auto_widget = widgets.FloatText(
    value=0,
    placeholder='Type something',
    description='pixis_centerx_px_auto',
    disabled=True
)
display(pixis_centerx_px_auto_widget)


pixis_profile_avg_centerx_px_auto_widget = widgets.FloatText(
    value=0,
    placeholder='Type something',
    description='pixis_profile_avg_centerx_px_auto',
    disabled=True
)
display(pixis_profile_avg_centerx_px_auto_widget)

use_pixis_profile_avg_centerx_widget = widgets.Checkbox(value=False, description = 'use_pixis_profile_avg_centerx')
pixis_center_autofind_widget = widgets.Checkbox(value=True, description = 'pixis_center_autofind')

plot_lineout_widget = widgets.Checkbox(value=False, description = 'plot_lineout')
plot_simulation_widget = widgets.Checkbox(value=False, description = 'plot_simulation')
plot_simulation_fit_widget = widgets.Checkbox(value=True, description = 'plot_simulation_fit')
plot_LightPipes_simulation_widget = widgets.Checkbox(value=False, description = 'plot_LightPipes_simulation')
plot_pixis_std_contour_widget = widgets.Checkbox(value=True, description = 'plot_pixis_std_contour')
levels_auto_widget = widgets.Checkbox(value=True, description = 'levels_auto')
levels_min_widget = widgets.FloatText(value=0.0, description = 'levels_min')
levels_max_widget = widgets.FloatText(value=0.1, description = 'levels_max')
savefigure_widget = widgets.Checkbox(value=False, description = 'savefigure')
savefigure_all_images_widget = widgets.Checkbox(value=False, description = 'savefigure_all_images')
PlotTextbox_widget = widgets.Checkbox(value=False, description = 'PlotTextbox')
initialize_fits_list_widget = widgets.Checkbox(value=True, description = 'initialize_fits_list')


df_fits_save_widget = widgets.ToggleButton(
    value=False,
    description='save to df_fits',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save to df_fits',
    icon='check'
)

df_fits_load_widget = widgets.ToggleButton(
    value=False,
    description='load from df_fits',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='load from df_fits',
    icon='check'
)

#['bgfactor', 'pixis_rotation', 'pixis_centerx_px', 'pixis_centery_px', 'pixis_profile_centerx_px_fit', 'pixis_profile_centery_px_fit', 'pinholes_cm_x_px', 'pinholes_cm_y_px', '_lambda_nm_fit', 'gamma_fit']

def update_df_fits_save_widget(change):
    if df_fits_save_widget.value == True:
        df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])), 'pixis_rotation'] = pixis_rotation_widget.value
        df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])), 'pixis_centerx_px'] = pixis_centerx_px_widget.value
        df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])), 'pixis_centery_px'] = pixis_centery_px_widget.value
        df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])), 'pinholes_centerx_px'] = pinholes_centerx_px_widget.value
        df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])), 'pinholes_centery_px'] = pinholes_centery_px_widget.value
        df_fits_save_widget.value = False

df_fits_save_widget.observe(update_df_fits_save_widget, names='value')


def update_df_fits_load_widget(change):
    if df_fits_load_widget.value == True:
        pixis_rotation_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_rotation'].iloc[0]
        pixis_centerx_px_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_centerx_px'].iloc[0]
        pixis_centery_px_widget.value = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_centery_px'].iloc[0]
        df_fits_load_widget.value = False

df_fits_load_widget.observe(update_df_fits_load_widget, names='value')

df_fits_csv_save_widget = widgets.ToggleButton(
    value=False,
    description='save df_fits to csv',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save df_fits to csv',
    icon='check'
)

def update_df_fits_csv_save_widget(change):
    if df_fits_csv_save_widget.value == True:
        # save fits to csv
        df_fits = df0[['timestamp_pulse_id'] + fits_header_list]
        save_df_fits = True
        if save_df_fits == True:
            df_fits.to_csv('/home/wodzinsk/PycharmProjects/coherence/'+'df_fits_v2.csv')
        df_fits_csv_save_widget.value = False

df_fits_csv_save_widget.observe(update_df_fits_csv_save_widget, names='value')






run_plot_data_and_simulation_widget = widgets.Checkbox(value=False,
                                              description='run_plot_data_and_simulation (Disable when changing multiple parameters!)')

   
ap5_x_px_widget = widgets.FloatText(value=523, step=1, description='ap5_x_px', disabled=False)
ap5_y_px_widget = widgets.FloatText(value=523, step=1, description='ap5_y_px', disabled=False)
ap5_width_px_widget = widgets.FloatText(value=200, step=1, description='ap5_width_px', disabled=False)
ap5_height_px_widget = widgets.FloatText(value=200, step=1, description='ap5_height_px', disabled=False)
screenhole_x_px_widget = widgets.FloatText(value=523, step=1, description='screenhole_x_px', disabled=False)
screenhole_y_px_widget = widgets.FloatText(value=523, step=1, description='screenhole_y_px', disabled=False)
screenhole_width_px_widget = widgets.FloatText(value=50, step=1, description='screenhole_width_px', disabled=False)
screenhole_height_px_widget = widgets.FloatText(value=50, step=1, description='screenhole_height_px', disabled=False)


cropforbp_widget = widgets.IntText(
    value=20,
    description='cropforbp:',
    disabled=False
)

do_deconvmethod_widget = widgets.Checkbox(
    value=False,
    description='do_deconvmethod',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='do_deconvmethod',
    icon='check'
)

# plot_data_and_simulation_interactive_input = widgets.VBox([
#                                                    run_plot_data_and_simulation_widget,
#                                                    imageid_widget,
#                                                    number_of_images_widget,
#                                                    pixis_rotation_widget,
#                                                    pixis_bgfactor_widget,
#                                                    pixis_avg_width_widget,
#                                                    pinholes_rotation_widget,
#                                                    pinholes_bgfactor_widget,
#                                                    pinholes_avg_width_widget,
#                                                    _lambda_widget,
#                                                    w1_widget,
#                                                    w2_widget,
#                                                    I_w1_widget,
#                                                    I_w2_widget,
#                                                    z_widget,
#                                                    I_Airy1_widget,
#                                                    I_Airy2_widget,
#                                                    sep_factor_widget,
#                                                    gamma_widget,
#                                                    d_um_widget,
#                                                    aperture5_mm_bg_widget,
#                                                    aperture7_mm_bg_widget,
#                                                    aperture5_mm_image_widget,
#                                                    aperture7_mm_image_widget,
#                                                    filter_bg_widget,
#                                                    filter_image_widget,
#                                                    use_pixis_profile_avg_centerx_widget,
#                                                    pixis_center_autofind_widget,
#                                                    pixis_centerx_px_widget,
#                                                    pixis_centery_px_widget,
#                                                    shiftx_um_adjust_widget,
#                                                    pinholes_centerx_px_widget,
#                                                    pinholes_centery_px_widget,
#                                                    pinholes_avg_width_alongy_widget,
#                                                    pinholes_avg_width_alongx_widget,
#                                                    xlim_lower_widget,
#                                                    xlim_upper_widget,
#                                                    plot_lineout_widget,
#                                                    plot_simulation_widget,
#                                                    plot_simulation_fit_widget,
#                                                    plot_LightPipes_simulation_widget,
#                                                    plot_pixis_std_contour_widget,
#                                                    levels_auto_widget,
#                                                    levels_min_widget,
#                                                    levels_max_widget,
#                                                    zoomwidth_widget,
#                                                    zoomborderleft_widget,
#                                                    zoomborderright_widget,
#                                                    savefigure_widget,
#                                                    savefigure_all_images_widget,
#                                                    PlotTextbox_widget,
#                                                    initialize_fits_list_widget])

plot_data_and_simulation_interactive_input_column1 = VBox([
                run_plot_data_and_simulation_widget,
                #imageid_widget,
                number_of_images_widget,
                #pixis_rotation_widget,
                #pixis_bgfactor_widget,
                pixis_avg_width_widget,
                pinholes_rotation_widget,
                pinholes_bgfactor_widget,
                pinholes_avg_width_widget,
                beamposition_horizontal_interval_widget,
                use_pixis_avg_widget
])

plot_data_and_simulation_interactive_input_column2 = VBox([
                _lambda_widget,
                w1_widget,
                w2_widget,
                I_w1_widget,
                I_w2_widget,
                z_widget,
                I_Airy1_widget,
                I_Airy2_widget,
                sep_factor_widget,
                gamma_widget,
                d_um_widget,
])


plot_data_and_simulation_interactive_input_column3 = VBox([
                aperture5_mm_bg_widget,
                aperture7_mm_bg_widget,
                aperture5_mm_image_widget,
                aperture7_mm_image_widget,
                filter_bg_widget,
                filter_image_widget,
])

plot_data_and_simulation_interactive_input_column4 = VBox([
                use_pixis_profile_avg_centerx_widget,
                pixis_center_autofind_widget,
                pixis_centerx_px_widget,
                pixis_centery_px_widget,
                shiftx_um_adjust_widget,
                pinholes_centerx_px_widget,
                pinholes_centery_px_widget,
                pinholes_avg_width_alongy_widget,
                pinholes_avg_width_alongx_widget,
                xlim_lower_widget,
                xlim_upper_widget
])

plot_data_and_simulation_interactive_input_column5 = VBox([
                plot_lineout_widget,
                plot_simulation_widget,
                plot_simulation_fit_widget,
                plot_LightPipes_simulation_widget,
                plot_pixis_std_contour_widget,
                levels_auto_widget,
                levels_min_widget,
                levels_max_widget,
                zoomwidth_widget,
                zoomborderleft_widget,
                zoomborderright_widget,
])

plot_data_and_simulation_interactive_input_column6 = VBox([
                savefigure_widget,
                savefigure_all_images_widget,
                PlotTextbox_widget,
                initialize_fits_list_widget,
    cropforbp_widget,
    do_deconvmethod_widget
])

plot_data_and_simulation_interactive_input_column7 = VBox([
    ap5_x_px_widget,
    ap5_y_px_widget,
    ap5_width_px_widget,
    ap5_height_px_widget,
    screenhole_x_px_widget,
    screenhole_y_px_widget,
    screenhole_width_px_widget,
    screenhole_height_px_widget
])

plot_data_and_simulation_interactive_input = VBox([
    HBox([
    plot_data_and_simulation_interactive_input_column1,
    plot_data_and_simulation_interactive_input_column2,
    plot_data_and_simulation_interactive_input_column3,
    plot_data_and_simulation_interactive_input_column4,
    plot_data_and_simulation_interactive_input_column5,
    plot_data_and_simulation_interactive_input_column6,
    plot_data_and_simulation_interactive_input_column7
    ]),
    HBox([imageid_widget,pixis_bgfactor_widget, pixis_rotation_widget, df_fits_save_widget, df_fits_load_widget, loadsettings_widget, df_fits_csv_save_widget])
])


def update_settings(change):
    loadsettings_widget.value = 0
    run_plot_data_and_simulation_widget.value = False
    hdf5_file_name_bg_widget.value = settings_widget.value[0]
    dataset_bg_args_widget.value = settings_widget.value[1]
    hdf5_file_name_image_widget.value = settings_widget.value[2]
    dataset_image_args_widget.value = settings_widget.value[3]

def update_LightPipes(change):
    global Int_LightPipes, Intlinex_LightPipes, Intliney_LightPipes
    Int_LightPipes, Intlinex_LightPipes, Intliney_LightPipes = run_doublepinholes_LightPipes_interactive.result
    #plot_data_and_simulation_interactive.update()

widget_LightPipesUpdate = widgets.Button(value=False, description='Update LightPipes Lineout in plot below')
widget_LightPipesUpdate.on_click(update_LightPipes)

settings_widget_layout = widgets.Layout(width='100%')
settings_widget = widgets.Dropdown(options=settings, layout = settings_widget_layout)
settings_widget.observe(update_settings, names='value')
display(settings_widget)

display(widget_LightPipesUpdate)


plot_data_and_simulation_interactive = interactive(plot_data_and_simulation,
                                                   run_plot_data_and_simulation = run_plot_data_and_simulation_widget,
                                                   imageid=imageid_widget,
                                                   use_pixis_avg = use_pixis_avg_widget,
                                                   imageid_max = number_of_images_widget,
                                                   pixis_rotation=pixis_rotation_widget,
                                                   pixis_bgfactor=pixis_bgfactor_widget,
                                                   pixis_avg_width=pixis_avg_width_widget,
                                                   pinholes_rotation=pinholes_rotation_widget,
                                                   pinholes_bgfactor=pinholes_bgfactor_widget,
                                                   pinholes_avg_width=pinholes_avg_width_widget,
                                                   _lambda_nm=_lambda_widget,
                                                   w1_um=w1_widget,
                                                   w2_um=w2_widget,
                                                   I_w1 = I_w1_widget,
                                                   I_w2 = I_w2_widget,
                                                   z_mm=z_widget,
                                                   I_Airy1 = I_Airy1_widget,
                                                   I_Airy2 = I_Airy2_widget,
                                                   sep_factor = sep_factor_widget,
                                                   gamma=gamma_widget,
                                                   d_um=d_um_widget,
                                                   aperture5_mm_bg=aperture5_mm_bg_widget,
                                                   aperture7_mm_bg=aperture7_mm_bg_widget,
                                                   aperture5_mm_image=aperture5_mm_image_widget,
                                                   aperture7_mm_image=aperture7_mm_image_widget,
                                                   filter_bg=filter_bg_widget,
                                                   filter_image = filter_image_widget,
                                                   use_pixis_profile_avg_centerx = use_pixis_profile_avg_centerx_widget,
                                                   pixis_center_autofind = pixis_center_autofind_widget,
                                                   pixis_centerx_px = pixis_centerx_px_widget,
                                                   pixis_centery_px = pixis_centery_px_widget,
                                                   shiftx_um_adjust=shiftx_um_adjust_widget,
                                                   pinholes_centerx_px = pinholes_centerx_px_widget,
                                                   pinholes_centery_px = pinholes_centery_px_widget,
                                                   pinholes_avg_width_alongy = pinholes_avg_width_alongy_widget,
                                                   pinholes_avg_width_alongx = pinholes_avg_width_alongx_widget,
                                                      ap5_x_px = ap5_x_px_widget,
                                                     ap5_y_px = ap5_y_px_widget,
                                                     ap5_width_px = ap5_width_px_widget,
                                                     ap5_height_px = ap5_height_px_widget,
                                                     screenhole_x_px = screenhole_x_px_widget,
                                                     screenhole_y_px = screenhole_y_px_widget,
                                                     screenhole_width_px = screenhole_width_px_widget,
                                                     screenhole_height_px = screenhole_height_px_widget,
                                                   xlim_lower=xlim_lower_widget,
                                                   xlim_upper=xlim_upper_widget,
                                                   plot_lineout = plot_lineout_widget,
                                                   plot_simulation = plot_simulation_widget,
                                                   plot_simulation_fit = plot_simulation_fit_widget,
                                                   plot_LightPipes_simulation = plot_LightPipes_simulation_widget,
                                                   plot_pixis_std_contour = plot_pixis_std_contour_widget,
                                                   levels_auto = levels_auto_widget,
                                                   levels_min = levels_min_widget,
                                                   levels_max = levels_max_widget,
                                                   zoomwidth = zoomwidth_widget,
                                                   zoomborderleft = zoomborderleft_widget,
                                                   zoomborderright = zoomborderright_widget,
                                                   savefigure= savefigure_widget,
                                                   savefigure_all_images = savefigure_all_images_widget,
                                                   PlotTextbox = PlotTextbox_widget,
                                                   initialize_fits_list = initialize_fits_list_widget,
                                                   cropforbp = cropforbp_widget,
                                                   do_deconvmethod = do_deconvmethod_widget
                                                  )

plot_data_and_simulation_interactive_output = interactive_output(plot_data_and_simulation,
                                                   {'run_plot_data_and_simulation' : run_plot_data_and_simulation_widget,
                                                   'imageid' : imageid_widget,
                                                    'use_pixis_avg': use_pixis_avg_widget,
                                                   'imageid_max' : number_of_images_widget,
                                                   'pixis_rotation' : pixis_rotation_widget,
#                                                    'pixis_rotation_save' : pixis_rotation_save_widget,
#                                                    'pixis_rotation_load' : pixis_rotation_load_widget,
                                                   'pixis_bgfactor' : pixis_bgfactor_widget,
                                                   'pixis_avg_width' : pixis_avg_width_widget,
                                                   'pinholes_rotation' : pinholes_rotation_widget,
                                                   'pinholes_bgfactor' : pinholes_bgfactor_widget,
                                                   'pinholes_avg_width' : pinholes_avg_width_widget,
                                                   '_lambda_nm' : _lambda_widget,
                                                   'w1_um' : w1_widget,
                                                   'w2_um' : w2_widget,
                                                   'I_w1' : I_w1_widget,
                                                   'I_w2' : I_w2_widget,
                                                   'z_mm' : z_widget,
                                                   'I_Airy1' : I_Airy1_widget,
                                                   'I_Airy2' : I_Airy2_widget,
                                                   'sep_factor' : sep_factor_widget,
                                                   'gamma' : gamma_widget,
                                                   'd_um' : d_um_widget,
                                                   'aperture5_mm_bg' : aperture5_mm_bg_widget,
                                                   'aperture7_mm_bg' : aperture7_mm_bg_widget,
                                                   'aperture5_mm_image' : aperture5_mm_image_widget,
                                                   'aperture7_mm_image' :aperture7_mm_image_widget,
                                                   'filter_bg' : filter_bg_widget,
                                                   'filter_image' : filter_image_widget,
                                                   'use_pixis_profile_avg_centerx' : use_pixis_profile_avg_centerx_widget,
                                                   'pixis_center_autofind' : pixis_center_autofind_widget,
                                                   'pixis_centerx_px' : pixis_centerx_px_widget,
                                                   'pixis_centery_px' : pixis_centery_px_widget,
                                                   'shiftx_um_adjust' : shiftx_um_adjust_widget,
                                                   'pinholes_centerx_px' : pinholes_centerx_px_widget,
                                                   'pinholes_centery_px' : pinholes_centery_px_widget,
                                                   'pinholes_avg_width_alongy' : pinholes_avg_width_alongy_widget,
                                                   'pinholes_avg_width_alongx' : pinholes_avg_width_alongx_widget,
                                                     'ap5_x_px' : ap5_x_px_widget,
                                                     'ap5_y_px' : ap5_y_px_widget,
                                                     'ap5_width_px' : ap5_width_px_widget,
                                                     'ap5_height_px' : ap5_height_px_widget,
                                                     'screenhole_x_px' : screenhole_x_px_widget,
                                                     'screenhole_y_px' : screenhole_y_px_widget,
                                                     'screenhole_width_px' : screenhole_width_px_widget,
                                                     'screenhole_height_px' : screenhole_height_px_widget,
                                                   'xlim_lower' : xlim_lower_widget,
                                                   'xlim_upper' : xlim_upper_widget,
                                                   'plot_lineout' : plot_lineout_widget,
                                                   'plot_simulation' : plot_simulation_widget,
                                                   'plot_simulation_fit' : plot_simulation_fit_widget,
                                                   'plot_LightPipes_simulation' : plot_LightPipes_simulation_widget,
                                                   'plot_pixis_std_contour' : plot_pixis_std_contour_widget,
                                                   'levels_auto' : levels_auto_widget,
                                                   'levels_min' : levels_min_widget,
                                                   'levels_max' : levels_max_widget,
                                                   'zoomwidth' : zoomwidth_widget,
                                                   'zoomborderleft' : zoomborderleft_widget,
                                                   'zoomborderright' : zoomborderright_widget,
                                                   'savefigure' :savefigure_widget,
                                                   'savefigure_all_images' : savefigure_all_images_widget,
                                                   'PlotTextbox' : PlotTextbox_widget,
                                                   'initialize_fits_list' : initialize_fits_list_widget,
                                                     'cropforbp': cropforbp_widget,
                                                   'do_deconvmethod': do_deconvmethod_widget}
                                                  )


#plot_data_and_simulation_interactive_output.layout.width = '100%'
plot_data_and_simulation_interactive_output.layout.height = '2000px'
plot_data_and_simulation_interactive_output.layout.border = 'solid'

from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})''')) # https://stackoverflow.com/a/57346765

display(VBox([plot_data_and_simulation_interactive_input,
              plot_data_and_simulation_interactive_output]))
##1 maingui
# %%
