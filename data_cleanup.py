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
        size_GB = os.path.getsize(Path.joinpath(useful_dir, filename, '_' , str(df_selection.begin.item()), 'to', str(df_selection.end.item()), '_useful.h5'))/(1024*1024*1024)
        
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



daq_parameter=[
'FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean wavelength',
'FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean phtoton energy',    
'FL2/Electron Diagnostic/Undulator setting/SASE03 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE04 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE05 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE06 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE07 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE08 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE09 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE10 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE11 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE12 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE13 gap',
'FL2/Electron Diagnostic/Undulator setting/SASE14 gap',
'FL2/Electron Diagnostic/Undulator setting/set wavelength',
'FL2/Photon Diagnostic/GMD/Average energy/hall',
'FL2/Photon Diagnostic/GMD/Average energy/tunnel',
'FL2/Photon Diagnostic/GMD/Beam position/Average/position hall horizontal',
'FL2/Photon Diagnostic/GMD/Beam position/Average/position hall vertical',
'FL2/Photon Diagnostic/GMD/Beam position/Average/position tunnel horizontal',
'FL2/Photon Diagnostic/GMD/Beam position/Average/position tunnel vertical',
'FL2/Photon Diagnostic/GMD/Beam position/Pulse resolved/hall x',
'FL2/Photon Diagnostic/GMD/Beam position/Pulse resolved/hall y',
'FL2/Photon Diagnostic/GMD/Beam position/Pulse resolved/tunnel x',
'FL2/Photon Diagnostic/GMD/Beam position/Pulse resolved/tunnel y',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy aux hall',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy aux tunnel',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/position horizontal hall',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/position horizontal tunnel',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/position vertical hall',
'FL2/Photon Diagnostic/GMD/Pulse resolved energy/position vertical tunnel'
]   



def get_datasets(data_dir, hdf5_file_name, start, end):
    hdf5_file = h5py.File(Path.joinpath(data_dir, hdf5_file_name), 'r')  # 'r' means that hdf5 file is open in read-only mode
    pixis_dataset = hdf5_file['/Experiment/Camera/Pixis 1/image'][start:end]
    pinholes_dataset = hdf5_file['/Experiment/Camera/FL24/Pinhole B/image'][start:end]
    timestamp_dataset = hdf5_file['/Timing/time stamp/fl2user1'][start:end]
    
    daq_parameter_dataset = []
    for parameter in daq_parameter:
        daq_parameter_dataset.append(hdf5_file[parameter][start:end])
    
    hdf5_file.close()
    return (pixis_dataset, pinholes_dataset, timestamp_dataset, daq_parameter_dataset)

def get_hdf5_file_length(data_dir, hdf5_file_name):
    hdf5_file = h5py.File(Path.joinpath(data_dir, hdf5_file_name), 'r')  # 'r' means that hdf5 file is open in read-only mode

    length = hdf5_file['/Experiment/Camera/Pixis 1/image'].len()

    hdf5_file.close()

    return length



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

    if os.path.exists(Path.joinpath(useful_dir, useful_hdf5_file_name)):
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




def plot_data_and_simulation(run_plot_data_and_simulation,
                             imageid,
                             use_pixis_avg,
                             imageid_max,
                             pixis_rotation,
                             #pixis_rotation_save,
                             #pixis_rotation_load,
                             pixis_bgfactor,
                             pixis_avg_width,
                             pinholes_rotation,
                             pinholes_bgfactor,
                             pinholes_avg_width,
                             _lambda_nm,
                             w1_um,
                             w2_um,
                             I_w1,
                             I_w2,
                             z_mm,
                             I_Airy1, I_Airy2, sep_factor,
                             gamma,
                             d_um,
                             aperture5_mm_bg,
                             aperture7_mm_bg,
                             aperture5_mm_image,
                             aperture7_mm_image,
                             filter_bg,
                             filter_image,
                             use_pixis_profile_avg_centerx,
                             pixis_center_autofind,
                             pixis_centerx_px,
                             pixis_centery_px,
                             shiftx_um_adjust,
                             pinholes_centerx_px,
                             pinholes_centery_px,
                             pinholes_avg_width_alongy,
                             pinholes_avg_width_alongx,
                             ap5_x_px,
                             ap5_y_px,
                             ap5_width_px,
                             ap5_height_px,
                             screenhole_x_px,
                             screenhole_y_px,
                             screenhole_width_px,
                             screenhole_height_px,
                             xlim_lower,
                             xlim_upper,
                             plot_lineout,
                             plot_simulation,
                             plot_simulation_fit,
                             plot_LightPipes_simulation,
                             plot_pixis_std_contour,
                             levels_auto,
                             levels_min,
                             levels_max,
                             zoomwidth,
                             zoomborderleft,
                             zoomborderright,
                             savefigure,
                            savefigure_all_images,
                            PlotTextbox,
                            initialize_fits_list,
                            cropforbp,
                            do_deconvmethod):
    
    global pixis_profile_centerx_px_fit
    global pinholes_cm_x_px
    global pinholes_cm_y_px
    global _lambda_nm_fit
    global gamma_fit
    global pixis_image_norm_dataset
    global pixis_profile_avg_dataset
    global pixis_centery_px_dataset
    global pixis_image_avg_norm
    global pinholes_image_norm
    global pinholes_image_norm_dataset
    global pinholes_bg_avg_dataset
    global xi_um
    global xi_um_of_average
    global sigma_F_gamma_um_opt
    global sigma_F_gamma_um_opt_of_average
    global pixis_image_minus_bg_rot_cropped_counts_dataset
    
    
    if run_plot_data_and_simulation == True:
        

        # Dataframe for this dataset:

        #df = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]
        
#         if pixis_rotation_save == True:
#             df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_rotation'] = pixis_rotation
#         if pixis_rotation_load == True:
#             pixis_rotation = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2])]['pixis_rotation']
#             pixis_rotation_widget.value = pixis_rotation
            
        # use df_all to get the values, since bg images are not in the combined df0 consisting df_all and df_settings!
        imageid_sequence_by_energy_hall = get_imageids_with_bgs(beamposition_horizontal_interval_widget.value)
        #df = df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2]) & df_all['imageid'].isin(imageid_sequence_by_energy_hall)]
        df = df0[(df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & df0['imageid'].isin(imageid_sequence_by_energy_hall)]
        
        
        fits_list_len = len(df_all[(df_all['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df_all['pinholes'] == dataset_image_args_widget.value[2])].index) + 1
        
        
        print('# images for this DPH: ' + str(fits_list_len))
        
        if use_pixis_avg == True:
            imageid = [-1]
        
        
        #with plot_data_and_simulation_interactive_output:
         #   print('running')
        # Initialize datset fits lists:
        
        if initialize_fits_list == True:
            pixis_profile_centerx_px_fit = [None] * (fits_list_len)
            pinholes_cm_x_px = [None] * (fits_list_len)
            pinholes_cm_y_px = [None] * (fits_list_len)
            _lambda_nm_fit = [None] * (fits_list_len)
            gamma_fit = [None] * (fits_list_len)
            pixis_image_norm_dataset = [None] * (fits_list_len)
            pixis_profile_avg_dataset = [None] * (fits_list_len)
            pixis_centery_px_dataset = [None] * (fits_list_len)
            pinholes_image_norm_dataset = [None] * (fits_list_len)
            pinholes_bg_avg_dataset = [None] * (fits_list_len)
            pixis_image_minus_bg_rot_cropped_counts_dataset = [None] * (fits_list_len)
        
        if savefigure_all_images == True:
            #imageid_sequence = range(imageid_max+1)
            imageid_sequence = imageid_sequence_by_energy_hall
            imageid_sequence.append(-1)
            
        else:
            if use_pixis_avg == True:
                imageid_sequence = [-1]
            else:
                imageid_sequence = [imageid]
        
        wavelength_nm = _lambda_nm
        slitwidth1_um = w1_um
        slitwidth2_um = w2_um
        separation_um = d_um
        z_mm = z_mm
            

#         pixis_image_norm_dataset = []
#         pixis_profile_avg_dataset = []
        for imageid_loop in imageid_sequence:
            
            print('imageid_loop=' + str(imageid_loop))
            print('imageid_sequence=' + str(imageid_sequence))
            if imageid_loop == -1:
                beamposx = df['beam position hall horizontal pulse resolved'].mean(axis=0)
                beamposy = df['beam position hall vertical pulse resolved'].mean(axis=0)
                energy_hall_uJ = df['energy hall'].mean(axis=0)
            else:
                beamposx = df[df['imageid']==imageid_loop]['beam position hall horizontal pulse resolved']
                beamposy = df[df['imageid']==imageid_loop]['beam position hall vertical pulse resolved']
                if df[df['imageid']==imageid_loop]['energy hall'].size > 0:
                    energy_hall_uJ = df[df['imageid']==imageid_loop]['energy hall'].iloc[0]
                else:
                    energy_hall_uJ = 45
                    df[df['imageid']==imageid_loop]['energy hall'] = energy_hall_uJ
                    
                    
            
            #plt.ioff()
            
            

            hdf5_file_name_image = hdf5_file_name_image_widget.value
            dataset_image_args = dataset_image_args_widget.value
            ph = dataset_image_args[2]

            global pixis_image_norm
            global pixis_profile
            global pixis_profile_avg
            global pixis_profile_alongy
            
            
            (pixis_image_norm, pixis_bg_std_norm, pixis_bg_std_avg, pixis_profile, pixis_profile_avg,
                 pixis_profile_avg_centerx_px, pixis_centerx_px, pixis_centery_px, pixis_profile_alongy,
                 pixis_cts, pixis_cm_x_px, pixis_cm_y_px, pinholes_bg_avg_dataset[imageid_loop], pixis_image_minus_bg_rot_cropped_counts_dataset[imageid_loop]) = get_pixis_profiles(imageid_loop,
                                                                               use_pixis_avg,
                                                                            pixis_bgfactor,
                                                                            pixis_avg_width,
                                                                            pixis_rotation,
                                                                            pixis_center_autofind,
                                                                            pixis_centerx_px,
                                                                            pixis_centery_px,
                                                                              cropforbp)

            
            
            #pixis_image_norm_dataset.append(pixis_image_norm)  # how to access later. these are sorted by energy ...
            #pixis_profile_avg_dataset.append(pixis_profile_avg)
            
            pixis_image_norm_dataset[imageid_loop] = pixis_image_norm
            pixis_profile_avg_dataset[imageid_loop] = pixis_profile_avg
            pixis_centery_px_dataset[imageid_loop] = pixis_centery_px
            
            
            if use_pixis_profile_avg_centerx == True:
                shiftx_px =  pixis_profile_avg_centerx_px - pixis_profile_avg.size/2                
            else:
                shiftx_px =  pixis_centerx_px - pixis_profile_avg.size/2

            shiftx_um = shiftx_px * 13 + shiftx_um_adjust
            
            pixis_centerx_px_auto_widget.value = pixis_centerx_px
            #pixis_profile_avg_centerx_px_auto_widget.value = pixis_profile_avg_centerx_px

            (pinholes_image_norm_dataset[imageid_loop], pinholes_profile_alongx,
             pinholes_profile_alongx_avg,
             pinholes_profile_alongy,
             pinholes_profile_alongx_avg,
             pinholes_cts, pinholes_cm_x_px[imageid_loop], pinholes_cm_y_px[imageid_loop]) = get_pinholes_profiles(imageid_loop,
                                                                  pinholes_bgfactor,
                                                                  pinholes_rotation,
                                                                  pinholes_centerx_px,
                                                                  pinholes_centery_px,
                                                                  pinholes_avg_width_alongy,
                                                                  pinholes_avg_width_alongx)
            
            df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'pinholes_cm_x_px'] = pinholes_cm_x_px[imageid_loop]
            df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'pinholes_cm_y_px'] = pinholes_cm_y_px[imageid_loop]
            df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'pixis_image_minus_bg_rot_cropped_counts'] = pixis_image_minus_bg_rot_cropped_counts_dataset[imageid_loop]
            
            

            # make a lower and upper profile along x to determine the proper rotation
            
            n = pixis_profile_avg.size # number of sampling point  # number of pixels
            dx = 13*1e-6 # sampling # pixel size
            xdata = list(range(n))
            ydata = pixis_profile_alongy
           
            if plot_simulation_fit == True:
                pixis_yshift_px = pixis_centery_px
                p0 = (pixis_yshift_px, 400)
                popt_gauss, pcov_gaussian = curve_fit(lambda x, m, w: gaussianbeam(x, 1, m ,w, 0), xdata, ydata, p0)
                pixis_beamwidth_px = popt_gauss[1]  # this is 2 sigma!
                pixis_centery_px = popt_gauss[0]
            else:
                pixis_beamwidth_px = 100
                     
            pixis_beamwidth_px
            pixis_uppery_px = pixis_centery_px + int(pixis_beamwidth_px/2) 
            pixis_lowery_px = pixis_centery_px - int(pixis_beamwidth_px/2)  # smaller pixel coordinate, but in upper part of image

            pixis_profile_avg_lower = get_pixis_profiles(imageid_loop,
                                                         use_pixis_avg,
                                                        pixis_bgfactor,
                                                        30,
                                                        pixis_rotation,
                                                        pixis_center_autofind,
                                                        pixis_centerx_px,
                                                        pixis_lowery_px,
                                                        cropforbp)[4]
            
            pixis_profile_avg_upper = get_pixis_profiles(imageid_loop,
                                                         use_pixis_avg,
                                                        pixis_bgfactor,
                                                        30,
                                                        pixis_rotation,
                                                        pixis_center_autofind,
                                                        pixis_centerx_px,
                                                        pixis_uppery_px,
                                                        cropforbp)[4]
            
            ydata_lower = pixis_profile_avg_lower
            ydata_upper = pixis_profile_avg_upper
            
            
            xdata = dx * np.linspace(-n/2+1, n/2, n)# coordinate
            ydata = pixis_profile_avg
            
            # Fitting of wavelength and shiftx_um
            if plot_simulation_fit == True:
                
                p0 = (shiftx_um,_lambda_nm,gamma)
                p0 = (shiftx_um,_lambda_nm,gamma) # fit for a gamma=1
                #bounds = ([dx*(-n/2+1)/um,_lambda_nm-1], [dx*n/2/um,_lambda_nm+1])
                #bounds = ([dx*(-n/2+1)/um,_lambda_nm-1,0], [dx*n/2/um,_lambda_nm+1,1]) 
                bounds = ([dx*(-n/2+1)/um,_lambda_nm-1,gamma], [dx*n/2/um,_lambda_nm+1,gamma]) # fit for a gamma given
                popt, pcov = curve_fit(lambda x,shiftx_um,_lambda_nm, gamma: simulation(x, shiftx_um, _lambda_nm, z_mm, d_um, w1_um,w2_um,I_w1,I_w2, I_Airy1, I_Airy2, sep_factor,gamma),xdata, ydata, p0)
                
                
                
                popt_lower, pcov_lower = curve_fit(lambda x,shiftx_um,_lambda_nm, gamma: simulation(x, shiftx_um, _lambda_nm, z_mm, d_um, w1_um,w2_um,I_w1,I_w2, I_Airy1, I_Airy2, sep_factor,gamma),xdata, ydata_lower, p0)
                popt_upper, pcov_upper = curve_fit(lambda x,shiftx_um,_lambda_nm, gamma: simulation(x, shiftx_um, _lambda_nm, z_mm, d_um, w1_um,w2_um,I_w1,I_w2, I_Airy1, I_Airy2, sep_factor,gamma),xdata, ydata_upper, p0)

                shiftx_um_fit = popt[0]
                shiftx_um_fit_lower = popt_lower[0]
                shiftx_um_fit_upper = popt_upper[0]
                _lambda_nm_fit[imageid_loop] = popt[1]
                df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), '_lambda_nm_fit'] = _lambda_nm_fit[imageid_loop]
                gamma_fit[imageid_loop] = popt[2]
                

                I_D_normalized_fit = simulation(xdata, shiftx_um_fit, _lambda_nm_fit[imageid_loop], z_mm, d_um, w1_um,w2_um,I_w1,I_w2,I_Airy1, I_Airy2, sep_factor, gamma_fit[imageid_loop])
            else:
                shiftx_um_fit = shiftx_um
                shiftx_um_fit_lower = 0
                shiftx_um_fit_upper = 0
                _lambda_nm_fit[imageid_loop] = _lambda_nm
                gamma_fit[imageid_loop] = gamma
                
                
                
           
                
            pixis_profile_centerx_px_fit[imageid_loop] = n/2 + shiftx_um_fit/13
            pixis_profile_centerx_px_fit_lower = n/2 + shiftx_um_fit_lower/13
            pixis_profile_centerx_px_fit_upper = n/2 + shiftx_um_fit_upper/13
            pixis_profile_centerx_px_fit_delta = pixis_profile_centerx_px_fit_lower - pixis_profile_centerx_px_fit_upper
 

            a1 = 1
            a2 = 1
            b1=1
            b2=1
            c1=1
            c2=1
            e1=1
            e2=1
            I_D_normalized = simulation(xdata, shiftx_um, _lambda_nm, z_mm, d_um, w1_um,w2_um,I_w1,I_w2, I_Airy1, I_Airy2, sep_factor,gamma)
            
            partiallycoherent = pixis_image_norm
            #do_deconvmethod = True
            if do_deconvmethod == True:
                
                z = 5781 * 1e-3
                z_0 = 1067 * mm
                z_T = z + z_0
                z_eff = z * z_0 / (z_T)
                dX_1 = 13 * 1e-6
                
                print('lambda_nm_fit='+str(_lambda_nm_fit[imageid_loop]))
                #sigma_F_gamma_um_max = 60
                #partiallycoherent_profile, fullycoherent, fullycoherent_profile, partiallycoherent_rec_profile, partiallycoherent_rec_profile, sigma_F_gamma_um_opt, F_gamma, abs_gamma, xi_um, I_bp, dX_2, cor = deconvmethod(partiallycoherent, z, dX_1, pixis_avg_width, int(pixis_centery_px),_lambda_nm_fit[imageid_loop]*1e-9, sigma_F_gamma_um_max)
                
                
                sigma_x_F_gamma_um_min = 7
                sigma_x_F_gamma_um_max = 40


                sigma_y_F_gamma_um_min = 7
                sigma_y_F_gamma_um_max = 40
                sigma_y_F_gamma_um_stepsize = 1

                partiallycoherent = pixis_image_norm_dataset[imageid_loop]
                #pixis_profile_avg_dataset[imageid]
                (partiallycoherent_profile, fullycoherent_opt_list, fullycoherent_profile_opt_list,  partiallycoherent_rec_list, partiallycoherent_rec_profile_list, 
                 partiallycoherent_rec_profile_min_list, delta_rec_min_list, delta_profiles_cropped_list, sigma_x_F_gamma_um_opt, sigma_y_F_gamma_um_list, F_gamma_list, 
                 abs_gamma_list, xi_x_um_list, xi_y_um_list, I_bp, dX_2, cor_list, cor_profiles_list, cor_profiles_cropped_list, index_opt) = deconvmethod(partiallycoherent, z, dX_1, pixis_avg_width, int(pixis_centery_px),
                                                                                                                                                   _lambda_nm_fit[imageid_loop]*1e-9, sigma_x_F_gamma_um_min, sigma_x_F_gamma_um_max, sigma_y_F_gamma_um_min, sigma_y_F_gamma_um_max, sigma_y_F_gamma_um_stepsize, 200)
                
#                 if delta_profiles_cropped_list[0] < 0 :
#                     index_opt = np.where(np.asarray(delta_profiles_cropped_list) > 0)[0][0]
#                 else:
#                     index_opt = np.where(np.asarray(delta_profiles_cropped_list) < 0)[0][0]
    
                chi2distance_list = []
                for partiallycoherent_rec in partiallycoherent_rec_list:
                    number_of_bins = 100
                    hist1, bin_edges1 = np.histogram(partiallycoherent.ravel(), bins=np.linspace(0,1,number_of_bins))
                    hist2, bin_edges2 = np.histogram(partiallycoherent_rec.ravel(), bins=np.linspace(0,1,number_of_bins))
                    chi2distance_list.append(chi2_distance(hist1, hist2))
    
                #index_opt = np.where(np.abs(np.asarray(delta_profiles_cropped_list)) == np.min(np.abs(np.asarray(delta_profiles_cropped_list))))[0][0]
                index_opt = np.where(np.asarray(chi2distance_list) == np.min(np.asarray(chi2distance_list)))[0][0]
                
                xi_um = xi_x_um_list[index_opt]
                print('sigma_x_F_gamma_um_opt='+str(sigma_x_F_gamma_um_opt))
                print('sigma_y_F_gamma_um_list[index_opt]='+str(sigma_y_F_gamma_um_list[index_opt]))
                print('xi_x_um_list[index_opt]='+str(xi_x_um_list[index_opt]))
                print('xi_y_um_list[index_opt]='+str(xi_y_um_list[index_opt]))

                fig, axs = plt.subplots(nrows=7,ncols=1, sharex=True, figsize=(5,15))
                ax = axs[0]
                ax.plot(sigma_y_F_gamma_um_list, cor_list)
                ax.set_ylabel('cor')

                ax = axs[1]
                ax.plot(sigma_y_F_gamma_um_list, cor_profiles_list)
                ax.set_ylabel('cor profiles')

                ax = axs[2]
                ax.plot(sigma_y_F_gamma_um_list, chi2distance_list)
                ax.set_ylabel('chi2distance')
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])

                ax = axs[3]
                ax.plot(sigma_y_F_gamma_um_list, delta_rec_min_list)
                ax.set_ylabel('delta minimum')

                ax = axs[4]
                ax.plot(sigma_y_F_gamma_um_list, delta_profiles_cropped_list)
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])
                ax.set_ylabel('delta profiles cropped')

                ax = axs[5]
                ax.plot(sigma_y_F_gamma_um_list, xi_x_um_list)
                ax.set_ylabel('xi_x')
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])

                ax = axs[6]
                ax.plot(sigma_y_F_gamma_um_list, xi_y_um_list)
                ax.set_ylabel('xi_y')
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])

                fig.tight_layout()
                
                if savefigure == True:
                    savefigure_dir = scratch_dir + str(settings_widget.label)
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/profiles_rec2d/'
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    plt.savefig(savefigure_dir + '/' + 'yscan_a_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                    plt.savefig(savefigure_dir + '/' + 'yscan_a_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.pdf', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)


                
                #### only the  profiles
                for idx in range(len(fullycoherent_profile_opt_list)):
                    n = partiallycoherent_profile.shape[0]

                    xdata = np.linspace((-n/2)*dX_1*1e3, (+n/2-1)*dX_1*1e3, n)

                    fig=plt.figure(figsize=(11.69,8.27), dpi= 300, facecolor='w', edgecolor='k')  # A4 sheet in landscape
                    ax = plt.subplot(1,1,1)
                    plt.plot(xdata, partiallycoherent_profile, 'b-', label='measured partially coherent', linewidth=1)
                    plt.plot(xdata, fullycoherent_profile_opt_list[idx], 'r-', label='recovered fully coherent', linewidth=1)
                    plt.plot(xdata, partiallycoherent_rec_profile_list[idx], 'g-', label='recovered partially coherent', linewidth=1)
                    #plt.plot(xdata, gaussianbeam(xdata, 1, popt_gauss[0] ,popt_gauss[1], 0), 'r-', label='fit: m=%5.1f px, w=%5.1f px' % tuple([popt_gauss[0] ,popt_gauss[1]]))
                    plt.axhline(0, color='k')
                    plt.xlabel('x / mm', fontsize = 14)
                    plt.ylabel('Intensity / a.u.', fontsize = 14)
                    plt.legend()

                    plt.title('d / $\mu$m = '+str(int(separation_um)) + ' coherence length $\\xi_x$ / $\mu$m = ' + str(round(xi_x_um_list[idx],2)) + ' $\\xi_y$ / $\mu$m = ' + str(round(xi_y_um_list[idx],2)), fontsize=12)

#                     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#                     textstr = 'corr=' + str(round(cor*100,2)) + '%'
#                     ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

                    if savefigure == True:
                        savefigure_dir = scratch_dir + str(settings_widget.label)
                        if os.path.isdir(savefigure_dir) == False:
                            os.mkdir(savefigure_dir)
                        #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                        savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                        if os.path.isdir(savefigure_dir) == False:
                            os.mkdir(savefigure_dir)
                        savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/profiles_rec2d_scan/'
                        if os.path.isdir(savefigure_dir) == False:
                            os.mkdir(savefigure_dir)
                        plt.savefig(savefigure_dir + '/' + 'profiles_rec2d_scan_' + hdf5_file_name_image \
                                + '_ph_'+str(ph) \
                                + '_d_'+str(separation_um) \
                                + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                                + '_image_'+str(imageid_loop) \
                                + '_sigmay_' +str(sigma_y_F_gamma_um_list[idx]) \
                                + '.png', dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None)
                        plt.savefig(savefigure_dir + '/' + 'profiles_rec2d_scan_' + hdf5_file_name_image \
                                + '_ph_'+str(ph) \
                                + '_d_'+str(separation_um) \
                                + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                                + '_image_'+str(imageid_loop) \
                                + '_sigmay_' +str(sigma_y_F_gamma_um_list[idx]) \
                                + '.pdf', dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None)
                        
#                         pl.dump(fig, file(savefigure_dir + '/' + 'profiles_rec2d_scan_' + hdf5_file_name_image \
#                                 + '_ph_'+str(ph) \
#                                 + '_d_'+str(separation_um) \
#                                 + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
#                                 + '_image_'+str(imageid_loop) \
#                                 + '_sigmay_' +str(sigma_y_F_gamma_um_list[idx]) \
#                                 + '.pickle', 'w'))
                
                    
                    
                

                sigma_y_F_gamma_um_min = sigma_y_F_gamma_um_list[index_opt] - 0.5
                sigma_y_F_gamma_um_max = sigma_y_F_gamma_um_list[index_opt] + 0.5
                sigma_y_F_gamma_um_stepsize = 0.1

                partiallycoherent = pixis_image_norm_dataset[imageid_loop]
                #pixis_profile_avg_dataset[imageid]
                (partiallycoherent_profile, fullycoherent_opt_list, fullycoherent_profile_opt_list,  partiallycoherent_rec_list, partiallycoherent_rec_profile_list, partiallycoherent_rec_profile_min_list,
                 delta_rec_min_list, delta_profiles_cropped_list, sigma_x_F_gamma_um_opt, sigma_y_F_gamma_um_list, F_gamma_list, abs_gamma_list, xi_x_um_list, xi_y_um_list, I_bp, dX_2, cor_list, cor_profiles_list, cor_profiles_cropped_list, index_opt) = deconvmethod(partiallycoherent, z, dX_1, pixis_avg_width, 
                 int(pixis_centery_px),_lambda_nm_fit[imageid_loop]*1e-9, sigma_x_F_gamma_um_min, sigma_x_F_gamma_um_max, sigma_y_F_gamma_um_min, sigma_y_F_gamma_um_max, sigma_y_F_gamma_um_stepsize, 200)
                
                chi2distance_list = []
                for partiallycoherent_rec in partiallycoherent_rec_list:
                    number_of_bins = 100
                    hist1, bin_edges1 = np.histogram(partiallycoherent.ravel(), bins=np.linspace(0,1,number_of_bins))
                    hist2, bin_edges2 = np.histogram(partiallycoherent_rec.ravel(), bins=np.linspace(0,1,number_of_bins))
                    chi2distance_list.append(chi2_distance(hist1, hist2))
    
                #index_opt = np.where(np.abs(np.asarray(delta_profiles_cropped_list)) == np.min(np.abs(np.asarray(delta_profiles_cropped_list))))[0][0]
                index_opt = np.where(np.asarray(chi2distance_list) == np.min(np.asarray(chi2distance_list)))[0][0]
                
                
                xi_um = xi_x_um_list[index_opt]
                xi_x_um = xi_x_um_list[index_opt]
                xi_y_um = xi_y_um_list[index_opt]
                
                print('sigma_x_F_gamma_um_opt='+str(sigma_x_F_gamma_um_opt))
                print('sigma_y_F_gamma_um_list[index_opt]='+str(sigma_y_F_gamma_um_list[index_opt]))
                print('xi_x_um_list[index_opt]='+str(xi_x_um_list[index_opt]))
                print('xi_y_um_list[index_opt]='+str(xi_y_um_list[index_opt]))

                fig, axs = plt.subplots(nrows=7,ncols=1, sharex=True, figsize=(5,15))
                ax = axs[0]
                ax.plot(sigma_y_F_gamma_um_list, cor_list)
                ax.set_ylabel('cor')

                ax = axs[1]
                ax.plot(sigma_y_F_gamma_um_list, cor_profiles_list)
                ax.set_ylabel('cor profiles')

                ax = axs[2]
                ax.plot(sigma_y_F_gamma_um_list, chi2distance_list)
                ax.set_ylabel('chi2distance')
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])

                ax = axs[3]
                ax.plot(sigma_y_F_gamma_um_list, delta_rec_min_list)
                ax.set_ylabel('delta minimum')

                ax = axs[4]
                ax.plot(sigma_y_F_gamma_um_list, delta_profiles_cropped_list)
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])
                ax.set_ylabel('delta profiles cropped')

                ax = axs[5]
                ax.plot(sigma_y_F_gamma_um_list, xi_x_um_list)
                ax.set_ylabel('xi_x')
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])

                ax = axs[6]
                ax.plot(sigma_y_F_gamma_um_list, xi_y_um_list)
                ax.set_ylabel('xi_y')
                ax.axvline(sigma_y_F_gamma_um_list[index_opt])

                fig.tight_layout()
                
                
                if savefigure == True:
                    savefigure_dir = scratch_dir + str(settings_widget.label)
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/profiles_rec2d/'
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    plt.savefig(savefigure_dir + '/' + 'yscan_b_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                    plt.savefig(savefigure_dir + '/' + 'yscan_b_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.pdf', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                
                #df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'sigma_F_gamma_um_opt'] = sigma_F_gamma_um_opt
                df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'xi_um'] = xi_um
                df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'xi_x_um'] = xi_x_um
                df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'xi_y_um'] = xi_y_um
                df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid'] == imageid_loop)), 'pixis_profile_centerx_px_fit'] = pixis_profile_centerx_px_fit[imageid_loop]
                
                if imageid_loop == -1:
                    sigma_F_gamma_um_opt_of_average = sigma_F_gamma_um_opt
                    xi_um_of_average = xi_um


                #A_bp = fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(np.sqrt(partiallycoherent))))  # amplitude
                #I_bp = np.abs(A_bp)**2  # intensity


                #imagesc(X2_axis*R_1,Y2_axis*R_1,log10(I_bp));


                #pixis_yshift_px = int(pixis_centery_px_widget.value)
                #p0 = (pixis_yshift_px, 400)
                #popt_gauss, pcov_gaussian = curve_fit(lambda x, m, w: gaussianbeam(x, 1, m ,w, 0), xdata, ydata, p0)
                #pixis_beamwidth_px = popt_gauss[1]

                xdata = list(range(n))

                fig=plt.figure(figsize=(48, 12), dpi= 80, facecolor='w', edgecolor='k')
                plt.subplot(1,4,1)
                plt.plot(xdata, partiallycoherent_profile, 'b-', label='measured partially coherent', linewidth=3)
                plt.plot(xdata, fullycoherent_profile_opt_list[index_opt], 'r-', label='recovered fully coherent')
                plt.plot(xdata, partiallycoherent_rec_profile_list[index_opt], 'g-', label='recovered partially coherent')
                #plt.plot(xdata, gaussianbeam(xdata, 1, popt_gauss[0] ,popt_gauss[1], 0), 'r-', label='fit: m=%5.1f px, w=%5.1f px' % tuple([popt_gauss[0] ,popt_gauss[1]]))
                plt.axhline(0, color='k')
                plt.legend()
                
                plt.title('coherence length $\\xi$ / $\mu$m = ' + str(round(xi_um,2)))

                plt.subplot(1,4,2)
                #plt.contourf(x,y,psf,cmap='jet')
                #plt.xlim(-5,5)
                #plt.ylim(-5,5)
                plt.imshow(F_gamma_list[index_opt],cmap='jet', extent=((-n/2)*dX_1, (+n/2-1)*dX_1, -n/2*dX_1, (+n/2-1)*dX_1))
                #plt.xlim(-5*dX_1,5*dX_1)
                #plt.ylim(-5*dX_1,5*dX_1)
                plt.title('$F(\\gamma)$ with $\sigma_x$ = ' + str(round(sigma_x_F_gamma_um_opt,2)) + '$\sigma_y$ = ' + str(round(sigma_y_F_gamma_um_list[index_opt],2)))


                plt.subplot(1,4,3)
                #plt.xlim(-5,5)
                #plt.ylim(-5,5)
                #plt.imshow(gamma,cmap='jet', extent=((-n/2)*dX_2, (+n/2-1)*dX_2, -n/2*dX_2, (+n/2-1)*dX_2))
                plt.contourf(abs_gamma_list[index_opt],cmap='jet', extent=((-n/2)*dX_2, (+n/2-1)*dX_2, -n/2*dX_2, (+n/2-1)*dX_2))

            
                n = abs_gamma_list[index_opt].shape[0]
                xdata = list(range(n))
                ydata = abs_gamma_list[index_opt][int(n/2),:]
                p0 = (int(n/2), 1)
                popt_gauss, pcov_gaussian = curve_fit(lambda x, m, w: gaussianbeam(x, 1, m ,w, 0), xdata, ydata, p0)
                plt.subplot(1,4,4)
                plt.plot(xdata, ydata, 'b-', label='abs(gamma)', linewidth=1)
                plt.plot(xdata, gaussianbeam(xdata, 1, popt_gauss[0] ,popt_gauss[1], 0), 'r-', label='fit: m=%5.1f px, w=%5.1f px' % tuple([popt_gauss[0] ,popt_gauss[1]]))
                plt.legend()
                
                
                if savefigure == True:
                    #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                    savefigure_dir = scratch_dir + str(settings_widget.label)
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/deconv_a/'
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    plt.savefig(savefigure_dir + '/' + 'deconv_a_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.png', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                
                
                #### only the  profiles
                n = partiallycoherent_profile.shape[0]
                
                xdata = np.linspace((-n/2)*dX_1*1e3, (+n/2-1)*dX_1*1e3, n)

                fig=plt.figure(figsize=(11.69,8.27), dpi= 300, facecolor='w', edgecolor='k')  # A4 sheet in landscape
                plt.subplot(1,1,1)
                plt.plot(xdata, partiallycoherent_profile, 'b-', label='measured partially coherent', linewidth=3)
                plt.plot(xdata, fullycoherent_profile_opt_list[index_opt], 'r-', label='recovered fully coherent', linewidth=1)
                #plt.plot(xdata, gaussianbeam(xdata, 1, popt_gauss[0] ,popt_gauss[1], 0), 'r-', label='fit: m=%5.1f px, w=%5.1f px' % tuple([popt_gauss[0] ,popt_gauss[1]]))
                plt.axhline(0, color='k')
                plt.xlabel('x / mm', fontsize = 14)
                plt.ylabel('Intensity / a.u.', fontsize = 14)
                plt.legend()
                
                plt.title('d / $\mu$m = '+str(int(separation_um)) + ' coherence length $\\xi_x$ / $\mu$m = ' + str(round(xi_x_um_list[index_opt],2)) + ' $\\xi_y$ / $\mu$m = ' + str(round(xi_y_um_list[index_opt],2)), fontsize=16)
             
                if savefigure == True:
                    savefigure_dir = scratch_dir + str(settings_widget.label)
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/profiles/'
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    plt.savefig(savefigure_dir + '/' + 'profiles_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                    plt.savefig(savefigure_dir + '/' + 'profiles_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.pdf', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)

                #plt.show()
                plt.close(fig)

                
                
                                #### only the  profiles
                n = partiallycoherent_profile.shape[0]
                
                xdata = np.linspace((-n/2)*dX_1*1e3, (+n/2-1)*dX_1*1e3, n)

                fig=plt.figure(figsize=(11.69,8.27), dpi= 300, facecolor='w', edgecolor='k')  # A4 sheet in landscape
                ax = plt.subplot(1,1,1)
                plt.plot(xdata, partiallycoherent_profile, 'b-', label='measured partially coherent', linewidth=1)
                plt.plot(xdata, fullycoherent_profile_opt_list[index_opt], 'r-', label='recovered fully coherent', linewidth=1)
                plt.plot(xdata, partiallycoherent_rec_profile_list[index_opt], 'g-', label='recovered partially coherent', linewidth=1)
                #plt.plot(xdata, gaussianbeam(xdata, 1, popt_gauss[0] ,popt_gauss[1], 0), 'r-', label='fit: m=%5.1f px, w=%5.1f px' % tuple([popt_gauss[0] ,popt_gauss[1]]))
                plt.axhline(0, color='k')
                plt.xlabel('x / mm', fontsize = 14)
                plt.ylabel('Intensity / a.u.', fontsize = 14)
                plt.legend()
                
                plt.title('d / $\mu$m = '+str(int(separation_um)) + ' coherence length $\\xi_x$ / $\mu$m = ' + str(round(xi_x_um_list[index_opt],2)) + ' $\\xi_y$ / $\mu$m = ' + str(round(xi_y_um_list[index_opt],2)), fontsize=16)
                
                #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                #textstr = 'corr=' + str(round(cor*100,2)) + '%'
                #ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
             
                if savefigure == True:
                    savefigure_dir = scratch_dir + str(settings_widget.label)
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/profiles_rec2d/'
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    plt.savefig(savefigure_dir + '/' + 'profiles_rec2d_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                    plt.savefig(savefigure_dir + '/' + 'profiles_rec2d_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.pdf', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)

                plt.show()
                #plt.close(fig)
                
                fig=plt.figure(figsize=(36, 16), dpi= 80, facecolor='w', edgecolor='k')
                plt.subplot(1,3,1)
                plt.imshow(partiallycoherent,origin='lower', interpolation='nearest', aspect=1, cmap='jet', vmin=0, vmax=1)
                plt.subplot(1,3,2)
                plt.imshow(fullycoherent_opt_list[index_opt],origin='lower', interpolation='nearest', aspect=1, cmap='jet', vmin=0, vmax=1)

                plt.subplot(1,3,3)
                #plt.xlim(-5,5)
                #plt.ylim(-5,5)
                #plt.imshow(gamma,cmap='jet', extent=((-n/2)*dX_2, (+n/2-1)*dX_2, -n/2*dX_2, (+n/2-1)*dX_2))
                n = partiallycoherent.shape[0]
                #plt.imshow(np.log10(I_bp),cmap='jet', extent=((-n/2)*dX_2, (+n/2-1)*dX_2, -n/2*dX_2, (+n/2-1)*dX_2))
                plt.imshow(np.log10(I_bp),cmap='jet')
                
                if savefigure == True:
                    savefigure_dir = scratch_dir + str(settings_widget.label)
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/deconv_b/'
                    if os.path.isdir(savefigure_dir) == False:
                        os.mkdir(savefigure_dir)
                    plt.savefig(savefigure_dir + '/' + 'deconv_b_' + hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) \
                            + '.png', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                plt.close(fig)
                
                
            
            
            # to be fixed:
            Error_min_idx = 0
        #     print(len(Error))
        #     if len(Error) > 0:
        #         Error_min_idx = np.where(Error==np.min(Error))[0][0]
        #     else:
        #         Error_min_idx=0
        #     #print('min. error=' + str(np.min(Error)) + 'at idx=' + str(Error_min_idx)
                  #+ ' for gamma=' + str(gamma_fit[Error_min_idx]))

            # Figure
            #fig = plt.figure(figsize=(48, 36))
            fig = plt.figure(figsize=(30, 18))
            

            gs = gridspec.GridSpec(4, 5, width_ratios=[1,1,1,1,1],height_ratios=[1, 2, 1, 2])
            gs.update(hspace=0.05)

            # Plot Profiles
            ax1 = plt.subplot(gs[0,0])
            if plot_lineout == True:
                ax1.scatter(list(range(pixis_profile.size)),pixis_profile,s=10,marker='+',color='c')
                ax1.scatter(np.where(pixis_profile<0)[0],pixis_profile[np.where(pixis_profile<0)[0]],s=10,marker='x',color='k')
            ax1.plot(list(range(pixis_profile_avg.size)),pixis_profile_avg, color='r')
            ax1.plot(list(range(pixis_profile_avg.size)),pixis_bg_std_avg, color='b')
           


            ax1.scatter(list(range(pixis_profile_avg.size)),pixis_profile_avg,s=15, color='r')

            if plot_simulation == True:
                ax1.plot(list(range(pixis_profile_avg.size)),I_D_normalized, color='k')
            if plot_LightPipes_simulation == True:
                ax1.plot(list(range(pixis_profile_avg.size)),Intlinex_LightPipes, 'g-')
            if plot_simulation_fit == True:
                ax1.plot(list(range(pixis_profile_avg.size)),I_D_normalized_fit, color='g')
            plt.ylim(-0.05,1)

            #plt.title(hdf5_file_name_image + 'cross section at y=' + str(centery) + ' \lambda=' + str(daq_parameter_image_dataset[0][imageid_widget.value]) + ' vs. ' + str(_lambda_nm) )

            ax1.hlines(0,0,pixis_profile_avg.size)

            # plot image

            ax2 = plt.subplot(gs[1,0], sharex=ax1)
            im_ax2 = plt.imshow(pixis_image_norm, origin='lower', interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)
            
            if plot_pixis_std_contour == True:
                if levels_auto == True:
                    CS = plt.contour(pixis_bg_std_norm,6,                                
                                 colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'))
                else:
                    levels = np.arange(levels_min,levels_max,levels_max-levels_min/6)
                    CS = plt.contour(pixis_bg_std_norm,6,                                
                                 colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'))
                plt.clabel(CS, fontsize=12, inline=1)
                CB = plt.colorbar(CS, ax=ax2, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            plot_center = True
            if plot_center == True:
                plt.scatter(pixis_centerx_px,pixis_centery_px, color='b')
                plt.scatter(pixis_cm_x_px,pixis_cm_y_px, color='r')

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            if plot_simulation_fit == True:
                #xdata = dx * np.linspace(-n/2+1, n/2, n)# coordinate
                n = len(pixis_profile_alongy)
                xdata = list(range(n))
                
                ax52 = plt.subplot(gs[1,4])
                ax52.plot(xdata, pixis_profile_alongy, 'b-', label='data')
                ax52.plot(xdata, gaussianbeam(xdata, 1, pixis_centery_px ,pixis_beamwidth_px, 0), 'r-', label='fit: m=%5.1f px, w=%5.1f px' % tuple([pixis_centery_px ,pixis_beamwidth_px]))
                ax52.legend()
            
            if do_deconvmethod == True:
                ax54 = plt.subplot(gs[3,4])
                n = partiallycoherent.shape[0]
                #plt.imshow(np.log10(I_bp),cmap='jet', extent=((-n/2)*dX_2, (+n/2-1)*dX_2, -n/2*dX_2, (+n/2-1)*dX_2))
                ax54.imshow(np.log10(I_bp),cmap='jet')
            
            
            # place a text box in upper left in axes coords
            if plot_simulation_fit == True:
                textstr = settings_widget.label + '\n' \
                        + 'imageid =' + str(imageid_loop) + '\n' \
                        + '$\lambda=$' + str(format(_lambda_nm, '.2f')) + '\n' \
                        + '$\lambda fit=$' + str(format(_lambda_nm_fit[imageid_loop], '.2f')) + '\n' \
                        + 'sep = ' + str(separation_um) + 'um' + '\n' \
                        + 'ap5,ap7 (image) = ' + str(aperture5_mm_image) + 'mm, ' + str(aperture7_mm_image) + 'mm' + '\n' \
                        + 'ap5,ap7 (bg) = ' + str(aperture5_mm_bg) + 'mm, ' + str(aperture7_mm_bg) + 'mm' + '\n' \
                        + 'pixis_centerx_px = ' + str(pixis_centerx_px)  + '\n' \
                        + 'pixis_profile_centerx_px_fit = ' + str(pixis_profile_centerx_px_fit[imageid_loop]) + '\n' \
                        + 'pixis_profile_centerx_px_fit_lower = ' + str(pixis_profile_centerx_px_fit_lower) + '\n' \
                        + 'pixis_profile_centerx_px_fit_upper = ' + str(pixis_profile_centerx_px_fit_upper) + '\n' \
                        + 'pixis_profile_centerx_px_fit_delta = ' + str(pixis_profile_centerx_px_fit_delta) + '\n' \
                        + 'w1_um = ' +str(w1_um) + ' w2_um = ' +str(w2_um) + '\n' + ' I_w1 = ' +str(I_w1) + ' I_w2 = ' +str(I_w2) + '\n' \
                        + 'gamma = ' +str(gamma) + '\n' \
                        + 'gamma_fit = ' +str(format(gamma_fit[imageid_loop], '.2f')) + '\n' \
                        + 'energy hall uJ = ' + str(format(energy_hall_uJ, '.2f')) + '\n' \
                        + 'pixis cts = ' + str(format(pixis_cts, '.2f'))
            else:
                textstr = settings_widget.label + '\n' \
                        + 'imageid =' + str(imageid_loop) + '\n' \
                        + '$\lambda=$' + str(format(_lambda_nm, '.2f')) + '\n' \
                        + 'sep = ' + str(separation_um) + 'um' + '\n' \
                        + 'ap5,ap7 (image) = ' + str(aperture5_mm_image) + 'mm, ' + str(aperture7_mm_image) + 'mm' + '\n' \
                        + 'ap5,ap7 (bg) = ' + str(aperture5_mm_bg) + 'mm, ' + str(aperture7_mm_bg) + 'mm' + '\n' \
                        + 'pixis_centerx_px = ' + str(pixis_centerx_px)  + '\n' \
                        + 'pixis_profile_centerx_px_fit = ' + str(pixis_profile_centerx_px_fit[imageid_loop]) + '\n' \
                        + 'w1_um = ' +str(w1_um) + ' w2_um = ' +str(w2_um) + '\n' + ' I_w1 = ' +str(I_w1) + ' I_w2 = ' +str(I_w2) + '\n' \
                        + 'gamma = ' +str(gamma) + '\n' \
                        + 'energy hall uJ = ' + str(format(energy_hall_uJ, '.2f')) + '\n' \
#                         + 'pixis cts = ' + str(format(pixis_cts, '.2f'))
            
            print('_lambda_nm=' + str(_lambda_nm))    
            print('_lambda_nm_fit=' + str(_lambda_nm_fit[imageid_loop]))
            print('energy_hall_uJ=' + str(energy_hall_uJ))
            print('pixis_cts=' + str(pixis_cts))
            
            if PlotTextbox == True: 
                ax1.text(0.01, 0.99, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

#             if plot_simulation == True or plot_simulation_fit == True:
#                 textstr = textstr + '\n' + '$\gamma=$' + str(format(gamma_fit[Error_min_idx], '.2f')) + ' (min error=' + str(format(np.min(Error), '.2f')) + ')'

                
            ax2.add_patch(
               patches.Rectangle(
                (0, pixis_centery_px-pixis_avg_width/2),
                pixis_profile.size,
                pixis_avg_width,
                color = 'w',
                linestyle = '--',
                fill=False      # remove background
                )
             )

            #to check the rotation:
            ax2.add_patch(
               patches.Rectangle(
                (0, pixis_lowery_px),
                pixis_profile.size,
                pixis_beamwidth_px,
                color = 'y',
                linestyle = '--',
                fill=False      # remove background
                )
             )

            #plt.title('background subtracted and rotation corrected (normalized)')

            if plot_pixis_std_contour == False:
                fig.colorbar(im_ax2, ax=ax2, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            #plt.xlim(0,pixis_profile.size)
            plt.xlim(xlim_lower,xlim_upper)

            #plt.tight_layout()

            plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)

            #fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)

            axisx = list(range(pixis_profile_avg.size))
            ### zoomed in plots ###


            ax21 = plt.subplot(gs[2,1])
            plt.setp(ax21, frame_on=False, xticks=(), yticks=())
            ax21.text(0.01, 0.95, textstr, transform=ax21.transAxes, fontsize=12, verticalalignment='top')

            ## column1 left zoom
            left = zoomborderleft
            right = zoomborderleft+zoomwidth

            ax11 = plt.subplot(gs[0,1])
            ax11.plot(axisx[left:right],pixis_profile_avg[left:right], color='r')
            ax11.scatter(axisx[left:right],pixis_profile_avg[left:right],s=15,marker='o', color='r')

            if plot_simulation == True:
                ax11.plot(axisx[left:right],I_D_normalized[left:right], color='k')
            if plot_simulation_fit == True:
                ax11.plot(axisx[left:right],I_D_normalized_fit[left:right], color='g')
            if plot_LightPipes_simulation == True:
                ax11.plot(axisx[left:right],Intlinex_LightPipes[left:right], 'g-')

            #plt.title(hdf5_file_name_image + 'cross section at y=' + str(centery) + ' \lambda=' + str(daq_parameter_image_dataset[0][imageid_widget.value]) + ' vs. ' + str(_lambda_nm) )

            ax11.hlines(0,0,pixis_profile_avg.size)

            # plot image

            ax21 = plt.subplot(gs[1,1], sharex=ax11)
            #im_ax21 = plt.imshow(pixis_image_norm, origin='lower', interpolation='nearest')
            im_ax21 = ax21.imshow(pixis_image_norm, origin='lower', interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)
            
            fig.colorbar(im_ax21, ax=ax21, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            ax21.add_patch(
               patches.Rectangle(
                (0, pixis_centery_px-pixis_avg_width/2),
                pixis_profile.size,
                pixis_avg_width,
                color = 'w',
                linestyle = '--',
                fill=False      # remove background
                )
            )

            ax11.set_ylim([-0.05,1])
            ax11.set_xlim([left,right])




            ## column2 center zoom
            left = int(pixis_centerx_px)-int(zoomwidth/2)
            right = int(pixis_centerx_px)+int(zoomwidth/2)
            
            #left = int(pixis_profile_centerx_px_fit[imageid_loop])-int(zoomwidth/2)
#             right = int(pixis_profile_centerx_px_fit[imageid_loop])+int(zoomwidth/2)

            ax12 = plt.subplot(gs[0,2])
            ax12.plot(axisx[left:right],pixis_profile_avg[left:right], color='r')
            ax12.scatter(axisx[left:right],pixis_profile_avg[left:right],s=15,marker='o', color='r')

            if plot_simulation == True:
                ax12.plot(axisx[left:right],I_D_normalized[left:right], color='k')
            if plot_simulation_fit == True:
                ax12.plot(axisx[left:right],I_D_normalized_fit[left:right], color='g')
            if plot_LightPipes_simulation == True:
                ax12.plot(axisx[left:right],Intlinex_LightPipes[left:right], 'g-')

            #plt.title(hdf5_file_name_image + 'cross section at y=' + str(centery) + ' \lambda=' + str(daq_parameter_image_dataset[0][imageid_widget.value]) + ' vs. ' + str(_lambda_nm) )

            ax12.hlines(0,0,pixis_profile_avg.size)

            # plot image

            ax22 = plt.subplot(gs[1,2], sharex=ax12)
            im_ax22 = ax22.imshow(pixis_image_norm, origin='lower', interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)
            
            fig.colorbar(im_ax22, ax=ax22, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            ax22.add_patch(
               patches.Rectangle(
                (0, pixis_centery_px-pixis_avg_width/2),
                pixis_profile.size,
                pixis_avg_width,
                color = 'w',
                linestyle = '--',
                fill=False      # remove background
                )
            )

            ax12.set_ylim([-0.05,1])
            ax12.set_xlim([left,right])

            ## column3 right zoom
            left = pixis_profile_avg.size-zoomborderright-zoomwidth
            right = pixis_profile_avg.size-zoomborderright

            ax13 = plt.subplot(gs[0,3])
            ax13.plot(axisx[left:right],pixis_profile_avg[left:right], color='r')
            ax13.scatter(axisx[left:right],pixis_profile_avg[left:right],s=15,marker='o', color='r')

            if plot_simulation == True:
                ax13.plot(axisx[left:right],I_D_normalized[left:right], color='k')
            if plot_simulation_fit == True:
                ax13.plot(axisx[left:right],I_D_normalized_fit[left:right], color='g')
            if plot_LightPipes_simulation == True:
                ax13.plot(axisx[left:right],Intlinex_LightPipes[left:right], 'g-')

            #plt.title(hdf5_file_name_image + 'cross section at y=' + str(centery) + ' \lambda=' + str(daq_parameter_image_dataset[0][imageid_widget.value]) + ' vs. ' + str(_lambda_nm) )

            ax13.hlines(0,0,pixis_profile_avg.size)

            # plot image

            ax23 = plt.subplot(gs[1,3], sharex=ax13)
            im_ax23 = plt.imshow(pixis_image_norm, origin='lower', interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)
            
            fig.colorbar(im_ax23, ax=ax23, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            ax23.add_patch(
               patches.Rectangle(
                (0, pixis_centery_px-pixis_avg_width/2),
                pixis_profile.size,
                pixis_avg_width,
                color = 'w',
                linestyle = '--',
                fill=False      # remove background
                )
            )

            ax13.set_ylim([-0.05,1])
            ax13.set_xlim([left,right])

            ### PINHOLES ####

            # Plot Profiles
            ax20 = plt.subplot(gs[2,0])
            #if plot_lineout == True:
            #    ax3.scatter(list(range(pinholes_profile.size)),pinholes_profile,s=10,marker='+',color='c')
            #ax3.scatter(np.where(pinholes_profile<0)[0],pinholes_profile[np.where(pinholes_profile<0)[0]],s=10,marker='x',color='k')
            ax20.plot(list(range(pinholes_profile_alongx_avg.size)),pinholes_profile_alongx_avg, color='r')


            ax20.scatter(list(range(pinholes_profile_alongx_avg.size)),pinholes_profile_alongx_avg,s=15,marker='o', color='r')


            #plt.title(hdf5_file_name_image + 'cross section at y=' + str(centery) + ' \lambda=' + str(daq_parameter_image_dataset[0][imageid_widget.value]) + ' vs. ' + str(_lambda_nm) )

            #ax1.hlines(0,0,pixis_profile_avg.size)

            # plot image

            pinholes_image_norm = pinholes_image_norm_dataset[imageid_loop]
            ax4 = plt.subplot(gs[3,0], sharex=ax20)
            im_ax4 = plt.imshow(pinholes_image_norm, origin='lower', interpolation='nearest', aspect='auto', cmap='nipy_spectral', vmin=0, vmax=1)
            
            plot_center = False
            #if plot_center == True:
            #    plt.scatter(centerx,centery, color='r')
            #ax4.scatter(pinholes_cm_x_px, pinholes_cm_y_px, c=list(df['energy hall']), marker='o', s=5, cmap='jet')
            #ax4.scatter(pinholes_cm_x_px[imageid_loop], pinholes_cm_y_px[imageid_loop], marker='x', s=10, color='k')

#             ax4.add_patch(
#                patches.Rectangle(
#                 (0, pinholes_centery_px-pinholes_avg_width_alongy/2),
#                 pinholes_profile_alongx.size,
#                 pinholes_avg_width_alongy,
#                 color = 'w',
#                 linestyle = '--',
#                 fill=False      # remove background
#                 )
#             )
#             ax4.add_patch(
#                patches.Ellipse(
#                 (ap5_x_px, ap5_y_px), 
#                 ap5_width_px, 
#                 ap5_height_px,
#                 color = 'w',
#                 linestyle = '--',
#                 fill=False      # remove background
#                 )
#             )    
#             ax4.add_patch(
#                patches.Ellipse(
#                 (screenhole_x_px, screenhole_y_px), 
#                 screenhole_width_px, 
#                 screenhole_height_px,
#                 color = 'w',
#                 linestyle = '--',
#                 fill=False      # remove background
#                 )
#             )

            #plt.title('background subtracted and rotation corrected (normalized)')

            fig.colorbar(im_ax4, ax=ax4, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            #plt.xlim(0,pixis_profile.size)
            #plt.xlim(xlim_lower,xlim_upper)

            #plt.tight_layout()

            plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)

            #fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)

            
            ### DATASET PLOTS ####
            
            ax31 = plt.subplot(gs[3,1])
        
            df.plot(kind='scatter', x='beam position hall horizontal pulse resolved', y='beam position hall vertical pulse resolved', c='xi_um', s=10, colormap='jet', ax=ax31)
            ax31.scatter(beamposx, beamposy, marker='x', s=60, color='k')

#             ax33.scatter(pinholes_cm_x_px,pinholes_cm_x_px, c=list(df['energy hall']), s=10, cmap='jet')
#             ax33.scatter(pinholes_cm_x_px[imageid_loop],pinholes_cm_y_px[imageid_loop], marker='x', s=60, color='k')

            ax31.set_aspect('equal')
            
            ax32 = plt.subplot(gs[3,2])
            
            

            df.plot(kind='scatter', x='beam position hall horizontal pulse resolved', y='beam position hall vertical pulse resolved', c='energy hall', s=10, colormap='jet', ax=ax32)
            ax32.scatter(beamposx, beamposy, marker='x', s=60, color='k')
            ax32.set_aspect('equal')

#             ax5.scatter(range(imageid_max),pixis_profile_centerx_px_fit, marker='.', color='r', label='pixis profile center / px')
#             ax5.scatter(imageid_loop,pixis_profile_centerx_px_fit[imageid_loop],  marker='.', color='k')
#             ax5.scatter(range(imageid_max),pinholes_cm_x_px, marker='o', color='b', label='pinholes CM x / px')
#             ax5.scatter(imageid_loop,pinholes_cm_x_px[imageid_loop], marker='o', color='k')
#             ax5.legend()
#             ax5.set_xlabel('image id')
            
            ax33 = plt.subplot(gs[3,3])
        
            df.plot(kind='scatter', x='pinholes_cm_x_px', y='pinholes_cm_y_px', c='energy hall', s=10, colormap='jet', ax=ax33)

            #ax33.scatter(pinholes_cm_x_px,pinholes_cm_x_px, c=list(df['energy hall']), s=10, cmap='jet')
            ax33.scatter(pinholes_cm_x_px[imageid_loop],pinholes_cm_y_px[imageid_loop], marker='x', s=60, color='k')
            ax33.set_xlabel('pinholes CM x / px')
            ax33.set_ylabel('pinholes CM y / px')
            ax33.set_aspect('equal')
            

            
            

            if savefigure == True:
                #savefigure_dir = scratch_dir + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
                
                savefigure_dir = scratch_dir + str(settings_widget.label)
                if os.path.isdir(savefigure_dir) == False:
                    os.mkdir(savefigure_dir)
                savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value))
                if os.path.isdir(savefigure_dir) == False:
                    os.mkdir(savefigure_dir)
                savefigure_dir = scratch_dir + str(settings_widget.label) + '/' + 'profilewidth_px_' + str(int(pixis_avg_width)) + '_' + 'bg_intervall_um_' + str(int(beamposition_horizontal_interval_widget.value)) + '/overview/'
                if os.path.isdir(savefigure_dir) == False:
                    os.mkdir(savefigure_dir)
                plt.savefig(savefigure_dir + '/'+ hdf5_file_name_image \
                        + '_ph_'+str(ph) \
                        + '_d_'+str(separation_um) \
                        + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                        + '_image_'+str(imageid_loop) \
                        + '.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
                
                
                
                # save to .mat for deconvolution in MATLAB
                pixis_image_norm_square = pixis_image_norm[0:min(pixis_image_norm.shape),0:min(pixis_image_norm.shape)]
                # Create a dictionary
                adict = {}
                adict['pixis_image_norm'] = pixis_image_norm_square
                adict['pinholes_image_norm'] = pinholes_image_norm
               #adict['wpg_image_norm_square'] = wpg_image_norm_square
                
                if imageid == -1:
                    sio.savemat(savefigure_dir + '/'+ hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_average'
                            + '.mat', adict)
                else:
                    sio.savemat(savefigure_dir + '/'+ hdf5_file_name_image \
                            + '_ph_'+str(ph) \
                            + '_d_'+str(separation_um) \
                            + '_E_' + str(format(energy_hall_uJ, '.4f')).zfill(6)  \
                            + '_image_'+str(imageid_loop) 
                            + '.mat', adict)
                
                #f = h5py.File(savefigure_dir + '.hdf5', 'w')
                
                

            #if savefigure_all_images == False:
                #plt.show(fig)
                #with plot_data_and_simulation_interactive_output:

                #display(fig)
                
                #plt.show(fig)
            
            clear_output(wait=True)
            display(fig)
            plt.close(fig)

        #pixis_image_avg_norm = np.average(pixis_image_norm_dataset, axis=0)
        #pixis_image_avg_norm = pixis_image_avg_norm / np.max(pixis_image_avg_norm)
        
#         if savefigure == True:

#             # save to .mat for deconvolution in MATLAB
#             import scipy.io as sio
#             # Create a dictionary
#             adict = {}
#             adict['pixis_image_norm'] = pixis_image_norm
#             adict['pixis_image_avg_norm'] = pixis_image_avg_norm
#             #adict['wpg_image_norm_square'] = wpg_image_norm_square
#             sio.savemat(savefigure_dir + '/'+ hdf5_file_name_image \
#                     + '_ph_'+str(ph) \
#                     + '_d_'+str(separation_um) \
#                     + '_E_' + str(format(energy_hall_uJ, '.2f')).zfill(6)  \
#                     + '_average'+ \
#                     + '.mat', adict)










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
    hdf5_file_name_bg_widget.value = dph_settings_widget.value[0]
    dataset_bg_args_widget.value = dph_settings_widget.value[1]
    hdf5_file_name_image_widget.value = dph_settings_widget.value[2]
    dataset_image_args_widget.value = dph_settings_widget.value[3]

def update_LightPipes(change):
    global Int_LightPipes, Intlinex_LightPipes, Intliney_LightPipes
    Int_LightPipes, Intlinex_LightPipes, Intliney_LightPipes = run_doublepinholes_LightPipes_interactive.result
    #plot_data_and_simulation_interactive.update()

widget_LightPipesUpdate = widgets.Button(value=False, description='Update LightPipes Lineout in plot below')
widget_LightPipesUpdate.on_click(update_LightPipes)

dph_settings_widget_layout = widgets.Layout(width='100%')
dph_settings_widget = widgets.Dropdown(options=dph_settings, layout = dph_settings_widget_layout)
dph_settings_widget.observe(update_settings, names='value')
display(dph_settings_widget)

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
