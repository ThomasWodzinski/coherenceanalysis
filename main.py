# <codecell>
# run current file in interactive window to see the frontend

from pathlib import Path  # see https://docs.python.org/3/library/pathlib.html#basic-use
import os.path

## Define paths
# Directory containing the data:
data_dir = Path('./data/')
# Directory containing the useful hdf5 files (cleaned)
useful_dir = Path('./data/useful/')
# Directory containing the background-subtracted hdf5 files
bgsubtracted_dir = Path('./data/bgsubtracted/')
# Directory for local temporary files:
scratch_dir = Path('./scratch/')
if os.path.isdir(scratch_dir) == False:
    if os.path.isdir(scratch_dir.parent.absolute()) == False:
        os.mkdir(scratch_dir.parent.absolute())    
    os.mkdir(scratch_dir)
results_dir = Path('./results/')
# prebgsubtracted_dir
# bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')

# <codecell>

# imports

# install missing packages --> see https://stackoverflow.com/a/63096701
import sys
import subprocess
import pkg_resources

required  = {'numpy', 'pandas', 'lmfit', 'wget', 'scipy', 'h5py', 'ipywidgets'} 
installed = {pkg.key for pkg in pkg_resources.working_set}
missing   = required - installed

if missing:
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])


import time
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.image as mpimg

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
    GridspecLayout
)
import ipywidgets as widgets
# import bqplot as bq

import h5py

import math
import scipy

import pandas as pd



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

from IPython.display import display, clear_output




from coherencefinder.deconvolution_module import calc_sigma_F_gamma_um, deconvmethod, deconvmethod_v1, normalize, chi2_distance
from coherencefinder.fitting_module import Airy, find_sigma, fit_profile_v1, fit_profile_v2, gaussian

# import pickle as pl

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


# %% settings for figures and latex

# download stixfonts with wget module if missing --> see https://stackoverflow.com/a/28313383

import wget
fonts_dir = './fonts'
if os.path.isfile(os.path.join(fonts_dir,'static_otf.zip')) == False:
    url='https://github.com/stipub/stixfonts/raw/master/zipfiles/static_otf.zip'
    if os.path.isdir(fonts_dir) == False:
        os.mkdir(fonts_dir)
    wget.download(url,out='./fonts')

# unzip --> see https://stackoverflow.com/a/3451150

import zipfile
if os.path.isdir(os.path.join(fonts_dir,'static_otf')) == False:
    with zipfile.ZipFile(os.path.join(fonts_dir,'static_otf.zip'), 'r') as zip_ref:
        zip_ref.extractall('./fonts/')

# add stixfonts -> see https://stackoverflow.com/a/65841091
from matplotlib import font_manager as fm
font_files = fm.findSystemFonts(fonts_dir)
for font_file in font_files:
    fm.fontManager.addfont(font_file)


# from
# # https://www.dmcdougall.co.uk/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib

# WIDTH = 350.0  # the number latex spits out
WIDTH = 379.41753  # optics express
# FACTOR = 0.45  # the fraction of the width you'd like the figure to occupy
FACTOR = 0.9  # the fraction of the width you'd like the figure to occupy
# FACTOR = 1  # the fraction of the width you'd like the figure to occupy
fig_width_pt  = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * golden_ratio   # figure height in inches
fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list

# adapted from https://tex.stackexchange.com/questions/391074/how-to-use-the-siunitx-package-within-python-matplotlib?noredirect=1 to make siunitx work with pdf

rcparams_with_latex_stix = {                      # setup matplotlib to use latex for output 
    "mathtext.fontset": 'stix', 
    "font.family": "serif",
    "font.serif": ['STIX Two Text'],                   # not working in texmode, just uses cm
    "font.sans-serif": ['Helvetica'],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 9,               # LaTeX default is 10pt font.
    "font.size": 9,
    "legend.fontsize": 8,               # Make the legend/label fonts 
    "xtick.labelsize": 8,               # a little smaller
    "ytick.labelsize": 8,
    "figure.figsize": fig_dims,     # default fig size of 0.9 textwidth
    "figure.dpi": 300, 
    "text.latex.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts 
        r"\usepackage[T1]{fontenc}",        # plots will be generated
        r'\usepackage{lmodern}', # otherwise savefig to pdf will produce an error!!
        r"\usepackage[detect-all,locale=US]{siunitx}",
#         r"\usepackage{amsmath}",
#         r"\usepackage{stix}"
#         r"\usepackage{stix2-type1}"
        ],                                   # using this preamble
    "text.usetex": True,                # use LaTeX to write all text
    }


#mpl.rcParams.update(pgf_with_latex)
# mpl.rcParams.update(rcparams_with_latex_stix)

# adapted from https://tex.stackexchange.com/questions/391074/how-to-use-the-siunitx-package-within-python-matplotlib?noredirect=1 to make siunitx work with pdf

rcparams_without_latex = {                      # setup matplotlib to use latex for output    
    "mathtext.fontset": 'stix', 
    "font.family": "serif",
    "font.serif": ['STIX Two Text'],                   # blank entries should cause plots 
    "font.sans-serif": ['stixsans'],              # to inherit fonts from the document    
    "font.monospace": [],
    "axes.labelsize": 9,               # LaTeX default is 10pt font.
    "font.size": 9,
    "legend.fontsize": 8,               # Make the legend/label fonts 
    "xtick.labelsize": 8,               # a little smaller
    "ytick.labelsize": 8,
    "figure.figsize": fig_dims,     # default fig size of 0.9 textwidth
    "figure.dpi": 150, 
    "text.usetex": False,                # use LaTeX to write all text
    }


#mpl.rcParams.update(pgf_with_latex)
mpl.rcParams.update(rcparams_without_latex)

# %%

# move this to a common module:
def get_sep_and_orient(pinholes):
        pinholes = pinholes[0:2]
        choices = {'1a': (50, 'vertical'), '1b': (707, 'vertical'),'1c': (50, 'horizontal'), '1d': (707, 'horizontal'),
                   '2a': (107, 'vertical'), '2b': (890, 'vertical'), '2c': (107, 'horizontal'), '2d': (890, 'horizontal'),
                   '3a': (215, 'vertical'), '3b': (1047, 'vertical'), '3c': (215, 'horizontal'), '3d': (1047, 'horizontal'),
                   '4a': (322, 'vertical'), '4b': (1335, 'vertical'), '4c': (322, 'horizontal'), '4d': (1335, 'horizontal'),
                   '5a': (445, 'vertical'), '5b': (1570, 'vertical'), '5c': (445, 'horizontal'), '5d': (1570, 'horizontal')}
        (sep, orient) = choices.get(pinholes,(np.nan,'bg'))
        return sep, orient



"""# Load dph settings and combinations"""

datasets_py_file = str(Path.joinpath(data_dir, "datasets.py"))
# datasets_py_file = str(Path.joinpath(data_dir, "datasets_deconvolution_failing.py"))
# datasets_py_file = str(Path.joinpath(data_dir, "datasets_fitting_failing.py"))

# Commented out IPython magic to ensure Python compatibility.
# %run -i $dph_settings_py # see https://stackoverflow.com/a/14411126 and http://ipython.org/ipython-doc/dev/interactive/magics.html#magic-run
# see also https://stackoverflow.com/questions/4383571/importing-files-from-different-folder to import as a module,
# requires however that it is located in a folder with an empty __init__.py
exec(open(datasets_py_file).read())

dph_settings_py_file = str(Path.joinpath(data_dir, "dph_settings.py"))

# Commented out IPython magic to ensure Python compatibility.
# %run -i $dph_settings_py # see https://stackoverflow.com/a/14411126 and http://ipython.org/ipython-doc/dev/interactive/magics.html#magic-run
# see also https://stackoverflow.com/questions/4383571/importing-files-from-different-folder to import as a module,
# requires however that it is located in a folder with an empty __init__.py
exec(open(dph_settings_py_file).read())

# %%
# import sys
# sys.path.append('g:\\My Drive\\PhD\\coherence\data\\dph_settings_package\\')
# from dph_settings_package import dph_settings_module


datasets_widget_layout = widgets.Layout(width="30%")
datasets_widget = widgets.Dropdown(options=list(datasets), layout=datasets_widget_layout, description='Dataset:')
# settings_widget.observe(update_settings, names='value')
# display(dph_settings_widget)
# initialize a dictionary holding a selection of measurements

datasets_selection_py_files = sorted(list(data_dir.glob("datasets_selection*.py")), reverse=True)
datasets_selection_py_files_widget_layout = widgets.Layout(width="30%")
datasets_selection_py_files_widget = widgets.Dropdown(
    options=datasets_selection_py_files,
    value=datasets_selection_py_files[0], # use newest available file per default
    layout=datasets_selection_py_files_widget_layout,
    description='Datasets selection file::'    
)

datasets_selection_py_file = datasets_selection_py_files_widget.value
if os.path.isfile(datasets_selection_py_file):
    exec(open(datasets_selection_py_file).read())

create_new_datasets_selection_py_file_widget = widgets.ToggleButton(
    value=False,
    description='create new file',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save df_fits to new csv file',
    icon='check'
)

def create_new_datasets_selection_py_file(change):
    datasets_selection_py_file = datasets_selection_py_files_widget.value
    new_datasets_selection_py_file = Path.joinpath(data_dir,str('datasets_selection_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.py'))
    with open(datasets_selection_py_file) as f:
        text = f.read()
    with open(new_datasets_selection_py_file,'w') as f:
        f.write(text)
    datasets_selection_py_files = sorted(list(data_dir.glob("datasets_selection*.py")), reverse=True)
    datasets_selection_py_files_widget.options = datasets_selection_py_files
    datasets_selection_py_files_widget.value = datasets_selection_py_files[0]

    create_new_datasets_selection_py_file_widget.value = False

create_new_datasets_selection_py_file_widget.observe(create_new_datasets_selection_py_file, names='value')



# else: 
#     datasets_selection = datasets.copy()

# dph_settings_widget_layout = widgets.Layout(width="100%")
# dph_settings_widget = widgets.Dropdown(options=dph_settings, layout=dph_settings_widget_layout)
# settings_widget.observe(update_settings, names='value')
# display(dph_settings_widget)

# dph_settings_bgsubtracted = list(bgsubtracted_dir.glob("*.h5"))
dph_settings_bgsubtracted = []
for pattern in ['*'+ s + '.h5' for s in datasets[datasets_widget.value]]: 
    dph_settings_bgsubtracted.extend(bgsubtracted_dir.glob(pattern))


dph_settings_bgsubtracted_widget_layout = widgets.Layout(width="50%")
dph_settings_bgsubtracted_widget = widgets.Dropdown(
    options=dph_settings_bgsubtracted,
    layout=dph_settings_bgsubtracted_widget_layout,
    description='Measurement:'
    # value=dph_settings_bgsubtracted[3],  # workaround, because some hdf5 files have no proper timestamp yet
)
# settings_widget.observe(update_settings, names='value')

measurements_selection_files = []
for pattern in ['*'+ s + '.h5' for s in datasets_selection[datasets_widget.value]]: 
    measurements_selection_files.extend(bgsubtracted_dir.glob(pattern))

measurements_selection_widget_layout = widgets.Layout(width="100%")
measurements_selection_widget = widgets.SelectMultiple(
    options=dph_settings_bgsubtracted,
    value=measurements_selection_files,
    layout=measurements_selection_widget_layout,
    description='Measurement:'
    # value=dph_settings_bgsubtracted[3],  # workaround, because some hdf5 files have no proper timestamp yet
)




# just hdf5_filename_bg_subtracted so we can use it to search in the dataframe
# dph_settings_bgsubtracted_widget.value.name

# how to get the hdf5_filename ?

# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
#     hdf5_file_useful_name = hdf5_file["/hdf5_file_useful_name"][0]
#     print(hdf5_file_useful_name)


"""# Load dataframes from csv"""

# dataframe of extracted from all available useful hdf5 files
df_all = pd.read_csv(Path.joinpath(data_dir, "df_all.csv"), index_col=0)
# maybe rename to df_hdf5_files? and then use df instead of df0?
df_all["imageid"] = df_all.index

# dataframe based on the dph_settings dictionary inside dph_settings.py

# del df_settings

dph_settings_keys = []
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
    dph_settings_keys.append(list(dph_settings.keys())[idx])
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
        "dph_settings": dph_settings_keys,
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

# dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]
df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]


# df_settings

# merge dataframe of hdf5files with dataframe of settings
df_temp = []
df_temp = pd.merge(df_all, df_settings)
df_temp["timestamp_pulse_id"] = df_temp["timestamp_pulse_id"].astype("int64")
# store this instead of df_all?

# definition of fits header columns
# needed in case we want to add new columns?

# preparation parameter and results
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
]

fits_header_list2 = [
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
   ]

# CDC results
fits_header_list3 = [
    "xi_um_fit",
    "xi_x_um_fit", 
    "xi_y_um_fit", 
    "zeta_x",
    "zeta_x_fit",
    "zeta_y",
    "zeta_y_fit", 
]

# fitting parameter
fits_header_list4 = [
    'pixis_profile_avg_width',
    'crop_px',
    'shiftx_um',
    'shiftx_um_range_0',
    'shiftx_um_range_1',
    'shiftx_um_do_fit',
    'wavelength_nm',
    'wavelength_nm_range_0',
    'wavelength_nm_range_1',
    'wavelength_nm_do_fit',
    'z_mm',
    'z_mm_range_0',
    'z_mm_range_1',
    'z_mm_do_fit',
    'd_um',
    'd_um_range_0',
    'd_um_range_1',
    'd_um_do_fit',
    'gamma',
    'gamma_range_0',
    'gamma_range_1',
    'gamma_do_fit',
    'w1_um',
    'w1_um_range_0',
    'w1_um_range_1',
    'w1_um_do_fit',
    'w2_um',
    'w2_um_range_0',
    'w2_um_range_1',
    'w2_um_do_fit',
    'I_Airy1',
    'I_Airy1_range_0',
    'I_Airy1_range_1',
    'I_Airy1_do_fit',
    'I_Airy2',
    'I_Airy2_range_0',
    'I_Airy2_range_1',
    'I_Airy2_do_fit',
    'x1_um',
    'x1_um_range_0',
    'x1_um_range_1',
    'x1_um_do_fit',
    'x2_um',
    'x2_um_range_0',
    'x2_um_range_1',
    'x2_um_do_fit',
    'normfactor',
    'normfactor_range_0',
    'normfactor_range_1',
    'normfactor_do_fit'
    ]

fits_header_list4_v1 = []
for header in fits_header_list4:
    fits_header_list4_v1.append(header + '_v1')

# fitting parameter of version 2
fits_header_list5 = [
    'mod_sigma_um',
    'mod_sigma_um_range_0',
    'mod_sigma_um_range_1',
    'mod_sigma_um_do_fit',
    'mod_shiftx_um',
    'mod_shiftx_um_range_0',
    'mod_shiftx_um_range_1',
    'mod_shiftx_um_do_fit'
]

# fitting results
fits_header_list6a = [
    "shiftx_um_fit",
    "wavelength_nm_fit",
    "z_mm_fit",
    "d_um_fit",
    "d_um_at_detector", # extra?
    "gamma_fit",
    "w1_um_fit",
    "w2_um_fit",
    "I_Airy1_fit",
    "I_Airy2_fit",
    "x1_um_fit",
    "x2_um_fit",
    'chi2distance_fitting'
]

fits_header_list6a_v1 = []
for header in fits_header_list6a:
    fits_header_list6a_v1.append(header + '_v1')

# fitting results
fits_header_list6b = [
    # fitting results of version 2
    'mod_sigma_um_fit',
    'mod_shiftx_um_fit',
    'gamma_fit_v2', # at center
    'xi_um_fit_v2' # at center
]

# deconvolution parameter
fits_header_list7 = [ 
    'balance',
    'snr_db',
    'pixis_profile_avg_width',
    'crop_px',
    'xi_um_guess',
    'sigma_x_F_gamma_um_multiplier',
    'xatol'
]

# deconvolution parameter v1
fits_header_list7_v1 = [
    'pixis_profile_avg_width',
    'crop_px',
    'sigma_x_F_gamma_um_min', 
    'sigma_x_F_gamma_um_max',
    'sigma_x_F_gamma_um_stepsize',
    'sigma_y_F_gamma_um_min', 
    'sigma_y_F_gamma_um_max', 
    'sigma_y_F_gamma_um_stepsize'
]

# deconvolution_1d results
fits_header_list8 = [   
    "xi_um",
    "chi2distance_deconvmethod_1d"
]

# deconvolution_2d results
fits_header_list9 = [   
    "sigma_F_gamma_um_opt",
    "xi_x_um",
    "xi_y_um",
    "chi2distance_deconvmethod_2d"
]
fits_header_list9_v1 = []
for header in fits_header_list9:
    fits_header_list9_v1.append(header + '_v1')

fits_header_list8_v2 = []
for header in fits_header_list9:
    fits_header_list8_v2.append(header + '_v2')

fits_header_list9_v2 = []
for header in fits_header_list9:
    fits_header_list9_v2.append(header + '_v2')

fits_header_list8_v3 = []
for header in fits_header_list9:
    fits_header_list8_v3.append(header + '_v3')

fits_header_list9_v3 = []
for header in fits_header_list9:
    fits_header_list9_v1.append(header + '_v3')


fits_header_list = fits_header_list1 + fits_header_list2 + fits_header_list3 + fits_header_list4 + fits_header_list5 + fits_header_list6a + fits_header_list6b + fits_header_list7 + fits_header_list8 + fits_header_list9


# fits_header_list1 already exists in saved csv, only adding fits_header_list2, only initiate when
initiate_df_fits = True
# if initiate_df_fits == True:
    # df0 = df0.reindex(columns = df0.columns.tolist() + fits_header_list)
    # df_fits = df0[['timestamp_pulse_id'] + fits_header_list]

# load saved df_fits from csv
df_fits_csv_files = sorted(list(results_dir.glob("df_fits*.csv")), reverse=True) # newest on top
df_fits_csv_file = df_fits_csv_files[0] # use the newest
df_fits = pd.read_csv(df_fits_csv_file, index_col=0)
df_fits_clean = df_fits[df_fits["pixis_rotation"].notna()].drop_duplicates()
df_fits = df_fits_clean
df_fits = df_fits.reindex(columns = df_fits.columns.tolist() + list(set(fits_header_list) - set(df_fits.columns.tolist())) )
df0 = pd.merge(df_temp, df_fits, on="timestamp_pulse_id", how="outer")


# %% default values per measurement

measurement_arr = []
dataset_arr = []

for dataset in list(datasets):
    for measurement in datasets[dataset]:
        measurement_arr.append(measurement)
        dataset_arr.append(dataset)




df_fitting_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

df_fitting_v1_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

df_deconvmethod_2d_v1_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

df_deconvmethod_1d_v2_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

df_deconvmethod_2d_v2_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

df_deconvmethod_1d_v3_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

df_deconvmethod_2d_v3_measurement_default = pd.DataFrame({'dataset' : dataset_arr,
                                    'measurement' : measurement_arr})

fitting_measurement_default_headers = []
for header in fits_header_list4 + fits_header_list5:
    fitting_measurement_default_headers.append(header + '_measurement_default')

fitting_v1_measurement_default_headers = []
for header in fits_header_list4_v1:
    fitting_v1_measurement_default_headers.append(header + '_measurement_default')

deconvmethod_2d_v1_measurement_default_headers = []
for header in fits_header_list7_v1:
    deconvmethod_2d_v1_measurement_default_headers.append(header + '_measurement_default')

deconvmethod_1d_v2_measurement_default_headers = []
for header in fits_header_list7:
    deconvmethod_1d_v2_measurement_default_headers.append(header + '_measurement_default')

deconvmethod_2d_v2_measurement_default_headers = []
for header in fits_header_list7:
    deconvmethod_2d_v2_measurement_default_headers.append(header + '_measurement_default')

deconvmethod_1d_v3_measurement_default_headers = []
for header in fits_header_list7:
    deconvmethod_1d_v3_measurement_default_headers.append(header + '_measurement_default')

deconvmethod_2d_v3_measurement_default_headers = []
for header in fits_header_list7:
    deconvmethod_2d_v3_measurement_default_headers.append(header + '_measurement_default')


df_fitting_measurement_default = df_fitting_measurement_default.reindex(columns = df_fitting_measurement_default.columns.tolist() + list(set(fitting_measurement_default_headers) - set(df_fitting_measurement_default.columns.tolist())) )
df_fitting_v1_measurement_default = df_fitting_v1_measurement_default.reindex(columns = df_fitting_v1_measurement_default.columns.tolist() + list(set(fitting_v1_measurement_default_headers) - set(df_fitting_v1_measurement_default.columns.tolist())) )
df_deconvmethod_2d_v1_measurement_default = df_deconvmethod_2d_v1_measurement_default.reindex(columns = df_deconvmethod_2d_v1_measurement_default.columns.tolist() + list(set(deconvmethod_2d_v1_measurement_default_headers) - set(df_deconvmethod_2d_v1_measurement_default.columns.tolist())) )
df_deconvmethod_1d_v2_measurement_default = df_deconvmethod_1d_v2_measurement_default.reindex(columns = df_deconvmethod_1d_v2_measurement_default.columns.tolist() + list(set(deconvmethod_1d_v2_measurement_default_headers) - set(df_deconvmethod_1d_v2_measurement_default.columns.tolist())) )
df_deconvmethod_2d_v2_measurement_default = df_deconvmethod_2d_v2_measurement_default.reindex(columns = df_deconvmethod_2d_v2_measurement_default.columns.tolist() + list(set(deconvmethod_2d_v2_measurement_default_headers) - set(df_deconvmethod_2d_v2_measurement_default.columns.tolist())) )
df_deconvmethod_1d_v3_measurement_default = df_deconvmethod_1d_v3_measurement_default.reindex(columns = df_deconvmethod_1d_v3_measurement_default.columns.tolist() + list(set(deconvmethod_1d_v3_measurement_default_headers) - set(df_deconvmethod_1d_v3_measurement_default.columns.tolist())) )
df_deconvmethod_2d_v3_measurement_default = df_deconvmethod_2d_v3_measurement_default.reindex(columns = df_deconvmethod_2d_v3_measurement_default.columns.tolist() + list(set(deconvmethod_2d_v3_measurement_default_headers) - set(df_deconvmethod_2d_v3_measurement_default.columns.tolist())) )



# store also 'measurement' into df_fits to be able to cross-correlate!

df_measurement_default_file = Path.joinpath(results_dir, 'df_fitting_v2_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_fitting_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

df_measurement_default_file = Path.joinpath(results_dir, 'df_fitting_v1_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_fitting_v1_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v1_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_deconvmethod_2d_v1_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v2_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_deconvmethod_1d_v2_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v2_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_deconvmethod_2d_v2_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v3_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_deconvmethod_1d_v3_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v3_measurement_default.csv')
if os.path.isfile(df_measurement_default_file):
    df_deconvmethod_2d_v3_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)




df_fitting_v1_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + fits_header_list4 + fits_header_list5 + fits_header_list6a_v1 )
df_fitting_v2_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + fits_header_list4 + fits_header_list5 + fits_header_list6a + fits_header_list6b )

df_deconvmethod_2d_v1_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + fits_header_list7_v1 + fits_header_list9_v1)
df_deconvmethod_1d_v2_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + list(set(fits_header_list7) - set(['xatol'])) + fits_header_list8_v2) 
# create fits_header_list_v2 and v3????
df_deconvmethod_2d_v2_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + fits_header_list7 + fits_header_list9_v2)
df_deconvmethod_1d_v3_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + list(set(fits_header_list7) - set(['xatol'])) + fits_header_list8_v3)
df_deconvmethod_2d_v3_results = pd.DataFrame(columns=['measurement','timestamp_pulse_id','imageid','separation_um'] + fits_header_list7 + fits_header_list9_v3)


# %%
# creating frontend


# Widget definitions

n = 1024  # number of sampling point  # number of pixels

fittingprogress_widget = widgets.IntProgress(
    value=0,
    min=0,
    max=10,
    step=1,
    description="Progress:",
    bar_style="success",  # 'success', 'info', 'warning', 'danger' or ''
    orientation="horizontal",
)

statustext_widget = widgets.Text(value="", placeholder="status", description="", disabled=False)

do_plot_fitting_v1_widget = widgets.Checkbox(value=False, description='', tooltip="do_fitting_v1", disabled=False, indent = False, layout=widgets.Layout(width='auto'))
do_plot_fitting_v2_widget = widgets.Checkbox(value=False, description='', tooltip="do_fitting_v2", disabled=False, indent = False, layout=widgets.Layout(width='auto'))

do_plot_deconvmethod_2d_v1_widget = widgets.Checkbox(value=False, description='', tooltip="deconvmethod_2d_v1", disabled=False, indent = False, layout=widgets.Layout(width='auto'))
do_plot_deconvmethod_1d_v2_widget = widgets.Checkbox(value=False, description='', tooltip="deconvmethod_1d_v2", disabled=False, indent = False, layout=widgets.Layout(width='auto'))
do_plot_deconvmethod_2d_v2_widget = widgets.Checkbox(value=False, description='', tooltip="deconvmethod_2d_v2", disabled=False, indent = False, layout=widgets.Layout(width='auto'))
do_plot_deconvmethod_1d_v3_widget = widgets.Checkbox(value=False, description='', tooltip="deconvmethod_1d_v3", disabled=False, indent = False, layout=widgets.Layout(width='auto'))
do_plot_deconvmethod_2d_v3_widget = widgets.Checkbox(value=False, description='', tooltip="deconvmethod_2d_v3", disabled=False, indent = False, layout=widgets.Layout(width='auto'))




timestamp_pulse_id_widget_layout = widgets.Layout(width="auto")
timestamp_pulse_id_widget = widgets.Dropdown(
    options=[],
    description="timestamp_pulse_id:",
    disabled=False,
    layout=timestamp_pulse_id_widget_layout,
    indent = False
)

imageid_widget_layout = widgets.Layout(width="auto")
imageid_widget = widgets.Dropdown(
    options=[],
    description="imageid:",
    disabled=False,
    layout=imageid_widget_layout,
    indent = False
)

imageid_index_widget_layout = widgets.Layout(width="auto")
imageid_index_widget = widgets.BoundedIntText(
    options=[],
    description="idx",
    disabled=False,
    layout=imageid_index_widget_layout,
    indent = False
)

savefigure_profile_fit_widget = widgets.Checkbox(value=False, description="savefigure", disabled=False)


# dataframe and csv widgets

save_to_df_widget = widgets.Checkbox(value=False, description="save_to_df", disabled=False)
load_from_df_widget = widgets.Checkbox(value=False, description="load_from_df", disabled=False)



df_fits_csv_files = sorted(list(results_dir.glob("df_fits*.csv")), reverse=True)
df_fits_csv_files_widget_layout = widgets.Layout(width="50%")
df_fits_csv_files_widget = widgets.Dropdown(
    options=df_fits_csv_files,
    value=df_fits_csv_files[0], # use newest available file per default
    layout=df_fits_csv_files_widget_layout,
    description='csv file:'    
)

scan_for_df_fits_csv_files_widget = widgets.ToggleButton(
    value=False,
    description='scan for df_fits*.csv',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='scan for df_fits*.csv',
    icon='check'
)

def update_df_fits_csv_files_widget(change):
    df_fits_csv_files = sorted(list(results_dir.glob("df_fits*.csv")), reverse=True)
    df_fits_csv_files_widget.options=df_fits_csv_files
    scan_for_df_fits_csv_files_widget.value = False
scan_for_df_fits_csv_files_widget.observe(update_df_fits_csv_files_widget)


load_csv_to_df_widget = widgets.ToggleButton(
    value=False,
    description='csv-->df',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='load from csv to dataframe',
    icon='check'
)


def update_load_csv_to_df_widget(change):
    global df0
    global df_fitting_v1_results
    global df_fitting_v2_results
    # global df_deconvmethod_1d_v1_results
    global df_deconvmethod_2d_v1_results
    global df_deconvmethod_1d_v2_results
    global df_deconvmethod_2d_v2_results
    global df_deconvmethod_1d_v3_results
    global df_deconvmethod_2d_v3_results

    df_fits_csv_file = df_fits_csv_files_widget.value
    df_fits = pd.read_csv(df_fits_csv_file, index_col=0)
    df_fits_clean = df_fits[df_fits["pixis_rotation"].notna()].drop_duplicates()
    df_fits = df_fits_clean
    df0 = pd.merge(df_temp, df_fits, on="timestamp_pulse_id", how="outer")

    datestring = os.path.splitext(os.path.basename(df_fits_csv_files_widget.value))[0].split('df_fits_')[1]
    df_fitting_v1_results_file = Path.joinpath(results_dir,str('df_fitting_v1_results_'+datestring+'.csv'))
    df_fitting_v1_results = pd.read_csv(df_fitting_v1_results_file, index_col=0)
    df_fitting_v2_results_file = Path.joinpath(results_dir,str('df_fitting_v2_results_'+datestring+'.csv'))
    df_fitting_v2_results = pd.read_csv(df_fitting_v2_results_file, index_col=0)
    # df_deconvmethod_1d_v1_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v1_results_'+datestring+'.csv'))
    # df_deconvmethod_1d_v1_results = pd.read_csv(df_deconvmethod_1d_v1_results_file, index_col=0)
    df_deconvmethod_2d_v1_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v1_results_'+datestring+'.csv'))
    df_deconvmethod_2d_v1_results = pd.read_csv(df_deconvmethod_2d_v1_results_file, index_col=0)
    df_deconvmethod_1d_v2_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v2_results_'+datestring+'.csv'))
    df_deconvmethod_1d_v2_results = pd.read_csv(df_deconvmethod_1d_v2_results_file, index_col=0)
    df_deconvmethod_2d_v2_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v2_results_'+datestring+'.csv'))
    df_deconvmethod_2d_v2_results = pd.read_csv(df_deconvmethod_2d_v2_results_file, index_col=0)
    df_deconvmethod_1d_v3_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v3_results_'+datestring+'.csv'))
    df_deconvmethod_1d_v3_results = pd.read_csv(df_deconvmethod_1d_v3_results_file, index_col=0)
    df_deconvmethod_2d_v3_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v3_results_'+datestring+'.csv'))
    df_deconvmethod_2d_v3_results = pd.read_csv(df_deconvmethod_2d_v3_results_file, index_col=0)

    load_csv_to_df_widget.value = False

load_csv_to_df_widget.observe(update_load_csv_to_df_widget)

df_fits_csv_save_widget = widgets.ToggleButton(
    value=False,
    description='df-->csv',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save df_fits to csv',
    icon='check'
)


def update_df_fits_csv_save_widget(change):
    if df_fits_csv_save_widget.value == True:
        df_fits = df0[['timestamp_pulse_id'] + list(set(fits_header_list) - set(['chi2distance_fitting', 'chi2distance_deconvmethod_1d', 'chi2distance_deconvmethod_2d']))]
        df_fits_csv_file = df_fits_csv_files_widget.value
        df_fits.to_csv(df_fits_csv_file)

        datestring = os.path.splitext(os.path.basename(df_fits_csv_files_widget.value))[0].split('df_fits_')[1]
        df_fitting_v1_results_file = Path.joinpath(results_dir,str('df_fitting_v1_results_'+datestring+'.csv'))
        df_fitting_v1_results.to_csv(df_fitting_v1_results_file)
        df_fitting_v2_results_file = Path.joinpath(results_dir,str('df_fitting_v2_results_'+datestring+'.csv'))
        df_fitting_v2_results.to_csv(df_fitting_v2_results_file)
        # df_deconvmethod_1d_v1_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v1_results_'+datestring+'.csv'))
        # df_deconvmethod_1d_v1_results.to_csv(df_deconvmethod_1d_v1_results_file)
        df_deconvmethod_2d_v1_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v1_results_'+datestring+'.csv'))
        df_deconvmethod_2d_v1_results.to_csv(df_deconvmethod_2d_v1_results_file)
        df_deconvmethod_1d_v2_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v2_results_'+datestring+'.csv'))
        df_deconvmethod_1d_v2_results.to_csv(df_deconvmethod_1d_v2_results_file)
        df_deconvmethod_2d_v2_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v2_results_'+datestring+'.csv'))
        df_deconvmethod_2d_v2_results.to_csv(df_deconvmethod_2d_v2_results_file)
        df_deconvmethod_1d_v3_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v3_results_'+datestring+'.csv'))
        df_deconvmethod_1d_v3_results.to_csv(df_deconvmethod_1d_v3_results_file)
        df_deconvmethod_2d_v3_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v3_results_'+datestring+'.csv'))
        df_deconvmethod_2d_v3_results.to_csv(df_deconvmethod_2d_v3_results_file)

        df_fits_csv_save_widget.value = False

df_fits_csv_save_widget.observe(update_df_fits_csv_save_widget, names='value')


create_new_csv_file_widget = widgets.ToggleButton(
    value=False,
    description='df-->new csv',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save df_fits to new csv file',
    icon='check'
)

def create_new_csv_file(change):
    df_fits_csv_file = Path.joinpath(results_dir,str('df_fits_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_fits = df0[['timestamp_pulse_id'] + list(set(fits_header_list) - set(['chi2distance_fitting', 'chi2distance_deconvmethod_1d', 'chi2distance_deconvmethod_2d']))]
    df_fits.to_csv(df_fits_csv_file)
    df_fits_csv_files = sorted(list(results_dir.glob("df_fits*.csv")), reverse=True)
    df_fits_csv_files_widget.options=df_fits_csv_files
    df_fits_csv_files_widget.value = df_fits_csv_file

    df_fitting_v1_results_file = Path.joinpath(results_dir,str('df_fitting_v1_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_fitting_v1_results.to_csv(df_fitting_v1_results_file)
    df_fitting_v2_results_file = Path.joinpath(results_dir,str('df_fitting_v2_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_fitting_v2_results.to_csv(df_fitting_v2_results_file)
    # df_deconvmethod_1d_v1_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v1_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    # df_deconvmethod_1d_v1_results.to_csv(df_deconvmethod_1d_v1_results_file)
    df_deconvmethod_2d_v1_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v1_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_deconvmethod_2d_v1_results.to_csv(df_deconvmethod_2d_v1_results_file)
    df_deconvmethod_1d_v2_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v2_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_deconvmethod_1d_v2_results.to_csv(df_deconvmethod_1d_v2_results_file)
    df_deconvmethod_2d_v2_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v2_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_deconvmethod_2d_v2_results.to_csv(df_deconvmethod_2d_v2_results_file)
    df_deconvmethod_1d_v3_results_file = Path.joinpath(results_dir,str('df_deconvmethod_1d_v3_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_deconvmethod_1d_v3_results.to_csv(df_deconvmethod_1d_v3_results_file)
    df_deconvmethod_2d_v3_results_file = Path.joinpath(results_dir,str('df_deconvmethod_2d_v3_results_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv'))
    df_deconvmethod_2d_v3_results.to_csv(df_deconvmethod_2d_v3_results_file)

    create_new_csv_file_widget.value = False

create_new_csv_file_widget.observe(create_new_csv_file, names='value')



# run widgets

## run_over_all_images_widget

run_over_all_images_widget = widgets.ToggleButton(
    value=False,
    description='run (all images in measurement)',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='run (all images in measurement)',
    icon='check'
)

run_over_all_images_progress_widget = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    step=1,
    description="Progress:",
    bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
    orientation="horizontal",
)

run_over_all_images_statustext_widget = widgets.Text(value="", placeholder="status", description="time taken|left:", disabled=False)

## run_over_all_measurements_widget

run_over_all_measurements_widget = widgets.ToggleButton(
    value=False,
    description='run (all measurements in dataset)',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='run (all measurements in dataset)',
    icon='check'
)

run_over_all_measurements_progress_widget = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    step=1,
    description="Progress:",
    bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
    orientation="horizontal",
)

run_over_all_measurements_statustext_widget = widgets.Text(value="", placeholder="status", description="time taken|left:", disabled=False)


## run_over_all_datasets_widget

run_over_all_datasets_widget = widgets.ToggleButton(
    value=False,
    description='run (all datasets in dataset)',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='run (all datasets in dataset)',
    icon='check'
)

run_over_all_datasets_progress_widget = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    step=1,
    description="Progress:",
    bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
    orientation="horizontal",
)

run_over_all_datasets_statustext_widget = widgets.Text(value="", placeholder="status", description="time taken|left:", disabled=False)


# result widgets

result_widget_style = {'description_width': 'initial'}

do_textbox_widget = widgets.Checkbox(
    value=False, description="do_textbox", disabled=False)

textarea_widget = widgets.Textarea(
    value="info", placeholder="Type something", description="Fitting:", disabled=False)
beamsize_text_widget = widgets.Text(
    value="", placeholder="beamsize in rms", description=r"beam rms", disabled=False, layout=widgets.Layout(width='auto'), style=result_widget_style
)
xi_um_fit_v1_widget = widgets.Text(
    # r"\({\xi}_{fit}_{center}\)"
    value="", placeholder="xi_fit (v1)", description='Fitting v1 ξ / μm', disabled=False, layout=widgets.Layout(width='auto'), 
)
fit_profile_text_widget = widgets.Text(
    value="", placeholder="xi_fit_v2_off_center", description='Fitting v2 ξ (fit) / μm', disabled=False, layout=widgets.Layout(width='auto')  # \({\xi}_{fit}\
)
xi_um_fit_v2_widget = widgets.Text(
    # r"\({\xi}_{fit}_{center}\)"
    value="", placeholder="xi_fit (v2)", description='Fitting v2 ξ / μm', disabled=False, layout=widgets.Layout(width='auto')
)

deconvmethod_2d_v1_result_widget = widgets.Text(
    # r"\({\xi}_x,{\xi}_y\)"
    value="", placeholder="(xi_x_um, xi_y_um) (v1)", description='2D-Deconvolution v1 (ξˣ, ξʸ) / μm', disabled=False, layout=widgets.Layout(width='auto')
)

deconvmethod_1d_v2_result_widget = widgets.Text(
    value="", placeholder="xi_um (v2)", description='1D-Deconvolution v2 ξ / μm', disabled=False, layout=widgets.Layout(width='auto')
)
deconvmethod_2d_v2_result_widget = widgets.Text(
    # r"\({\xi}_x,{\xi}_y\)"
    value="", placeholder="(xi_x_um, xi_y_um) (v2)", description='2D-Deconvolution v2 (ξˣ, ξʸ) / μm', disabled=False, layout=widgets.Layout(width='auto')
)

deconvmethod_1d_v3_result_widget = widgets.Text(
    value="", placeholder="xi_um (v3)", description='1D-Deconvolution v3 ξ / μm', disabled=False, layout=widgets.Layout(width='auto')
)

deconvmethod_2d_v3_result_widget = widgets.Text(
    # r"\({\xi}_x,{\xi}_y\)"
    value="", placeholder="(xi_x_um, xi_y_um) (v3)", description='2D-Deconvolution v3 (ξˣ, ξʸ) / μm', disabled=False, layout=widgets.Layout(width='auto')
)

# general parameter widgets

crop_px_widget = widgets.FloatText(value=200, description='crop_px')
pixis_profile_avg_width_widget = widgets.FloatText(value=200, description='profile width / px')

# fitting parameter widgets

shiftx_um_widget = widgets.FloatSlider(min=-n / 2 * 13, max=n / 2 * 13, value=477, step=1, description="shiftx_um")
# wavelength_nm_widget = widgets.FloatSlider(value=_lambda_widget.value, description='wavelength_nm')
wavelength_nm_widget = widgets.FloatSlider(value=8.0, description="wavelength_nm")
z_mm_widget = widgets.FloatSlider(min=5000.0, max=6000.0, value=5781.0, description="z_mm")
# d_um_widget = widgets.FloatSlider(min=107, max= 1337, value=d_um_widget.value, description='d_um')
d_um_widget = widgets.FloatSlider(min=107, max=1337, value=215.0, description="d_um")
gamma_widget = widgets.FloatSlider(min=0, max=2.0, value=0.8, description="gamma")
w1_um_widget = widgets.FloatSlider(min=8, max=16, value=11.00, description="w1_um")
w2_um_widget = widgets.FloatSlider(min=8, max=16, value=11.00, description="w2_um")
I_Airy1_widget = widgets.FloatSlider(min=0, max=10, value=1.0, description="I_Airy1")
I_Airy2_widget = widgets.FloatSlider(min=0, max=10, value=0.8, description="I_Airy2")
x1_um_widget = widgets.FloatSlider(
    min=-n * 13 / 2 - 5000, max=0, value=-d_um_widget.value * 10 / 2, step=0.1, description="x1_um"
)
x2_um_widget = widgets.FloatSlider(
    min=0, max=2 * n * 13 + 5000, value=d_um_widget.value * 10 / 2, step=0.1, description="x2_um"
)
normfactor_widget = widgets.FloatSlider(
    min=0.00, max=10, value=1.0, step=0.1, description="normfactor", readout_format=".2f"
)
mod_sigma_um_widget = widgets.FloatSlider(
    min=0, max=100000, value=3000, step=100, description="mod_sigma_um", readout_format=".2f"
)
mod_shiftx_um_widget = widgets.FloatSlider(min=-30000, max=30000, value=3000, step=1, description="mod_shiftx_um")


shiftx_um_range_widget = widgets.FloatRangeSlider(
    min=-n / 2 * 13, max=n / 2 * 13, value=[-1500, 1500], step=1, description="shiftx_um"
)
wavelength_nm_range_widget = widgets.FloatRangeSlider(
    min=7,
    max=19,
    value=[wavelength_nm_widget.value - 0.1, wavelength_nm_widget.value + 0.1],
    description="wavelength_nm",
)
z_mm_range_widget = widgets.FloatRangeSlider(min=5000.0, max=6000.0, value=[5770.0, 5790], description="z_mm")
d_um_range_widget = widgets.FloatRangeSlider(min=50, max=1337, value=[50.0, 1337.0], description="d_um")
gamma_range_widget = widgets.FloatRangeSlider(min=0, max=2.0, value=[0.01, 1.0], description="gamma")
w1_um_range_widget = widgets.FloatRangeSlider(min=5, max=20, value=[8, 15], description="w1_um")
w2_um_range_widget = widgets.FloatRangeSlider(min=5, max=20, value=[8, 15], description="w2_um")
I_Airy1_range_widget = widgets.FloatRangeSlider(min=0, max=10, value=[0.2, 1.5], description="I_Airy1")
I_Airy2_range_widget = widgets.FloatRangeSlider(min=0, max=10, value=[0.2, 5.5], description="I_Airy2")
x1_um_range_widget = widgets.FloatRangeSlider(
    min=-n * 13, max=0, value=[-d_um_widget.value * 10 / 2 - 1000, 0], step=0.1, description="x1_um"
)
x2_um_range_widget = widgets.FloatRangeSlider(
    min=0, max=n * 13, value=[0, d_um_widget.value * 10 / 2 + 1000], step=0.1, description="x2_um"
)
normfactor_range_widget = widgets.FloatRangeSlider(
    min=0, max=10, value=[0.5, 1.5], step=0.01, description="normfactor", readout_format=".2f"
)
mod_sigma_um_range_widget = widgets.FloatRangeSlider(
    min=0, max=100000, value=[1500.0, 100000.0], step=100, description="mod_sigma_um", readout_format=".2f"
)
mod_shiftx_um_range_widget = widgets.FloatRangeSlider(
    min=-30000, max=30000, value=[-10000, 10000], step=100, description="mod_shiftx_um"
)

do_fit_widget_layout = widgets.Layout(width="auto")
shiftx_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
wavelength_nm_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
z_mm_do_fit_widget = widgets.Checkbox(value=False, description="", indent=False, layout=do_fit_widget_layout)
d_um_do_fit_widget = widgets.Checkbox(value=False, description="", indent=False, layout=do_fit_widget_layout)
gamma_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
w1_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
w2_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
I_Airy1_do_fit_widget = widgets.Checkbox(value=False, description="", indent=False, layout=do_fit_widget_layout)
I_Airy2_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
x1_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
x2_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
normfactor_do_fit_widget = widgets.Checkbox(value=False, description="", indent=False, layout=do_fit_widget_layout)
mod_sigma_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)
mod_shiftx_um_do_fit_widget = widgets.Checkbox(value=True, description="", indent=False, layout=do_fit_widget_layout)

value_widget_layout = widgets.Layout(width="80px")
shiftx_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
wavelength_nm_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
z_mm_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
d_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
gamma_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
w1_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
w2_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
I_Airy1_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
I_Airy2_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
x1_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
x2_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
normfactor_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
mod_sigma_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)
mod_shiftx_um_value_widget = widgets.Text(value="", description="",layout=value_widget_layout)


# deconvolution_2d_v1 parameter widgets
crop_px_2d_v1_widget = widgets.FloatText(description='crop_px (2dv1)')
pixis_profile_avg_width_2d_v1_widget = widgets.FloatText(description='profile width / px (2dv1)')

sigma_x_F_gamma_um_min_2d_v1_widget = widgets.FloatText(description='sigma_x_F_gamma_um_min (2dv1)')
sigma_x_F_gamma_um_max_2d_v1_widget = widgets.FloatText(description='sigma_x_F_gamma_um_max (2dv1)')
sigma_x_F_gamma_um_stepsize_2d_v1_widget = widgets.FloatText(description='sigma_x_F_gamma_um_stepsize (2dv1)')
sigma_y_F_gamma_um_min_2d_v1_widget = widgets.FloatText(description='sigma_y_F_gamma_um_min (2dv1)')
sigma_y_F_gamma_um_max_2d_v1_widget = widgets.FloatText(description='sigma_y_F_gamma_um_max (2dv1)')
sigma_y_F_gamma_um_stepsize_2d_v1_widget = widgets.FloatText(description='sigma_y_F_gamma_um_stepsize (2dv1)')

# deconvolution 1d v2 parameter widgets
crop_px_1d_v2_widget = widgets.FloatText(description='crop_px (1dv2)')
pixis_profile_avg_width_1d_v2_widget = widgets.FloatText(description='profile width / px (1dv2)')
balance_1d_v2_widget = widgets.FloatText(description='balance (1dv2)')
xi_um_guess_1d_v2_widget = widgets.FloatText(description='xi_um_guess (1dv2)')
xatol_1d_v2_widget = widgets.FloatText(description='xatol (1dv2')
sigma_x_F_gamma_um_multiplier_1d_v2_widget = widgets.FloatText(description='sigma_x_F_gamma_um_multiplier_widget (1dv2)')

# deconvolution 2d v2 parameter widgets
crop_px_2d_v2_widget = widgets.FloatText(description='crop_px (2dv2)')
pixis_profile_avg_width_2d_v2_widget = widgets.FloatText(description='profile width / px (2dv2)')
balance_2d_v2_widget = widgets.FloatText(description='balance (2dv2)')
xi_um_guess_2d_v2_widget = widgets.FloatText(description='xi_um_guess (2dv2)')
xatol_2d_v2_widget = widgets.FloatText(description='xatol (2dv2')
sigma_x_F_gamma_um_multiplier_2d_v2_widget = widgets.FloatText(description='sigma_x_F_gamma_um_multiplier_widget (2dv2)')

# deconvolution 1d v3 parameter widgets
crop_px_1d_v3_widget = widgets.FloatText(description='crop_px (1dv3)')
pixis_profile_avg_width_1d_v3_widget = widgets.FloatText(description='profile width / px (1dv3)')
snr_db_1d_v3_widget = widgets.FloatText(description='snr_db (1dv3)', step=0.1)
xi_um_guess_1d_v3_widget = widgets.FloatText(description='xi_um_guess (1dv3)')
xatol_1d_v3_widget = widgets.FloatText(description='xatol (1dv3)')
sigma_x_F_gamma_um_multiplier_1d_v3_widget = widgets.FloatText(description='sigma_x_F_gamma_um_multiplier_widget (1dv3)')

# deconvolution 2d v3 parameter widgets
crop_px_2d_v3_widget = widgets.FloatText(description='crop_px (2dv3)')
pixis_profile_avg_width_2d_v3_widget = widgets.FloatText(description='profile width / px (2dv3)')
snr_db_2d_v3_widget = widgets.FloatText(description='snr_db (2dv3)', step=0.1)
xi_um_guess_2d_v3_widget = widgets.FloatText(description='xi_um_guess (2dv3)')
xatol_2d_v3_widget = widgets.FloatText(description='xatol (2dv3)')
sigma_x_F_gamma_um_multiplier_2d_v3_widget = widgets.FloatText(description='sigma_x_F_gamma_um_multiplier_widget (2dv3)')


# plot result widgets
do_plot_fitting_vs_deconvolution_widget = widgets.Checkbox(value=False, description="do fitting vs deconv plot")
do_list_results_widget = widgets.Checkbox(value=False, description="do list results")

xi_um_deconv_options = [
    ('xi_um_v3',('xi_um_v3',r"$\xi$ / um (deconv1d_v3)")),
    ('xi_x_um_v3',('xi_x_um_v3',r"$\xi_x$ / um (deconv2d_v2)")),
    ('xi_um_v2',('xi_um_v2',r"$\xi$ / um (deconv1d_v2)")),
    ('xi_x_um_v2',('xi_x_um_v2',r"$\xi_x$ / um (deconv2d_v2)")),
    ('xi_x_um_v1',('xi_x_um_v1',r"$\xi_x$ / um (deconv2d_v1)")),
    ('xi_x_um_measurement_default_result',('xi_x_um_measurement_default_result',r"$\xi_x$ / um (deconv)")),
    ('xi_um_measurement_default_result',('xi_um_measurement_default_result',r"$\xi$ / um (deconv)")),]
xi_um_fit_options = [
    ('xi_um_fit_v2',('xi_um_fit_v2',r"$\xi$ / um (fitting_v2)")), \
    ('xi_um_fit_v1',('xi_um_fit_v1',r"$\xi$ / um (fitting_v1)")), \
    ('xi_um_fit_v2_off_center',('xi_um_fit',r"$\xi$ / um (fitting_v2_off_center)")),
    ('xi_um_fit_measurement_default_result',('xi_um_fit_measurement_default_result',r"$\xi$ / um (fit)")), \
    ('xi_um_fit_v2_measurement_default_result',('xi_um_fit_v2_measurement_default_result',r"$\xi_c$ / um (fit)"))
    ]

chi2distance_options = [
    ('chi2distance_deconvmethod_1d_v3',('chi2distance_deconvmethod_1d_v3',r"$\chi^2$ (deconv1d_v3)")), \
    ('chi2distance_deconvmethod_2d_v3',('chi2distance_deconvmethod_2d_v3',r"$\chi^2$ (deconv2d_v3)")), \
    ('chi2distance_deconvmethod_1d_v2',('chi2distance_deconvmethod_1d_v2',r"$\chi^2$ (deconv1d_v2)")), \
    ('chi2distance_deconvmethod_2d_v2',('chi2distance_deconvmethod_2d_v2',r"$\chi^2$ (deconv2d_v2)")), \
    ('chi2distance_deconvmethod_2d_v1',('chi2distance_deconvmethod_2d_v1',r"$\chi^2$ (deconv2d_v1)")), \
    ('chi2distance_fitting_v1',('chi2distance_fitting_v1',r"$\chi^2$ (fitting_v1)")), \
    ('chi2distance_fitting_v2',('chi2distance_fitting',r"$\chi^2$ (fitting_v2)"))
                        ]

xi_um_deconv_column_and_label_widget = widgets.Dropdown(
    options=xi_um_deconv_options + xi_um_fit_options,
    description="x-data",
    description_tooltip='deconvolution variant',
    disabled=False,
    layout=widgets.Layout(width='auto')
)

xi_um_fit_column_and_label_widget = widgets.Dropdown(
    options= xi_um_fit_options + xi_um_deconv_options,
    description="y-data",
    description_tooltip='fitting variant',
    disabled=False,
    layout=widgets.Layout(width='auto')
)

chi2distance_column_and_label_widget = widgets.Dropdown(
    options= chi2distance_options,
    description="c-data",
    description_tooltip='chi2distance variant',
    disabled=False,
    layout=widgets.Layout(width='auto')
)

deconvmethod_outlier_limit_widget = widgets.FloatText(value = 2000, description='ξ>', description_tooltip='list values above this threshold',layout=widgets.Layout(width='auto'))
fitting_outlier_limit_widget = widgets.FloatText(value = 2000, description='ξ>', description_tooltip='list values above this threshold',layout=widgets.Layout(width='auto'))

xaxisrange_widget = widgets.IntRangeSlider(min=0, max=4000, value=[0,2000], description='x-range', description_tooltip='x-axis range', layout=widgets.Layout(width='auto'))
yaxisrange_widget = widgets.IntRangeSlider(min=0, max=4000, value=[0,2000], description='y-range', description_tooltip='y-axis range', layout=widgets.Layout(width='auto'))


do_plot_CDCs_widget = widgets.Checkbox(value=False, description="do plot CDCs")
do_plot_xi_um_fit_vs_I_Airy2_fit_widget = widgets.Checkbox(value=False, description="do plot xi_um_fit vs I_Airy2_fit")

# define what should happen when the hdf5 file widget is changed:


# function using the widgets:



def plot_fitting_v1(
    do_plot_fitting_v1,
    pixis_profile_avg_width,
    crop_px,
    # hdf5_file_path,
    # imageid,
    savefigure,
    save_to_df,
    do_textbox,
    shiftx_um,
    shiftx_um_range,
    shiftx_um_do_fit,
    wavelength_nm,
    wavelength_nm_range,
    wavelength_nm_do_fit,
    z_mm,
    z_mm_range,
    z_mm_do_fit,
    d_um,
    d_um_range,
    d_um_do_fit,
    gamma,
    gamma_range,
    gamma_do_fit,
    w1_um,
    w1_um_range,
    w1_um_do_fit,
    w2_um,
    w2_um_range,
    w2_um_do_fit,
    I_Airy1,
    I_Airy1_range,
    I_Airy1_do_fit,
    I_Airy2,
    I_Airy2_range,
    I_Airy2_do_fit,
    x1_um,
    x1_um_range,
    x1_um_do_fit,
    x2_um,
    x2_um_range,
    x2_um_do_fit,
    normfactor,
    normfactor_range,
    normfactor_do_fit,
):

    if do_plot_fitting_v1 == True:  # workaround, so that the function is not executed while several inputs are changed

        global df_fitting_v1_results

        # fittingprogress_widget.bar_style = 'info'
        # fittingprogress_widget.value = 0
        # statustext_widget.value = 'fitting ...'
        # textarea_widget.value = ''

        xi_um_fit_v1_widget.value = ''


        # Loading and preparing

        imageid = imageid_widget.value
        hdf5_file_path = dph_settings_bgsubtracted_widget.value

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]
            pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][0]

        pinholes = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["pinholes"].iloc[0]
        separation_um = get_sep_and_orient(pinholes)[0]
        orientation = get_sep_and_orient(pinholes)[1]
        setting_wavelength_nm = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_wavelength_nm"].iloc[0]
        energy_hall_uJ = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["energy hall"].iloc[0]
        
        # pixis_profile_avg_width = 200  # read from df0 instead!

        # fittingprogress_widget.value = 2
        #     hdf5_file_name_image = hdf5_file_name_image_widget.value
        #     dataset_image_args = dataset_image_args_widget.value
        xi_um_fit_v1_widget.value = 'calculating ...'

        


        # imageids_by_energy_hall = get_imageids_with_bgs(beamposition_horizontal_interval)
        imageids_by_energy_hall = imageids

        # if imageid == -1:
        #     beamposx = df['beam position hall horizontal pulse resolved'].mean(axis=0)
        #     beamposy = df['beam position hall vertical pulse resolved'].mean(axis=0)
        #     energy_hall_uJ = df['energy hall'].mean(axis=0)
        # else:
        #     beamposx = df[df['imageid']==imageid]['beam position hall horizontal pulse resolved']
        #     beamposy = df[df['imageid']==imageid]['beam position hall vertical pulse resolved']
        #     energy_hall_uJ = df[df['imageid']==imageid]['energy hall'].iloc[0]

        pixis_profile_avg = np.average(pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:],axis=0)
        pixis_profile_avg = pixis_profile_avg / np.max(pixis_profile_avg)

        n = pixis_profile_avg.size  # number of sampling point  # number of pixels
        dX_1 = 13e-6
        xdata = np.linspace((-n / 2) * dX_1, (+n / 2 - 1) * dX_1, n)
        # ydata = pixis_profile_avg_dataset[imageid]*datafactor
        ydata = pixis_profile_avg  # defined in the cells above, still to implement: select
       
        #still to average over y!

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # Fitting

        result = fit_profile_v1(
            pixis_image_norm,
            pixis_profile_avg,
            shiftx_um,
            shiftx_um_range,
            shiftx_um_do_fit,
            wavelength_nm,
            wavelength_nm_range,
            wavelength_nm_do_fit,
            z_mm,
            z_mm_range,
            z_mm_do_fit,
            d_um,
            d_um_range,
            d_um_do_fit,
            gamma,
            gamma_range,
            gamma_do_fit,
            w1_um,
            w1_um_range,
            w1_um_do_fit,
            w2_um,
            w2_um_range,
            w2_um_do_fit,
            I_Airy1,
            I_Airy1_range,
            I_Airy1_do_fit,
            I_Airy2,
            I_Airy2_range,
            I_Airy2_do_fit,
            x1_um,
            x1_um_range,
            x1_um_do_fit,
            x2_um,
            x2_um_range,
            x2_um_do_fit,
            normfactor,
            normfactor_range,
            normfactor_do_fit,
        )

        shiftx_um_fit = result.params["shiftx_um"].value
        wavelength_nm_fit = result.params["wavelength_nm"].value
        z_mm_fit = result.params["z_mm"].value
        d_um_fit = result.params["d_um"].value
        w1_um_fit = result.params["w1_um"].value
        w2_um_fit = result.params["w2_um"].value
        I_Airy1_fit = result.params["I_Airy1"].value
        I_Airy2_fit = result.params["I_Airy2"].value
        x1_um_fit = result.params["x1_um"].value
        x2_um_fit = result.params["x2_um"].value
        gamma_fit = result.params["gamma"].value
        normfactor_fit = result.params["normfactor"].value

        textarea_widget.value = result.fit_report()
        chi2distance = result.chisqr

        # # print number of function efvals
        # print result.nfev
        # # print number of data points
        # print result.ndata
        # # print number of variables
        # print result.nvarys
        # # chi-sqr
        # print result.chisqr
        # # reduce chi-sqr
        # print result.redchi
        # #Akaike info crit
        # print result.aic
        # #Bayesian info crit
        # print result.bic

        shiftx_um_value_widget.value = r"%.2f" % (shiftx_um_fit)
        wavelength_nm_value_widget.value = r"%.2f" % (wavelength_nm_fit)
        z_mm_value_widget.value = r"%.2f" % (z_mm_fit)
        d_um_value_widget.value = r"%.2f" % (d_um_fit)
        gamma_value_widget.value = r"%.2f" % (gamma_fit)
        w1_um_value_widget.value = r"%.2f" % (w1_um_fit)
        w2_um_value_widget.value = r"%.2f" % (w2_um_fit)
        I_Airy1_value_widget.value = r"%.2f" % (I_Airy1_fit)
        I_Airy2_value_widget.value = r"%.2f" % (I_Airy2_fit)
        x1_um_value_widget.value = r"%.2f" % (x1_um_fit)
        x2_um_value_widget.value = r"%.2f" % (x2_um_fit)
        normfactor_value_widget.value = r"%.2f" % (normfactor_fit)

        # calculate gamma_fit at the center between the two airy disks
        
        d_um_at_detector = x2_um_fit - x1_um_fit

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm_fit * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # lmfit throws RuntimeWarnings, maybe its a bug. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit, xi_um_fit_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit], [0, 0], 470, False)
                
        xi_um_fit_v1_widget.value = r"%.2fum" % (xi_um_fit)

        if save_to_df == True:
            # guess parameters - fitting

            measurement = os.path.splitext(os.path.basename(dph_settings_bgsubtracted_widget.value))[0]
            df_fitting_v1_results = df_fitting_v1_results.append(
                    {
                        # image identifiers
                        'measurement' : measurement,
                        'timestamp_pulse_id' : timestamp_pulse_id,
                        'imageid' : imageid,
                        'separation_um' : separation_um,
                        # fitting parameters
                        'pixis_profile_avg_width' : pixis_profile_avg_width,
                        'shiftx_um' : shiftx_um,
                        'shiftx_um_range_0' : shiftx_um_range[0],
                        'shiftx_um_range_1' : shiftx_um_range[1],
                        'shiftx_um_do_fit' : shiftx_um_do_fit,
                        'wavelength_nm' : wavelength_nm,
                        'wavelength_nm_range_0' : wavelength_nm_range[0],
                        'wavelength_nm_range_1' : wavelength_nm_range[1],
                        'wavelength_nm_do_fit' : wavelength_nm_do_fit,
                        'z_mm' : z_mm,
                        'z_mm_range_0' : z_mm_range[0],
                        'z_mm_range_1' : z_mm_range[1],
                        'z_mm_do_fit' : z_mm_do_fit,
                        'd_um' : d_um,
                        'd_um_range_0' : d_um_range[0],
                        'd_um_range_1' : d_um_range[1],
                        'd_um_do_fit' : d_um_do_fit,
                        'gamma' : gamma,
                        'gamma_range_0' : gamma_range[0],
                        'gamma_range_1' : gamma_range[1],
                        'gamma_do_fit' : gamma_do_fit,
                        'w1_um' : w1_um,
                        'w1_um_range_0' : w1_um_range[0],
                        'w1_um_range_1' : w1_um_range[1],
                        'w1_um_do_fit' : w1_um_do_fit,
                        'w2_um' : w2_um,
                        'w2_um_range_0' : w2_um_range[0],
                        'w2_um_range_1' : w2_um_range[1],
                        'w2_um_do_fit' : w2_um_do_fit,
                        'I_Airy1' : I_Airy1,
                        'I_Airy1_range_0' : I_Airy1_range[0],
                        'I_Airy1_range_1' : I_Airy1_range[1],
                        'I_Airy1_do_fit' : I_Airy1_do_fit,
                        'I_Airy2' : I_Airy2,
                        'I_Airy2_range_0' : I_Airy2_range[0],
                        'I_Airy2_range_1' : I_Airy2_range[1],
                        'I_Airy2_do_fit' : I_Airy2_do_fit,
                        'x1_um' : x1_um,
                        'x1_um_range_0' : x1_um_range[0],
                        'x1_um_range_1' : x1_um_range[1],
                        'x1_um_do_fit' : x1_um_do_fit,
                        'x2_um' : x2_um,
                        'x2_um_range_0' : x2_um_range[0],
                        'x2_um_range_1' : x2_um_range[1],
                        'x2_um_do_fit' : x2_um_do_fit,
                        'normfactor' :  normfactor,
                        'normfactor_range_0' : normfactor_range[0],
                        'normfactor_range_1' : normfactor_range[1],
                        'normfactor_do_fit' : normfactor_do_fit,
                        # fitting results
                        'gamma_fit_v1' :  gamma_fit,
                        'xi_um_fit_v1' :  xi_um_fit,  # add this first to the df_fits dataframe
                        'wavelength_nm_fit_v1' :  wavelength_nm_fit,
                        'd_um_at_detector_v1' :  d_um_at_detector,
                        'I_Airy1_fit_v1' :  I_Airy1_fit,
                        'I_Airy2_fit_v1' :  I_Airy2_fit,
                        'w1_um_fit_v1' :  w1_um_fit,
                        'w2_um_fit_v1' :  w2_um_fit,
                        'shiftx_um_fit_v1' :  shiftx_um_fit,
                        'x1_um_fit_v1' :  x1_um_fit,
                        'x2_um_fit_v1' :  x2_um_fit,
                        'chi2distance_fitting_v1' : chi2distance                       
                    }, ignore_index = True
                )
            df_fitting_v1_results = df_fitting_v1_results.drop_duplicates()

            




        # print('fringeseparation_px=' + str(round(fringeseparation_px,2)))

        # textarea_widget.value = result.fit_report()

        # fittingprogress_widget.value = 8
        # statustext_widget.value = 'Generating Plot ...'

        # Plotting

        #     fig=plt.figure(figsize=(11.69,8.27), dpi= 150, facecolor='w', edgecolor='k')  # A4 sheet in landscape
        fig = plt.figure(constrained_layout=False, figsize=(8.27, 11.69), dpi=150)

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2])
        gs.update(hspace=0.1)

        #     ax2 = plt.subplot(2,1,2)
        ax10 = fig.add_subplot(gs[1, 0])

        im_ax10 = ax10.imshow(
            pixis_image_norm,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            cmap="jet",
            vmin=0,
            vmax=1,
            extent=((-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3, -n / 2 * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3),
        )

        # fig.colorbar(im_ax2, ax=ax2, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

        ax10.add_patch(
            patches.Rectangle(
                ((-n / 2) * dX_1 * 1e3, (int(round(pixis_centery_px)) - n / 2 - pixis_profile_avg_width / 2) * dX_1 * 1e3),
                n * dX_1 * 1e3,
                pixis_profile_avg_width * dX_1 * 1e3,
                color="w",
                linestyle="-",
                alpha=0.8,
                fill=False,  # remove background
            )
        )

        ax10.set_xlabel("x / mm", fontsize=14)
        ax10.set_ylabel("y / mm", fontsize=14)
        ax10.grid(color="w", linewidth=1, alpha=0.5, linestyle="--", which="major")

        ax00 = fig.add_subplot(gs[0, 0], sharex=ax10)
        #     ax = plt.subplot(2,1,1)

        #     plt.plot(list(range(pixis_profile_avg.size)),ydata, color='r', linewidth=2)
        #     plt.plot(list(range(pixis_profile_avg.size)),result.best_fit, color='b', linewidth=0.5)
        ax00.plot(xdata * 1e3, ydata, color="r", linewidth=2, label="data")
        ax00.plot(xdata * 1e3, result.best_fit, color="b", linewidth=1, label="fit")

        Airy1 = [
            I_Airy1_fit
            * Airy(
                (x - shiftx_um_fit * 1e-6),
                w1_um_fit * 1e-6,
                wavelength_nm_fit * 1e-9,
                z_mm_fit * 1e-3,
                x1_um_fit * 1e-6,
            )
            ** 2
            for x in xdata
        ]
        Airy1 = normalize(Airy1) / I_Airy2_fit
        Airy2 = [
            I_Airy2_fit
            * Airy(
                (x - shiftx_um_fit * 1e-6),
                w2_um_fit * 1e-6,
                wavelength_nm_fit * 1e-9,
                z_mm_fit * 1e-3,
                x2_um_fit * 1e-6,
            )
            ** 2
            for x in xdata
        ]
        Airy2 = normalize(Airy2)

        do_plot_Airys = False
        if do_plot_Airys == True:
            plt.plot(xdata * 1e3, Airy1, color="k", label="Airy1", linewidth=1)
            plt.plot(xdata * 1e3, Airy2, color="grey", label="Airy2", linewidth=1)

        # plt.vlines([x1_loc_px_fit, pixis_centerx_px, x2_loc_px_fit],0,1)
        ax00.vlines(
            [(shiftx_um_fit + x1_um_fit) * 1e-3, shiftx_um_fit * 1e-3, (shiftx_um_fit + x2_um_fit) * 1e-3], 0, 0.1
        )
        #     ax00.annotate('xshift',
        #                xy=((shiftx_um_fit)*1e-3,0), xycoords='data',
        #                xytext=(0,-20), textcoords='offset points',
        #                 bbox=dict(boxstyle="round", fc="w"),
        #                 arrowprops=dict(arrowstyle="->"))
        ax00.annotate(
            "$x_2$",
            xy=((shiftx_um_fit + x2_um_fit) * 1e-3, 0),
            xycoords="data",
            xytext=(0, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        ax00.annotate(
            "$x_1$",
            xy=((shiftx_um_fit + x1_um_fit) * 1e-3, 0),
            xycoords="data",
            xytext=(0, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )

        # plt.hlines(0,0,n)

        textstr = "\n".join(
            (
                r"imageid=%.2f" % (imageid,),
                r"shiftx_um=%.2f" % (shiftx_um_fit,),
                r"$\lambda=%.2f$nm" % (wavelength_nm_fit,),
                r"fringesepar_um=%.2f" % (fringeseparation_um,),
                r"w1_um=%.2f" % (w1_um_fit,),
                r"w2_um=%.2f" % (w2_um_fit,),
                r"I_Airy1=%.2f" % (I_Airy1_fit,),
                r"I_Airy2=%.2f" % (I_Airy2_fit,),
                r"x1_um=%.2f" % (x1_um_fit,),
                r"x2_um=%.2f" % (x2_um_fit,),
                r"$\gamma=%.2f$" % (gamma_fit,),
                r"normfactor=%.2f" % (normfactor_fit,),
                r"d_um_at_detector=%.2f" % (d_um_at_detector,),
            )
        )

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords

        if do_textbox == True:
            ax.text(1, 0.95, textstr, transform=ax.transAxes, fontsize=6, verticalalignment="top", bbox=props)

        ax00.set_xlim([(-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3])
        ax00.set_ylim([0, 1])

        ax00.set_ylabel("Intensity / a.u.", fontsize=14)
        ax00.legend()

        textstr = " ".join(
            (
                "ph-" + pinholes + ".id" + str(int(imageid)),
                r"$\lambda=%.2f$nm" % (result.params["wavelength_nm"].value,),
                orientation,
                "$d$=" + str(int(separation_um)) + "um",
                r"$d_{det}$=%.2fum" % (d_um_at_detector,),
                "\n",
                r"$w_1$=%.2fum" % (result.params["w1_um"].value,),
                r"$w_2$=%.2fum" % (result.params["w2_um"].value,),
                r"$I_1$=%.2f" % (result.params["I_Airy1"].value,),
                r"$I_2=$%.2f" % (result.params["I_Airy2"].value,),
                r"$\gamma=%.2f$" % (gamma_fit,),
                r"$\xi=%.2fum$" % (xi_um_fit,),
            )
        )
        ax00.set_title(textstr, fontsize=10)

        if savefigure == True:
            savefigure_dir = str(scratch_dir) + "/" + dph_settings_bgsubtracted_widget.value.name
            if os.path.isdir(savefigure_dir) == False:
                os.mkdir(savefigure_dir)
            # savefigure_dir = str(scratch_dir) + '/' + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
            savefigure_dir = (
                str(scratch_dir)
                + "/"
                + dph_settings_bgsubtracted_widget.value.name
                + "/"
                + "profilewidth_px_"
                + str(int(pixis_avg_width))
                + "_"
                + "bg_intervall_um_"
                + str(int(beamposition_horizontal_interval))
            )
            if os.path.isdir(savefigure_dir) == False:
                os.mkdir(savefigure_dir)
            savefigure_dir = (
                str(scratch_dir)
                + "/"
                + dph_settings_bgsubtracted_widget.value.name
                + "/"
                + "profilewidth_px_"
                + str(int(pixis_avg_width))
                + "_"
                + "bg_intervall_um_"
                + str(int(beamposition_horizontal_interval))
                + "/profiles_fit/"
            )
            if os.path.isdir(savefigure_dir) == False:
                os.mkdir(savefigure_dir)
            plt.savefig(
                savefigure_dir
                + "/"
                + "profiles_fit_"
                + hdf5_file_name_image_widget.value
                + "_ph_"
                + str(pinholes)
                + "_d_"
                + str(separation_um)
                + "_E_"
                + str(format(energy_hall_uJ, ".4f")).zfill(6)
                + "_image_"
                + str(imageid)
                + ".png",
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                papertype=None,
                format=None,
                transparent=False,
                bbox_inches=None,
                pad_inches=0.1,
                frameon=None,
            )
            plt.savefig(
                savefigure_dir
                + "/"
                + "profiles_fit_"
                + hdf5_file_name_image_widget.value
                + "_ph_"
                + str(pinholes)
                + "_d_"
                + str(separation_um)
                + "_E_"
                + str(format(energy_hall_uJ, ".4f")).zfill(6)
                + "_image_"
                + str(imageid)
                + ".pdf",
                dpi=None,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                papertype=None,
                format=None,
                transparent=False,
                bbox_inches=None,
                pad_inches=0.1,
                frameon=None,
            )

        

            
            
            


        plt.show()
        # fittingprogress_widget.value = 10
        # fittingprogress_widget.bar_style = 'success'
        # statustext_widget.value = 'done'

        # print(gamma_fit)
















def plot_fitting_v2(
    do_plot_fitting_v2,
    pixis_profile_avg_width,
    crop_px,
    # hdf5_file_path,
    # imageid,
    savefigure,
    save_to_df,
    do_textbox,
    shiftx_um,
    shiftx_um_range,
    shiftx_um_do_fit,
    wavelength_nm,
    wavelength_nm_range,
    wavelength_nm_do_fit,
    z_mm,
    z_mm_range,
    z_mm_do_fit,
    d_um,
    d_um_range,
    d_um_do_fit,
    gamma,
    gamma_range,
    gamma_do_fit,
    w1_um,
    w1_um_range,
    w1_um_do_fit,
    w2_um,
    w2_um_range,
    w2_um_do_fit,
    I_Airy1,
    I_Airy1_range,
    I_Airy1_do_fit,
    I_Airy2,
    I_Airy2_range,
    I_Airy2_do_fit,
    x1_um,
    x1_um_range,
    x1_um_do_fit,
    x2_um,
    x2_um_range,
    x2_um_do_fit,
    normfactor,
    normfactor_range,
    normfactor_do_fit,
    mod_sigma_um,
    mod_sigma_um_range,
    mod_sigma_um_do_fit,
    mod_shiftx_um,
    mod_shiftx_um_range,
    mod_shiftx_um_do_fit,
):

    if do_plot_fitting_v2 == True:  # workaround, so that the function is not executed while several inputs are changed

        global df_fitting_v1_results
        global df_fitting_v2_results

        # fittingprogress_widget.bar_style = 'info'
        # fittingprogress_widget.value = 0
        # statustext_widget.value = 'fitting ...'
        # textarea_widget.value = ''

        fit_profile_text_widget.value = ''
        xi_um_fit_v2_widget.value = ''

        # Loading and preparing

        imageid = imageid_widget.value
        hdf5_file_path = dph_settings_bgsubtracted_widget.value

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]
            pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][0]

        pinholes = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["pinholes"].iloc[0]
        separation_um = get_sep_and_orient(pinholes)[0]
        orientation = get_sep_and_orient(pinholes)[1]
        setting_wavelength_nm = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_wavelength_nm"].iloc[0]
        energy_hall_uJ = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["energy hall"].iloc[0]

        # fittingprogress_widget.value = 2
        #     hdf5_file_name_image = hdf5_file_name_image_widget.value
        #     dataset_image_args = dataset_image_args_widget.value
        fit_profile_text_widget.value = 'calculating ...'
        xi_um_fit_v2_widget.value = 'calculating ...'
        


        # imageids_by_energy_hall = get_imageids_with_bgs(beamposition_horizontal_interval)
        imageids_by_energy_hall = imageids

        # if imageid == -1:
        #     beamposx = df['beam position hall horizontal pulse resolved'].mean(axis=0)
        #     beamposy = df['beam position hall vertical pulse resolved'].mean(axis=0)
        #     energy_hall_uJ = df['energy hall'].mean(axis=0)
        # else:
        #     beamposx = df[df['imageid']==imageid]['beam position hall horizontal pulse resolved']
        #     beamposy = df[df['imageid']==imageid]['beam position hall vertical pulse resolved']
        #     energy_hall_uJ = df[df['imageid']==imageid]['energy hall'].iloc[0]

        pixis_profile_avg = np.average(pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:],axis=0)
        pixis_profile_avg = pixis_profile_avg / np.max(pixis_profile_avg)

        n = pixis_profile_avg.size  # number of sampling point  # number of pixels
        dX_1 = 13e-6
        xdata = np.linspace((-n / 2) * dX_1, (+n / 2 - 1) * dX_1, n)
        # ydata = pixis_profile_avg_dataset[imageid]*datafactor
        ydata = pixis_profile_avg  # defined in the cells above, still to implement: select
       
        #still to average over y!

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # Fitting

        result = fit_profile_v2(
            pixis_image_norm,
            pixis_profile_avg,
            shiftx_um,
            shiftx_um_range,
            shiftx_um_do_fit,
            wavelength_nm,
            wavelength_nm_range,
            wavelength_nm_do_fit,
            z_mm,
            z_mm_range,
            z_mm_do_fit,
            d_um,
            d_um_range,
            d_um_do_fit,
            gamma,
            gamma_range,
            gamma_do_fit,
            w1_um,
            w1_um_range,
            w1_um_do_fit,
            w2_um,
            w2_um_range,
            w2_um_do_fit,
            I_Airy1,
            I_Airy1_range,
            I_Airy1_do_fit,
            I_Airy2,
            I_Airy2_range,
            I_Airy2_do_fit,
            x1_um,
            x1_um_range,
            x1_um_do_fit,
            x2_um,
            x2_um_range,
            x2_um_do_fit,
            normfactor,
            normfactor_range,
            normfactor_do_fit,
            mod_sigma_um,
            mod_sigma_um_range,
            mod_sigma_um_do_fit,
            mod_shiftx_um,
            mod_shiftx_um_range,
            mod_shiftx_um_do_fit,
        )

        shiftx_um_fit = result.params["shiftx_um"].value
        wavelength_nm_fit = result.params["wavelength_nm"].value
        z_mm_fit = result.params["z_mm"].value
        d_um_fit = result.params["d_um"].value
        w1_um_fit = result.params["w1_um"].value
        w2_um_fit = result.params["w2_um"].value
        I_Airy1_fit = result.params["I_Airy1"].value
        I_Airy2_fit = result.params["I_Airy2"].value
        x1_um_fit = result.params["x1_um"].value
        x2_um_fit = result.params["x2_um"].value
        gamma_fit = result.params["gamma"].value
        normfactor_fit = result.params["normfactor"].value
        mod_sigma_um_fit = result.params["mod_sigma_um"].value
        mod_shiftx_um_fit = result.params["mod_shiftx_um"].value

        textarea_widget.value = result.fit_report()
        chi2distance = result.chisqr

        # # print number of function efvals
        # print result.nfev
        # # print number of data points
        # print result.ndata
        # # print number of variables
        # print result.nvarys
        # # chi-sqr
        # print result.chisqr
        # # reduce chi-sqr
        # print result.redchi
        # #Akaike info crit
        # print result.aic
        # #Bayesian info crit
        # print result.bic

        shiftx_um_value_widget.value = r"%.2f" % (shiftx_um_fit)
        wavelength_nm_value_widget.value = r"%.2f" % (wavelength_nm_fit)
        z_mm_value_widget.value = r"%.2f" % (z_mm_fit)
        d_um_value_widget.value = r"%.2f" % (d_um_fit)
        gamma_value_widget.value = r"%.2f" % (gamma_fit)
        w1_um_value_widget.value = r"%.2f" % (w1_um_fit)
        w2_um_value_widget.value = r"%.2f" % (w2_um_fit)
        I_Airy1_value_widget.value = r"%.2f" % (I_Airy1_fit)
        I_Airy2_value_widget.value = r"%.2f" % (I_Airy2_fit)
        x1_um_value_widget.value = r"%.2f" % (x1_um_fit)
        x2_um_value_widget.value = r"%.2f" % (x2_um_fit)
        normfactor_value_widget.value = r"%.2f" % (normfactor_fit)
        mod_sigma_um_value_widget.value = r"%.2f" % (mod_sigma_um_fit)
        mod_shiftx_um_value_widget.value = r"%.2f" % (mod_shiftx_um_fit)

        # calculate gamma_fit at the center between the two airy disks
        gamma_fit_v2 = gaussian(0,1,mod_shiftx_um_fit,mod_sigma_um_fit)*gamma_fit

        d_um_at_detector = x2_um_fit - x1_um_fit

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm_fit * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # lmfit throws RuntimeWarnings, maybe its a bug. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit, xi_um_fit_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit], [0, 0], 470, False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit_v2, xi_um_fit_v2_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit_v2], [0, 0], 470, False)
                
        fit_profile_text_widget.value = r"%.2fum" % (xi_um_fit)
        xi_um_fit_v2_widget.value = r"%.2fum" % (xi_um_fit_v2)

        if save_to_df == True:
            # guess parameters - fitting
            measurement = os.path.splitext(os.path.basename(dph_settings_bgsubtracted_widget.value))[0]
            df_fitting_v2_results = df_fitting_v2_results.append(
                    {
                        # image identifiers
                        'measurement' : measurement,
                        'timestamp_pulse_id' : timestamp_pulse_id,
                        'imageid' : imageid,
                        'separation_um' : separation_um,
                        # fitting parameters
                        'pixis_profile_avg_width' : pixis_profile_avg_width,
                        'shiftx_um' : shiftx_um,
                        'shiftx_um_range_0' : shiftx_um_range[0],
                        'shiftx_um_range_1' : shiftx_um_range[1],
                        'shiftx_um_do_fit' : shiftx_um_do_fit,
                        'wavelength_nm' : wavelength_nm,
                        'wavelength_nm_range_0' : wavelength_nm_range[0],
                        'wavelength_nm_range_1' : wavelength_nm_range[1],
                        'wavelength_nm_do_fit' : wavelength_nm_do_fit,
                        'z_mm' : z_mm,
                        'z_mm_range_0' : z_mm_range[0],
                        'z_mm_range_1' : z_mm_range[1],
                        'z_mm_do_fit' : z_mm_do_fit,
                        'd_um' : d_um,
                        'd_um_range_0' : d_um_range[0],
                        'd_um_range_1' : d_um_range[1],
                        'd_um_do_fit' : d_um_do_fit,
                        'gamma' : gamma,
                        'gamma_range_0' : gamma_range[0],
                        'gamma_range_1' : gamma_range[1],
                        'gamma_do_fit' : gamma_do_fit,
                        'w1_um' : w1_um,
                        'w1_um_range_0' : w1_um_range[0],
                        'w1_um_range_1' : w1_um_range[1],
                        'w1_um_do_fit' : w1_um_do_fit,
                        'w2_um' : w2_um,
                        'w2_um_range_0' : w2_um_range[0],
                        'w2_um_range_1' : w2_um_range[1],
                        'w2_um_do_fit' : w2_um_do_fit,
                        'I_Airy1' : I_Airy1,
                        'I_Airy1_range_0' : I_Airy1_range[0],
                        'I_Airy1_range_1' : I_Airy1_range[1],
                        'I_Airy1_do_fit' : I_Airy1_do_fit,
                        'I_Airy2' : I_Airy2,
                        'I_Airy2_range_0' : I_Airy2_range[0],
                        'I_Airy2_range_1' : I_Airy2_range[1],
                        'I_Airy2_do_fit' : I_Airy2_do_fit,
                        'x1_um' : x1_um,
                        'x1_um_range_0' : x1_um_range[0],
                        'x1_um_range_1' : x1_um_range[1],
                        'x1_um_do_fit' : x1_um_do_fit,
                        'x2_um' : x2_um,
                        'x2_um_range_0' : x2_um_range[0],
                        'x2_um_range_1' : x2_um_range[1],
                        'x2_um_do_fit' : x2_um_do_fit,
                        'normfactor' :  normfactor,
                        'normfactor_range_0' : normfactor_range[0],
                        'normfactor_range_1' : normfactor_range[1],
                        'normfactor_do_fit' : normfactor_do_fit,
                        'mod_sigma_um' : mod_sigma_um,
                        'mod_sigma_um_range_0' : mod_sigma_um_range[0],
                        'mod_sigma_um_range_1' : mod_sigma_um_range[1],
                        'mod_sigma_um_do_fit' : mod_sigma_um_do_fit,
                        'mod_shiftx_um' : mod_shiftx_um,
                        'mod_shiftx_um_range_0' : mod_shiftx_um_range[0],
                        'mod_shiftx_um_range_1' : mod_shiftx_um_range[1],
                        'mod_shiftx_um_do_fit' : mod_shiftx_um_do_fit,
                        # fitting results
                        'gamma_fit' :  gamma_fit,
                        'gamma_fit_v2' :  gamma_fit_v2,
                        'xi_um_fit' :  xi_um_fit,  # add this first to the df_fits dataframe
                        'xi_um_fit_v2' :  xi_um_fit_v2,  # add this first to the df_fits dataframe
                        'wavelength_nm_fit' :  wavelength_nm_fit,
                        'd_um_at_detector' :  d_um_at_detector,
                        'I_Airy1_fit' :  I_Airy1_fit,
                        'I_Airy2_fit' :  I_Airy2_fit,
                        'w1_um_fit' :  w1_um_fit,
                        'w2_um_fit' :  w2_um_fit,
                        'shiftx_um_fit' :  shiftx_um_fit,
                        'x1_um_fit' :  x1_um_fit,
                        'x2_um_fit' :  x2_um_fit,
                        'mod_sigma_um_fit' :  mod_sigma_um_fit,
                        'mod_shiftx_um_fit' :  mod_shiftx_um_fit,
                        'chi2distance_fitting' : chi2distance                       
                    }, ignore_index = True
                )
            df_fitting_v2_results = df_fitting_v2_results.drop_duplicates()

            




        # print('fringeseparation_px=' + str(round(fringeseparation_px,2)))

        # textarea_widget.value = result.fit_report()

        # fittingprogress_widget.value = 8
        # statustext_widget.value = 'Generating Plot ...'

        # Plotting

        #     fig=plt.figure(figsize=(11.69,8.27), dpi= 150, facecolor='w', edgecolor='k')  # A4 sheet in landscape
        fig = plt.figure(constrained_layout=False, figsize=(8.27, 11.69), dpi=150)

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2])
        gs.update(hspace=0.1)

        #     ax2 = plt.subplot(2,1,2)
        ax10 = fig.add_subplot(gs[1, 0])

        im_ax10 = ax10.imshow(
            pixis_image_norm,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            cmap="jet",
            vmin=0,
            vmax=1,
            extent=((-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3, -n / 2 * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3),
        )

        # fig.colorbar(im_ax2, ax=ax2, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

        ax10.add_patch(
            patches.Rectangle(
                ((-n / 2) * dX_1 * 1e3, (int(round(pixis_centery_px)) - n / 2 - pixis_profile_avg_width / 2) * dX_1 * 1e3),
                n * dX_1 * 1e3,
                pixis_profile_avg_width * dX_1 * 1e3,
                color="w",
                linestyle="-",
                alpha=0.8,
                fill=False,  # remove background
            )
        )

        ax10.set_xlabel("x / mm", fontsize=14)
        ax10.set_ylabel("y / mm", fontsize=14)
        ax10.grid(color="w", linewidth=1, alpha=0.5, linestyle="--", which="major")

        ax00 = fig.add_subplot(gs[0, 0], sharex=ax10)
        #     ax = plt.subplot(2,1,1)

        #     plt.plot(list(range(pixis_profile_avg.size)),ydata, color='r', linewidth=2)
        #     plt.plot(list(range(pixis_profile_avg.size)),result.best_fit, color='b', linewidth=0.5)
        ax00.plot(xdata * 1e3, ydata, color="r", linewidth=2, label="data")
        ax00.plot(xdata * 1e3, result.best_fit, color="b", linewidth=1, label="fit")

        Airy1 = [
            I_Airy1_fit
            * Airy(
                (x - shiftx_um_fit * 1e-6),
                w1_um_fit * 1e-6,
                wavelength_nm_fit * 1e-9,
                z_mm_fit * 1e-3,
                x1_um_fit * 1e-6,
            )
            ** 2
            for x in xdata
        ]
        Airy1 = normalize(Airy1) / I_Airy2_fit
        Airy2 = [
            I_Airy2_fit
            * Airy(
                (x - shiftx_um_fit * 1e-6),
                w2_um_fit * 1e-6,
                wavelength_nm_fit * 1e-9,
                z_mm_fit * 1e-3,
                x2_um_fit * 1e-6,
            )
            ** 2
            for x in xdata
        ]
        Airy2 = normalize(Airy2)

        do_plot_Airys = False
        if do_plot_Airys == True:
            plt.plot(xdata * 1e3, Airy1, color="k", label="Airy1", linewidth=1)
            plt.plot(xdata * 1e3, Airy2, color="grey", label="Airy2", linewidth=1)

        # plt.vlines([x1_loc_px_fit, pixis_centerx_px, x2_loc_px_fit],0,1)
        ax00.vlines(
            [(shiftx_um_fit + x1_um_fit) * 1e-3, shiftx_um_fit * 1e-3, (shiftx_um_fit + x2_um_fit) * 1e-3], 0, 0.1
        )
        #     ax00.annotate('xshift',
        #                xy=((shiftx_um_fit)*1e-3,0), xycoords='data',
        #                xytext=(0,-20), textcoords='offset points',
        #                 bbox=dict(boxstyle="round", fc="w"),
        #                 arrowprops=dict(arrowstyle="->"))
        ax00.annotate(
            "$x_2$",
            xy=((shiftx_um_fit + x2_um_fit) * 1e-3, 0),
            xycoords="data",
            xytext=(0, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        ax00.annotate(
            "$x_1$",
            xy=((shiftx_um_fit + x1_um_fit) * 1e-3, 0),
            xycoords="data",
            xytext=(0, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )

        # plt.hlines(0,0,n)

        textstr = "\n".join(
            (
                r"imageid=%.2f" % (imageid,),
                r"shiftx_um=%.2f" % (shiftx_um_fit,),
                r"$\lambda=%.2f$nm" % (wavelength_nm_fit,),
                r"fringesepar_um=%.2f" % (fringeseparation_um,),
                r"w1_um=%.2f" % (w1_um_fit,),
                r"w2_um=%.2f" % (w2_um_fit,),
                r"I_Airy1=%.2f" % (I_Airy1_fit,),
                r"I_Airy2=%.2f" % (I_Airy2_fit,),
                r"x1_um=%.2f" % (x1_um_fit,),
                r"x2_um=%.2f" % (x2_um_fit,),
                r"$\gamma=%.2f$" % (gamma_fit,),
                r"$\gamma=%.2f$" % (gamma_fit_v2,),
                r"normfactor=%.2f" % (normfactor_fit,),
                r"d_um_at_detector=%.2f" % (d_um_at_detector,),
            )
        )

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords

        if do_textbox == True:
            ax.text(1, 0.95, textstr, transform=ax.transAxes, fontsize=6, verticalalignment="top", bbox=props)

        ax00.set_xlim([(-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3])
        ax00.set_ylim([0, 1])

        ax00.set_ylabel("Intensity / a.u.", fontsize=14)
        ax00.legend()

        textstr = " ".join(
            (
                "ph-" + pinholes + ".id" + str(int(imageid)),
                r"$\lambda=%.2f$nm" % (result.params["wavelength_nm"].value,),
                orientation,
                "$d$=" + str(int(separation_um)) + "um",
                r"$d_{det}$=%.2fum" % (d_um_at_detector,),
                "\n",
                r"$w_1$=%.2fum" % (result.params["w1_um"].value,),
                r"$w_2$=%.2fum" % (result.params["w2_um"].value,),
                r"$I_1$=%.2f" % (result.params["I_Airy1"].value,),
                r"$I_2=$%.2f" % (result.params["I_Airy2"].value,),
                r"$\gamma=%.2f$" % (gamma_fit,),
                r"$\gamma_c=%.2f$" % (gamma_fit_v2,),
                r"$\xi=%.2fum$" % (xi_um_fit,),
                r"$\xi_c=%.2fum$" % (xi_um_fit_v2,),
            )
        )
        ax00.set_title(textstr, fontsize=10)

        if savefigure == True:
            savefigure_dir = str(scratch_dir) + "/" + dph_settings_bgsubtracted_widget.value.name
            if os.path.isdir(savefigure_dir) == False:
                os.mkdir(savefigure_dir)
            # savefigure_dir = str(scratch_dir) + '/' + hdf5_file_name_image + '_ph_'+str(ph) + '_d_'+str(separation_um)
            savefigure_dir = (
                str(scratch_dir)
                + "/"
                + dph_settings_bgsubtracted_widget.value.name
                + "/"
                + "profilewidth_px_"
                + str(int(pixis_avg_width))
                + "_"
                + "bg_intervall_um_"
                + str(int(beamposition_horizontal_interval))
            )
            if os.path.isdir(savefigure_dir) == False:
                os.mkdir(savefigure_dir)
            savefigure_dir = (
                str(scratch_dir)
                + "/"
                + dph_settings_bgsubtracted_widget.value.name
                + "/"
                + "profilewidth_px_"
                + str(int(pixis_avg_width))
                + "_"
                + "bg_intervall_um_"
                + str(int(beamposition_horizontal_interval))
                + "/profiles_fit/"
            )
            if os.path.isdir(savefigure_dir) == False:
                os.mkdir(savefigure_dir)
            plt.savefig(
                savefigure_dir
                + "/"
                + "profiles_fit_"
                + hdf5_file_name_image_widget.value
                + "_ph_"
                + str(pinholes)
                + "_d_"
                + str(separation_um)
                + "_E_"
                + str(format(energy_hall_uJ, ".4f")).zfill(6)
                + "_image_"
                + str(imageid)
                + ".png",
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                papertype=None,
                format=None,
                transparent=False,
                bbox_inches=None,
                pad_inches=0.1,
                frameon=None,
            )
            plt.savefig(
                savefigure_dir
                + "/"
                + "profiles_fit_"
                + hdf5_file_name_image_widget.value
                + "_ph_"
                + str(pinholes)
                + "_d_"
                + str(separation_um)
                + "_E_"
                + str(format(energy_hall_uJ, ".4f")).zfill(6)
                + "_image_"
                + str(imageid)
                + ".pdf",
                dpi=None,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                papertype=None,
                format=None,
                transparent=False,
                bbox_inches=None,
                pad_inches=0.1,
                frameon=None,
            )

        

            
            
            


        plt.show()
        # fittingprogress_widget.value = 10
        # fittingprogress_widget.bar_style = 'success'
        # statustext_widget.value = 'done'

        # print(gamma_fit)






def plot_deconvmethod(
    do_plot_deconvmethod,
    wienerimplementation,
    balance,
    snr_db,
    pixis_profile_avg_width,
    xi_um_guess,
    scan_x,
    xatol,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    # hdf5_file_path,
    # imageid,
    save_to_df,
    create_steps_figures,
    create_figure    
):
    global df_deconvmethod_1d_v2_results
    global df_deconvmethod_2d_v2_results
    global df_deconvmethod_1d_v3_results
    global df_deconvmethod_2d_v3_results

    

    if do_plot_deconvmethod == True:

        start = datetime.now()


        # Loading and preparing

        imageid = imageid_widget.value
        hdf5_file_path = dph_settings_bgsubtracted_widget.value

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]
            # pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
            #     np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            # ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][0]

        end = datetime.now()
        time_taken = end - start
        statustext_widget.value = 'Loading from HDF5: ' + str(time_taken)    

        pinholes = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["pinholes"].iloc[0]
        separation_um = get_sep_and_orient(pinholes)[0]
        orientation = get_sep_and_orient(pinholes)[1]
        setting_wavelength_nm = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_wavelength_nm"].iloc[0]
        energy_hall_uJ = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["energy hall"].iloc[0]

        pixis_profile_avg = pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:]

        if scan_x == True:
            if wienerimplementation == 'scikit':
                deconvmethod_2d_v2_result_widget.value = 'calculating ...'
            if wienerimplementation == 'opencv':
                deconvmethod_2d_v3_result_widget.value =  'calculating ...'
        else:
            if wienerimplementation == 'scikit':
                deconvmethod_1d_v2_result_widget.value = 'calculating ...'
            if wienerimplementation == 'opencv':
                deconvmethod_1d_v3_result_widget.value = 'calculating ...'
        partiallycoherent = pixis_image_norm
        z = 5781 * 1e-3
        dX_1 = 13 * 1e-6
        profilewidth = 200  # pixis_avg_width  # defined where?
        pixis_centery_px = int(pixis_centery_px)
        wavelength = setting_wavelength_nm * 1e-9
        # xi_um_guess = 475
        # guess sigma_y_F_gamma_um based on the xi_um_guess assuming to be the beams intensity rms width

        pixis_profile_avg = np.average(pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:],axis=0)
        pixis_profile_avg = pixis_profile_avg / np.max(pixis_profile_avg)

        n = pixis_profile_avg.size  # number of sampling point  # number of pixels
        dX_1 = 13e-6
        xdata = np.linspace((-n / 2) * dX_1, (+n / 2 - 1) * dX_1, n)
        ydata = pixis_profile_avg  # defined in the cells above, still to implement: select
        
        sigma_y_F_gamma_um_guess = calc_sigma_F_gamma_um(xi_um_guess, n, dX_1, setting_wavelength_nm, False)
        # create_steps_figures = True
        savefigure_dir = scratch_dir


        end = datetime.now()
        time_taken = end - start
        statustext_widget.value = 'Start Deconvmethod (' + wienerimplementation +'): ' + str(time_taken) 

        
        # Ignoring OptimizeWarning. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                (
                    partiallycoherent_profile,
                    fullycoherent_opt,
                    fullycoherent_profile_opt,
                    partiallycoherent_rec,
                    partiallycoherent_rec_profile,
                    sigma_x_F_gamma_um_opt,
                    sigma_y_F_gamma_um,
                    F_gamma,
                    abs_gamma,
                    xi_x_um,
                    xi_y_um,
                    I_bp,
                    dX_2,
                    chi2distance,
                ) = deconvmethod(
                    wienerimplementation,
                    balance,
                    snr_db,
                    partiallycoherent,
                    z,
                    dX_1,
                    profilewidth,
                    pixis_centery_px,
                    wavelength,
                    xi_um_guess,
                    sigma_y_F_gamma_um_guess,
                    crop_px,
                    sigma_x_F_gamma_um_multiplier,
                    scan_x,
                    xatol,
                    create_steps_figures,
                    savefigure_dir
                )
            except:
                print('deconvmethod called in plot_deconvmethod failed!')
                xi_x_um = np.nan
        
        if scan_x == True:
            if wienerimplementation == 'scikit':
                deconvmethod_2d_v2_result_widget.value = r"%.2fum" % (xi_x_um) + r", %.2fum" % (xi_y_um)
            if wienerimplementation == 'opencv':
                deconvmethod_2d_v3_result_widget.value =  r"%.2fum" % (xi_x_um) + r", %.2fum" % (xi_y_um)
        else:
            if wienerimplementation == 'scikit':
                deconvmethod_1d_v2_result_widget.value = r"%.2fum" % (xi_x_um)
            if wienerimplementation == 'opencv':
                deconvmethod_1d_v3_result_widget.value = r"%.2fum" % (xi_x_um)
        # str(round(xi_x_um, 2)) + ', ' + str(round(xi_y_um, 2))

        if save_to_df == True:
            measurement = os.path.splitext(os.path.basename(dph_settings_bgsubtracted_widget.value))[0]

            if scan_x == True:
                if wienerimplementation == 'scikit':
                    df_deconvmethod_2d_v2_results = df_deconvmethod_2d_v2_results.append(
                        {
                            # image identifiers
                            'measurement' : measurement,
                            'timestamp_pulse_id' : timestamp_pulse_id,
                            'imageid' : imageid,
                            'separation_um' : separation_um,
                            # deconvolution parameters
                            'pixis_profile_avg_width' : pixis_profile_avg_width,
                            'crop_px' : crop_px,
                            'balance' : balance,
                            'xi_um_guess' : xi_um_guess,
                            'sigma_x_F_gamma_um_multiplier' : sigma_x_F_gamma_um_multiplier,
                            'xatol' : xatol,
                            # deconvolution results
                            'xi_x_um_v2' : xi_x_um,
                            'xi_y_um_v2' : xi_y_um,
                            'chi2distance_deconvmethod_2d_v2' : chi2distance                        
                        }, ignore_index = True
                    )
                    df_deconvmethod_2d_v2_results = df_deconvmethod_2d_v2_results.drop_duplicates()
                if wienerimplementation == 'opencv':
                    df_deconvmethod_2d_v3_results = df_deconvmethod_2d_v3_results.append(
                        {
                            # image identifiers
                            'measurement' : measurement,
                            'timestamp_pulse_id' : timestamp_pulse_id,
                            'imageid' : imageid,
                            'separation_um' : separation_um,
                            # deconvolution parameters
                            'pixis_profile_avg_width' : pixis_profile_avg_width,
                            'crop_px' : crop_px,
                            'snr_db' : snr_db,
                            'xi_um_guess' : xi_um_guess,
                            'sigma_x_F_gamma_um_multiplier' : sigma_x_F_gamma_um_multiplier,
                            'xatol' : xatol,
                            # deconvolution results
                            'xi_x_um_v3' : xi_x_um,
                            'xi_y_um_v3' : xi_y_um,
                            'chi2distance_deconvmethod_2d_v3' : chi2distance                        
                        }, ignore_index = True
                    )
                    df_deconvmethod_2d_v3_results = df_deconvmethod_2d_v3_results.drop_duplicates()
            else:
                if wienerimplementation == 'scikit':
                    df_deconvmethod_1d_v2_results = df_deconvmethod_1d_v2_results.append(
                        {
                            # image identifiers
                            'measurement' : measurement,
                            'timestamp_pulse_id' : timestamp_pulse_id,
                            'imageid' : imageid,
                            'separation_um' : separation_um,
                            # deconvolution parameters
                            'pixis_profile_avg_width' : pixis_profile_avg_width,
                            'crop_px' : crop_px,
                            'balance' : balance,
                            'xi_um_guess' : xi_um_guess,
                            'sigma_x_F_gamma_um_multiplier' : sigma_x_F_gamma_um_multiplier,
                            # deconvolution results
                            # 'sigma_F_gamma_um_opt' : sigma_F_gamma_um_opt, not calculated?
                            'xi_um_v2' : xi_x_um,
                            'chi2distance_deconvmethod_1d_v2' : chi2distance    
                        }, ignore_index = True
                    )
                    df_deconvmethod_1d_v2_results = df_deconvmethod_1d_v2_results.drop_duplicates()
                if wienerimplementation == 'opencv':
                    df_deconvmethod_1d_v3_results = df_deconvmethod_1d_v3_results.append(
                        {
                            # image identifiers
                            'measurement' : measurement,
                            'timestamp_pulse_id' : timestamp_pulse_id,
                            'imageid' : imageid,
                            'separation_um' : separation_um,
                            # deconvolution parameters
                            'pixis_profile_avg_width' : pixis_profile_avg_width,
                            'crop_px' : crop_px,
                            'snr_db' : snr_db,
                            'xi_um_guess' : xi_um_guess,
                            'sigma_x_F_gamma_um_multiplier' : sigma_x_F_gamma_um_multiplier,
                            # deconvolution results
                            # 'sigma_F_gamma_um_opt' : sigma_F_gamma_um_opt, not calculated?
                            'xi_um_v3' : xi_x_um,
                            'chi2distance_deconvmethod_1d_v3' : chi2distance    
                        }, ignore_index = True
                    )
                    df_deconvmethod_1d_v3_results = df_deconvmethod_1d_v3_results.drop_duplicates()

        if create_figure == True:
            if np.isnan(xi_x_um) == False:
                fig = plt.figure(constrained_layout=False, figsize=(8.27, 11.69), dpi=150)

                gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2])
                gs.update(hspace=0.1)

                #     ax2 = plt.subplot(2,1,2)
                ax10 = fig.add_subplot(gs[1, 0])

                im_ax10 = ax10.imshow(
                    pixis_image_norm,
                    origin="lower",
                    interpolation="nearest",
                    aspect="auto",
                    cmap="jet",
                    vmin=0,
                    vmax=1,
                    extent=((-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3, -n / 2 * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3),
                )

                # fig.colorbar(im_ax2, ax=ax2, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

                ax10.add_patch(
                    patches.Rectangle(
                        ((-n / 2) * dX_1 * 1e3, (int(round(pixis_centery_px)) - n / 2 - pixis_profile_avg_width / 2) * dX_1 * 1e3),
                        n * dX_1 * 1e3,
                        pixis_profile_avg_width * dX_1 * 1e3,
                        color="w",
                        linestyle="-",
                        alpha=0.8,
                        fill=False,  # remove background
                    )
                )

                ax10.set_xlabel("x / mm", fontsize=14)
                ax10.set_ylabel("y / mm", fontsize=14)
                ax10.grid(color="w", linewidth=1, alpha=0.5, linestyle="--", which="major")

                ax00 = fig.add_subplot(gs[0, 0], sharex=ax10)
                #     ax = plt.subplot(2,1,1)

                #     plt.plot(list(range(pixis_profile_avg.size)),ydata, color='r', linewidth=2)
                #     plt.plot(list(range(pixis_profile_avg.size)),result.best_fit, color='b', linewidth=0.5)
                ax00.plot(xdata * 1e3, ydata, color="r", linewidth=2, label="data")
                ax00.plot(xdata * 1e3, partiallycoherent_rec_profile, color="g", linewidth=1, label="recovered partially coherent")
                ax00.plot(xdata * 1e3, fullycoherent_profile_opt, color="k", linewidth=0.5, label="fully coherent")
                

                ax00.set_xlim([(-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3])
                ax00.set_ylim([0, 1])

                ax00.set_ylabel("Intensity / a.u.", fontsize=14)
                ax00.legend()

                textstr = " ".join(
                    (
                        "ph-" + pinholes + ".id" + str(int(imageid)),
                        r"$\lambda=%.2f$nm" % (df0[df0['timestamp_pulse_id'] == timestamp_pulse_id]['wavelength_nm_fit'],),
                        orientation,
                        "\n",
                        "$d$=" + str(int(separation_um)) + "um",
                        r"$\gamma=%.2f$" % (df0[df0['timestamp_pulse_id'] == timestamp_pulse_id]['gamma_fit'],),
                        r"$\xi_x=%.2fum$" % (xi_x_um,),
                    )
                )
                ax00.set_title(textstr, fontsize=10)


                plt.show()
                # fittingprogress_widget.value = 10
                # fittingprogress_widget.bar_style = 'success'
                # statustext_widget.value = 'done'

                # print(gamma_fit)

        end = datetime.now()
        time_taken = end - start
        statustext_widget.value = 'End Deconvmethod: ' + str(time_taken)    


def plot_deconvmethod_1d_v2(
    do_plot_deconvmethod_1d_v2,
    balance,
    pixis_profile_avg_width,
    xi_um_guess,
    xatol,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    # hdf5_file_path,
    # imageid,
    save_to_df,
    create_steps_figures,
    create_figure 
):
    do_plot_deconvmethod = do_plot_deconvmethod_1d_v2
    wienerimplementation = 'scikit'
    scan_x = False
    plot_deconvmethod(
        do_plot_deconvmethod,
        wienerimplementation,
        balance,
        _, #snr_db
        pixis_profile_avg_width,
        xi_um_guess,
        scan_x,
        xatol,
        sigma_x_F_gamma_um_multiplier,
        crop_px,
        # hdf5_file_path,
        # imageid,
        save_to_df,
        create_steps_figures,
        create_figure
        )

def plot_deconvmethod_2d_v2(
    do_plot_deconvmethod_2d_v2,
    balance,
    pixis_profile_avg_width,
    xi_um_guess,
    xatol,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    # hdf5_file_path,
    # imageid,
    save_to_df,
    create_steps_figures,
    create_figure 
):
    do_plot_deconvmethod = do_plot_deconvmethod_2d_v2
    wienerimplementation = 'scikit'
    scan_x = True
    plot_deconvmethod(
        do_plot_deconvmethod,
        wienerimplementation,
        balance,
        _, #snr_db
        pixis_profile_avg_width,
        xi_um_guess,
        scan_x,
        xatol,
        sigma_x_F_gamma_um_multiplier,
        crop_px,
        # hdf5_file_path,
        # imageid,
        save_to_df,
        create_steps_figures,
        create_figure
        )

def plot_deconvmethod_1d_v3(
    do_plot_deconvmethod_1d_v3,
    snr_db,
    pixis_profile_avg_width,
    xi_um_guess,
    xatol,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    # hdf5_file_path,
    # imageid,
    save_to_df,
    create_steps_figures,
    create_figure 
):
    do_plot_deconvmethod = do_plot_deconvmethod_1d_v3
    wienerimplementation = 'opencv'
    scan_x = False
    plot_deconvmethod(
        do_plot_deconvmethod,
        wienerimplementation,
        _, #balance
        snr_db,
        pixis_profile_avg_width,
        xi_um_guess,
        scan_x,
        xatol,
        sigma_x_F_gamma_um_multiplier,
        crop_px,
        # hdf5_file_path,
        # imageid,
        save_to_df,
        create_steps_figures,
        create_figure
        )

def plot_deconvmethod_2d_v3(
    do_plot_deconvmethod_2d_v3,
    snr_db,
    pixis_profile_avg_width,
    xi_um_guess,
    xatol,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    # hdf5_file_path,
    # imageid,
    save_to_df,
    create_steps_figures,
    create_figure 
):
    do_plot_deconvmethod = do_plot_deconvmethod_2d_v3
    wienerimplementation = 'opencv'
    
    scan_x = True
    plot_deconvmethod(
        do_plot_deconvmethod,
        wienerimplementation,
        _, #balance
        snr_db,
        pixis_profile_avg_width,
        xi_um_guess,
        scan_x,
        xatol,
        sigma_x_F_gamma_um_multiplier,
        crop_px,
        # hdf5_file_path,
        # imageid,
        save_to_df,
        create_steps_figures,
        create_figure
        )


create_steps_figures_widget = widgets.Checkbox(value=False, description="create step figure")
create_figure_widget = widgets.Checkbox(value=False, description="create figure")

do_plot_deconvmethod_steps_widget = widgets.Checkbox(value=False, description="Do")
clear_plot_deconvmethod_steps_widget = widgets.Checkbox(value=False, description="Clear")

deconvmethod_step_widget = widgets.BoundedIntText(
    value=2,
    min=0,
    max=10,
    description='Step:',
    disabled=False
)

deconvmethod_ystep_widget = widgets.BoundedIntText(
    value=0,
    min=0,
    description='yStep:',
    disabled=False
)


def plot_deconvmethod_steps(do_plot_deconvmethod_steps, clear_plot_deconvmethod_steps, step, ystep):

    if do_plot_deconvmethod_steps == True:
        savefigure_dir = str(scratch_dir) + '/' + 'deconvmethod_steps'
        image_path_name = savefigure_dir + '/' + 'ystep_' + str(ystep) + '_step_' + str(step) + '.png'
        image = mpimg.imread(image_path_name)
        plt.figure(figsize=(10, 6), dpi=300)
        plt.imshow(image) 
        plt.axis('off')
        plt.show()  # display it

        df_deconv_scany = pd.read_csv(Path.joinpath(scratch_dir, 'deconvmethod_steps', "sigma_y_F_gamma_um_guess_scan.csv"),
                              header=None, names=['ystep', 'sigma_y_F_gamma_um_guess', 'chi2distance'])
        df_deconv_scany.plot.scatter('ystep', 'chi2distance')

    if clear_plot_deconvmethod_steps == True:
        clear_output()


def plot_deconvmethod_2d_v1(
    do_plot_deconvmethod_2d_v1,
    pixis_profile_avg_width,
    crop_px,
    sigma_x_F_gamma_um_min, 
    sigma_x_F_gamma_um_max, 
    sigma_x_F_gamma_um_stepsize,
    sigma_y_F_gamma_um_min, 
    sigma_y_F_gamma_um_max, 
    sigma_y_F_gamma_um_stepsize,
    save_to_df    
):
    if do_plot_deconvmethod_2d_v1 == True:

        global df_deconvmethod_2d_v1_results

        

        # Loading and preparing

        imageid = imageid_widget.value
        hdf5_file_path = dph_settings_bgsubtracted_widget.value

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]
            # pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
            #     np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            # ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][0]

        pinholes = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["pinholes"].iloc[0]
        separation_um = get_sep_and_orient(pinholes)[0]
        orientation = get_sep_and_orient(pinholes)[1]
        setting_wavelength_nm = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_wavelength_nm"].iloc[0]
        energy_hall_uJ = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["energy hall"].iloc[0]

        pixis_profile_avg = pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:]


        partiallycoherent = pixis_image_norm
        z = 5781 * 1e-3
        dX_1 = 13 * 1e-6
        profilewidth = 200  # pixis_avg_width  # defined where?
        pixis_centery_px = int(pixis_centery_px)
        wavelength = setting_wavelength_nm * 1e-9
        # xi_um_guess = 475
        # guess sigma_y_F_gamma_um based on the xi_um_guess assuming to be the beams intensity rms width

        pixis_profile_avg = np.average(pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:],axis=0)
        pixis_profile_avg = pixis_profile_avg / np.max(pixis_profile_avg)

        n = pixis_profile_avg.size  # number of sampling point  # number of pixels
        dX_1 = 13e-6
        xdata = np.linspace((-n / 2) * dX_1, (+n / 2 - 1) * dX_1, n)
        ydata = pixis_profile_avg  # defined in the cells above, still to implement: select
        

        create_figure = True
        savefigure_dir = scratch_dir


        # sigma_x_F_gamma_um_min = 7
        # sigma_x_F_gamma_um_max = 40

        # sigma_y_F_gamma_um_min = 7
        # sigma_y_F_gamma_um_max = 40
        # sigma_y_F_gamma_um_stepsize = 1

        statustext_widget.value = 'deconvmethod_2d_v1 rough scan ...'

        (
            partiallycoherent_profile, 
            fullycoherent_opt_list, 
            fullycoherent_profile_opt_list,  
            partiallycoherent_rec_list, 
            partiallycoherent_rec_profile_list, 
            partiallycoherent_rec_profile_min_list, 
            delta_rec_min_list, 
            delta_profiles_cropped_list, 
            sigma_x_F_gamma_um_opt, 
            sigma_y_F_gamma_um_list, 
            F_gamma_list, 
            abs_gamma_list, 
            xi_x_um_list, 
            xi_y_um_list, 
            I_bp, 
            dX_2, 
            cor_list, 
            cor_profiles_list, 
            cor_profiles_cropped_list, 
            index_opt,
            chi2distance_list
        ) = deconvmethod_v1(
            partiallycoherent, 
            z, 
            dX_1, 
            profilewidth, 
            pixis_centery_px, 
            wavelength, 
            sigma_x_F_gamma_um_min, 
            sigma_x_F_gamma_um_max, 
            sigma_y_F_gamma_um_min, 
            sigma_y_F_gamma_um_max, 
            sigma_y_F_gamma_um_stepsize, 
            crop_px
        )

        # chi2distance_list = []
        # for partiallycoherent_rec in partiallycoherent_rec_list:
        #     number_of_bins = 100
        #     hist1, bin_edges1 = np.histogram(partiallycoherent.ravel(), bins=np.linspace(0,1,number_of_bins))
        #     hist2, bin_edges2 = np.histogram(partiallycoherent_rec.ravel(), bins=np.linspace(0,1,number_of_bins))
        #     chi2distance_list.append(chi2_distance(hist1, hist2))

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

        plt.close(fig)

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

            plt.close(fig)


        statustext_widget.value = 'deconvmethod_2d_v1 fine scan ...'

        sigma_y_F_gamma_um_min = sigma_y_F_gamma_um_list[index_opt] - 0.5
        sigma_y_F_gamma_um_max = sigma_y_F_gamma_um_list[index_opt] + 0.5
        sigma_y_F_gamma_um_stepsize = 0.1

        (
        partiallycoherent_profile, 
        fullycoherent_opt_list, 
        fullycoherent_profile_opt_list,  
        partiallycoherent_rec_list, 
        partiallycoherent_rec_profile_list, 
        partiallycoherent_rec_profile_min_list, 
        delta_rec_min_list, 
        delta_profiles_cropped_list, 
        sigma_x_F_gamma_um_opt, 
        sigma_y_F_gamma_um_list, 
        F_gamma_list, 
        abs_gamma_list, 
        xi_x_um_list, 
        xi_y_um_list, 
        I_bp, 
        dX_2, 
        cor_list, 
        cor_profiles_list, 
        cor_profiles_cropped_list, 
        index_opt,
        chi2distance_list
        ) = deconvmethod_v1(
            partiallycoherent, 
            z, 
            dX_1, 
            profilewidth, 
            pixis_centery_px, 
            wavelength, 
            sigma_x_F_gamma_um_min, 
            sigma_x_F_gamma_um_max, 
            sigma_y_F_gamma_um_min, 
            sigma_y_F_gamma_um_max, 
            sigma_y_F_gamma_um_stepsize, 
            crop_px
        )


        # chi2distance_list = []
        # for partiallycoherent_rec in partiallycoherent_rec_list:
        #     number_of_bins = 100
        #     hist1, bin_edges1 = np.histogram(partiallycoherent.ravel(), bins=np.linspace(0,1,number_of_bins))
        #     hist2, bin_edges2 = np.histogram(partiallycoherent_rec.ravel(), bins=np.linspace(0,1,number_of_bins))
        #     chi2distance_list.append(chi2_distance(hist1, hist2))

        index_opt = np.where(np.asarray(chi2distance_list) == np.min(np.asarray(chi2distance_list)))[0][0]


        xi_um = xi_x_um_list[index_opt]
        xi_x_um = xi_x_um_list[index_opt]
        xi_y_um = xi_y_um_list[index_opt]

        deconvmethod_2d_v1_result_widget.value = r"%.2fum" % (xi_x_um) + r", %.2fum" % (xi_y_um)

        print('sigma_x_F_gamma_um_opt='+str(sigma_x_F_gamma_um_opt))
        print('sigma_y_F_gamma_um_list[index_opt]='+str(sigma_y_F_gamma_um_list[index_opt]))
        print('xi_x_um_list[index_opt]='+str(xi_x_um_list[index_opt]))
        print('xi_y_um_list[index_opt]='+str(xi_y_um_list[index_opt]))

        
        if save_to_df == True:
            df_deconvmethod_2d_v1_results = df_deconvmethod_2d_v1_results.append(
                    {
                        # image identifiers
                        'measurement' : measurement,
                        'timestamp_pulse_id' : timestamp_pulse_id,
                        'imageid' : imageid,
                        'separation_um' : separation_um,
                        # deconvolution parameters
                        'pixis_profile_avg_width' : pixis_profile_avg_width,
                        'crop_px' : crop_px,
                        'sigma_x_F_gamma_um_min' : sigma_x_F_gamma_um_min,
                        'sigma_x_F_gamma_um_max' : sigma_x_F_gamma_um_max,
                        'sigma_y_F_gamma_um_min' : sigma_y_F_gamma_um_min,
                        'sigma_y_F_gamma_um_max' : sigma_y_F_gamma_um_max,
                        'sigma_y_F_gamma_um_stepsize' : sigma_y_F_gamma_um_stepsize,
                        # deconvolution results
                        'xi_x_um_v1' : xi_x_um,
                        'xi_y_um_v1' : xi_y_um,
                        'chi2distance_deconvmethod_2d_v1' : chi2distance_list[index_opt]                     
                    }, ignore_index = True
                )
            df_deconvmethod_2d_v1_results = df_deconvmethod_2d_v1_results.drop_duplicates()




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
        plt.close(fig)


        if np.isnan(xi_x_um) == False:
            n = pixis_profile_avg.size  # number of sampling point  # number of pixels
            dX_1 = 13e-6
            xdata = np.linspace((-n / 2) * dX_1, (+n / 2 - 1) * dX_1, n)


            fig = plt.figure(constrained_layout=False, figsize=(8.27, 11.69), dpi=150)

            gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2])
            gs.update(hspace=0.1)

            #     ax2 = plt.subplot(2,1,2)
            ax10 = fig.add_subplot(gs[1, 0])

            im_ax10 = ax10.imshow(
                pixis_image_norm,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                cmap="jet",
                vmin=0,
                vmax=1,
                extent=((-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3, -n / 2 * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3),
            )

            # fig.colorbar(im_ax2, ax=ax2, pad=0.05, fraction=0.1, shrink=1.00, aspect=20, orientation='horizontal')

            ax10.add_patch(
                patches.Rectangle(
                    ((-n / 2) * dX_1 * 1e3, (int(round(pixis_centery_px)) - n / 2 - pixis_profile_avg_width / 2) * dX_1 * 1e3),
                    n * dX_1 * 1e3,
                    pixis_profile_avg_width * dX_1 * 1e3,
                    color="w",
                    linestyle="-",
                    alpha=0.8,
                    fill=False,  # remove background
                )
            )

            ax10.set_xlabel("x / mm", fontsize=14)
            ax10.set_ylabel("y / mm", fontsize=14)
            ax10.grid(color="w", linewidth=1, alpha=0.5, linestyle="--", which="major")

            ax00 = fig.add_subplot(gs[0, 0], sharex=ax10)
            #     ax = plt.subplot(2,1,1)

            #     plt.plot(list(range(pixis_profile_avg.size)),ydata, color='r', linewidth=2)
            #     plt.plot(list(range(pixis_profile_avg.size)),result.best_fit, color='b', linewidth=0.5)
            ax00.plot(xdata * 1e3, partiallycoherent_profile, color="r", linewidth=2, label="data")
            ax00.plot(xdata * 1e3, fullycoherent_profile_opt_list[index_opt], color="g", linewidth=1, label="recovered partially coherent")
            ax00.plot(xdata * 1e3, partiallycoherent_rec_profile_list[index_opt], color="k", linewidth=0.5, label="fully coherent")
            

            ax00.set_xlim([(-n / 2) * dX_1 * 1e3, (+n / 2 - 1) * dX_1 * 1e3])
            ax00.set_ylim([0, 1])

            ax00.set_ylabel("Intensity / a.u.", fontsize=14)
            ax00.legend()

            textstr = " ".join(
                (
                    "ph-" + pinholes + ".id" + str(int(imageid)),
                    r"$\lambda=%.2f$nm" % (df0[df0['timestamp_pulse_id'] == timestamp_pulse_id]['wavelength_nm_fit'],),
                    orientation,
                    "\n",
                    "$d$=" + str(int(separation_um)) + "um",
                    r"$\gamma=%.2f$" % (df0[df0['timestamp_pulse_id'] == timestamp_pulse_id]['gamma_fit'],),
                    r"$\xi_x=%.2fum$" % (xi_x_um,),
                )
            )
            ax00.set_title(textstr, fontsize=10)


            plt.show()


def plot_fitting_vs_deconvolution(
    do_plot_fitting_vs_deconvolution,
    dataset,
    measurement_file,
    timestamp_pulse_id,
    xi_um_deconv_column_and_label,
    xi_um_fit_column_and_label,
    chi2distance_column_and_label,
    deconvmethod_outlier_limit,
    fitting_outlier_limit,
    xaxisrange,
    yaxisrange,
    use_measurement_default_result
):

    if do_plot_fitting_vs_deconvolution == True:

        timestamp_pulse_id = timestamp_pulse_id_widget.value

        xi_um_deconv_column = xi_um_deconv_column_and_label[0]
        xi_um_deconv_label = xi_um_deconv_column_and_label[1]
        xi_um_fit_column = xi_um_fit_column_and_label[0]
        xi_um_fit_label = xi_um_fit_column_and_label[1]
        chi2distance_column = chi2distance_column_and_label[0]
        chi2distance_label = chi2distance_column_and_label[1]

        # Loading and preparing

        # get all the files in a dataset:
        files = []
        # for set in [list(datasets)[0]]:
        
        for measurement in datasets_selection[dataset]:
            # print(measurement)
            files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

        # testing:
        files = measurements_selection_widget.value

        # get all the timestamps in these files:        
        # datasets[list(datasets)[0]][0]

        pd.set_option('display.max_rows', None)

        timestamp_pulse_ids = []
        
        for f in files:
            timestamp_pulse_ids_measurement = []
            measurement = os.path.splitext(os.path.basename(f))[0]
            with h5py.File(f, "r") as hdf5_file:         
                timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
                timestamp_pulse_ids_measurement.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
            if use_measurement_default_result == True:
                if xi_um_deconv_column == 'xi_x_um_v1':
                    # deconvolution 2d v1 defaults:
                    balance_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]
                    df_deconvmethod_result = df_deconvmethod_2d_v1_results[(df_deconvmethod_2d_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_2d_v1_results['balance'] == balance_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['xatol'] == xatol_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v1','chi2distance_deconvmethod_2d_v1']].sort_values('chi2distance_deconvmethod_2d_v1',ascending=False)               
                if xi_um_deconv_column == 'xi_um_v2':
                    # deconvolution 2d v2 defaults:
                    balance_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0] 
                    df_deconvmethod_result = df_deconvmethod_1d_v2_results[(df_deconvmethod_1d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_1d_v2_results['balance'] == balance_measurement_default) & \
                        (df_deconvmethod_1d_v2_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_1d_v2_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_1d_v2_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','balance','xi_um_guess','xi_um_v2','chi2distance_deconvmethod_1d_v2']].sort_values('chi2distance_deconvmethod_1d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v2':
                    # deconvolution 2d v2 defaults:
                    balance_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]
                    df_deconvmethod_result = df_deconvmethod_2d_v2_results[(df_deconvmethod_2d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_2d_v2_results['balance'] == balance_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['xatol'] == xatol_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','balance','xi_um_guess','xi_x_um_v2','chi2distance_deconvmethod_2d_v2']].sort_values('chi2distance_deconvmethod_2d_v2',ascending=False)               
                if xi_um_deconv_column == 'xi_um_v3':
                    # deconvolution v3 defaults:
                    snr_db_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['snr_db_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0] 
                    df_deconvmethod_result = df_deconvmethod_1d_v3_results[(df_deconvmethod_1d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_1d_v3_results['snr_db'] == snr_db_measurement_default) & \
                        (df_deconvmethod_1d_v3_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_1d_v3_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_1d_v3_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','snr_db','xi_um_guess','xi_um_v3','chi2distance_deconvmethod_1d_v3']].sort_values('chi2distance_deconvmethod_1d_v3',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v3':
                    # deconvolution v3 defaults:
                    snr_db_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['snr_db_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]
                    df_deconvmethod_result = df_deconvmethod_2d_v3_results[(df_deconvmethod_2d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_2d_v3_results['snr_db'] == snr_db_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['xatol'] == xatol_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','snr_db','xi_um_guess','xi_x_um_v3','chi2distance_deconvmethod_2d_v3']].sort_values('chi2distance_deconvmethod_2d_v3',ascending=False)               
                 
                
                if xi_um_fit_column == 'xi_um_fit_v1':
                    df_fitting_result = df_fitting_v1_results[(df_fitting_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))][['separation_um','imageid','timestamp_pulse_id',xi_um_fit_column,'chi2distance_fitting_v1']].sort_values('chi2distance_fitting_v1',ascending=False)        
                if (xi_um_fit_column == 'xi_um_fit_v2') or (xi_um_fit_column == 'xi_um_fit'):
                    # fitting v2 defaults:
                    mod_sigma_um_measurement_default = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_sigma_um_measurement_default'].iloc[0]
                    mod_shiftx_um_measurement_default = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_shiftx_um_measurement_default'].iloc[0]
                    df_fitting_result = df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_fitting_v2_results['mod_sigma_um'] == mod_sigma_um_measurement_default) & \
                        (df_fitting_v2_results['mod_shiftx_um'] == mod_shiftx_um_measurement_default)][['separation_um','imageid','timestamp_pulse_id','mod_sigma_um', 'mod_sigma_um_fit','mod_shiftx_um','mod_shiftx_um_fit',xi_um_fit_column,'chi2distance_fitting']].sort_values('chi2distance_fitting',ascending=False)
        

            # https://datascienceparichay.com/article/pandas-groupby-minimum/
            if use_measurement_default_result == False:


                if xi_um_deconv_column == 'xi_x_um_v1':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_2d_v1_results,df_deconvmethod_2d_v1_results[(df_deconvmethod_2d_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v1']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v1'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v1','chi2distance_deconvmethod_2d_v1']].sort_values('chi2distance_deconvmethod_2d_v1',ascending=False)
                if xi_um_deconv_column == 'xi_um_v2':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_1d_v2_results,df_deconvmethod_1d_v2_results[(df_deconvmethod_1d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v2'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_um_v2','chi2distance_deconvmethod_1d_v2']].sort_values('chi2distance_deconvmethod_1d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v2':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_2d_v2_results,df_deconvmethod_2d_v2_results[(df_deconvmethod_2d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v2'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v2','chi2distance_deconvmethod_2d_v2']].sort_values('chi2distance_deconvmethod_2d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_um_v3':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_1d_v3_results,df_deconvmethod_1d_v3_results[(df_deconvmethod_1d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v3'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_um_v3','chi2distance_deconvmethod_1d_v3']].sort_values('chi2distance_deconvmethod_1d_v3',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v3':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_2d_v3_results,df_deconvmethod_2d_v3_results[(df_deconvmethod_2d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v3'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v3','chi2distance_deconvmethod_2d_v3']].sort_values('chi2distance_deconvmethod_2d_v3',ascending=False)
             

                if xi_um_fit_column == 'xi_um_fit_v1':
                    df_fitting_result = pd.merge(df_fitting_v1_results,df_fitting_v1_results[(df_fitting_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting_v1']].min(), on=['timestamp_pulse_id','chi2distance_fitting_v1'])[['separation_um','imageid','timestamp_pulse_id',xi_um_fit_column,'chi2distance_fitting_v1']].sort_values('chi2distance_fitting_v1',ascending=False)
                if (xi_um_fit_column == 'xi_um_fit_v2') or (xi_um_fit_column == 'xi_um_fit'):
                    df_fitting_result = pd.merge(df_fitting_v2_results,df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting']].min(), on=['timestamp_pulse_id','chi2distance_fitting'])[['separation_um','imageid','timestamp_pulse_id','mod_sigma_um', 'mod_sigma_um_fit','mod_shiftx_um','mod_shiftx_um_fit',xi_um_fit_column,'chi2distance_fitting']].sort_values('chi2distance_fitting',ascending=False)
            
            df_result = pd.merge(df_deconvmethod_result,df_fitting_result, on='timestamp_pulse_id', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)').sort_values(chi2distance_column,ascending=False)
            
            x = df_result[xi_um_deconv_column]
            y = df_result[xi_um_fit_column]
            c = df_result[chi2distance_column]

            
            plt.scatter(x=x, y=y, c=c, marker='x', s=2)

            x0 = df_result[(df_result['timestamp_pulse_id']== timestamp_pulse_id)][xi_um_deconv_column]
            y0 = df_result[(df_result['timestamp_pulse_id']== timestamp_pulse_id)][xi_um_fit_column]
            c0 = df_result[(df_result['timestamp_pulse_id']== timestamp_pulse_id)][chi2distance_column]

            plt.scatter(x=x0, y=y0, c='red', marker='x', s=10)

        plt.clim(vmin=c.min(), vmax=c.max())
        plt.colorbar(label=chi2distance_label)

        x = np.linspace(0,2000)
        plt.plot(x,x, c='grey', linewidth=1, alpha=0.5, linestyle="--")

        


        # deconvmethod_outliers = df_deconvmethod_1d_results[(df_deconvmethod_1d_results["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df_deconvmethod_1d_results[xi_um_deconv_column] > deconvmethod_outlier_limit)][['imageid', 'separation_um', xi_um_deconv_column]].sort_values(by=xi_um_deconv_column, ascending=False)
        # print('Deconvmethod outliers > ' + str(deconvmethod_outlier_limit))
        # print(deconvmethod_outliers)
        
        # fitting_outliers = df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df_fitting_v2_results[xi_um_fit_column] > fitting_outlier_limit)][['imageid','separation_um', xi_um_fit_column]].sort_values(by=xi_um_fit_column, ascending=False)
        # print('Fitting method outliers > ' + str(fitting_outlier_limit))
        # print(fitting_outliers)

        # timestamp_pulse_ids = []
        # with h5py.File(measurement_file, "r") as hdf5_file:
        #     timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

        # plt.scatter(df_deconvmethod_1d_results[(df_deconvmethod_1d_results["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df_deconvmethod_1d_results["imageid"] == int(imageid))][xi_um_deconv_column] , \
        #     df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df_fitting_v2_results["imageid"] == int(imageid))][xi_um_fit_column], \
        #         c='red',\
        #             marker='x', s=10)

  
        # x = np.linspace(0,2000)
        # plt.plot(x,x, c='grey', linewidth=1, alpha=0.5, linestyle="--")
        

        plt.xlim(xaxisrange[0],xaxisrange[1])
        plt.ylim(yaxisrange[0],yaxisrange[1])
        plt.xlabel(xi_um_deconv_label)
        plt.ylabel(xi_um_fit_label)

        plt.gca().set_aspect('equal')






use_measurement_default_result_widget = widgets.Checkbox(value=False, description="use_measurement_default_result", disabled=False)


def list_results(
    do_list_results,
    dataset,
    measurement_file,
    timestamp_pulse_id,
    xi_um_deconv_column_and_label,
    xi_um_fit_column_and_label,
    chi2distance_column_and_label,
    use_measurement_default_result
):

    if do_list_results == True:

        global df_result

        # timestamp_pulse_id = timestamp_pulse_id_widget.value

        xi_um_deconv_column = xi_um_deconv_column_and_label[0]
        xi_um_deconv_label = xi_um_deconv_column_and_label[1]
        xi_um_fit_column = xi_um_fit_column_and_label[0]
        xi_um_fit_label = xi_um_fit_column_and_label[1]
        chi2distance_column = chi2distance_column_and_label[0]
        chi2distance_label = chi2distance_column_and_label[1]

        # Loading and preparing

        # get all the files in a dataset:
        files = []
        # for set in [list(datasets)[0]]:
        
        for measurement in datasets_selection[dataset]:
            # print(measurement)
            files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

        # testing:
        files = measurements_selection_widget.value

        # get all the timestamps in these files:        
        # datasets[list(datasets)[0]][0]

        pd.set_option('display.max_rows', None)

        timestamp_pulse_ids = []
        
        for f in files:
            timestamp_pulse_ids_measurement = []
            measurement = os.path.splitext(os.path.basename(f))[0]
            with h5py.File(f, "r") as hdf5_file:         
                timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
                timestamp_pulse_ids_measurement.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
            if use_measurement_default_result == True:
                if xi_um_deconv_column == 'xi_x_um_v1':
                    # deconvolution 2d v1 defaults:
                    balance_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]
                    df_deconvmethod_result = df_deconvmethod_2d_v1_results[(df_deconvmethod_2d_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_2d_v1_results['balance'] == balance_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['xatol'] == xatol_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_2d_v1_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v1','chi2distance_deconvmethod_2d_v1']].sort_values('chi2distance_deconvmethod_2d_v1',ascending=False)               
                if xi_um_deconv_column == 'xi_um_v2':
                    # deconvolution 2d v2 defaults:
                    balance_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0] 
                    df_deconvmethod_result = df_deconvmethod_1d_v2_results[(df_deconvmethod_1d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_1d_v2_results['balance'] == balance_measurement_default) & \
                        (df_deconvmethod_1d_v2_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_1d_v2_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_1d_v2_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','balance','xi_um_guess','xi_um_v2','chi2distance_deconvmethod_1d_v2']].sort_values('chi2distance_deconvmethod_1d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v2':
                    # deconvolution 2d v2 defaults:
                    balance_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]
                    df_deconvmethod_result = df_deconvmethod_2d_v2_results[(df_deconvmethod_2d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_2d_v2_results['balance'] == balance_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['xatol'] == xatol_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_2d_v2_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','balance','xi_um_guess','xi_x_um_v2','chi2distance_deconvmethod_2d_v2']].sort_values('chi2distance_deconvmethod_2d_v2',ascending=False)               
                if xi_um_deconv_column == 'xi_um_v3':
                    # deconvolution v3 defaults:
                    snr_db_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['snr_db_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0] 
                    df_deconvmethod_result = df_deconvmethod_1d_v3_results[(df_deconvmethod_1d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_1d_v3_results['snr_db'] == snr_db_measurement_default) & \
                        (df_deconvmethod_1d_v3_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_1d_v3_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_1d_v3_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','snr_db','xi_um_guess','xi_um_v3','chi2distance_deconvmethod_1d_v3']].sort_values('chi2distance_deconvmethod_1d_v3',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v3':
                    # deconvolution v3 defaults:
                    snr_db_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['snr_db_measurement_default'].iloc[0]
                    xi_um_guess_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
                    xatol_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
                    sigma_x_F_gamma_um_multiplier_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
                    crop_px_measurement_default = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]
                    df_deconvmethod_result = df_deconvmethod_2d_v3_results[(df_deconvmethod_2d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_deconvmethod_2d_v3_results['snr_db'] == snr_db_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['xi_um_guess'] == xi_um_guess_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['xatol'] == xatol_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['sigma_x_F_gamma_um_multiplier'] == sigma_x_F_gamma_um_multiplier_measurement_default) & \
                        (df_deconvmethod_2d_v3_results['crop_px'] == crop_px_measurement_default)][['separation_um','imageid','timestamp_pulse_id','snr_db','xi_um_guess','xi_x_um_v3','chi2distance_deconvmethod_2d_v3']].sort_values('chi2distance_deconvmethod_2d_v3',ascending=False)               
                
                
                if xi_um_fit_column == 'xi_um_fit_v1':
                    df_fitting_result = df_fitting_v1_results[(df_fitting_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))][['separation_um','imageid','timestamp_pulse_id',xi_um_fit_column,'chi2distance_fitting_v1']].sort_values('chi2distance_fitting_v1',ascending=False)        
                if (xi_um_fit_column == 'xi_um_fit_v2') or (xi_um_fit_column == 'xi_um_fit'):
                    # fitting v2 defaults:
                    mod_sigma_um_measurement_default = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_sigma_um_measurement_default'].iloc[0]
                    mod_shiftx_um_measurement_default = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_shiftx_um_measurement_default'].iloc[0]
                    df_fitting_result = df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement)) & \
                        (df_fitting_v2_results['mod_sigma_um'] == mod_sigma_um_measurement_default) & \
                        (df_fitting_v2_results['mod_shiftx_um'] == mod_shiftx_um_measurement_default)][['separation_um','imageid','timestamp_pulse_id','mod_sigma_um', 'mod_sigma_um_fit','mod_shiftx_um','mod_shiftx_um_fit',xi_um_fit_column,'chi2distance_fitting']].sort_values('chi2distance_fitting',ascending=False)
        

            # https://datascienceparichay.com/article/pandas-groupby-minimum/
            if use_measurement_default_result == False:


                if xi_um_deconv_column == 'xi_x_um_v1':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_2d_v1_results,df_deconvmethod_2d_v1_results[(df_deconvmethod_2d_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v1']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v1'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v1','chi2distance_deconvmethod_2d_v1']].sort_values('chi2distance_deconvmethod_2d_v1',ascending=False)
                if xi_um_deconv_column == 'xi_um_v2':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_1d_v2_results,df_deconvmethod_1d_v2_results[(df_deconvmethod_1d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v2'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_um_v2','chi2distance_deconvmethod_1d_v2']].sort_values('chi2distance_deconvmethod_1d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v2':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_2d_v2_results,df_deconvmethod_2d_v2_results[(df_deconvmethod_2d_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v2'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v2','chi2distance_deconvmethod_2d_v2']].sort_values('chi2distance_deconvmethod_2d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_um_v3':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_1d_v3_results,df_deconvmethod_1d_v3_results[(df_deconvmethod_1d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v3'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_um_v3','chi2distance_deconvmethod_1d_v3']].sort_values('chi2distance_deconvmethod_1d_v3',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v3':
                    df_deconvmethod_result = pd.merge(df_deconvmethod_2d_v3_results,df_deconvmethod_2d_v3_results[(df_deconvmethod_2d_v3_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v3'])[['separation_um','imageid','timestamp_pulse_id','xi_um_guess','xi_x_um_v3','chi2distance_deconvmethod_2d_v3']].sort_values('chi2distance_deconvmethod_2d_v3',ascending=False)
                


                if xi_um_fit_column == 'xi_um_fit_v1':
                    df_fitting_result = pd.merge(df_fitting_v1_results,df_fitting_v1_results[(df_fitting_v1_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting_v1']].min(), on=['timestamp_pulse_id','chi2distance_fitting_v1'])[['separation_um','imageid','timestamp_pulse_id',xi_um_fit_column,'chi2distance_fitting_v1']].sort_values('chi2distance_fitting_v1',ascending=False)
                if (xi_um_fit_column == 'xi_um_fit_v2') or (xi_um_fit_column == 'xi_um_fit'):
                    df_fitting_result = pd.merge(df_fitting_v2_results,df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_measurement))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting']].min(), on=['timestamp_pulse_id','chi2distance_fitting'])[['separation_um','imageid','timestamp_pulse_id','mod_sigma_um', 'mod_sigma_um_fit','mod_shiftx_um','mod_shiftx_um_fit',xi_um_fit_column,'chi2distance_fitting']].sort_values('chi2distance_fitting',ascending=False)

            df_result = pd.merge(df_deconvmethod_result,df_fitting_result, on='timestamp_pulse_id', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)').sort_values(chi2distance_column,ascending=False)
            
            display(df_result.style.apply(
                lambda x: ['background-color: yellow' if x.timestamp_pulse_id == timestamp_pulse_id else '' for i in x],
                axis=1
            ))
                







# sort_imageids_by_chi2distance

sort_imageids_by_chi2distance_widget = widgets.ToggleButton(
    value=False,
    description='sort by chi2distance',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='sort images by chi2distance',
    icon=''
)

def sort_imageids_by_chi2distance_widget_changed(change):
    if sort_imageids_by_chi2distance_widget.value == True:
        imageid_widget.options = df_result['imageid']
        timestamp_pulse_id_widget.options = df_result['timestamp_pulse_id']
        
        sort_imageids_by_chi2distance_widget.value = False

sort_imageids_by_chi2distance_widget.observe(sort_imageids_by_chi2distance_widget_changed, names="value")


# CDC from Deconvolution (green) and Fitting (red)
def plot_CDCs(
    do_plot_CDCs,
    xi_um_deconv_column_and_label,
    xi_um_fit_column_and_label
):

    if do_plot_CDCs == True:

        xi_um_deconv_column = xi_um_deconv_column_and_label[0]
        xi_um_deconv_label = xi_um_deconv_column_and_label[1]
        xi_um_fit_column = xi_um_fit_column_and_label[0]
        xi_um_fit_label = xi_um_fit_column_and_label[1]
        gamma_fit_column = 'gamma_fit' + xi_um_fit_column[9:]

        fig = plt.figure(figsize=[6, 8], constrained_layout=True)

        gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
        gs.update(hspace=0, wspace=0.0)

        i=0
        j=0

        for dataset in list(datasets_selection):
            statustext_widget.value = 'plotting: ' + dataset
            timestamp_pulse_ids_dataset=[]

            ax = plt.subplot(gs[i,j])
        
            # get all the files in a dataset:
            files = []
            # for set in [list(datasets)[0]]:
            
            for measurement in datasets_selection[dataset]:
                # print(measurement)
                statustext_widget.value = 'plotting: ' + measurement
                files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

            # get all the timestamps in these files:        
            # datasets[list(datasets)[0]][0]
            
            if xi_um_deconv_column == 'xi_um_v2':
                df_deconvmethod_results = df_deconvmethod_1d_v2_results
            if xi_um_deconv_column == 'xi_x_um_v2':
                df_deconvmethod_results = df_deconvmethod_2d_v2_results

            if xi_um_deconv_column == 'xi_um_v3':
                df_deconvmethod_results = df_deconvmethod_1d_v3_results
            if xi_um_deconv_column == 'xi_x_um_v3':
                df_deconvmethod_results = df_deconvmethod_2d_v3_results
            
            for f in files:
                timestamp_pulse_ids = []
                with h5py.File(f, "r") as hdf5_file:
                    timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
                    timestamp_pulse_ids_dataset.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

            # create plot for the determined timestamps:
        
                # ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'] , \
                #     gaussian(x=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'], amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_x_um']), \
                #         c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['I_Airy2_fit'])

                # Deconvolution (green)
                # todo: implement als deconvmethod_2d_result
                x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
                if xi_um_deconv_column == 'xi_um_v2':
                    df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v2'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_1d_v2']].sort_values('chi2distance_deconvmethod_1d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v2':
                    df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v2'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_2d_v2']].sort_values('chi2distance_deconvmethod_2d_v2',ascending=False)
                if xi_um_deconv_column == 'xi_um_v3':
                    df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v3'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_1d_v3']].sort_values('chi2distance_deconvmethod_1d_v3',ascending=False)
                if xi_um_deconv_column == 'xi_x_um_v3':
                    df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v3'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_2d_v3']].sort_values('chi2distance_deconvmethod_2d_v3',ascending=False)
                
                for separation_um in x:
                    y_nans = df_deconvmethod_results_min[df_deconvmethod_results_min[xi_um_deconv_column].isna()]
                    if len(y_nans) > 0:
                        print('Deconvolution failed in file: ' + str(f))
                        print('separation='+str(x))
                        print('imageids:')
                        display(y_nans)
                y = [gaussian(x=x, amp=1, cen=0, sigma=df_deconvmethod_results_min[df_deconvmethod_results_min["separation_um"]==x][xi_um_deconv_column].max()) for x in x]
                ax.scatter(x, y, marker='v', s=20, color='darkgreen', facecolors='none', label='maximum')
                
                # Fitting (red)
                x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
                df_fitting_v2_results_min = pd.merge(df_fitting_v2_results,df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting']].min(), on=['timestamp_pulse_id','chi2distance_fitting'])[['separation_um','imageid','mod_sigma_um', 'mod_sigma_um_fit','mod_shiftx_um','mod_shiftx_um_fit','chi2distance_fitting',gamma_fit_column]].sort_values('chi2distance_fitting',ascending=False)
                y = [df_fitting_v2_results_min[(df_fitting_v2_results_min["separation_um"]==x)][gamma_fit_column].max() for x in x]
                ax.scatter(x, y, marker='v', s=20, color='darkred', facecolors='none', label='maximum')
                
            
            # fit a gaussian on all max of each measurement
            x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))]['separation_um'].unique()
            if xi_um_deconv_column == 'xi_um_v2':
                df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v2'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_1d_v2']].sort_values('chi2distance_deconvmethod_1d_v2',ascending=False)
            if xi_um_deconv_column == 'xi_x_um_v2':
                df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v2']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v2'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_2d_v2']].sort_values('chi2distance_deconvmethod_2d_v2',ascending=False)
            if xi_um_deconv_column == 'xi_um_v3':
                df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_1d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_1d_v3'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_1d_v3']].sort_values('chi2distance_deconvmethod_1d_v3',ascending=False)
            if xi_um_deconv_column == 'xi_x_um_v3':
                df_deconvmethod_results_min = pd.merge(df_deconvmethod_results,df_deconvmethod_results[(df_deconvmethod_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))].groupby(['timestamp_pulse_id'])[['chi2distance_deconvmethod_2d_v3']].min(), on=['timestamp_pulse_id','chi2distance_deconvmethod_2d_v3'])[['separation_um','imageid','xi_um_guess',xi_um_deconv_column,'chi2distance_deconvmethod_2d_v3']].sort_values('chi2distance_deconvmethod_2d_v3',ascending=False)
            y = [gaussian(x=x, amp=1, cen=0, sigma=df_deconvmethod_results_min[df_deconvmethod_results_min["separation_um"]==x][xi_um_deconv_column].max()) for x in x]
        
            xx = np.arange(0.0, 2000, 10)
            gamma_xi_x_um_max = y
            d_gamma = x
            # gamma_xi_x_um_max = gamma_xi_x_um_max[~np.isnan(gamma_xi_x_um_max)]
            (xi_x_um_max_sigma, xi_x_um_max_sigma_std) = find_sigma(d_gamma,gamma_xi_x_um_max,0, 400, False)
            
            y1 = [gaussian(x=x, amp=1, cen=0, sigma=xi_x_um_max_sigma) for x in xx]
            ax.plot(xx, y1, '-', color='green', label='') # xi_x_um_max plot
            y_min = [gaussian(x=x, amp=1, cen=0, sigma=xi_x_um_max_sigma-xi_x_um_max_sigma_std) for x in xx]
            y_max = [gaussian(x=x, amp=1, cen=0, sigma=xi_x_um_max_sigma+xi_x_um_max_sigma_std) for x in xx]
            ax.fill_between(xx, y_min, y_max, facecolor='green', alpha=0.3)
            # ax.hlines(0.606, 0, np.nanmean(xi_x_um_max), linestyles = '-', color='green')
            ax.hlines(0.606, 0, np.nanmean(xi_x_um_max_sigma), linestyles = '-', color='green')
            # ax.hlines(0.606, 0, np.nanmean(sigma_B_um), linestyles = '-', color='black')


            # TO DO: find mean sigma and error of the max(gamma_fit) of each separation

            x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))]['separation_um'].unique()
            # y = [df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset)) & (df0["separation_um"]==x)][gamma_fit_column].max() for x in x]
            df_fitting_v2_results_min = pd.merge(df_fitting_v2_results,df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting']].min())[['separation_um','imageid','mod_sigma_um', 'mod_sigma_um_fit','mod_shiftx_um','mod_shiftx_um_fit','chi2distance_fitting',gamma_fit_column]].sort_values('chi2distance_fitting',ascending=False)
            y = [df_fitting_v2_results_min[(df_fitting_v2_results_min["separation_um"]==x)][gamma_fit_column].max() for x in x]
        
            xx = np.arange(0.0, 2000, 10)
            gamma_fit_max = y
            d_gamma = x
                
            (xi_x_um_max_sigma, xi_x_um_max_sigma_std) = find_sigma(d_gamma,gamma_fit_max,0, 400, False)
            
            

            if xi_x_um_max_sigma_std is None:
                xi_x_um_max_sigma_std = 0
                print(xi_x_um_max_sigma)
                print(xi_x_um_max_sigma_std)

            y1 = [gaussian(x=x, amp=1, cen=0, sigma=xi_x_um_max_sigma) for x in xx]
            ax.plot(xx, y1, '-', color='red', label='') # xi_x_um_max plot
            y_min = [gaussian(x=x, amp=1, cen=0, sigma=xi_x_um_max_sigma-xi_x_um_max_sigma_std) for x in xx]
            y_max = [gaussian(x=x, amp=1, cen=0, sigma=xi_x_um_max_sigma+xi_x_um_max_sigma_std) for x in xx]
            ax.fill_between(xx, y_min, y_max, facecolor='red', alpha=0.3)
            # ax.hlines(0.606, 0, np.nanmean(xi_x_um_max), linestyles = '-', color='green')
            ax.hlines(0.606, 0, np.nanmean(xi_x_um_max_sigma), linestyles = '-', color='red')
            # ax.hlines(0.606, 0, np.nanmean(sigma_B_um), linestyles = '-', color='black')

        
            ax.set_xlim(0,2000)
            ax.set_ylim(0,1)
            
            ax.set_title(dataset)
            
            
            if j==0:
                j+=1
            else:
                j=0
                i=i+1




# create plots fitting vs deconvolution
def plot_xi_um_fit_vs_I_Airy2_fit(
    do_plot_xi_um_fit_vs_I_Airy2_fit,
    xi_um_fit_column_and_label
):

    if do_plot_xi_um_fit_vs_I_Airy2_fit == True:

        xi_um_fit_column = xi_um_fit_column_and_label[0]
        xi_um_fit_label = xi_um_fit_column_and_label[1]

        fig = plt.figure(figsize=[6, 8], constrained_layout=True)

        gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
        gs.update(hspace=0, wspace=0.0)

        i=0
        j=0
        for dataset in list(datasets_selection):

            ax = plt.subplot(gs[i,j])

            # get all the files in a dataset:
            files = []
            # for set in [list(datasets)[0]]:
            
            for measurement in datasets_selection[dataset]:
                # print(measurement)
                files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

            # get all the timestamps in these files:        
            # datasets[list(datasets)[0]][0]
            timestamp_pulse_ids = []
            for f in files:
                with h5py.File(f, "r") as hdf5_file:
                    timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

            # create plot for the determined timestamps:
            df_fitting_v2_results_min = pd.merge(df_fitting_v2_results,df_fitting_v2_results[(df_fitting_v2_results["timestamp_pulse_id"].isin(timestamp_pulse_ids))].groupby(['timestamp_pulse_id'])[['chi2distance_fitting']].min()).sort_values('chi2distance_fitting',ascending=False)
            plt.scatter(df_fitting_v2_results_min['I_Airy2_fit'] , \
                df_fitting_v2_results_min[xi_um_fit_column], \
                    c=df_fitting_v2_results_min['separation_um'],\
                        marker='x', s=2)
            plt.xlabel(r"$I_2$")
            plt.ylabel(xi_um_fit_label)
            plt.axvline(x=1, color='black')
            plt.colorbar()

            # plt.xlim(0,2000)
            # plt.ylim(0,2000)
            
            plt.title(dataset)

            if j==0:
                j+=1
            else:
                j=0
                i=i+1


# Structuring the input widgets



column1a_v2 = widgets.VBox(
    [
        crop_px_1d_v2_widget,
        pixis_profile_avg_width_1d_v2_widget,
        balance_1d_v2_widget,
        xi_um_guess_1d_v2_widget,
        sigma_x_F_gamma_um_multiplier_1d_v2_widget,      
    ]
)
column1b_v2 = widgets.VBox(
    [
        crop_px_2d_v2_widget,
        pixis_profile_avg_width_2d_v2_widget,
        balance_2d_v2_widget,
        xi_um_guess_2d_v2_widget,
        sigma_x_F_gamma_um_multiplier_2d_v2_widget,
        xatol_2d_v2_widget,
    ]
)

column1a_v3 = widgets.VBox(
    [
        crop_px_1d_v3_widget,
        pixis_profile_avg_width_1d_v3_widget,
        snr_db_1d_v3_widget,
        xi_um_guess_1d_v3_widget,
        sigma_x_F_gamma_um_multiplier_1d_v3_widget,      
    ]
)
column1b_v3 = widgets.VBox(
    [
        crop_px_2d_v3_widget,
        pixis_profile_avg_width_2d_v3_widget,
        snr_db_2d_v3_widget,
        xi_um_guess_2d_v3_widget,
        sigma_x_F_gamma_um_multiplier_2d_v3_widget,
        xatol_2d_v3_widget,
    ]
)

# for v1:
column1c = widgets.VBox(
    [
        crop_px_2d_v1_widget,
        pixis_profile_avg_width_2d_v1_widget,
        sigma_x_F_gamma_um_min_2d_v1_widget,
        sigma_x_F_gamma_um_max_2d_v1_widget,
        sigma_x_F_gamma_um_stepsize_2d_v1_widget,        
        sigma_y_F_gamma_um_min_2d_v1_widget,
        sigma_y_F_gamma_um_max_2d_v1_widget,
        sigma_y_F_gamma_um_stepsize_2d_v1_widget,        
    ]
)

column2a = widgets.VBox(
    [
        shiftx_um_widget,
        wavelength_nm_widget,
        z_mm_widget,
        d_um_widget,
        gamma_widget,
        w1_um_widget,
        w2_um_widget,
        I_Airy1_widget,
        I_Airy2_widget,
        x1_um_widget,
        x2_um_widget,
        normfactor_widget,

    ]
)

column2b = widgets.VBox(
    [
        shiftx_um_widget,
        wavelength_nm_widget,
        z_mm_widget,
        d_um_widget,
        gamma_widget,
        w1_um_widget,
        w2_um_widget,
        I_Airy1_widget,
        I_Airy2_widget,
        x1_um_widget,
        x2_um_widget,
        normfactor_widget,
        mod_sigma_um_widget,
        mod_shiftx_um_widget
    ]
)


column3a = widgets.VBox(
    [
        shiftx_um_value_widget,
        wavelength_nm_value_widget,
        z_mm_value_widget,
        d_um_value_widget,
        gamma_value_widget,
        w1_um_value_widget,
        w2_um_value_widget,
        I_Airy1_value_widget,
        I_Airy2_value_widget,
        x1_um_value_widget,
        x2_um_value_widget,
        normfactor_value_widget,
    ]
)

column3b = widgets.VBox(
    [
        shiftx_um_value_widget,
        wavelength_nm_value_widget,
        z_mm_value_widget,
        d_um_value_widget,
        gamma_value_widget,
        w1_um_value_widget,
        w2_um_value_widget,
        I_Airy1_value_widget,
        I_Airy2_value_widget,
        x1_um_value_widget,
        x2_um_value_widget,
        normfactor_value_widget,
        mod_sigma_um_value_widget,
        mod_shiftx_um_value_widget
    ]
)


column4a = widgets.VBox(
    [
        shiftx_um_do_fit_widget,
        wavelength_nm_do_fit_widget,
        z_mm_do_fit_widget,
        d_um_do_fit_widget,
        gamma_do_fit_widget,
        w1_um_do_fit_widget,
        w2_um_do_fit_widget,
        I_Airy1_do_fit_widget,
        I_Airy2_do_fit_widget,
        x1_um_do_fit_widget,
        x2_um_do_fit_widget,
        normfactor_do_fit_widget,
    ]
)

column4b = widgets.VBox(
    [
        shiftx_um_do_fit_widget,
        wavelength_nm_do_fit_widget,
        z_mm_do_fit_widget,
        d_um_do_fit_widget,
        gamma_do_fit_widget,
        w1_um_do_fit_widget,
        w2_um_do_fit_widget,
        I_Airy1_do_fit_widget,
        I_Airy2_do_fit_widget,
        x1_um_do_fit_widget,
        x2_um_do_fit_widget,
        normfactor_do_fit_widget,
        mod_sigma_um_do_fit_widget,
        mod_shiftx_um_do_fit_widget
    ]
)

column5a = widgets.VBox(
    [
        shiftx_um_range_widget,
        wavelength_nm_range_widget,
        z_mm_range_widget,
        d_um_range_widget,
        gamma_range_widget,
        w1_um_range_widget,
        w2_um_range_widget,
        I_Airy1_range_widget,
        I_Airy2_range_widget,
        x1_um_range_widget,
        x2_um_range_widget,
        normfactor_range_widget,
    ]
)

column5b = widgets.VBox(
    [
        shiftx_um_range_widget,
        wavelength_nm_range_widget,
        z_mm_range_widget,
        d_um_range_widget,
        gamma_range_widget,
        w1_um_range_widget,
        w2_um_range_widget,
        I_Airy1_range_widget,
        I_Airy2_range_widget,
        x1_um_range_widget,
        x2_um_range_widget,
        normfactor_range_widget,
        mod_sigma_um_range_widget,
        mod_shiftx_um_range_widget
    ]
)



fitting_parameter_tab = widgets.Tab()
fitting_parameter_tab.children = [widgets.HBox([column2b]),
                                  widgets.HBox([column5b])]
fitting_parameter_tab.set_title(0,'Guess')
fitting_parameter_tab.set_title(1,'Range')

fitting_result_tab = widgets.Tab()
fitting_result_tab.children = [widgets.HBox([column3b])]
fitting_result_tab.set_title(0,'Result')

fitting_do_fit_tab = widgets.Tab()
fitting_do_fit_tab.children = [widgets.HBox([column4b])]
fitting_do_fit_tab.set_title(0,'fit')

fitting_columns = widgets.HBox([
                                fitting_do_fit_tab, 
                                fitting_parameter_tab,
                                fitting_result_tab
                                ])

fitting_v1_parameter_tab = widgets.Tab()
fitting_v1_parameter_tab.children = [widgets.HBox([column2a]),
                                  widgets.HBox([column5a])]
fitting_v1_parameter_tab.set_title(0,'Guess')
fitting_v1_parameter_tab.set_title(1,'Range')

fitting_v1_result_tab = widgets.Tab()
fitting_v1_result_tab.children = [widgets.HBox([column3a])]
fitting_v1_result_tab.set_title(0,'Result')

fitting_v1_do_fit_tab = widgets.Tab()
fitting_v1_do_fit_tab.children = [widgets.HBox([column4a])]
fitting_v1_do_fit_tab.set_title(0,'do fit')

fitting_v1_columns = widgets.HBox([
                                fitting_v1_do_fit_tab, 
                                fitting_v1_parameter_tab,
                                fitting_v1_result_tab
                                ])

parameter_tabs_children = [ fitting_columns,
                                fitting_v1_columns,
                                column1a_v3,
                                column1b_v3,
                                column1a_v2,
                                column1b_v2,
                                column1c #v1
]
parameter_tabs = widgets.Tab()
parameter_tabs.children = parameter_tabs_children
parameter_tabs.set_title(0, 'Fitting_v2')
parameter_tabs.set_title(1, 'Fitting_v1')
parameter_tabs.set_title(2, 'Deconv_1d_v3')
parameter_tabs.set_title(3, 'Deconv_2d_v3')
parameter_tabs.set_title(4, 'Deconv_1d_v2')
parameter_tabs.set_title(5, 'Deconv_2d_v2')
parameter_tabs.set_title(6, 'Deconv_2d_v1')




plot_fitting_v1_interactive_output = interactive_output(
    plot_fitting_v1,
    {
        "do_plot_fitting_v1": do_plot_fitting_v1_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_widget,
        "crop_px" : crop_px_widget,
        # "hdf5_file_path": dph_settings_bgsubtracted_widget,
        # "imageid": imageid_widget,
        "savefigure": savefigure_profile_fit_widget,
        "save_to_df": save_to_df_widget,
        "do_textbox": do_textbox_widget,
        "shiftx_um": shiftx_um_widget,
        "shiftx_um_range": shiftx_um_range_widget,
        "shiftx_um_do_fit": shiftx_um_do_fit_widget,
        "wavelength_nm": wavelength_nm_widget,
        "wavelength_nm_range": wavelength_nm_range_widget,
        "wavelength_nm_do_fit": wavelength_nm_do_fit_widget,
        "z_mm": z_mm_widget,
        "z_mm_range": z_mm_range_widget,
        "z_mm_do_fit": z_mm_do_fit_widget,
        "d_um": d_um_widget,
        "d_um_range": d_um_range_widget,
        "d_um_do_fit": d_um_do_fit_widget,
        "gamma": gamma_widget,
        "gamma_range": gamma_range_widget,
        "gamma_do_fit": gamma_do_fit_widget,
        "w1_um": w1_um_widget,
        "w1_um_range": w1_um_range_widget,
        "w1_um_do_fit": w1_um_do_fit_widget,
        "w2_um": w2_um_widget,
        "w2_um_range": w2_um_range_widget,
        "w2_um_do_fit": w2_um_do_fit_widget,
        "I_Airy1": I_Airy1_widget,
        "I_Airy1_range": I_Airy1_range_widget,
        "I_Airy1_do_fit": I_Airy1_do_fit_widget,
        "I_Airy2": I_Airy2_widget,
        "I_Airy2_range": I_Airy2_range_widget,
        "I_Airy2_do_fit": I_Airy2_do_fit_widget,
        "x1_um": x1_um_widget,
        "x1_um_range": x1_um_range_widget,
        "x1_um_do_fit": x1_um_do_fit_widget,
        "x2_um": x2_um_widget,
        "x2_um_range": x2_um_range_widget,
        "x2_um_do_fit": x2_um_do_fit_widget,
        "normfactor": normfactor_widget,
        "normfactor_range": normfactor_range_widget,
        "normfactor_do_fit": normfactor_do_fit_widget,
    },
)

plot_fitting_v2_interactive_output = interactive_output(
    plot_fitting_v2,
    {
        "do_plot_fitting_v2": do_plot_fitting_v2_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_widget,
        "crop_px" : crop_px_widget,
        # "hdf5_file_path": dph_settings_bgsubtracted_widget,
        # "imageid": imageid_widget,
        "savefigure": savefigure_profile_fit_widget,
        "save_to_df": save_to_df_widget,
        "do_textbox": do_textbox_widget,
        "shiftx_um": shiftx_um_widget,
        "shiftx_um_range": shiftx_um_range_widget,
        "shiftx_um_do_fit": shiftx_um_do_fit_widget,
        "wavelength_nm": wavelength_nm_widget,
        "wavelength_nm_range": wavelength_nm_range_widget,
        "wavelength_nm_do_fit": wavelength_nm_do_fit_widget,
        "z_mm": z_mm_widget,
        "z_mm_range": z_mm_range_widget,
        "z_mm_do_fit": z_mm_do_fit_widget,
        "d_um": d_um_widget,
        "d_um_range": d_um_range_widget,
        "d_um_do_fit": d_um_do_fit_widget,
        "gamma": gamma_widget,
        "gamma_range": gamma_range_widget,
        "gamma_do_fit": gamma_do_fit_widget,
        "w1_um": w1_um_widget,
        "w1_um_range": w1_um_range_widget,
        "w1_um_do_fit": w1_um_do_fit_widget,
        "w2_um": w2_um_widget,
        "w2_um_range": w2_um_range_widget,
        "w2_um_do_fit": w2_um_do_fit_widget,
        "I_Airy1": I_Airy1_widget,
        "I_Airy1_range": I_Airy1_range_widget,
        "I_Airy1_do_fit": I_Airy1_do_fit_widget,
        "I_Airy2": I_Airy2_widget,
        "I_Airy2_range": I_Airy2_range_widget,
        "I_Airy2_do_fit": I_Airy2_do_fit_widget,
        "x1_um": x1_um_widget,
        "x1_um_range": x1_um_range_widget,
        "x1_um_do_fit": x1_um_do_fit_widget,
        "x2_um": x2_um_widget,
        "x2_um_range": x2_um_range_widget,
        "x2_um_do_fit": x2_um_do_fit_widget,
        "normfactor": normfactor_widget,
        "normfactor_range": normfactor_range_widget,
        "normfactor_do_fit": normfactor_do_fit_widget,
        "mod_sigma_um": mod_sigma_um_widget,
        "mod_sigma_um_range": mod_sigma_um_range_widget,
        "mod_sigma_um_do_fit": mod_sigma_um_do_fit_widget,
        "mod_shiftx_um": mod_shiftx_um_widget,
        "mod_shiftx_um_range": mod_shiftx_um_range_widget,
        "mod_shiftx_um_do_fit": mod_shiftx_um_do_fit_widget,
    },
)

plot_deconvmethod_1d_v2_interactive_output = interactive_output(
    plot_deconvmethod_1d_v2,
    {
        "do_plot_deconvmethod_1d_v2": do_plot_deconvmethod_1d_v2_widget,
        "balance" : balance_1d_v2_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_1d_v2_widget,
        "xi_um_guess" : xi_um_guess_1d_v2_widget,
        "xatol" : xatol_1d_v2_widget,
        "sigma_x_F_gamma_um_multiplier" : sigma_x_F_gamma_um_multiplier_1d_v2_widget,
        "crop_px" : crop_px_1d_v2_widget,
        # "hdf5_file_path": dph_settings_bgsubtracted_widget,
        # "imageid": imageid_widget,
        "save_to_df": save_to_df_widget,
        "create_steps_figures": create_steps_figures_widget,
        "create_figure": create_figure_widget
    },
)

plot_deconvmethod_2d_v2_interactive_output = interactive_output(
    plot_deconvmethod_2d_v2,
    {
        "do_plot_deconvmethod_2d_v2": do_plot_deconvmethod_2d_v2_widget,
        "balance" : balance_2d_v2_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_2d_v2_widget,
        "xi_um_guess" : xi_um_guess_2d_v2_widget,
        "xatol" : xatol_2d_v2_widget,
        "sigma_x_F_gamma_um_multiplier" : sigma_x_F_gamma_um_multiplier_2d_v2_widget,
        "crop_px" : crop_px_2d_v2_widget,
        # "hdf5_file_path": dph_settings_bgsubtracted_widget,
        # "imageid": imageid_widget,
        "save_to_df": save_to_df_widget,
        "create_steps_figures": create_steps_figures_widget,
        "create_figure": create_figure_widget
    },
)

plot_deconvmethod_1d_v3_interactive_output = interactive_output(
    plot_deconvmethod_1d_v3,
    {
        "do_plot_deconvmethod_1d_v3": do_plot_deconvmethod_1d_v3_widget,
        "snr_db" : snr_db_1d_v3_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_1d_v3_widget,
        "xi_um_guess" : xi_um_guess_1d_v3_widget,
        "xatol" : xatol_1d_v3_widget,
        "sigma_x_F_gamma_um_multiplier" : sigma_x_F_gamma_um_multiplier_1d_v3_widget,
        "crop_px" : crop_px_1d_v3_widget,
        # "hdf5_file_path": dph_settings_bgsubtracted_widget,
        # "imageid": imageid_widget,
        "save_to_df": save_to_df_widget,
        "create_steps_figures": create_steps_figures_widget,
        "create_figure": create_figure_widget
    },
)

plot_deconvmethod_2d_v3_interactive_output = interactive_output(
    plot_deconvmethod_2d_v3,
    {
        "do_plot_deconvmethod_2d_v3": do_plot_deconvmethod_2d_v3_widget,
        "snr_db" : snr_db_2d_v3_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_2d_v3_widget,
        "xi_um_guess" : xi_um_guess_2d_v3_widget,
        "xatol" : xatol_2d_v3_widget,
        "sigma_x_F_gamma_um_multiplier" : sigma_x_F_gamma_um_multiplier_2d_v3_widget,
        "crop_px" : crop_px_2d_v3_widget,
        # "hdf5_file_path": dph_settings_bgsubtracted_widget,
        # "imageid": imageid_widget,
        "save_to_df": save_to_df_widget,
        "create_steps_figures": create_steps_figures_widget,
        "create_figure": create_figure_widget
    },
)

plot_deconvmethod_steps_interactive_output = interactive_output(
    plot_deconvmethod_steps,
    {
        "do_plot_deconvmethod_steps" : do_plot_deconvmethod_steps_widget,
        "clear_plot_deconvmethod_steps" : clear_plot_deconvmethod_steps_widget,
        "step" : deconvmethod_step_widget,
        "ystep" : deconvmethod_ystep_widget
    },
)

plot_deconvmethod_2d_v1_interactive_output = interactive_output(
    plot_deconvmethod_2d_v1,
    {
        "do_plot_deconvmethod_2d_v1": do_plot_deconvmethod_2d_v1_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_2d_v1_widget,
        "crop_px" : crop_px_2d_v1_widget,
        "sigma_x_F_gamma_um_min" : sigma_x_F_gamma_um_min_2d_v1_widget, 
        "sigma_x_F_gamma_um_max" : sigma_x_F_gamma_um_max_2d_v1_widget, 
        "sigma_x_F_gamma_um_stepsize" : sigma_x_F_gamma_um_stepsize_2d_v1_widget, 
        "sigma_y_F_gamma_um_min" : sigma_y_F_gamma_um_min_2d_v1_widget, 
        "sigma_y_F_gamma_um_max" : sigma_y_F_gamma_um_max_2d_v1_widget, 
        "sigma_y_F_gamma_um_stepsize" : sigma_y_F_gamma_um_stepsize_2d_v1_widget, 
        "save_to_df": save_to_df_widget,
    },
)

plot_fitting_vs_deconvolution_output = interactive_output(
    plot_fitting_vs_deconvolution,
    {
        "do_plot_fitting_vs_deconvolution": do_plot_fitting_vs_deconvolution_widget,
        "dataset" : datasets_widget,
        "measurement_file" : dph_settings_bgsubtracted_widget,
        "timestamp_pulse_id": timestamp_pulse_id_widget,
        "xi_um_deconv_column_and_label" : xi_um_deconv_column_and_label_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget,
        "chi2distance_column_and_label" : chi2distance_column_and_label_widget,
        "deconvmethod_outlier_limit" : deconvmethod_outlier_limit_widget,
        "fitting_outlier_limit" : fitting_outlier_limit_widget,
        'xaxisrange' : xaxisrange_widget,
        'yaxisrange' : yaxisrange_widget,
        'use_measurement_default_result' : use_measurement_default_result_widget
    },
)

list_results_output = interactive_output(
    list_results,
    {
        "do_list_results": do_list_results_widget,
        "dataset" : datasets_widget,
        "measurement_file" : dph_settings_bgsubtracted_widget,
        "timestamp_pulse_id": timestamp_pulse_id_widget,
        "xi_um_deconv_column_and_label" : xi_um_deconv_column_and_label_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget,
        "chi2distance_column_and_label" : chi2distance_column_and_label_widget,
        'use_measurement_default_result' : use_measurement_default_result_widget
    },
)

plot_CDCs_output = interactive_output(
    plot_CDCs,
    {
        "do_plot_CDCs": do_plot_CDCs_widget,
        "xi_um_deconv_column_and_label" : xi_um_deconv_column_and_label_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget},
)

plot_xi_um_fit_vs_I_Airy2_fit_output = interactive_output(
    plot_xi_um_fit_vs_I_Airy2_fit,
    {
        "do_plot_xi_um_fit_vs_I_Airy2_fit": do_plot_xi_um_fit_vs_I_Airy2_fit_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget
    },
)


def dph_settings_bgsubtracted_widget_changed(change):
    # plotprofile_interactive_output.clear_output()

    fittingprogress_widget.value = 0

    imageid_widget.disabled = True
    # imageid_widget.options = None
    # imageid_index_widget.disabled = True   
    with h5py.File(dph_settings_bgsubtracted_widget.value, "r") as hdf5_file:
        imageids_float=[]
        imageids_float = hdf5_file["/bgsubtracted/imageid"][:]
        imageids=[]
        imageid_widget.value = None
        imageid_widget.options = None
        for imageid in imageids_float:
            imageids.append(int(imageid[0]))
        imageid_widget.options = imageids
        timestamp_pulse_ids=[]
        timestamps = hdf5_file["Timing/time stamp/fl2user1"][:]
        timestamp_pulse_ids = []
        for timestamp in timestamps:
            timestamp_pulse_ids.append(timestamp[2])
        timestamp_pulse_id_widget.value = None
        timestamp_pulse_id_widget.options = None
        timestamp_pulse_id_widget.options = timestamp_pulse_ids
        imageid_index_widget.min = 0
        imageid_index_widget.max = len(imageid_widget.options) - 1
        imageid_index_widget.value = 0
        imageid_widget.disabled = False
        imageid_index_widget.disabled = False



dph_settings_bgsubtracted_widget.observe(dph_settings_bgsubtracted_widget_changed, names="value") # this was the root cause!

def imageid_index_widget_changed(change):
    imageid_widget.value = imageid_widget.options[imageid_index_widget.value]
    timestamp_pulse_id_widget.value = timestamp_pulse_id_widget.options[imageid_index_widget.value]
    
imageid_index_widget.observe(imageid_index_widget_changed, names="value")



def datasets_widget_changed(change):
    datasets_selection_py_file = datasets_selection_py_files_widget.value
    statustext_widget.value = str(datasets_selection_py_files_widget.value)
    if os.path.isfile(datasets_selection_py_file):
        exec(open(datasets_selection_py_file).read())  
    dph_settings_bgsubtracted = []
    for pattern in ['*'+ s + '.h5' for s in datasets[datasets_widget.value]]: 
        dph_settings_bgsubtracted.extend(bgsubtracted_dir.glob(pattern))
    dph_settings_bgsubtracted_widget.options=dph_settings_bgsubtracted
    measurements_selection_widget.options = dph_settings_bgsubtracted
    measurements_selection_files = []
    for pattern in ['*'+ s + '.h5' for s in datasets_selection[datasets_widget.value]]:  # is this not using the above read file? how to verify?
        measurements_selection_files.extend(bgsubtracted_dir.glob(pattern))
    measurements_selection_widget.value = measurements_selection_files
datasets_widget.observe(datasets_widget_changed, names="value")
datasets_selection_py_files_widget.observe(datasets_widget_changed, names="value") # does not work


def measurements_selection_widget_changed(change):
    datasets_selection_py_file = datasets_selection_py_files_widget.value
    if os.path.isfile(datasets_selection_py_file):
        exec(open(datasets_selection_py_file).read())
    if len(measurements_selection_widget.value) > 0: # avoid the empty array that is generated during datasets_widget_changed
        measurements_selection = []
        for f in measurements_selection_widget.value:
            measurements_selection.append(f.stem)
        datasets_selection.update({ datasets_widget.value : measurements_selection })
    datasets_selection_py_file = datasets_selection_py_files_widget.value
    with open(datasets_selection_py_file, 'w') as f:
        print(datasets_selection, file=f)
    text1 = 'global datasets_selection; datasets_selection = collections.' # see https://stackoverflow.com/questions/23168282/setting-variables-with-exec-inside-a-function
    with open(datasets_selection_py_file) as fpin:
        text2 = fpin.read()
    text = text1 + text2
    with open(datasets_selection_py_file, "w") as fpout:
        fpout.write(text)
    # update some outputs:
    if do_plot_fitting_vs_deconvolution_widget.value == True:
        do_plot_fitting_vs_deconvolution_widget.value = False
        do_plot_fitting_vs_deconvolution_widget.value = True
    if do_plot_CDCs_widget.value == True:
        do_plot_CDCs_widget.value = False
        do_plot_CDCs_widget.value = True
    if do_plot_xi_um_fit_vs_I_Airy2_fit_widget.value == True:
        do_plot_xi_um_fit_vs_I_Airy2_fit_widget.value = False
        do_plot_xi_um_fit_vs_I_Airy2_fit_widget.value = True
measurements_selection_widget.observe(measurements_selection_widget_changed, names="value")


def imageid_widget_changed(change):    
    
    if imageid_widget.value is not None and imageid_widget.options is not None:
    # if do_plot_fitting_v2_widget.value == True:
    
        imageid_index_widget.value = np.where(np.array(imageid_widget.options) == imageid_widget.value)[0][0]

        clear_plot_deconvmethod_steps_widget.value = True
        clear_plot_deconvmethod_steps_widget.value = False

        do_plot_fitting_v1_widget_was_active = False
        if do_plot_fitting_v1_widget.value == True:
            do_plot_fitting_v1_widget_was_active = True
            do_plot_fitting_v1_widget.value = False

        do_plot_fitting_v2_widget_was_active = False
        if do_plot_fitting_v2_widget.value == True:
            do_plot_fitting_v2_widget_was_active = True
            do_plot_fitting_v2_widget.value = False

        do_plot_deconvmethod_2d_v1_widget_was_active = False
        if do_plot_deconvmethod_2d_v1_widget.value == True:
            do_plot_deconvmethod_2d_v1_widget_was_active = True
            do_plot_deconvmethod_2d_v1_widget.value = False

        do_plot_deconvmethod_1d_v2_widget_was_active = False
        if do_plot_deconvmethod_1d_v2_widget.value == True:
            do_plot_deconvmethod_1d_v2_widget_was_active = True
            do_plot_deconvmethod_1d_v2_widget.value = False

        do_plot_deconvmethod_2d_v2_widget_was_active = False
        if do_plot_deconvmethod_2d_v2_widget.value == True:
            do_plot_deconvmethod_2d_v2_widget_was_active = True
            do_plot_deconvmethod_2d_v2_widget.value = False

        do_plot_deconvmethod_1d_v3_widget_was_active = False
        if do_plot_deconvmethod_1d_v3_widget.value == True:
            do_plot_deconvmethod_1d_v3_widget_was_active = True
            do_plot_deconvmethod_1d_v3_widget.value = False

        do_plot_deconvmethod_2d_v3_widget_was_active = False
        if do_plot_deconvmethod_2d_v3_widget.value == True:
            do_plot_deconvmethod_2d_v3_widget_was_active = True
            do_plot_deconvmethod_2d_v3_widget.value = False


        hdf5_file_path = dph_settings_bgsubtracted_widget.value
        imageid = imageid_widget.value
        shiftx_um = np.nan
        xi_um_guess = np.nan
        
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ][0]  # needed for what?
            
            pinholes_bg_avg_sx_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sx_um"].iloc[0]
            pinholes_bg_avg_sy_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sy_um"].iloc[0]
            
            ph = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["pinholes"].iloc[0]
            separation_um = get_sep_and_orient(pinholes)[0]
            orientation = get_sep_and_orient(pinholes)[1]
            setting_wavelength_nm = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_wavelength_nm"].iloc[0]
            energy_hall_uJ = setting_energy_uJ = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_energy_uJ"].iloc[0]

            if orientation == "horizontal":
                beamsize_text_widget.value = r"%.2fum" % (pinholes_bg_avg_sx_um,)
            if orientation == "vertical":
                beamsize_text_widget.value = r"%.2fum" % (pinholes_bg_avg_sy_um,)

            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                    np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
            ]

            # determine how far the maximum of the image is shifted from the center
            pixis_image_norm_max_x_px = np.where(pixis_image_norm==np.max(pixis_image_norm))[1][0]
            pixis_image_norm_max_y_px = np.where(pixis_image_norm==np.max(pixis_image_norm))[0][0]
            pixis_image_norm_min_x_px = np.where(pixis_image_norm==np.min(pixis_image_norm))[1][0]
            pixis_image_norm_min_y_px = np.where(pixis_image_norm==np.min(pixis_image_norm))[0][0]
            delta_max_x_px = pixis_image_norm_max_x_px - int(np.shape(pixis_image_norm)[1]/2)
            delta_max_x_um = delta_max_x_px*13
            delta_min_x_px = pixis_image_norm_min_x_px - int(np.shape(pixis_image_norm)[1]/2)
            textarea_widget.value = 'max_x_px='+str(pixis_image_norm_max_x_px)+'\n'+'min_x_px='+str(pixis_image_norm_min_x_px) +'\n' + \
                'delta_max_x_um='+str(delta_max_x_px*13)+'\n'+'delta_min_x_um='+str(delta_min_x_px*13)
            # if the peaks of the two airy disks are two far away from the center set the shift to 0. Choose the range of shiftx_um empirically
            if abs(delta_max_x_um) > abs(max(shiftx_um_range_widget.value)):
                shiftx_um_widget.value = 0
            else:
                shiftx_um_widget.value = delta_max_x_um

        if load_from_df_widget.value == True:

            

            pixis_profile_avg_width = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pixis_profile_avg_width"].iloc[0]

            # guess parameter - fitting
            df_fitting_v2_best = df_fitting_v2_results[df_fitting_v2_results["timestamp_pulse_id"] == timestamp_pulse_id].sort_values('chi2distance_fitting',ascending=True)

            shiftx_um = df_fitting_v2_best["shiftx_um"].iloc[0]        
            shiftx_um_range_0 = df_fitting_v2_best["shiftx_um_range_0"].iloc[0]
            shiftx_um_range_1 = df_fitting_v2_best["shiftx_um_range_1"].iloc[0]
            shiftx_um_do_fit = df_fitting_v2_best["shiftx_um_do_fit"].iloc[0]
            wavelength_nm = df_fitting_v2_best["wavelength_nm"].iloc[0]
            wavelength_nm_range_0 = df_fitting_v2_best["wavelength_nm_range_0"].iloc[0]
            wavelength_nm_range_1 = df_fitting_v2_best["wavelength_nm_range_1"].iloc[0]
            wavelength_nm_do_fit = df_fitting_v2_best["wavelength_nm_do_fit"].iloc[0]
            z_mm = df_fitting_v2_best["z_mm"].iloc[0]
            z_mm_range_0 = df_fitting_v2_best["z_mm_range_0"].iloc[0]
            z_mm_range_1 = df_fitting_v2_best["z_mm_range_1"].iloc[0]
            z_mm_do_fit = df_fitting_v2_best["z_mm_do_fit"].iloc[0]
            d_um = df_fitting_v2_best["d_um"].iloc[0]
            d_um_range_0 = df_fitting_v2_best["d_um_range_0"].iloc[0]
            d_um_range_1 = df_fitting_v2_best["d_um_range_1"].iloc[0]
            d_um_do_fit = df_fitting_v2_best["d_um_do_fit"].iloc[0]
            gamma = df_fitting_v2_best["gamma"].iloc[0]
            gamma_range_0 = df_fitting_v2_best["gamma_range_0"].iloc[0]
            gamma_range_1 = df_fitting_v2_best["gamma_range_1"].iloc[0]
            gamma_do_fit = df_fitting_v2_best["gamma_do_fit"].iloc[0]
            w1_um = df_fitting_v2_best["w1_um"].iloc[0]
            w1_um_range_0 = df_fitting_v2_best["w1_um_range_0"].iloc[0]
            w1_um_range_1 = df_fitting_v2_best["w1_um_range_1"].iloc[0]
            w1_um_do_fit = df_fitting_v2_best["w1_um_do_fit"].iloc[0]
            w2_um = df_fitting_v2_best["w2_um"].iloc[0]
            w2_um_range_0 = df_fitting_v2_best["w2_um_range_0"].iloc[0]
            w2_um_range_1 = df_fitting_v2_best["w2_um_range_1"].iloc[0]
            w2_um_do_fit = df_fitting_v2_best["w2_um_do_fit"].iloc[0]
            I_Airy1 = df_fitting_v2_best["I_Airy1"].iloc[0]
            I_Airy1_range_0 = df_fitting_v2_best["I_Airy1_range_0"].iloc[0]
            I_Airy1_range_1 = df_fitting_v2_best["I_Airy1_range_1"].iloc[0]
            I_Airy1_do_fit = df_fitting_v2_best["I_Airy1_do_fit"].iloc[0]
            I_Airy2 = df_fitting_v2_best["I_Airy2"].iloc[0]
            I_Airy2_range_0 = df_fitting_v2_best["I_Airy2_range_0"].iloc[0]
            I_Airy2_range_1 = df_fitting_v2_best["I_Airy2_range_1"].iloc[0]
            I_Airy2_do_fit = df_fitting_v2_best["I_Airy2_do_fit"].iloc[0]
            x1_um = df_fitting_v2_best["x1_um"].iloc[0]
            x1_um_range_0 = df_fitting_v2_best["x1_um_range_0"].iloc[0]
            x1_um_range_1 = df_fitting_v2_best["x1_um_range_1"].iloc[0]
            x1_um_do_fit = df_fitting_v2_best["x1_um_do_fit"].iloc[0]
            x2_um = df_fitting_v2_best["x2_um"].iloc[0]
            x2_um_range_0 = df_fitting_v2_best["x2_um_range_0"].iloc[0]
            x2_um_range_1 = df_fitting_v2_best["x2_um_range_1"].iloc[0]
            x2_um_do_fit = df_fitting_v2_best["x2_um_do_fit"].iloc[0]
            normfactor = df_fitting_v2_best["normfactor"].iloc[0]
            normfactor_range_0 = df_fitting_v2_best["normfactor_range_0"].iloc[0]
            normfactor_range_1 = df_fitting_v2_best["normfactor_range_1"].iloc[0]
            normfactor_do_fit = df_fitting_v2_best["normfactor_do_fit"].iloc[0]
            mod_sigma_um = df_fitting_v2_best["mod_sigma_um"].iloc[0]
            mod_sigma_um_range_0 = df_fitting_v2_best["mod_sigma_um_range_0"].iloc[0]
            mod_sigma_um_range_1 = df_fitting_v2_best["mod_sigma_um_range_1"].iloc[0]
            mod_sigma_um_do_fit = df_fitting_v2_best["mod_sigma_um_do_fit"].iloc[0]
            mod_shiftx_um = df_fitting_v2_best["mod_shiftx_um"].iloc[0]
            mod_shiftx_um_range_0 = df_fitting_v2_best["mod_shiftx_um_range_0"].iloc[0]
            mod_shiftx_um_range_1 = df_fitting_v2_best["mod_shiftx_um_range_1"].iloc[0]
            mod_shiftx_um_do_fit = df_fitting_v2_best["mod_shiftx_um_do_fit"].iloc[0]

            if np.isnan(shiftx_um) == False:

                # fitting widgets
                shiftx_um_widget.value = shiftx_um
                shiftx_um_range_widget.value = [shiftx_um_range_0, shiftx_um_range_1]
                shiftx_um_do_fit_widget.value = bool(shiftx_um_do_fit)
                wavelength_nm_widget.value = wavelength_nm
                wavelength_nm_range_widget.value = [wavelength_nm_range_0, wavelength_nm_range_1]
                wavelength_nm_do_fit_widget.value = bool(wavelength_nm_do_fit)
                z_mm_widget.value = z_mm
                z_mm_range_widget.value = [z_mm_range_0, z_mm_range_1]
                z_mm_do_fit_widget.value = bool(z_mm_do_fit)
                d_um_widget.value = d_um
                d_um_range_widget.value = [d_um_range_0, d_um_range_1]
                d_um_do_fit_widget.value = bool(d_um_do_fit)
                gamma_widget.value = gamma
                gamma_range_widget.value = [gamma_range_0, gamma_range_1]
                gamma_do_fit_widget.value = bool(gamma_do_fit)
                w1_um_widget.value = w1_um
                w1_um_range_widget.value = [w1_um_range_0, w1_um_range_1]
                w1_um_do_fit_widget.value = bool(w1_um_do_fit)
                w2_um_widget.value = w2_um
                w2_um_range_widget.value = [w2_um_range_0, w2_um_range_1]
                w2_um_do_fit_widget.value = bool(w2_um_do_fit)
                I_Airy1_widget.value = I_Airy1
                I_Airy1_range_widget.value = [I_Airy1_range_0, I_Airy1_range_1]
                I_Airy1_do_fit_widget.value = bool(I_Airy1_do_fit)
                I_Airy2_widget.value = I_Airy2
                I_Airy2_range_widget.value = [I_Airy2_range_0, I_Airy2_range_1]
                I_Airy2_do_fit_widget.value = bool(I_Airy2_do_fit)
                x1_um_widget.value = x1_um
                x1_um_range_widget.value = [x1_um_range_0, x1_um_range_1]
                x1_um_do_fit_widget.value = bool(x1_um_do_fit)
                x2_um_widget.value = x2_um
                x2_um_range_widget.value = [x2_um_range_0, x2_um_range_1]
                x2_um_do_fit_widget.value = bool(x2_um_do_fit)
                normfactor_widget.value = normfactor
                normfactor_range_widget.value = [normfactor_range_0, normfactor_range_1]
                normfactor_do_fit_widget.value = bool(normfactor_do_fit)
                mod_sigma_um_widget.value = mod_sigma_um
                mod_sigma_um_range_widget.value = [mod_sigma_um_range_0, mod_sigma_um_range_1]
                mod_sigma_um_do_fit_widget.value = bool(mod_sigma_um_do_fit)
                mod_shiftx_um_widget.value = mod_shiftx_um
                mod_shiftx_um_range_widget.value = [mod_shiftx_um_range_0, mod_shiftx_um_range_1]
                mod_shiftx_um_do_fit_widget.value = bool(mod_shiftx_um_do_fit)

            # guess parameter - df_deconvmethod_2d_v1
            df_deconvmethod_2d_v1_best = df_deconvmethod_2d_v1_results[df_deconvmethod_2d_v1_results["timestamp_pulse_id"] == timestamp_pulse_id].sort_values('chi2distance_deconvmethod_2d_v1',ascending=True)

            pixis_profile_avg_width_2d_v1_best = df_deconvmethod_2d_v1_best['pixis_profile_avg_width'].iloc[0]
            crop_px_2d_v1_best = df_deconvmethod_2d_v1_best['crop_px'].iloc[0] 

            sigma_x_F_gamma_um_min_2d_v1_best = df_deconvmethod_2d_v1_best['sigma_x_F_gamma_um_min'].iloc[0]
            sigma_x_F_gamma_um_max_2d_v1_best = df_deconvmethod_2d_v1_best['sigma_x_F_gamma_um_max'].iloc[0]
            sigma_x_F_gamma_um_stepsize_2d_v1_best = df_deconvmethod_2d_v1_best['sigma_x_F_gamma_um_stepsize'].iloc[0]
            sigma_y_F_gamma_um_min_2d_v1_best = df_deconvmethod_2d_v1_best['sigma_y_F_gamma_um_min'].iloc[0]
            sigma_y_F_gamma_um_max_2d_v1_best = df_deconvmethod_2d_v1_best['sigma_y_F_gamma_um_max'].iloc[0]
            sigma_y_F_gamma_um_stepsize_2d_v1_best = df_deconvmethod_2d_v1_best['sigma_y_F_gamma_um_stepsize'].iloc[0]

            if np.isnan(xi_um_guess_1d_v2_best) == False:               
                pixis_profile_avg_width_2d_v1_widget.value = pixis_profile_avg_width_2d_v1_best
                crop_px_2d_v1_widget.value = crop_px_2d_v1_best

                sigma_x_F_gamma_um_min_2d_v1_widget.value = sigma_x_F_gamma_um_min_2d_v1_best
                sigma_x_F_gamma_um_max_2d_v1_widget.value = sigma_x_F_gamma_um_max_2d_v1_best
                sigma_x_F_gamma_um_stepsize_2d_v1_widget.value = sigma_x_F_gamma_um_stepsize_2d_v1_best
                sigma_y_F_gamma_um_min_2d_v1_widget.value = sigma_y_F_gamma_um_min_2d_v1_best
                sigma_y_F_gamma_um_max_2d_v1_widget.value = sigma_y_F_gamma_um_max_2d_v1_best
                sigma_y_F_gamma_um_stepsize_2d_v1_widget.value = sigma_y_F_gamma_um_stepsize_2d_v1_best


            # guess parameter - df_deconvmethod_1d_v2
            df_deconvmethod_1d_v2_best = df_deconvmethod_1d_v2_results[df_deconvmethod_1d_v2_results["timestamp_pulse_id"] == timestamp_pulse_id].sort_values('chi2distance_deconvmethod_1d_v2',ascending=True)

            pixis_profile_avg_width_1d_v2_best = df_deconvmethod_1d_v2_best['pixis_profile_avg_width'].iloc[0]
            crop_px_1d_v2_best = df_deconvmethod_1d_v2_best['crop_px'].iloc[0]         
            balance_1d_v2_best = df_deconvmethod_1d_v2_best['balance'].iloc[0]
            xi_um_guess_1d_v2_best = df_deconvmethod_1d_v2_best['xi_um_guess'].iloc[0]
            sigma_x_F_gamma_um_multiplier_1d_v2_best = df_deconvmethod_1d_v2_best['sigma_x_F_gamma_um_multiplier'].iloc[0]
            xatol_1d_v2_best = df_deconvmethod_1d_v2_best['xatol'].iloc[0]

            if np.isnan(xi_um_guess_1d_v2_best) == False:               
                pixis_profile_avg_width_1d_v2_widget.value = pixis_profile_avg_width_1d_v2_best
                crop_px_1d_v2_widget.value = crop_px_1d_v2_best

                balance_1d_v2_widget.value = balance_1d_v2_best
                xi_um_guess_1d_v2_widget.value = xi_um_guess_1d_v2_best
                sigma_x_F_gamma_um_multiplier_1d_v2_widget.value = sigma_x_F_gamma_um_multiplier_1d_v2_best
                xatol_1d_v2_widget.value = xatol_1d_v2_best

            # guess parameter - df_deconvmethod_2d_v2
            df_deconvmethod_2d_v2_best = df_deconvmethod_2d_v2_results[df_deconvmethod_2d_v2_results["timestamp_pulse_id"] == timestamp_pulse_id].sort_values('chi2distance_deconvmethod_2d_v2',ascending=True)

            pixis_profile_avg_width_2d_v2_best = df_deconvmethod_2d_v2_best['pixis_profile_avg_width'].iloc[0]
            crop_px_2d_v2_best = df_deconvmethod_2d_v2_best['crop_px'].iloc[0]        
            balance_2d_v2_best = df_deconvmethod_2d_v2_best['balance'].iloc[0]
            xi_um_guess_2d_v2_best = df_deconvmethod_2d_v2_best['xi_um_guess'].iloc[0]
            sigma_x_F_gamma_um_multiplier_2d_v2_best = df_deconvmethod_2d_v2_best['sigma_x_F_gamma_um_multiplier'].iloc[0]
            xatol_2d_v2_best = df_deconvmethod_2d_v2_best['xatol'].iloc[0]

            if np.isnan(xi_um_guess_2d_v2_best) == False:               
                pixis_profile_avg_width_2d_v2_widget.value = pixis_profile_avg_width_2d_v2_best
                crop_px_2d_v2_widget.value = crop_px_2d_v2_best

                balance_2d_v2_widget.value = balance_2d_v2_best
                xi_um_guess_2d_v2_widget.value = xi_um_guess_2d_v2_best
                sigma_x_F_gamma_um_multiplier_2d_v2_widget.value = sigma_x_F_gamma_um_multiplier_2d_v2_best
                xatol_2d_v2_widget.value = xatol_2d_v2_best

            # guess parameter - df_deconvmethod_1d_v3
            df_deconvmethod_1d_v3_best = df_deconvmethod_1d_v3_results[df_deconvmethod_1d_v3_results["timestamp_pulse_id"] == timestamp_pulse_id].sort_values('chi2distance_deconvmethod_1d_v3',ascending=True)

            pixis_profile_avg_width_1d_v3_best = df_deconvmethod_1d_v3_best['pixis_profile_avg_width'].iloc[0]
            crop_px_1d_v3_best = df_deconvmethod_1d_v3_best['crop_px'].iloc[0]        
            snr_db_1d_v3_best = df_deconvmethod_1d_v3_best['snr_db'].iloc[0]
            xi_um_guess_1d_v3_best = df_deconvmethod_1d_v3_best['xi_um_guess'].iloc[0]
            sigma_x_F_gamma_um_multiplier_1d_v3_best = df_deconvmethod_1d_v3_best['sigma_x_F_gamma_um_multiplier'].iloc[0]
            xatol_1d_v3_best = df_deconvmethod_1d_v3_best['xatol'].iloc[0]

            if np.isnan(xi_um_guess_1d_v3_best) == False:
                pixis_profile_avg_width_1d_v3_widget.value = pixis_profile_avg_width_1d_v3_best
                crop_px_1d_v3_widget.value = crop_px_1d_v3_best

                snr_db_1d_v3_widget.value = snr_db_1d_v3_best # 26.8 # to do
                xi_um_guess_1d_v3_widget.value = xi_um_guess_1d_v3_best
                sigma_x_F_gamma_um_multiplier_1d_v3_widget.value = sigma_x_F_gamma_um_multiplier_1d_v3_best
                xatol_1d_v3_widget.value = xatol_1d_v3_best

            # guess parameter - df_deconvmethod_2d_v3
            df_deconvmethod_2d_v3_best = df_deconvmethod_2d_v3_results[df_deconvmethod_2d_v3_results["timestamp_pulse_id"] == timestamp_pulse_id].sort_values('chi2distance_deconvmethod_2d_v3',ascending=True)

            pixis_profile_avg_width_2d_v3_best = df_deconvmethod_2d_v3_best['pixis_profile_avg_width'].iloc[0]
            crop_px_2d_v3_best = df_deconvmethod_2d_v3_best['crop_px'].iloc[0]        
            snr_db_2d_v3_best = df_deconvmethod_2d_v3_best['snr_db'].iloc[0]
            xi_um_guess_2d_v3_best = df_deconvmethod_2d_v3_best['xi_um_guess'].iloc[0]
            sigma_x_F_gamma_um_multiplier_2d_v3_best = df_deconvmethod_2d_v3_best['sigma_x_F_gamma_um_multiplier'].iloc[0]
            xatol_2d_v3_best = df_deconvmethod_2d_v3_best['xatol'].iloc[0]

            if np.isnan(xi_um_guess_2d_v3_best) == False:
                pixis_profile_avg_width_2d_v3_widget.value = pixis_profile_avg_width_2d_v3_best
                crop_px_2d_v3_widget.value = crop_px_2d_v3_best

                snr_db_2d_v3_widget.value = snr_db_2d_v3_best # 26.8 # to do
                xi_um_guess_2d_v3_widget.value = xi_um_guess_2d_v3_best
                sigma_x_F_gamma_um_multiplier_2d_v3_widget.value = sigma_x_F_gamma_um_multiplier_2d_v3_best
                xatol_2d_v3_widget.value = xatol_2d_v3_best
            
                

        measurement = os.path.splitext(os.path.basename(dph_settings_bgsubtracted_widget.value))[0]        
        # Set default values for fitting
        if load_from_df_widget.value == False or np.isnan(shiftx_um) == True:
            # load default values instead and inform that there are no saved values!
            # determine how far the maximum of the image is shifted from the center
            pixis_image_norm_max_x_px = np.where(pixis_image_norm==np.max(pixis_image_norm))[1][0]
            pixis_image_norm_max_y_px = np.where(pixis_image_norm==np.max(pixis_image_norm))[0][0]
            pixis_image_norm_min_x_px = np.where(pixis_image_norm==np.min(pixis_image_norm))[1][0]
            pixis_image_norm_min_y_px = np.where(pixis_image_norm==np.min(pixis_image_norm))[0][0]
            delta_max_x_px = pixis_image_norm_max_x_px - int(np.shape(pixis_image_norm)[1]/2)
            delta_max_x_um = delta_max_x_px*13
            delta_min_x_px = pixis_image_norm_min_x_px - int(np.shape(pixis_image_norm)[1]/2)
            textarea_widget.value = 'max_x_px='+str(pixis_image_norm_max_x_px)+'\n'+'min_x_px='+str(pixis_image_norm_min_x_px) +'\n' + \
                'delta_max_x_um='+str(delta_max_x_px*13)+'\n'+'delta_min_x_um='+str(delta_min_x_px*13)
            # if the peaks of the two airy disks are two far away from the center set the shift to 0. Choose the range of shiftx_um empirically
            if abs(delta_max_x_um) > abs(max(shiftx_um_range_widget.value)):
                shiftx_um_widget.value = 0
            else:
                shiftx_um_widget.value = delta_max_x_um
            

            wavelength_nm_widget.value = setting_wavelength_nm
            wavelength_nm_range_widget.value = value = [wavelength_nm_widget.value - 0.1, wavelength_nm_widget.value + 0.1]
            d_um_widget.value = separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
            x1_um_widget.value = -d_um_widget.value * 10 / 2
            x2_um_widget.value = d_um_widget.value * 10 / 2
            x1_um_range_widget.value = [-d_um_widget.value * 10 / 2 - 1000, 0]
            x2_um_range_widget.value = [0, d_um_widget.value * 10 / 2 + 1000]

            # add more default values

            # shiftx_um_widget.value = shiftx_um
            shiftx_um_range_widget.value = [-1500, 1500]
            shiftx_um_do_fit_widget.value = True

            # wavelength_nm_widget.value = wavelength_nm
            # wavelength_nm_range_widget.value = [wavelength_nm_range_0, wavelength_nm_range_1]
            wavelength_nm_do_fit_widget.value = True
            z_mm_widget.value = 5781
            z_mm_range_widget.value = [5770.0, 5790.0]
            z_mm_do_fit_widget.value = False
            # d_um_widget.value = d_um
            d_um_range_widget.value = [50.0, 1337.0]
            d_um_do_fit_widget.value = False
            gamma_widget.value = 0.8
            gamma_range_widget.value = [0.01, 1]
            gamma_do_fit_widget.value = True
            w1_um_widget.value = 11.0
            w1_um_range_widget.value = [8.0, 15.0]
            w1_um_do_fit_widget.value = True
            w2_um_widget.value = 11.0
            w2_um_range_widget.value = [8.0, 15.0]
            w2_um_do_fit_widget.value = True
            I_Airy1_widget.value = 1.0
            I_Airy1_range_widget.value = [0.2, 1.5]
            I_Airy1_do_fit_widget.value = False
            I_Airy2_widget.value = 0.8
            I_Airy2_range_widget.value = [0.2, 5.5]
            I_Airy2_do_fit_widget.value = True
            # x1_um_widget.value = x1_um
            # x1_um_range_widget.value = [x1_um_range_0, x1_um_range_1]
            x1_um_do_fit_widget.value = True
            # x2_um_widget.value = x2_um
            # x2_um_range_widget.value = [x2_um_range_0, x2_um_range_1]
            x2_um_do_fit_widget.value = True
            normfactor_widget.value = 1.0
            normfactor_range_widget.value = [0.1, 1.5]
            normfactor_do_fit_widget.value = False

            # load measurement defaults
            # --> add all the others and distinguish between old a new versions!
            mod_sigma_um_widget.value = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_sigma_um_measurement_default'].iloc[0]
            mod_sigma_um_range_0 = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_sigma_um_range_0_measurement_default'].iloc[0]
            mod_sigma_um_range_1 = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_sigma_um_range_1_measurement_default'].iloc[0]
            mod_sigma_um_range_widget.value = [mod_sigma_um_range_0, mod_sigma_um_range_1]
            # mod_sigma_um_do_fit_widget.value = df_measurement_default[df_measurement_default['measurement']==measurement]['mod_sigma_um_do_fit_measurement_default'].iloc[0] # boolean leads to a problem in the exported csv when importing back!
            mod_shiftx_um_widget.value = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_shiftx_um_measurement_default'].iloc[0]
            mod_shiftx_um_range_0 = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_shiftx_um_range_0_measurement_default'].iloc[0]
            mod_shiftx_um_range_1 = df_fitting_measurement_default[df_fitting_measurement_default['measurement']==measurement]['mod_shiftx_um_range_1_measurement_default'].iloc[0]
            mod_shiftx_um_range_widget.value = [mod_shiftx_um_range_0, mod_shiftx_um_range_1]
            # mod_shiftx_um_do_fit_widget.value = df_measurement_default[df_measurement_default['measurement']==measurement]['mod_shiftx_um_do_fit_measurement_default'].iloc[0]

       
        if load_from_df_widget.value == False or np.isnan(sigma_x_F_gamma_um_min_2d_v1_best) == True: 
            pixis_profile_avg_width_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['pixis_profile_avg_width_measurement_default'].iloc[0]
            crop_px_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]

            sigma_x_F_gamma_um_min_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_min_measurement_default'].iloc[0]
            sigma_x_F_gamma_um_max_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_max_measurement_default'].iloc[0]
            sigma_x_F_gamma_um_stepsize_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_stepsize_measurement_default'].iloc[0]
            sigma_y_F_gamma_um_min_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_y_F_gamma_um_min_measurement_default'].iloc[0]
            sigma_y_F_gamma_um_max_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_y_F_gamma_um_max_measurement_default'].iloc[0]
            sigma_y_F_gamma_um_stepsize_2d_v1_widget.value = df_deconvmethod_2d_v1_measurement_default[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement]['sigma_y_F_gamma_um_stepsize_measurement_default'].iloc[0]
        
        if load_from_df_widget.value == False or np.isnan(xi_um_guess_1d_v2_best) == True:           
            # Load default values for Deconvmethod 1d v2
            pixis_profile_avg_width_1d_v2_widget.value = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['pixis_profile_avg_width_measurement_default'].iloc[0]
            crop_px_1d_v2_widget.value = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]

            balance_1d_v2_widget.value = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
            xi_um_guess_1d_v2_widget.value = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
            xatol_1d_v2_widget.value = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
            sigma_x_F_gamma_um_multiplier_1d_v2_widget.value = df_deconvmethod_1d_v2_measurement_default[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]

        if load_from_df_widget.value == False or np.isnan(xi_um_guess_2d_v2_best) == True:    
            # Load default values for Deconvmethod 2d v2
            pixis_profile_avg_width_2d_v2_widget.value = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['pixis_profile_avg_width_measurement_default'].iloc[0]
            crop_px_2d_v2_widget.value = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]

            balance_2d_v2_widget.value = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['balance_measurement_default'].iloc[0]
            xi_um_guess_2d_v2_widget.value = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
            xatol_2d_v2_widget.value = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
            sigma_x_F_gamma_um_multiplier_2d_v2_widget.value = df_deconvmethod_2d_v2_measurement_default[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]

        if load_from_df_widget.value == False or np.isnan(xi_um_guess_1d_v3_best) == True:
            # Load default values for Deconvmethod 1d v3
            pixis_profile_avg_width_1d_v3_widget.value = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['pixis_profile_avg_width_measurement_default'].iloc[0]
            crop_px_1d_v3_widget.value = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]

            snr_db_1d_v3_widget.value = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['snr_db_measurement_default'].iloc[0]
            xi_um_guess_1d_v3_widget.value = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
            xatol_1d_v3_widget.value = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
            sigma_x_F_gamma_um_multiplier_1d_v3_widget.value = df_deconvmethod_1d_v3_measurement_default[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]

        if load_from_df_widget.value == False or np.isnan(xi_um_guess_2d_v3_best) == True:
            # Load default values for Deconvmethod 2d v3
            pixis_profile_avg_width_2d_v3_widget.value = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['pixis_profile_avg_width_measurement_default'].iloc[0]
            crop_px_2d_v3_widget.value = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['crop_px_measurement_default'].iloc[0]

            snr_db_2d_v3_widget.value = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['snr_db_measurement_default'].iloc[0]
            xi_um_guess_2d_v3_widget.value = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['xi_um_guess_measurement_default'].iloc[0]
            xatol_2d_v3_widget.value = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['xatol_measurement_default'].iloc[0]
            sigma_x_F_gamma_um_multiplier_2d_v3_widget.value = df_deconvmethod_2d_v3_measurement_default[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement]['sigma_x_F_gamma_um_multiplier_measurement_default'].iloc[0]
            

        if do_plot_fitting_v1_widget_was_active == True:
            do_plot_fitting_v1_widget.value = True

        if do_plot_fitting_v2_widget_was_active == True:
            do_plot_fitting_v2_widget.value = True

        if do_plot_deconvmethod_1d_v2_widget_was_active == True:
            do_plot_deconvmethod_1d_v2_widget.value = True
        
        if do_plot_deconvmethod_2d_v2_widget_was_active == True:
            do_plot_deconvmethod_2d_v2_widget.value = True

        if do_plot_deconvmethod_1d_v3_widget_was_active == True:
            do_plot_deconvmethod_1d_v3_widget.value = True
        
        if do_plot_deconvmethod_2d_v3_widget_was_active == True:
            do_plot_deconvmethod_2d_v3_widget.value = True

        if do_plot_deconvmethod_2d_v1_widget_was_active == True:
            do_plot_deconvmethod_2d_v1_widget.value = True

imageid_widget.observe(imageid_widget_changed, names="value")


# set measurement default widget

set_measurement_default_widget = widgets.ToggleButton(
    value=False,
    description='set measurement default',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='set measurement default',
    icon=''
)

def set_measurement_default(change):
    if set_measurement_default_widget.value == True:
        measurement = os.path.splitext(os.path.basename(dph_settings_bgsubtracted_widget.value))[0]
        
        # Set default values for Fitting
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'shiftx_um_measurement_default'] = shiftx_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'shiftx_um_range_0_measurement_default'] = shiftx_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'shiftx_um_range_1_measurement_default'] = shiftx_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'shiftx_um_do_fit_measurement_default'] = shiftx_um_do_fit_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'wavelength_nm_measurement_default'] = wavelength_nm_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'wavelength_nm_range_0_measurement_default'] = wavelength_nm_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'wavelength_nm_range_1_measurement_default'] = wavelength_nm_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'wavelength_nm_do_fit_measurement_default'] = wavelength_nm_do_fit_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'z_mm_measurement_default'] = z_mm_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'z_mm_range_0_measurement_default'] = z_mm_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'z_mm_range_1_measurement_default'] = z_mm_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'd_um_measurement_default'] = d_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'd_um_range_0_measurement_default'] = d_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'd_um_range_1_measurement_default'] = d_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'gamma_measurement_default'] = gamma_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'gamma_range_0_measurement_default'] = gamma_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'gamma_range_1_measurement_default'] = gamma_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'w1_um_measurement_default'] = w1_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'w1_um_range_0_measurement_default'] = w1_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'w1_um_range_1_measurement_default'] = w1_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'w2_um_measurement_default'] = w2_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'w2_um_range_0_measurement_default'] = w2_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'w2_um_range_1_measurement_default'] = w2_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'I_Airy1_measurement_default'] = I_Airy1_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'I_Airy1_range_0_measurement_default'] = I_Airy1_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'I_Airy1_range_1_measurement_default'] = I_Airy1_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'I_Airy2_measurement_default'] = I_Airy2_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'I_Airy2_range_0_measurement_default'] = I_Airy2_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'I_Airy2_range_1_measurement_default'] = I_Airy2_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'x1_um_measurement_default'] = x1_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'x1_um_range_0_measurement_default'] = x1_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'x1_um_range_1_measurement_default'] = x1_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'x2_um_measurement_default'] = x2_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'x2_um_range_0_measurement_default'] = x2_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'x2_um_range_1_measurement_default'] = x2_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'normfactor_measurement_default'] = normfactor_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'normfactor_range_0_measurement_default'] = normfactor_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'normfactor_range_1_measurement_default'] = normfactor_range_widget.value[1]

        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_sigma_um_measurement_default'] = mod_sigma_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_sigma_um_range_0_measurement_default'] = mod_sigma_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_sigma_um_range_1_measurement_default'] = mod_sigma_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_sigma_um_do_fit_measurement_default'] = mod_sigma_um_do_fit_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_shiftx_um_measurement_default'] = mod_shiftx_um_widget.value
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_shiftx_um_range_0_measurement_default'] = mod_shiftx_um_range_widget.value[0]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_shiftx_um_range_1_measurement_default'] = mod_shiftx_um_range_widget.value[1]
        df_fitting_measurement_default.loc[df_fitting_measurement_default['measurement']==measurement, 'mod_shiftx_um_do_fit_measurement_default'] = mod_shiftx_um_do_fit_widget.value

        # Set default values for Fitting_v1
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'shiftx_um_measurement_default'] = shiftx_um_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'shiftx_um_range_0_measurement_default'] = shiftx_um_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'shiftx_um_range_1_measurement_default'] = shiftx_um_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'shiftx_um_do_fit_measurement_default'] = shiftx_um_do_fit_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'wavelength_nm_measurement_default'] = wavelength_nm_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'wavelength_nm_range_0_measurement_default'] = wavelength_nm_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'wavelength_nm_range_1_measurement_default'] = wavelength_nm_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'wavelength_nm_do_fit_measurement_default'] = wavelength_nm_do_fit_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'z_mm_measurement_default'] = z_mm_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'z_mm_range_0_measurement_default'] = z_mm_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'z_mm_range_1_measurement_default'] = z_mm_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'd_um_measurement_default'] = d_um_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'd_um_range_0_measurement_default'] = d_um_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'd_um_range_1_measurement_default'] = d_um_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'gamma_measurement_default'] = gamma_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'gamma_range_0_measurement_default'] = gamma_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'gamma_range_1_measurement_default'] = gamma_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'w1_um_measurement_default'] = w1_um_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'w1_um_range_0_measurement_default'] = w1_um_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'w1_um_range_1_measurement_default'] = w1_um_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'w2_um_measurement_default'] = w2_um_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'w2_um_range_0_measurement_default'] = w2_um_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'w2_um_range_1_measurement_default'] = w2_um_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'I_Airy1_measurement_default'] = I_Airy1_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'I_Airy1_range_0_measurement_default'] = I_Airy1_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'I_Airy1_range_1_measurement_default'] = I_Airy1_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'I_Airy2_measurement_default'] = I_Airy2_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'I_Airy2_range_0_measurement_default'] = I_Airy2_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'I_Airy2_range_1_measurement_default'] = I_Airy2_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'x1_um_measurement_default'] = x1_um_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'x1_um_range_0_measurement_default'] = x1_um_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'x1_um_range_1_measurement_default'] = x1_um_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'x2_um_measurement_default'] = x2_um_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'x2_um_range_0_measurement_default'] = x2_um_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'x2_um_range_1_measurement_default'] = x2_um_range_widget.value[1]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'normfactor_measurement_default'] = normfactor_widget.value
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'normfactor_range_0_measurement_default'] = normfactor_range_widget.value[0]
        df_fitting_v1_measurement_default.loc[df_fitting_v1_measurement_default['measurement']==measurement, 'normfactor_range_1_measurement_default'] = normfactor_range_widget.value[1]

        # Set default values for Deconvmethod 2d v1
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_min_measurement_default'] = sigma_x_F_gamma_um_min_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_max_measurement_default'] = sigma_x_F_gamma_um_max_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_stepsize_measurement_default'] = sigma_x_F_gamma_um_stepsize_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'sigma_y_F_gamma_um_min_measurement_default'] = sigma_y_F_gamma_um_min_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'sigma_y_F_gamma_um_max_measurement_default'] = sigma_y_F_gamma_um_max_2d_v1_widget.value
        df_deconvmethod_2d_v1_measurement_default.loc[df_deconvmethod_2d_v1_measurement_default['measurement']==measurement, 'sigma_y_F_gamma_um_stepsize_measurement_default'] = sigma_y_F_gamma_um_stepsize_2d_v1_widget.value

        # Set default values for Deconvmethod 1d v2
        df_deconvmethod_1d_v2_measurement_default.loc[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement, 'balance_measurement_default'] = balance_1d_v2_widget.value
        df_deconvmethod_1d_v2_measurement_default.loc[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_1d_v2_widget.value
        df_deconvmethod_1d_v2_measurement_default.loc[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_1d_v2_widget.value
        df_deconvmethod_1d_v2_measurement_default.loc[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement, 'xi_um_guess_measurement_default'] = xi_um_guess_1d_v2_widget.value
        df_deconvmethod_1d_v2_measurement_default.loc[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_multiplier_measurement_default'] = sigma_x_F_gamma_um_multiplier_1d_v2_widget.value
        df_deconvmethod_1d_v2_measurement_default.loc[df_deconvmethod_1d_v2_measurement_default['measurement']==measurement, 'xatol_measurement_default'] = xatol_1d_v2_widget.value
        
        # Set default values for Deconvmethod 2d v2
        df_deconvmethod_2d_v2_measurement_default.loc[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement, 'balance_measurement_default'] = balance_2d_v2_widget.value
        df_deconvmethod_2d_v2_measurement_default.loc[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_2d_v2_widget.value
        df_deconvmethod_2d_v2_measurement_default.loc[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_2d_v2_widget.value
        df_deconvmethod_2d_v2_measurement_default.loc[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement, 'xi_um_guess_measurement_default'] = xi_um_guess_2d_v2_widget.value
        df_deconvmethod_2d_v2_measurement_default.loc[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_multiplier_measurement_default'] = sigma_x_F_gamma_um_multiplier_2d_v2_widget.value
        df_deconvmethod_2d_v2_measurement_default.loc[df_deconvmethod_2d_v2_measurement_default['measurement']==measurement, 'xatol_measurement_default'] = xatol_2d_v2_widget.value

        # Set default values for Deconvmethod 1d v3
        df_deconvmethod_1d_v3_measurement_default.loc[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement, 'snr_db_measurement_default'] = snr_db_1d_v3_widget.value
        df_deconvmethod_1d_v3_measurement_default.loc[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_1d_v3_widget.value
        df_deconvmethod_1d_v3_measurement_default.loc[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_1d_v3_widget.value
        df_deconvmethod_1d_v3_measurement_default.loc[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement, 'xi_um_guess_measurement_default'] = xi_um_guess_1d_v3_widget.value
        df_deconvmethod_1d_v3_measurement_default.loc[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_multiplier_measurement_default'] = sigma_x_F_gamma_um_multiplier_1d_v3_widget.value
        df_deconvmethod_1d_v3_measurement_default.loc[df_deconvmethod_1d_v3_measurement_default['measurement']==measurement, 'xatol_measurement_default'] = xatol_1d_v3_widget.value

        # Set default values for Deconvmethod 2d v3
        df_deconvmethod_2d_v3_measurement_default.loc[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement, 'snr_db_measurement_default'] = snr_db_2d_v3_widget.value
        df_deconvmethod_2d_v3_measurement_default.loc[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement, 'pixis_profile_avg_width_measurement_default'] = pixis_profile_avg_width_2d_v3_widget.value
        df_deconvmethod_2d_v3_measurement_default.loc[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement, 'crop_px_measurement_default'] = crop_px_2d_v3_widget.value
        df_deconvmethod_2d_v3_measurement_default.loc[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement, 'xi_um_guess_measurement_default'] = xi_um_guess_2d_v3_widget.value
        df_deconvmethod_2d_v3_measurement_default.loc[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement, 'sigma_x_F_gamma_um_multiplier_measurement_default'] = sigma_x_F_gamma_um_multiplier_2d_v3_widget.value
        df_deconvmethod_2d_v3_measurement_default.loc[df_deconvmethod_2d_v3_measurement_default['measurement']==measurement, 'xatol_measurement_default'] = xatol_2d_v3_widget.value


        set_measurement_default_widget.value = False

set_measurement_default_widget.observe(set_measurement_default, names="value")

# save measurement default to csv widget

save_measurement_default_to_csv_widget = widgets.ToggleButton(
    value=False,
    description='measurement default --> csv',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save measurement default to csv',
    icon=''
)

def save_measurement_default_to_csv(change):
    if save_measurement_default_to_csv_widget.value == True:

        df_measurement_default_file = Path.joinpath(results_dir, 'df_fitting_v2_measurement_default.csv')
        df_fitting_measurement_default.to_csv(df_measurement_default_file)

        df_measurement_default_file = Path.joinpath(results_dir, 'df_fitting_v1_measurement_default.csv')
        df_fitting_v1_measurement_default.to_csv(df_measurement_default_file)

        df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v1_measurement_default.csv')
        df_deconvmethod_2d_v1_measurement_default.to_csv(df_measurement_default_file)

        df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v2_measurement_default.csv')
        df_deconvmethod_1d_v2_measurement_default.to_csv(df_measurement_default_file)
        
        df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v2_measurement_default.csv')
        df_deconvmethod_2d_v2_measurement_default.to_csv(df_measurement_default_file)

        df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v3_measurement_default.csv')
        df_deconvmethod_2d_v3_measurement_default.to_csv(df_measurement_default_file)
        
        df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v3_measurement_default.csv')
        df_deconvmethod_2d_v3_measurement_default.to_csv(df_measurement_default_file)

        save_measurement_default_to_csv_widget.value = False

save_measurement_default_to_csv_widget.observe(save_measurement_default_to_csv, names="value")

# load measurement default from csv widget

load_measurement_default_from_csv_widget = widgets.ToggleButton(
    value=False,
    description='csv --> measurement default',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='load measurement default from csv',
    icon=''
)

def load_measurement_default_from_csv(change):
    global df_fitting_measurement_default
    global df_fitting_v1_measurement_default
    global df_deconvmethod_2d_v1_measurement_default
    global df_deconvmethod_1d_v2_measurement_default
    global df_deconvmethod_2d_v2_measurement_default
    global df_deconvmethod_1d_v3_measurement_default
    global df_deconvmethod_2d_v3_measurement_default
    
    if load_measurement_default_from_csv_widget.value == True:
        df_measurement_default_file = Path.joinpath(results_dir, 'df_fitting_v2_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_fitting_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

    df_measurement_default_file = Path.joinpath(results_dir, 'df_fitting_v1_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_fitting_v1_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

    df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v1_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_deconvmethod_2d_v1_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

    df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v2_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_deconvmethod_1d_v2_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)
    
    df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v2_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_deconvmethod_2d_v2_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)

    df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_1d_v3_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_deconvmethod_1d_v3_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)
    
    df_measurement_default_file = Path.joinpath(results_dir, 'df_deconvmethod_2d_v3_measurement_default.csv')
    if os.path.isfile(df_measurement_default_file):
        df_deconvmethod_2d_v3_measurement_default = pd.read_csv(df_measurement_default_file,index_col=0)


    load_measurement_default_from_csv_widget.value = False

load_measurement_default_from_csv_widget.observe(load_measurement_default_from_csv, names="value")


# run widgets behaviour

## run_over_all_images
run_over_all_images_continue_file = 'delete_this_file_to_abort_run_over_all_images.py'
def run_over_all_images():
    start = datetime.now()
    with open(run_over_all_images_continue_file,'w') as f:
        f.write('from os import remove;from sys import argv;remove(argv[0])')
    run_over_all_images_progress_widget.bar_style = 'info'
    
    run_over_all_images_progress_widget.value = 0
    i = 0
    for imageid in imageid_widget.options:
        if os.path.isfile(run_over_all_images_continue_file):
            imageid_widget.value = imageid
            i = i+1
            run_over_all_images_progress_widget.value = int(i/len(imageid_widget.options)*100)
            end = datetime.now()
            time_taken = end - start
            time_left = time_taken/i * (len(imageid_widget.options) - i)
            run_over_all_images_statustext_widget.value = str(time_taken) + "|" + str(time_left)

    # disable for now, there is a problem with the fits_header_list, possible duplicate columns?
    # df_fits = df0[['timestamp_pulse_id'] + fits_header_list]
    # df_fits_csv_file = df_fits_csv_files_widget.value
    # df_fits.to_csv(df_fits_csv_file)

    if os.path.isfile(run_over_all_images_continue_file):
        run_over_all_images_progress_widget.bar_style = 'success'
    else:
        run_over_all_images_progress_widget.bar_style = 'danger'


def update_run_over_all_images_widget(change):
    if run_over_all_images_widget.value == True:

        run_over_all_images_widget.button_style = 'info'
        run_over_all_images()
        run_over_all_images_widget.button_style = 'success'
        run_over_all_images_widget.value = False
        run_over_all_images_widget.button_style = ''
        

run_over_all_images_widget.observe(update_run_over_all_images_widget, names='value')


## run_over_all_measurements
run_over_all_measurements_continue_file = 'delete_this_file_to_abort_run_over_all_measurements.py'
def run_over_all_measurements():
    start = datetime.now()
    with open(run_over_all_measurements_continue_file,'w') as f:
        f.write('from os import remove;from sys import argv;remove(argv[0])')
    run_over_all_measurements_progress_widget.bar_style = 'info'
    
    run_over_all_measurements_progress_widget.value = 0
    i = 0
    for measurement in measurements_selection_widget.value:
        if os.path.isfile(run_over_all_measurements_continue_file):
            dph_settings_bgsubtracted_widget.value = measurement
            run_over_all_images()
            i = i+1
            run_over_all_measurements_progress_widget.value = int(i/len(measurements_selection_widget.value)*100)
            end = datetime.now()
            time_taken = end - start
            time_left = time_taken/i * (len(measurements_selection_widget.value) - i)
            run_over_all_measurements_statustext_widget.value = str(time_taken) + "|" + str(time_left)
    if os.path.isfile(run_over_all_measurements_continue_file):
        run_over_all_measurements_progress_widget.bar_style = 'success'
    else:
        run_over_all_measurements_progress_widget.bar_style = 'danger'


def update_run_over_all_measurements_widget(change):
    if run_over_all_measurements_widget.value == True:

        run_over_all_measurements_widget.button_style = 'info'
        run_over_all_measurements()
        run_over_all_measurements_widget.button_style = 'success'
        run_over_all_measurements_widget.value = False
        run_over_all_measurements_widget.button_style = ''
        

run_over_all_measurements_widget.observe(update_run_over_all_measurements_widget, names='value')


## run_over_all_datasets
run_over_all_datasets_continue_file = 'delete_this_file_to_abort_run_over_all_datasets.py'
def run_over_all_datasets():
    start = datetime.now()
    with open(run_over_all_datasets_continue_file,'w') as f:
        f.write('from os import remove;from sys import argv;remove(argv[0])')
    run_over_all_datasets_progress_widget.bar_style = 'info'
    
    run_over_all_datasets_progress_widget.value = 0
    i = 0
    for dataset in list(datasets_selection):
        if os.path.isfile(run_over_all_datasets_continue_file):
            datasets_widget.value = dataset
            run_over_all_measurements()
            i = i+1
            run_over_all_datasets_progress_widget.value = int(i/len(list(datasets_selection))*100)
            end = datetime.now()
            time_taken = end - start
            time_left = time_taken/i * (len(list(datasets_selection)) - i)
            run_over_all_datasets_statustext_widget.value = str(time_taken) + "|" + str(time_left)
    if os.path.isfile(run_over_all_datasets_continue_file):
        run_over_all_datasets_progress_widget.bar_style = 'success'
    else:
        run_over_all_datasets_progress_widget.bar_style = 'danger'


def update_run_over_all_datasets_widget(change):
    if run_over_all_datasets_widget.value == True:

        run_over_all_datasets_widget.button_style = 'info'
        run_over_all_datasets()
        run_over_all_datasets_widget.button_style = 'success'
        run_over_all_datasets_widget.value = False
        run_over_all_datasets_widget.button_style = ''
        

run_over_all_datasets_widget.observe(update_run_over_all_datasets_widget, names='value')




# getting all parameters from the file



with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
    imageids = hdf5_file["/bgsubtracted/imageid"][:]

    imageid = imageids[0]

    hdf5_file_path = dph_settings_bgsubtracted_widget.value
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
        ]
        pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
        ]
        timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
        ][2]
        pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == float(imageid))[0][0]
        ][0]

    pinholes = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["pinholes"].iloc[0]
    separation_um = get_sep_and_orient(pinholes)[0]
    orientation = get_sep_and_orient(pinholes)[1]
    setting_wavelength_nm = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_wavelength_nm"].iloc[0]
    setting_energy_uJ = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["setting_energy_uJ"].iloc[0]
    
    hdf5_file_name_image = df_settings[df_settings['dph_settings'] == dph_settings_bgsubtracted_widget.value.name.split('.h5')[0]]["hdf5_file_name"].iloc[0]

    beamposition_horizontal_interval = 1000  # random number, store in hdf5?


# Increase output of Jupyer Notebook Cell:
from IPython.display import Javascript

display(
    Javascript("""google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})""")
)  # https://stackoverflow.com/a/57346765


output_tabs_right_ratio_widget = widgets.IntSlider(value=35, description='right box width / %')


children_left = [plot_fitting_v2_interactive_output,
                 plot_fitting_v1_interactive_output,
                 plot_deconvmethod_1d_v3_interactive_output,
                 plot_deconvmethod_2d_v3_interactive_output,
                 plot_deconvmethod_1d_v2_interactive_output,
                 plot_deconvmethod_2d_v2_interactive_output,
                 VBox([HBox([do_plot_deconvmethod_steps_widget, clear_plot_deconvmethod_steps_widget,
                      deconvmethod_ystep_widget, deconvmethod_step_widget]), plot_deconvmethod_steps_interactive_output]),
                 plot_deconvmethod_2d_v1_interactive_output,
                 VBox([
                     do_plot_CDCs_widget,
                     plot_CDCs_output
                 ]),
                 VBox([
                     do_plot_xi_um_fit_vs_I_Airy2_fit_widget,
                     plot_xi_um_fit_vs_I_Airy2_fit_output,
                 ]),
                 VBox([
                     do_list_results_widget,
                     list_results_output
                 ])
                 ]



# tabs_left = widgets.Tab(layout=widgets.Layout(height='1000px', width='67%'))
tabs_left = widgets.Tab(layout=widgets.Layout(height='1000px', width=str(100-output_tabs_right_ratio_widget.value)+'%'))
tabs_left.children = children_left
tabs_left.set_title(0, 'Fitting_v2')
tabs_left.set_title(1, 'Fitting_v1')
tabs_left.set_title(2, 'Deconv_1d_v3')
tabs_left.set_title(3, 'Deconv_2d_v3')
tabs_left.set_title(4, 'Deconv_1d_v2')
tabs_left.set_title(5, 'Deconv_2d_v2')
tabs_left.set_title(6, 'Deconv Steps')
tabs_left.set_title(7, 'Deconv_2d_v1')
tabs_left.set_title(8, 'CDCs')
tabs_left.set_title(9, 'plot_xi_um_fit_vs_I_Airy2_fit')
tabs_left.set_title(10, 'list_results')


column0 = widgets.VBox(
    [
        HBox([do_plot_fitting_v2_widget,xi_um_fit_v2_widget]),
        HBox([do_plot_fitting_v1_widget,xi_um_fit_v1_widget]),        
        HBox([do_plot_deconvmethod_1d_v2_widget,deconvmethod_1d_v2_result_widget]),
        HBox([do_plot_deconvmethod_2d_v2_widget,deconvmethod_2d_v2_result_widget]),
        HBox([do_plot_deconvmethod_1d_v3_widget,deconvmethod_1d_v3_result_widget]),
        HBox([do_plot_deconvmethod_2d_v3_widget,deconvmethod_2d_v3_result_widget]),
        HBox([do_plot_deconvmethod_2d_v1_widget,deconvmethod_2d_v1_result_widget]),
        create_steps_figures_widget,
        create_figure_widget
    ]
)

column6 = widgets.VBox(
    [        
        xi_um_fit_v1_widget, 
        xi_um_fit_v2_widget,
        deconvmethod_1d_v2_result_widget, 
        deconvmethod_2d_v2_result_widget,
        deconvmethod_2d_v1_result_widget
    ]
)



children_right = [
                    column0,
                    VBox([
                        column0,
                        VBox([
                            HBox([
                                load_measurement_default_from_csv_widget,
                                set_measurement_default_widget,
                                save_measurement_default_to_csv_widget,
                                ]),
                            HBox([
                                save_to_df_widget,
                                load_from_df_widget,
                                ]),
                            ]),
                        parameter_tabs,
                        ]),
                    VBox([column0,
                        do_plot_fitting_vs_deconvolution_widget,
                        HBox([VBox([use_measurement_default_result_widget, \
                                                        xi_um_deconv_column_and_label_widget, \
                                                        xi_um_fit_column_and_label_widget, \
                                                        chi2distance_column_and_label_widget, \
                                                        sort_imageids_by_chi2distance_widget]),
                    VBox([deconvmethod_outlier_limit_widget,fitting_outlier_limit_widget]),
                    VBox([xaxisrange_widget, yaxisrange_widget])]), 
                    plot_fitting_vs_deconvolution_output]),
                    VBox([
                        textarea_widget, 
                        beamsize_text_widget,
                        pixis_profile_avg_width_widget,
                        crop_px_widget,
                        savefigure_profile_fit_widget,
                        do_textbox_widget,
                        output_tabs_right_ratio_widget
                        ]),
]




# tabs_right = widgets.Tab(layout=widgets.Layout(height='1000px', width='33%'))
tabs_right = widgets.Tab(layout=widgets.Layout(height='1000px', width=str(output_tabs_right_ratio_widget.value)+'%'))
tabs_right.children = children_right
tabs_right.set_title(0, 'Methods & Results')
tabs_right.set_title(1, 'Parameter')
tabs_right.set_title(2, 'Fitting vs. Deconvolution')
tabs_right.set_title(3, 'other')


grid = widgets.GridspecLayout(1, 3, height='1000px', width='100%')
grid[0, 0:1] = tabs_left
grid[0, 2] = tabs_right


def update_output_tabs_widths(change):
    tabs_right.layout.width = str(output_tabs_right_ratio_widget.value)+'%'
    tabs_left.layout.width = str(100-output_tabs_right_ratio_widget.value)+'%'
output_tabs_right_ratio_widget.observe(update_output_tabs_widths)


input_widgets = VBox([
    HBox([datasets_widget,
          dph_settings_bgsubtracted_widget,
          timestamp_pulse_id_widget,
          imageid_widget, imageid_index_widget]),
    HBox([run_over_all_datasets_widget, run_over_all_datasets_progress_widget, run_over_all_datasets_statustext_widget,
          run_over_all_measurements_widget, run_over_all_measurements_progress_widget, run_over_all_measurements_statustext_widget,
          run_over_all_images_widget, run_over_all_images_progress_widget, run_over_all_images_statustext_widget]),

    ])

measurement_selection_settings_widgets = VBox([
    datasets_widget,
    measurements_selection_widget,
    HBox([
        datasets_selection_py_files_widget,
        create_new_datasets_selection_py_file_widget,
    ]),
    
])

import_export_results_widgets = HBox([
            scan_for_df_fits_csv_files_widget,
            df_fits_csv_files_widget,
            load_csv_to_df_widget,
            df_fits_csv_save_widget,
            create_new_csv_file_widget
        ])

input_tabs_children = [ input_widgets,
                        measurement_selection_settings_widgets,
                        import_export_results_widgets
                      ]
input_tabs = widgets.Tab()
input_tabs.children = input_tabs_children
input_tabs.set_title(0, 'Input')
input_tabs.set_title(1, 'Settings')
input_tabs.set_title(2, 'Import/Export results')


# Display widgets and outputs
display(
    VBox(
        [
            HBox([fittingprogress_widget, statustext_widget]),
            input_tabs,
            HBox([tabs_left,tabs_right])
        ]
    )
)
dph_settings_bgsubtracted_widget_changed(None)
 



