# <codecell>

from pathlib import Path  # see https://docs.python.org/3/library/pathlib.html#basic-use

## Define paths

data_dir = Path("g:/My Drive/PhD/coherence/data/")
useful_dir = Path("g:/My Drive/PhD/coherence/data/useful/")
bgsubtracted_dir = Path("g:/My Drive/PhD/coherence/data/bgsubtracted/")
print(useful_dir)
scratch_dir = Path("g:/My Drive/PhD/coherence/data/scratch_cc/")
# prebgsubtracted_dir
# bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')

# <codecell>

# imports

from coherencefinder.deconvolution_module import calc_sigma_F_gamma_um, deconvmethod, normalize
from coherencefinder.fitting_module import Airy, find_sigma, fit_profile, fit_profile_v2, gaussian

# <codecell>

#!pip install bqplot

# <codecell>

import time
from datetime import datetime
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
    GridspecLayout
)
import ipywidgets as widgets
# import bqplot as bq

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

from IPython.display import display, clear_output

import os.path

# import pickle as pl

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


# %% settings for figures and latex

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


datasets_widget_layout = widgets.Layout(width="100%")
datasets_widget = widgets.Dropdown(options=list(datasets), layout=datasets_widget_layout, description='Dataset:')
# settings_widget.observe(update_settings, names='value')
# display(dph_settings_widget)
# initialize a dictionary holding a selection of measurements
datasets_selection = datasets.copy()


# dph_settings_widget_layout = widgets.Layout(width="100%")
# dph_settings_widget = widgets.Dropdown(options=dph_settings, layout=dph_settings_widget_layout)
# settings_widget.observe(update_settings, names='value')
# display(dph_settings_widget)

# dph_settings_bgsubtracted = list(bgsubtracted_dir.glob("*.h5"))
dph_settings_bgsubtracted = []
for pattern in ['*'+ s + '.h5' for s in datasets[datasets_widget.value]]: 
    dph_settings_bgsubtracted.extend(bgsubtracted_dir.glob(pattern))


dph_settings_bgsubtracted_widget_layout = widgets.Layout(width="100%")
dph_settings_bgsubtracted_widget = widgets.Dropdown(
    options=dph_settings_bgsubtracted,
    layout=dph_settings_bgsubtracted_widget_layout,
    description='Measurement:'
    # value=dph_settings_bgsubtracted[3],  # workaround, because some hdf5 files have no proper timestamp yet
)
# settings_widget.observe(update_settings, names='value')

measurements_selection_widget_layout = widgets.Layout(width="100%")
measurements_selection_widget = widgets.SelectMultiple(
    options=dph_settings_bgsubtracted,
    value=dph_settings_bgsubtracted,
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
fits_header_list4 = ["xi_y_um_fit", "zeta_y", "zeta_y_fit", "xi_um_fit"]

fits_header_list5 = ['gamma_fit_at_center', 'xi_um_fit_at_center', 'mod_sigma_um_fit', 'mod_shiftx_um_fit']

fits_header_list = fits_header_list1 + fits_header_list2 + fits_header_list3 + fits_header_list4 + fits_header_list5


# fits_header_list1 already exists in saved csv, only adding fits_header_list2, only initiate when
initiate_df_fits = True
# if initiate_df_fits == True:
    # df0 = df0.reindex(columns = df0.columns.tolist() + fits_header_list)
    # df_fits = df0[['timestamp_pulse_id'] + fits_header_list]

# load saved df_fits from csv
# df_fits_csv_filename = 'df_fits_v2.csv'
df_fits_csv_filename = 'df_fits_2022-02-07--19h54.csv'
load_df_fits_csv = True
if load_df_fits_csv == True:
    df_fits = pd.read_csv(Path.joinpath(data_dir, df_fits_csv_filename), index_col=0)
    df_fits_clean = df_fits[df_fits["pixis_rotation"].notna()].drop_duplicates()
    df_fits = df_fits_clean


df0 = pd.merge(df0, df_fits, on="timestamp_pulse_id", how="outer")


# """# List all groups inside the hd5file"""

# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:

#     def printname(name):
#         print(name)

#     hdf5_file.visit(printname)


# """# display bgsubtracted images"""


# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
#     imageids = hdf5_file["/bgsubtracted/imageid"][:]
#     imageid = imageids[0]
#     pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
#         np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
#     ]
#     plt.imshow(pixis_image_norm)
#     print(
#         "imageid="
#         + str(hdf5_file["/bgsubtracted/imageid"][np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]])
#     )

# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
#     imageids = hdf5_file["/bgsubtracted/imageid"][:]
#     imageid = imageids[0]
#     pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
#         np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
#     ]
#     plt.plot(pixis_profile_avg)
#     print(
#         "imageid="
#         + str(hdf5_file["/bgsubtracted/imageid"][np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]])
#     )

# # reproducing
# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
#     imageids = hdf5_file["/bgsubtracted/imageid"][:]
#     imageid = imageids[0]
#     # use here 1 sigma of the gaussian or something similar, so it is comparable to different profile sizes
#     pixis_avg_width = 200
#     pixis_centery_px = int(
#         hdf5_file["/bgsubtracted/pixis_centery_px"][np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]][0]
#     )
#     print(pixis_centery_px)
#     pixis_profile_avg = np.average(
#         hdf5_file["/bgsubtracted/pixis_image_norm"][np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]][
#             int(pixis_centery_px - pixis_avg_width / 2) : int(pixis_centery_px + pixis_avg_width / 2), :
#         ],
#         axis=0,
#     )
#     pixis_profile_avg = normalize(pixis_profile_avg)
#     plt.plot(pixis_profile_avg)
#     # why is this not giving the same profile?? in the GUI a width of 200 is defined. what was actually calculated?


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

plotprofile_active_widget = widgets.Checkbox(value=False, description="active", disabled=False)
do_deconvmethod_widget = widgets.Checkbox(value=False, description="do_deconvmethod", disabled=False)
xi_um_guess_widget = widgets.FloatText(value=900, description='xi_um_guess')
scan_x_widget = widgets.Checkbox(value=False, description="scan_x", disabled=False)
sigma_x_F_gamma_um_multiplier_widget = widgets.FloatText(value=1.5, description='sigma_x_F_gamma_um_multiplier_widget')
crop_px_widget = widgets.IntText(value=200, description='crop_px')
pixis_profile_avg_width_widget = widgets.IntText(value=200, description='profile width / px')

imageid_profile_fit_widget = widgets.Dropdown(
    # options=imageid_widget.options,
    options=[],
    description="imageid:",
    disabled=False,
)

savefigure_profile_fit_widget = widgets.Checkbox(value=False, description="savefigure", disabled=False)

save_to_df_widget = widgets.Checkbox(value=False, description="save_to_df", disabled=False)

df_fits_csv_save_widget = widgets.ToggleButton(
    value=False,
    description='save df_fits to csv',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='save df_fits to csv',
    icon='check'
)


do_textbox_widget = widgets.Checkbox(value=False, description="do_textbox", disabled=False)

textarea_widget = widgets.Textarea(value="info", placeholder="Type something", description="Fitting:", disabled=False)
beamsize_text_widget = widgets.Text(
    value="", placeholder="beamsize in rms", description=r"beam rms", disabled=False
)
fit_profile_text_widget = widgets.Text(
    value="", placeholder="xi_fit_um", description=r"\({\xi}_{fit}\)", disabled=False
)
xi_um_fit_at_center_text_widget = widgets.Text(
    value="", placeholder="xi_fit_um_at_center", description=r"\({\xi}_{fit}_{center}\)", disabled=False
)
deconvmethod_simple_text_widget = widgets.Text(
    value="", placeholder="xi_um", description=r"{\xi}", disabled=False
) 
deconvmethod_text_widget = widgets.Text(
    value="", placeholder="(xi_x_um, xi_y_um)", description=r"\({\xi}_x,{\xi}_y\)", disabled=False
)  # latex only working in browser?

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

shiftx_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")
wavelength_nm_do_fit_widget = widgets.Checkbox(value=True, description="fit")
z_mm_do_fit_widget = widgets.Checkbox(value=False, description="fit")
d_um_do_fit_widget = widgets.Checkbox(value=False, description="fit")
gamma_do_fit_widget = widgets.Checkbox(value=True, description="fit")
w1_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")
w2_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")
I_Airy1_do_fit_widget = widgets.Checkbox(value=False, description="fit")
I_Airy2_do_fit_widget = widgets.Checkbox(value=True, description="fit")
x1_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")
x2_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")
normfactor_do_fit_widget = widgets.Checkbox(value=False, description="fit")
mod_sigma_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")
mod_shiftx_um_do_fit_widget = widgets.Checkbox(value=True, description="fit")

value_widget_layout = widgets.Layout(width="100px")
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


do_plot_fitting_vs_deconvolution_widget = widgets.Checkbox(value=False, description="do fitting vs deconv plot")

xi_um_deconv_column_and_label_widget = widgets.Dropdown(
    options=[('xi_x_um',('xi_x_um',r"$\xi_x$ / um (deconv)")),('xi_um',('xi_um',r"$\xi$ / um (deconv)"))],
    description="deconv variant:",
    disabled=False,
)

xi_um_fit_column_and_label_widget = widgets.Dropdown(
    options=[('xi_um_fit',('xi_um_fit',r"$\xi$ / um (fit)")),('xi_um_fit_at_center',('xi_um_fit_at_center',r"$\xi_c$ / um (fit)"))],
    description="fitting variant:",
    disabled=False,
)


do_plot_CDCs_widget = widgets.Checkbox(value=False, description="do plot CDCs")
do_plot_xi_um_fit_vs_I_Airy2_fit_widget = widgets.Checkbox(value=False, description="do plot xi_um_fit vs I_Airy2_fit")

# define what should happen when the hdf5 file widget is changed:


# function using the widgets:


def plotprofile(
    plotprofile_active,
    do_deconvmethod,
    xi_um_guess,
    scan_x,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    hdf5_file_path,
    imageid,
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

    if plotprofile_active == True:  # workaround, so that the function is not executed while several inputs are changed
        # fittingprogress_widget.bar_style = 'info'
        # fittingprogress_widget.value = 0
        # statustext_widget.value = 'fitting ...'
        # textarea_widget.value = ''

        fit_profile_text_widget.value = ''
        deconvmethod_text_widget.value = ''

        # Loading and preparing

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][0]

        pinholes = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes"].iloc[0]
        separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
        orientation = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["orientation"].iloc[0]
        setting_wavelength_nm = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[0]
        pinholes_bg_avg_sx_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sx_um"].iloc[0]
        pinholes_bg_avg_sy_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sy_um"].iloc[0]
        pixis_avg_width = 200  # read from df0 instead!

        # fittingprogress_widget.value = 2
        #     hdf5_file_name_image = hdf5_file_name_image_widget.value
        #     dataset_image_args = dataset_image_args_widget.value
        fit_profile_text_widget.value = 'calculating ...'
        


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

        n = pixis_profile_avg.size  # number of sampling point  # number of pixels
        dX_1 = 13e-6
        xdata = np.linspace((-n / 2) * dX_1, (+n / 2 - 1) * dX_1, n)
        # ydata = pixis_profile_avg_dataset[imageid]*datafactor
        ydata = pixis_profile_avg  # defined in the cells above, still to implement: select

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # Fitting

        result = fit_profile(
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

        d_um_at_detector = x2_um_fit - x1_um_fit

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm_fit * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # lmfit throws RuntimeWarnings, maybe its a bug. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit, xi_um_fit_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit], [0, 0], 470, False)
                
        fit_profile_text_widget.value = r"%.2fum" % (xi_um_fit)

        if save_to_df == True:
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'gamma_fit'] = gamma_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um_fit'] = xi_um_fit  # add this first to the df_fits dataframe
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'wavelength_nm_fit'] = wavelength_nm_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'd_um_at_detector'] = d_um_at_detector
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy1_fit'] = I_Airy1_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy2_fit'] = I_Airy2_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w1_um_fit'] = w1_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w2_um_fit'] = w2_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'shiftx_um_fit'] = shiftx_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'x1_um_fit'] = x1_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'x2_um_fit'] = x2_um_fit

        if do_deconvmethod == True:
            deconvmethod_text_widget.value = 'calculating ...'
            partiallycoherent = pixis_image_norm
            z = 5781 * 1e-3
            dX_1 = 13 * 1e-6
            profilewidth = 200  # pixis_avg_width  # defined where?
            pixis_centery_px = int(pixis_centery_px)
            wavelength = setting_wavelength_nm * 1e-9
            # xi_um_guess = 475
            # guess sigma_y_F_gamma_um based on the xi_um_guess assuming to be the beams intensity rms width
            sigma_y_F_gamma_um_guess = calc_sigma_F_gamma_um(xi_um_guess, n, dX_1, setting_wavelength_nm, False)
            create_figure = True

            # Ignoring OptimizeWarning. Supressing warning as described in https://stackoverflow.com/a/14463362:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
                    create_figure,
                )
            deconvmethod_text_widget.value = r"%.2fum" % (xi_x_um) + r", %.2fum" % (xi_y_um)
            # str(round(xi_x_um, 2)) + ', ' + str(round(xi_y_um, 2))

            if save_to_df == True:
                df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_x_um'] = xi_x_um
                df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_y_um'] = xi_y_um

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
                ((-n / 2) * dX_1 * 1e3, (int(round(pixis_centery_px)) - n / 2 - pixis_avg_width / 2) * dX_1 * 1e3),
                n * dX_1 * 1e3,
                pixis_avg_width * dX_1 * 1e3,
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
        ax00.plot(xdata * 1e3, result.best_fit, color="b", linewidth=0.5, label="fit")

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
                r"$\gamma=%.2f$" % (result.params["gamma"].value,),
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

def plot_fitting(
    plotprofile_active,
    pixis_profile_avg_width,
    crop_px,
    hdf5_file_path,
    imageid,
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

    if plotprofile_active == True:  # workaround, so that the function is not executed while several inputs are changed
        # fittingprogress_widget.bar_style = 'info'
        # fittingprogress_widget.value = 0
        # statustext_widget.value = 'fitting ...'
        # textarea_widget.value = ''

        fit_profile_text_widget.value = ''
        deconvmethod_text_widget.value = ''

        # Loading and preparing

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][0]

        pinholes = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes"].iloc[0]
        separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
        orientation = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["orientation"].iloc[0]
        setting_wavelength_nm = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[0]
        pinholes_bg_avg_sx_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sx_um"].iloc[0]
        pinholes_bg_avg_sy_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sy_um"].iloc[0]
        # pixis_profile_avg_width = 200  # read from df0 instead!

        # fittingprogress_widget.value = 2
        #     hdf5_file_name_image = hdf5_file_name_image_widget.value
        #     dataset_image_args = dataset_image_args_widget.value
        fit_profile_text_widget.value = 'calculating ...'
        


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

        result = fit_profile(
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

        d_um_at_detector = x2_um_fit - x1_um_fit

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm_fit * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # lmfit throws RuntimeWarnings, maybe its a bug. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit, xi_um_fit_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit], [0, 0], 470, False)
                
        fit_profile_text_widget.value = r"%.2fum" % (xi_um_fit)

        if save_to_df == True:
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'gamma_fit'] = gamma_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um_fit'] = xi_um_fit  # add this first to the df_fits dataframe
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'wavelength_nm_fit'] = wavelength_nm_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'd_um_at_detector'] = d_um_at_detector
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy1_fit'] = I_Airy1_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy2_fit'] = I_Airy2_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w1_um_fit'] = w1_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w2_um_fit'] = w2_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'shiftx_um_fit'] = shiftx_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'x1_um_fit'] = x1_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'x2_um_fit'] = x2_um_fit

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
                r"$\gamma=%.2f$" % (result.params["gamma"].value,),
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
    plotprofile_active,
    pixis_profile_avg_width,
    crop_px,
    hdf5_file_path,
    imageid,
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

    if plotprofile_active == True:  # workaround, so that the function is not executed while several inputs are changed
        # fittingprogress_widget.bar_style = 'info'
        # fittingprogress_widget.value = 0
        # statustext_widget.value = 'fitting ...'
        # textarea_widget.value = ''

        fit_profile_text_widget.value = ''
        xi_um_fit_at_center_text_widget.value = ''
        deconvmethod_text_widget.value = ''

        # Loading and preparing

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][0]

        pinholes = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes"].iloc[0]
        separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
        orientation = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["orientation"].iloc[0]
        setting_wavelength_nm = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[0]
        pinholes_bg_avg_sx_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sx_um"].iloc[0]
        pinholes_bg_avg_sy_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sy_um"].iloc[0]
        # pixis_profile_avg_width = 200  # read from df0 instead!

        # fittingprogress_widget.value = 2
        #     hdf5_file_name_image = hdf5_file_name_image_widget.value
        #     dataset_image_args = dataset_image_args_widget.value
        fit_profile_text_widget.value = 'calculating ...'
        xi_um_fit_at_center_text_widget.value = 'calculating ...'
        


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
        gamma_fit_at_center = gaussian(0,1,mod_shiftx_um_fit,mod_sigma_um_fit)*gamma_fit

        d_um_at_detector = x2_um_fit - x1_um_fit

        fringeseparation_um = z_mm * 1e-3 * wavelength_nm_fit * 1e-9 / (d_um * 1e-6) * 1e6
        fringeseparation_px = fringeseparation_um / 13

        # lmfit throws RuntimeWarnings, maybe its a bug. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit, xi_um_fit_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit], [0, 0], 470, False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (xi_um_fit_at_center, xi_um_fit_at_center_stderr) = find_sigma([0.0, d_um], [1.0, gamma_fit_at_center], [0, 0], 470, False)
                
        fit_profile_text_widget.value = r"%.2fum" % (xi_um_fit)
        xi_um_fit_at_center_text_widget.value = r"%.2fum" % (xi_um_fit_at_center)

        if save_to_df == True:
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'gamma_fit'] = gamma_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'gamma_fit_at_center'] = gamma_fit_at_center
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um_fit'] = xi_um_fit  # add this first to the df_fits dataframe
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um_fit_at_center'] = xi_um_fit_at_center  # add this first to the df_fits dataframe
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'wavelength_nm_fit'] = wavelength_nm_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'd_um_at_detector'] = d_um_at_detector
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy1_fit'] = I_Airy1_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy2_fit'] = I_Airy2_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w1_um_fit'] = w1_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w2_um_fit'] = w2_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'shiftx_um_fit'] = shiftx_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'x1_um_fit'] = x1_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'mod_sigma_um_fit'] = mod_sigma_um_fit
            df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'mod_shiftx_um_fit'] = mod_shiftx_um_fit

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
                r"$\gamma=%.2f$" % (gamma_fit_at_center,),
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
                r"$\gamma_c=%.2f$" % (gamma_fit_at_center,),
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





def plot_deconvmethod(
    do_deconvmethod,
    pixis_profile_avg_width,
    xi_um_guess,
    scan_x,
    sigma_x_F_gamma_um_multiplier,
    crop_px,
    hdf5_file_path,
    imageid,
    save_to_df    
):

    if do_deconvmethod == True:

        if scan_x == True:
            deconvmethod_text_widget.value = ''
        else:
            deconvmethod_simple_text_widget.value = ''

        # Loading and preparing

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ]
            # pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
            #     np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            # ]
            timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][2]
            pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
            ][0]

        pinholes = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes"].iloc[0]
        separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
        orientation = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["orientation"].iloc[0]
        setting_wavelength_nm = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[0]
        pinholes_bg_avg_sx_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sx_um"].iloc[0]
        pinholes_bg_avg_sy_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sy_um"].iloc[0]
        # pixis_avg_width = 200  # read from df0 instead!

        pixis_profile_avg = pixis_image_norm[int(pixis_centery_px-pixis_profile_avg_width/2):int(pixis_centery_px+pixis_profile_avg_width/2),:]

        if scan_x == True:
            deconvmethod_text_widget.value = 'calculating ...'
        else:
            deconvmethod_simple_text_widget.value = 'calculating ...'
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
        create_figure = False

        # Ignoring OptimizeWarning. Supressing warning as described in https://stackoverflow.com/a/14463362:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                create_figure,
            )
        if scan_x == True:
            deconvmethod_text_widget.value = r"%.2fum" % (xi_x_um) + r", %.2fum" % (xi_y_um)
        else:
            deconvmethod_simple_text_widget.value = r"%.2fum" % (xi_x_um)
        # str(round(xi_x_um, 2)) + ', ' + str(round(xi_y_um, 2))

        if save_to_df == True:
            if scan_x == True:
                df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_x_um'] = xi_x_um
                df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_y_um'] = xi_y_um
            else:
                df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um'] = xi_x_um



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

def plot_fitting_vs_deconvolution(
    do_plot_fitting_vs_deconvolution,
    dataset,
    measurement_file,
    imageid,
    xi_um_deconv_column_and_label,
    xi_um_fit_column_and_label
):

    if do_plot_fitting_vs_deconvolution == True:

        xi_um_deconv_column = xi_um_deconv_column_and_label[0]
        xi_um_deconv_label = xi_um_deconv_column_and_label[1]
        xi_um_fit_column = xi_um_fit_column_and_label[0]
        xi_um_fit_label = xi_um_fit_column_and_label[1]

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
        timestamp_pulse_ids = []
        for f in files:
            with h5py.File(f, "r") as hdf5_file:
                timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

        # create plot for the determined timestamps:
        # plt.scatter(df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_x_um'], df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_um_fit'], cmap=df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['separation_um'])
        plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)][xi_um_deconv_column] , \
            df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)][xi_um_fit_column], \
                c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'],\
                    marker='x', s=2)

        plt.colorbar()


        timestamp_pulse_ids = []
        with h5py.File(measurement_file, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

        plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["imageid"] == int(imageid)) & (df0["xi_um_fit"]<2000)][xi_um_deconv_column] , \
            df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["imageid"] == int(imageid)) & (df0["xi_um_fit"]<2000)][xi_um_fit_column], \
                c='red',\
                    marker='x', s=10)


        x = np.linspace(0,2000)
        plt.plot(x,x, c='grey')
        

        plt.xlim(0,2000)
        plt.ylim(0,2000)
        plt.xlabel(xi_um_deconv_label)
        plt.ylabel(xi_um_fit_label)
        plt.gca().set_aspect('equal')
        



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
            timestamp_pulse_ids_dataset=[]

            ax = plt.subplot(gs[i,j])
        
            # get all the files in a dataset:
            files = []
            # for set in [list(datasets)[0]]:
            
            for measurement in datasets_selection[dataset]:
                # print(measurement)
                files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

            # get all the timestamps in these files:        
            # datasets[list(datasets)[0]][0]
            
            
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
                x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
                y = [gaussian(x=x, amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["separation_um"]==x)][xi_um_deconv_column].max()) for x in x]
                ax.scatter(x, y, marker='v', s=20, color='darkgreen', facecolors='none', label='maximum')
                
                # Fitting (red)
                x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
                y = [df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["separation_um"]==x)][gamma_fit_column].max() for x in x]
                ax.scatter(x, y, marker='v', s=20, color='darkred', facecolors='none', label='maximum')
                
            x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))]['separation_um'].unique()
            y = [gaussian(x=x, amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset)) & (df0["separation_um"]==x)][xi_um_deconv_column].max()) for x in x]
        
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
            y = [df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset)) & (df0["separation_um"]==x)][gamma_fit_column].max() for x in x]
        
            xx = np.arange(0.0, 2000, 10)
            gamma_fit_max = y
            d_gamma = x
                
            (xi_x_um_max_sigma, xi_x_um_max_sigma_std) = find_sigma(d_gamma,gamma_fit_max,0, 400, False)

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
            # plt.scatter(df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_x_um'], df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_um_fit'], cmap=df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['separation_um'])
            plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['I_Airy2_fit'] , \
                df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)][xi_um_fit_column], \
                    c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'],\
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

column0 = widgets.VBox(
    [
        plotprofile_active_widget,
        do_deconvmethod_widget,
        do_plot_fitting_vs_deconvolution_widget,
        do_plot_CDCs_widget,
        do_plot_xi_um_fit_vs_I_Airy2_fit_widget,
        pixis_profile_avg_width_widget,
        xi_um_guess_widget,
        scan_x_widget,
        sigma_x_F_gamma_um_multiplier_widget,
        crop_px_widget,
        imageid_profile_fit_widget,
        savefigure_profile_fit_widget,
        save_to_df_widget,
        df_fits_csv_save_widget,
        do_textbox_widget,
    ]
)

column1 = widgets.VBox(
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

column2 = widgets.VBox(
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


column3 = widgets.VBox(
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

column4 = widgets.VBox(
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

column5 = widgets.VBox([textarea_widget, beamsize_text_widget, fit_profile_text_widget, xi_um_fit_at_center_text_widget, deconvmethod_simple_text_widget, deconvmethod_text_widget])

plotprofile_interactive_input = widgets.HBox([column0, column1, column2, column3, column4, column5])

# plotprofile_interactive_output = interactive_output(
#     plotprofile,
#     {
#         "plotprofile_active": plotprofile_active_widget,
#         "do_deconvmethod": do_deconvmethod_widget,
#         "scan_x" : scan_x_widget,
#         "xi_um_guess" : xi_um_guess_widget,
#         "sigma_x_F_gamma_um_multiplier" : sigma_x_F_gamma_um_multiplier_widget,
#         "crop_px" : crop_px_widget,
#         "hdf5_file_path": dph_settings_bgsubtracted_widget,
#         "imageid": imageid_profile_fit_widget,
#         "savefigure": savefigure_profile_fit_widget,
#         "save_to_df": save_to_df_widget,
#         "do_textbox": do_textbox_widget,
#         "shiftx_um": shiftx_um_widget,
#         "shiftx_um_range": shiftx_um_range_widget,
#         "shiftx_um_do_fit": shiftx_um_do_fit_widget,
#         "wavelength_nm": wavelength_nm_widget,
#         "wavelength_nm_range": wavelength_nm_range_widget,
#         "wavelength_nm_do_fit": wavelength_nm_do_fit_widget,
#         "z_mm": z_mm_widget,
#         "z_mm_range": z_mm_range_widget,
#         "z_mm_do_fit": z_mm_do_fit_widget,
#         "d_um": d_um_widget,
#         "d_um_range": d_um_range_widget,
#         "d_um_do_fit": d_um_do_fit_widget,
#         "gamma": gamma_widget,
#         "gamma_range": gamma_range_widget,
#         "gamma_do_fit": gamma_do_fit_widget,
#         "w1_um": w1_um_widget,
#         "w1_um_range": w1_um_range_widget,
#         "w1_um_do_fit": w1_um_do_fit_widget,
#         "w2_um": w2_um_widget,
#         "w2_um_range": w2_um_range_widget,
#         "w2_um_do_fit": w2_um_do_fit_widget,
#         "I_Airy1": I_Airy1_widget,
#         "I_Airy1_range": I_Airy1_range_widget,
#         "I_Airy1_do_fit": I_Airy1_do_fit_widget,
#         "I_Airy2": I_Airy2_widget,
#         "I_Airy2_range": I_Airy2_range_widget,
#         "I_Airy2_do_fit": I_Airy2_do_fit_widget,
#         "x1_um": x1_um_widget,
#         "x1_um_range": x1_um_range_widget,
#         "x1_um_do_fit": x1_um_do_fit_widget,
#         "x2_um": x2_um_widget,
#         "x2_um_range": x2_um_range_widget,
#         "x2_um_do_fit": x2_um_do_fit_widget,
#         "normfactor": normfactor_widget,
#         "normfactor_range": normfactor_range_widget,
#         "normfactor_do_fit": normfactor_do_fit_widget,
#     },
# )

# plot_fitting_interactive_output = interactive_output(
#     plot_fitting,
#     {
#         "plotprofile_active": plotprofile_active_widget,
#         "pixis_profile_avg_width" : pixis_profile_avg_width_widget,
#         "crop_px" : crop_px_widget,
#         "hdf5_file_path": dph_settings_bgsubtracted_widget,
#         "imageid": imageid_profile_fit_widget,
#         "savefigure": savefigure_profile_fit_widget,
#         "save_to_df": save_to_df_widget,
#         "do_textbox": do_textbox_widget,
#         "shiftx_um": shiftx_um_widget,
#         "shiftx_um_range": shiftx_um_range_widget,
#         "shiftx_um_do_fit": shiftx_um_do_fit_widget,
#         "wavelength_nm": wavelength_nm_widget,
#         "wavelength_nm_range": wavelength_nm_range_widget,
#         "wavelength_nm_do_fit": wavelength_nm_do_fit_widget,
#         "z_mm": z_mm_widget,
#         "z_mm_range": z_mm_range_widget,
#         "z_mm_do_fit": z_mm_do_fit_widget,
#         "d_um": d_um_widget,
#         "d_um_range": d_um_range_widget,
#         "d_um_do_fit": d_um_do_fit_widget,
#         "gamma": gamma_widget,
#         "gamma_range": gamma_range_widget,
#         "gamma_do_fit": gamma_do_fit_widget,
#         "w1_um": w1_um_widget,
#         "w1_um_range": w1_um_range_widget,
#         "w1_um_do_fit": w1_um_do_fit_widget,
#         "w2_um": w2_um_widget,
#         "w2_um_range": w2_um_range_widget,
#         "w2_um_do_fit": w2_um_do_fit_widget,
#         "I_Airy1": I_Airy1_widget,
#         "I_Airy1_range": I_Airy1_range_widget,
#         "I_Airy1_do_fit": I_Airy1_do_fit_widget,
#         "I_Airy2": I_Airy2_widget,
#         "I_Airy2_range": I_Airy2_range_widget,
#         "I_Airy2_do_fit": I_Airy2_do_fit_widget,
#         "x1_um": x1_um_widget,
#         "x1_um_range": x1_um_range_widget,
#         "x1_um_do_fit": x1_um_do_fit_widget,
#         "x2_um": x2_um_widget,
#         "x2_um_range": x2_um_range_widget,
#         "x2_um_do_fit": x2_um_do_fit_widget,
#         "normfactor": normfactor_widget,
#         "normfactor_range": normfactor_range_widget,
#         "normfactor_do_fit": normfactor_do_fit_widget,
#     },
# )

plot_fitting_v2_interactive_output = interactive_output(
    plot_fitting_v2,
    {
        "plotprofile_active": plotprofile_active_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_widget,
        "crop_px" : crop_px_widget,
        "hdf5_file_path": dph_settings_bgsubtracted_widget,
        "imageid": imageid_profile_fit_widget,
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

plot_deconvmethod_interactive_output = interactive_output(
    plot_deconvmethod,
    {
        "do_deconvmethod": do_deconvmethod_widget,
        "pixis_profile_avg_width" : pixis_profile_avg_width_widget,
        "xi_um_guess" : xi_um_guess_widget,
        "scan_x" : scan_x_widget,
        "sigma_x_F_gamma_um_multiplier" : sigma_x_F_gamma_um_multiplier_widget,
        "crop_px" : crop_px_widget,
        "hdf5_file_path": dph_settings_bgsubtracted_widget,
        "imageid": imageid_profile_fit_widget,
        "save_to_df": save_to_df_widget,
    },
)


plot_fitting_vs_deconvolution_output = interactive_output(
    plot_fitting_vs_deconvolution,
    {
        "do_plot_fitting_vs_deconvolution": do_plot_fitting_vs_deconvolution_widget,
        "dataset" : datasets_widget,
        "measurement_file" : dph_settings_bgsubtracted_widget,
        "imageid": imageid_profile_fit_widget,
        "xi_um_deconv_column_and_label" : xi_um_deconv_column_and_label_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget
    },
)

plot_CDCs_output = interactive_output(
    plot_CDCs,
    {
        "do_plot_CDCs": do_plot_CDCs_widget,
        "xi_um_deconv_column_and_label" : xi_um_deconv_column_and_label_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget
    },
)

plot_xi_um_fit_vs_I_Airy2_fit_output = interactive_output(
    plot_xi_um_fit_vs_I_Airy2_fit,
    {
        "do_plot_xi_um_fit_vs_I_Airy2_fit": do_plot_xi_um_fit_vs_I_Airy2_fit_widget,
        "xi_um_fit_column_and_label" : xi_um_fit_column_and_label_widget
    },
)


def dph_settings_bgsubtracted_widget_changed(change):
    statustext_widget.value = "updating widgets ..."
    # plotprofile_interactive_output.clear_output()
    fittingprogress_widget.value = 0
    plotprofile_active_widget.value = False
    statustext_widget.value = "plotprofile_active_widget.value = False"
    imageid_profile_fit_widget.disabled = True
    imageid_profile_fit_widget.options = None
    with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
        imageids = hdf5_file["/bgsubtracted/imageid"][:]
        imageid_profile_fit_widget.options = imageids
        imageid_profile_fit_widget.disabled = False
        imageid = imageid_profile_fit_widget.value
        timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
        ][2]
        pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
        ][
            0
        ]  # needed for what?
        setting_wavelength_nm = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[0]
        pinholes_bg_avg_sx_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sx_um"].iloc[0]
        pinholes_bg_avg_sy_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes_bg_avg_sy_um"].iloc[0]
        ph = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes"].iloc[0]
        separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
        orientation = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["orientation"].iloc[0]

        pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
                np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
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
        

    wavelength_nm_widget.value = setting_wavelength_nm
    wavelength_nm_range_widget.value = value = [wavelength_nm_widget.value - 0.1, wavelength_nm_widget.value + 0.1]
    d_um_widget.value = separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
    x1_um_widget.value = -d_um_widget.value * 10 / 2
    x2_um_widget.value = d_um_widget.value * 10 / 2
    x1_um_range_widget.value = [-d_um_widget.value * 10 / 2 - 1000, 0]
    x2_um_range_widget.value = [0, d_um_widget.value * 10 / 2 + 1000]
    statustext_widget.value = "widgets updated"
    if orientation == "horizontal":
        beamsize_text_widget.value = r"%.2fum" % (pinholes_bg_avg_sx_um,)
    if orientation == "vertical":
        beamsize_text_widget.value = r"%.2fum" % (pinholes_bg_avg_sy_um,)


dph_settings_bgsubtracted_widget.observe(dph_settings_bgsubtracted_widget_changed, names="label")


def datasets_widget_changed(change):
    dph_settings_bgsubtracted = []
    for pattern in ['*'+ s + '.h5' for s in datasets[datasets_widget.value]]: 
        dph_settings_bgsubtracted.extend(bgsubtracted_dir.glob(pattern))
    dph_settings_bgsubtracted_widget.options=dph_settings_bgsubtracted
    measurements_selection_widget.options = dph_settings_bgsubtracted
    measurements_selection_files = []
    for pattern in ['*'+ s + '.h5' for s in datasets_selection[datasets_widget.value]]: 
        measurements_selection_files.extend(bgsubtracted_dir.glob(pattern))
    measurements_selection_widget.value = measurements_selection_files
datasets_widget.observe(datasets_widget_changed, names="value")


def measurements_selection_widget_changed(change):
    if len(measurements_selection_widget.value) > 0: # avoid the empty array that is generated during datasets_widget_changed
        measurements_selection = []
        for f in measurements_selection_widget.value:
            measurements_selection.append(f.stem)
        datasets_selection.update({ datasets_widget.value : measurements_selection })
    datasets_selection_py_file = str(Path.joinpath(data_dir, "datasets_selection.py"))
    with open(datasets_selection_py_file, 'w') as f:
        print(datasets_selection, file=f)
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



def update_df_fits_csv_save_widget(change):
    if df_fits_csv_save_widget.value == True:
        # save fits to csv
        df_fits = df0[['timestamp_pulse_id'] + fits_header_list]
        save_df_fits = True
        if save_df_fits == True:
            # df_fits.to_csv(Path.joinpath(data_dir,str('df_fits_'+datetime.now()+'.csv')))
            df_fits.to_csv(Path.joinpath(data_dir,str('df_fits_test.csv')))
        df_fits_csv_save_widget.value = False

df_fits_csv_save_widget.observe(update_df_fits_csv_save_widget, names='value')
# not working, why? impement also which file to load/import, which ones to export/save, ...



# getting all parameters from the file


with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
    imageids = hdf5_file["/bgsubtracted/imageid"][:]

    imageid = imageids[0]

    hdf5_file_path = dph_settings_bgsubtracted_widget.value
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        pixis_image_norm = hdf5_file["/bgsubtracted/pixis_image_norm"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
        ]
        pixis_profile_avg = hdf5_file["/bgsubtracted/pixis_profile_avg"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
        ]
        timestamp_pulse_id = hdf5_file["Timing/time stamp/fl2user1"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
        ][2]
        pixis_centery_px = hdf5_file["/bgsubtracted/pixis_centery_px"][
            np.where(hdf5_file["/bgsubtracted/imageid"][:] == imageid)[0][0]
        ][0]

    pinholes = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["pinholes"].iloc[0]
    separation_um = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["separation_um"].iloc[0]
    orientation = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["orientation"].iloc[0]
    setting_wavelength_nm = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[0]
    energy_hall_uJ = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["energy hall"].iloc[0]
    _lambda_nm_fit = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["setting_wavelength_nm"].iloc[
        0
    ]  # is this stored in df0? get it from profile_fitting?

    hdf5_file_name_image = df0[df0["timestamp_pulse_id"] == timestamp_pulse_id]["hdf5_file_name"].iloc[0]

    beamposition_horizontal_interval = 1000  # random number, store in hdf5?


# Increase output of Jupyer Notebook Cell:
from IPython.display import Javascript

display(
    Javascript("""google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})""")
)  # https://stackoverflow.com/a/57346765

children_left = [plot_fitting_v2_interactive_output, 
plot_deconvmethod_interactive_output, 
plot_CDCs_output, 
plot_xi_um_fit_vs_I_Airy2_fit_output]
tabs_left = widgets.Tab()
tabs_left.children = children_left
tabs_left.set_title(0, 'Fitting')
tabs_left.set_title(1, 'Deconvolution')
tabs_left.set_title(2, 'CDCs')
tabs_left.set_title(3, 'plot_xi_um_fit_vs_I_Airy2_fit')

children_right = [VBox([xi_um_deconv_column_and_label_widget, xi_um_fit_column_and_label_widget, plot_fitting_vs_deconvolution_output])]
tabs_right = widgets.Tab()
tabs_right.children = children_right
tabs_right.set_title(0, 'Fitting vs. Deconvolution')

grid = widgets.GridspecLayout(1, 3, height='1000px')
grid[0, :2] = tabs_left
grid[0, 2] = tabs_right


# Display widgets and outputs
display(
    VBox(
        [
            HBox([fittingprogress_widget, statustext_widget]),
            datasets_widget,
            dph_settings_bgsubtracted_widget,
            measurements_selection_widget,
            plotprofile_interactive_input,
            grid
        ]
    )
)
dph_settings_bgsubtracted_widget_changed(None)

 

# How to get only the timestamp_pulse_id of the datasets?

# name of dph_settings in datasets accessible by index:
# print(datasets[list(datasets)[0]][0])

# timestamp_pulse_ids = []
# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:
#     timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

# timestamp_pulse_ids
# <codecell>
# loop over all datasets and create coherence plots:
for dataset in list(datasets):
    print(dataset)

    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    plt.scatter(df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['separation_um'], df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['gamma_fit'],color='blue')
    plt.xlim(0,2000)
    plt.ylim(0,1)
    plt.show()

# <codecell>
# loop over all datasets and create coherence plots:
for dataset in list(datasets):
    print(dataset)

    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    plt.scatter(df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['separation_um'], df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_x_um'],color='blue')
    plt.xlim(0,2000)
    
    plt.show()


# <codecell>
# loop over all datasets and delete all fits and deconvolution results:
remove_fits_from_df = True
if remove_fits_from_df == True:
    for dataset in list(datasets):
        print(dataset)

        # get all the files in a dataset:
        files = []
        # for set in [list(datasets)[0]]:
        
        for measurement in datasets[dataset]:
            # print(measurement)
            files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

        # get all the timestamps in these files:        
        # datasets[list(datasets)[0]][0]
        timestamp_pulse_ids = []
        for f in files:
            with h5py.File(f, "r") as hdf5_file:
                timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'gamma_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'gamma_fit_at_center'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'xi_um_fit_at_center'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'wavelength_nm_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'd_um_at_detector'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy1_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'I_Airy2_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w1_um_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'w2_um_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'shiftx_um_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'x1_um_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'mod_sigma_um_fit'] =  np.nan
        df0.loc[(df0['timestamp_pulse_id'] == timestamp_pulse_id), 'mod_shiftx_um_fit'] =  np.nan

        

# <codecell>
# iterate over all images

# delete all fits, create new df columns for new fits, etc.

for imageid in imageid_profile_fit_widget.options:
    imageid_profile_fit_widget.value = imageid



# <codecell>
# iterate over all images in a given measurement
start = datetime.now()
plotprofile_active_widget.value = True
for imageid in imageid_profile_fit_widget.options:
    imageid_profile_fit_widget.value = imageid
end = datetime.now()
time_taken = end - start
print(time_taken)


# <codecell>
# # iterate over all measurements and images in a given dataset
for measurement in dph_settings_bgsubtracted_widget.options:
    dph_settings_bgsubtracted_widget.value = measurement
    plotprofile_active_widget.value = True
    for imageid in imageid_profile_fit_widget.options:
        imageid_profile_fit_widget.value = imageid


# <codecell>
# iterate over all datasets
for dataset in list(datasets):
    datasets_widget.value = dataset
    plotprofile_active_widget.value = True
print('done')


# <codecell>
# iterate over everything
start = datetime.now()
for dataset in list(datasets):
    print(dataset)
    datasets_widget.value = dataset
    plotprofile_active_widget.value = True
    for measurement in dph_settings_bgsubtracted_widget.options:
        print(measurement)
        dph_settings_bgsubtracted_widget.value = measurement
        plotprofile_active_widget.value = True
        start_measurement = datetime.now()
        for imageid in imageid_profile_fit_widget.options:
            imageid_profile_fit_widget.value = imageid
        end_measurement = datetime.now()
        time_taken = end_measurement - start_measurement
        print(time_taken)
end = datetime.now()
time_taken = end - start
print(time_taken)





# %%

# <codecell>
# create plots showing those measurements where the deconvolution algorithm did not cross zero
fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)



i=0
j=0
for dataset in list(datasets):
   

    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    
    ax = plt.subplot(gs[i,j])

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0['xi_x_um'].isin([np.nan]))]['separation_um'], df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0['xi_x_um'].isin([np.nan]))]['gamma_fit'],color='blue')
    plt.xlim(0,2000)
    plt.ylim(0,1)
    plt.show()
    plt.title(dataset)

    i=+1
    if j==1:
        j=j+1
    else:
        j=0


# %%


# <codecell>
# Coherence length from Deconvolution (x-axis) against from Fitting (y-axis)

fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)



i=0
j=0
for dataset in list(datasets):
 
    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    
    ax = plt.subplot(gs[i,j])

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['xi_x_um'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['xi_um_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'], \
                marker='x', s=2)
    ax.set_xlim(0,2000)
    ax.set_ylim(0,2000)

    ax.set_title(dataset)
    
    
    if j==0:
        j+=1
    else:
        j=0
        i=i+1



# <codecell>
# create plots fitting vs deconvolution
fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)

i=0
j=0
for dataset in list(datasets):

    ax = plt.subplot(gs[i,j])

    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    # plt.scatter(df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_x_um'], df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_um_fit'], cmap=df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['separation_um'])
    plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['xi_x_um'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['xi_um_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'],\
                marker='x', s=2)
    plt.xlabel(r"$\xi$ (fits)")
    plt.ylabel(r"$\xi$ (deconv)")
    plt.gca().set_aspect('equal')
    plt.colorbar()

    # plt.xlim(0,2000)
    # plt.ylim(0,2000)
    
    plt.title(dataset)

    if j==0:
        j+=1
    else:
        j=0
        i=i+1



# <codecell>
# CDC from Fitting

fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)



i=0
j=0
for dataset in list(datasets):
 
    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    
    ax = plt.subplot(gs[i,j])

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['gamma_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['I_Airy2_fit'])
    ax.set_xlim(0,2000)
    ax.set_ylim(0,1)
    
    ax.set_title(dataset)
    
    
    if j==0:
        j+=1
    else:
        j=0
        i=i+1




# <codecell>
# CDC from Deconvolution

fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)



i=0
j=0
for dataset in list(datasets):
 
    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    
    ax = plt.subplot(gs[i,j])

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'] , \
        gaussian(x=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'], \
             amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_x_um']), \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['I_Airy2_fit'])
    # after next run make another one for xi_um
    ax.set_xlim(0,2000)
    ax.set_ylim(0,1)
    
    ax.set_title(dataset)
    
    
    if j==0:
        j+=1
    else:
        j=0
        i=i+1

# gaussian(x=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'], amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['xi_um_fit'])
# %%

# <codecell>
# CDC from Deconvolution (green) and Fitting (red)


fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)

i=0
j=0

for dataset in list(datasets):
    timestamp_pulse_ids_dataset=[]

    ax = plt.subplot(gs[i,j])
 
    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    
    
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
        x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
        y = [gaussian(x=x, amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["separation_um"]==x)]['xi_x_um'].max()) for x in x]
        ax.scatter(x, y, marker='v', s=20, color='darkgreen', facecolors='none', label='maximum')
        
        # Fitting (red)
        x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
        y = [df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["separation_um"]==x)]['gamma_fit'].max() for x in x]
        ax.scatter(x, y, marker='v', s=20, color='darkred', facecolors='none', label='maximum')
        
    x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset))]['separation_um'].unique()
    y = [gaussian(x=x, amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset)) & (df0["separation_um"]==x)]['xi_x_um'].max()) for x in x]
   
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
    y = [df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids_dataset)) & (df0["separation_um"]==x)]['gamma_fit'].max() for x in x]
   
    xx = np.arange(0.0, 2000, 10)
    gamma_fit_max = y
    d_gamma = x
        
    (xi_x_um_max_sigma, xi_x_um_max_sigma_std) = find_sigma(d_gamma,gamma_fit_max,0, 400, False)

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
# %%

# measurement = '2017-11-27T1529 8.0nm 45uJ 12Und. KOAS=PMMA 1047um (3d) ap5=7.0 ap7=50.0 (bg3cd) ap5=7.0 ap7=50.0'
measurement = '2017-11-27T1529 8.0nm 45uJ 12Und. KOAS=PMMA 0707um (1d) ap5=7.0 ap7=50.0 (bg1cd) ap5=7.0 ap7=50.0'

# get all the files in a dataset:
files = []
files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

for f in files:
        timestamp_pulse_ids = []
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
            timestamp_pulse_ids_dataset.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'].unique()
y = [df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["separation_um"]==x)]['gamma_fit'].max() for x in x]

print(x)
print(y)
# %%

# %% find the imageids for the highest and lowest gamma_fit (those images where the fitting seemed to have issues)

# measurement = '2017-11-27T1529 8.0nm 45uJ 12Und. KOAS=PMMA 1047um (3d) ap5=7.0 ap7=50.0 (bg3cd) ap5=7.0 ap7=50.0'
# measurement = '2017-11-27T1529 8.0nm 45uJ 12Und. KOAS=PMMA 0707um (1d) ap5=7.0 ap7=50.0 (bg1cd) ap5=7.0 ap7=50.0'
measurement = '2017-11-26T2300 18.0nm 70uJ 7Und. KOAS=1.5mm 1047um (3b) ap5=7.0 ap7=1.5 (bg3ab) ap5=7.0 ap7=1.5'

# get all the files in a dataset:
files = []
files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

for f in files:
        timestamp_pulse_ids = []
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])
            

x = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um']
y = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['gamma_fit']
plt.scatter(x,y)

gamma_fit_max = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['gamma_fit'].max()
gamma_fit_min = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['gamma_fit'].min()
print(gamma_fit_max)

imageid_max = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0['gamma_fit']==gamma_fit_max)][['imageid', 'separation_um', 'gamma_fit', 'xi_x_um']]
imageid_min = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0['gamma_fit']==gamma_fit_min)][['imageid', 'separation_um', 'gamma_fit', 'xi_x_um']]
# df0.loc[(df0['timestamp_pulse_id'].isin(timestamp_pulse_ids)), 'gamma_fit'] = np.nan
# imageid_max = df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["gamma_fit"]==1.0)][['imageid', 'separation_um', 'gamma_fit']]

print(imageid_max)
print(imageid_min)



# %% Remove gamma_fit in the dataframe for certain imageids



measurement = '2017-11-26T2300 18.0nm 70uJ 7Und. KOAS=1.5mm 1047um (3b) ap5=7.0 ap7=1.5 (bg3ab) ap5=7.0 ap7=1.5'

# get all the files in a dataset:
files = []
files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

for f in files:
        timestamp_pulse_ids = []
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

imageids_to_remove = [61]

df0.loc[(df0['imageid'].isin(imageids_to_remove)), 'gamma_fit'] = np.nan


# %% save df_fits

df_fits = df0[['timestamp_pulse_id'] + fits_header_list]
save_df_fits = True
if save_df_fits == True:
    df_fits.to_csv(Path.joinpath(data_dir,str('df_fits_'+datetime.now().strftime("%Y-%m-%d--%Hh%M")+'.csv')))
    # df_fits.to_csv(Path.joinpath(data_dir,str('df_fits_'+'test'+'.csv')))

# %% check the beam size

fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)

i=0
j=0

for dataset in list(datasets):
    timestamp_pulse_ids_dataset=[]

    # ax = plt.subplot(gs[i,j])
    # print(dataset)
 
    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    
    
    for f in files:
        timestamp_pulse_ids = []
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

        print(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['pinholes_bg_avg_sx_um'].mean())

# %% remove duplicated index and columns
df0 = df0.loc[:,~df0.columns.duplicated()]
df0 = df0[~df0.index.duplicated()]


# %% plot the beamsizes ...

fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)



i=0
j=0
for dataset in list(datasets):
   
    ax = plt.subplot(gs[i,j])

    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:

    

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'],
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['pinholes_bg_avg_sx_um'],
            color='black')
    ax.set_xlim(0,2000)
    # plt.ylim(0,1)
    ax.set_title(dataset)

    if j==0:
        j=j+1
    else:
        j=0
        i=i+1


## how was this determined?

# %%


# <codecell>
# create plots fitting vs deconvolution
fig = plt.figure(figsize=[6, 8], constrained_layout=True)

gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)
gs.update(hspace=0, wspace=0.0)

i=0
j=0
for dataset in list(datasets):

    ax = plt.subplot(gs[i,j])

    # get all the files in a dataset:
    files = []
    # for set in [list(datasets)[0]]:
    
    for measurement in datasets[dataset]:
        # print(measurement)
        files.extend(bgsubtracted_dir.glob('*'+ measurement + '.h5'))

    # get all the timestamps in these files:        
    # datasets[list(datasets)[0]][0]
    timestamp_pulse_ids = []
    for f in files:
        with h5py.File(f, "r") as hdf5_file:
            timestamp_pulse_ids.extend(hdf5_file["Timing/time stamp/fl2user1"][:][:,2])

    # create plot for the determined timestamps:
    # plt.scatter(df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_x_um'], df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['xi_um_fit'], cmap=df0[df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)]['separation_um'])
    plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['I_Airy2_fit'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['xi_um_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids)) & (df0["xi_um_fit"]<2000)]['separation_um'],\
                marker='x', s=2)
    plt.xlabel(r"$I_2$")
    plt.ylabel(r"$\xi$ (fits)")
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
# %%


datasets_selection = datasets
# %%
datasets_selection.update({ datasets_widget.value : measurements_selection_widget.value })

# %%
datasets_selection_py_file = str(Path.joinpath(data_dir, "datasets_selection.py"))
with open(datasets_selection_py_file, 'w') as f:
    print(datasets_selection, file=f)
# %%
