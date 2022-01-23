# %% imports

from coherencefinder.deconvolution_module import calc_sigma_F_gamma_um, deconvmethod, normalize
from coherencefinder.fitting_module import Airy, find_sigma, fit_profile

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


"""# Load dph settings and combinations"""

datasets_py_file = str(Path.joinpath(data_dir, "datasets.py"))

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

# import sys
# sys.path.append('g:\\My Drive\\PhD\\coherence\data\\dph_settings_package\\')
# from dph_settings_package import dph_settings_module


datasets_widget_layout = widgets.Layout(width="100%")
datasets_widget = widgets.Dropdown(options=datasets, layout=datasets_widget_layout)
# settings_widget.observe(update_settings, names='value')
# display(dph_settings_widget)


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


#%% """# List all groups inside the hd5file"""

# with h5py.File(dph_settings_bgsubtracted_widget.label, "r") as hdf5_file:

#     def printname(name):
#         print(name)

#     hdf5_file.visit(printname)


# %% """# display bgsubtracted images"""


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


# %% creating frontend


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

imageid_profile_fit_widget = widgets.Dropdown(
    # options=imageid_widget.options,
    options=[],
    description="imageid:",
    disabled=False,
)

savefigure_profile_fit_widget = widgets.Checkbox(value=False, description="savefigure", disabled=False)

save_to_df_widget = widgets.Checkbox(value=False, description="save_to_df", disabled=False)

do_textbox_widget = widgets.Checkbox(value=False, description="do_textbox", disabled=False)

textarea_widget = widgets.Textarea(value="info", placeholder="Type something", description="Fitting:", disabled=False)
beamsize_text_widget = widgets.Text(
    value="", placeholder="beamsize in rms", description=r"beam rms", disabled=False
)
fit_profile_text_widget = widgets.Text(
    value="", placeholder="xi_fit_um", description=r"\({\xi}_{fit}\)", disabled=False
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

shiftx_um_range_widget = widgets.FloatRangeSlider(
    min=-n / 2 * 13, max=n / 2 * 13, value=[-600, 1000], step=1, description="shiftx_um"
)
wavelength_nm_range_widget = widgets.FloatRangeSlider(
    min=7,
    max=19,
    value=[wavelength_nm_widget.value - 0.1, wavelength_nm_widget.value + 0.1],
    description="wavelength_nm",
)
z_mm_range_widget = widgets.FloatRangeSlider(min=5000.0, max=6000.0, value=[5770.0, 5790], description="z_mm")
d_um_range_widget = widgets.FloatRangeSlider(min=50, max=1337, value=[50.0, 1337.0], description="d_um")
gamma_range_widget = widgets.FloatRangeSlider(min=0, max=2.0, value=[0.2, 1.0], description="gamma")
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


# define what should happen when the hdf5 file widget is changed:


# function using the widgets:


def plotprofile(
    plotprofile_active,
    do_deconvmethod,
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

        if do_deconvmethod == True:
            deconvmethod_text_widget.value = 'calculating ...'
            partiallycoherent = pixis_image_norm
            z = 5781 * 1e-3
            dX_1 = 13 * 1e-6
            profilewidth = 200  # pixis_avg_width  # defined where?
            pixis_centery_px = int(pixis_centery_px)
            wavelength = setting_wavelength_nm * 1e-9
            xi_um_guess = 475
            # guess sigma_y_F_gamma_um based on the xi_um_guess assuming to be the beams intensity rms width
            sigma_y_F_gamma_um_guess = calc_sigma_F_gamma_um(xi_um_guess, n, dX_1, setting_wavelength_nm, False)
            crop_px = 200
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
                    create_figure,
                )
            deconvmethod_text_widget.value = r"%.2fum" % (xi_x_um) + r", %.2fum" % (xi_y_um)
            # str(round(xi_x_um, 2)) + ', ' + str(round(xi_y_um, 2))

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

        # if save_to_df == True:
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'gamma_fit'] = gamma_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'wavelength_nm_fit'] = wavelength_nm_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'd_um_at_detector'] = d_um_at_detector
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'I_Airy1_fit'] = I_Airy1_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'I_Airy2_fit'] = I_Airy2_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'w1_um_fit'] = w1_um_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'w2_um_fit'] = w2_um_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'shiftx_um_fit'] = shiftx_um_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'x1_um_fit'] = x1_um_fit
        #     df0.loc[((df0['hdf5_file_name'] == hdf5_file_name_image_widget.value) & (df0['pinholes'] == dataset_image_args_widget.value[2]) & (df0['imageid']==imageid)), 'x2_um_fit'] = x2_um_fit

        plt.show()
        # fittingprogress_widget.value = 10
        # fittingprogress_widget.bar_style = 'success'
        # statustext_widget.value = 'done'

        # print(gamma_fit)


# Structuring the input widgets

column0 = widgets.VBox(
    [
        plotprofile_active_widget,
        do_deconvmethod_widget,
        imageid_profile_fit_widget,
        savefigure_profile_fit_widget,
        save_to_df_widget,
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
    ]
)

column2 = widgets.VBox(
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

column3 = widgets.VBox(
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

column4 = widgets.VBox([textarea_widget, beamsize_text_widget, fit_profile_text_widget, deconvmethod_text_widget])

plotprofile_interactive_input = widgets.HBox([column0, column1, column2, column3, column4])

plotprofile_interactive_output = interactive_output(
    plotprofile,
    {
        "plotprofile_active": plotprofile_active_widget,
        "do_deconvmethod": do_deconvmethod_widget,
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
    },
)


def dph_settings_bgsubtracted_widget_changed(change):
    statustext_widget.value = "updating widgets ..."
    plotprofile_interactive_output.clear_output()
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


dph_settings_bgsubtracted_widget.observe(dph_settings_bgsubtracted_widget_changed, names="value")


#%% getting all parameters from the file


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

# Display widgets and outputs
display(
    VBox(
        [
            HBox([fittingprogress_widget, statustext_widget]),
            datasets_widget,
            dph_settings_bgsubtracted_widget,
            plotprofile_interactive_input,
            plotprofile_interactive_output,
        ]
    )
)
dph_settings_bgsubtracted_widget_changed(None)


# %%


# %%
