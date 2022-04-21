
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
do_fitting_widget.value = True
for imageid in imageid_profile_fit_widget.options:
    imageid_profile_fit_widget.value = imageid
end = datetime.now()
time_taken = end - start
print(time_taken)


# <codecell>
# # iterate over all measurements and images in a given dataset
for measurement in dph_settings_bgsubtracted_widget.options:
    dph_settings_bgsubtracted_widget.value = measurement
    do_fitting_widget.value = True
    for imageid in imageid_profile_fit_widget.options:
        imageid_profile_fit_widget.value = imageid


# <codecell>
# iterate over all datasets
for dataset in list(datasets):
    datasets_widget.value = dataset
    do_fitting_widget.value = True
print('done')


# <codecell>
# iterate over everything
start = datetime.now()
for dataset in list(datasets):
    print(dataset)
    datasets_widget.value = dataset
    do_fitting_widget.value = True
    for measurement in dph_settings_bgsubtracted_widget.options:
        print(measurement)
        dph_settings_bgsubtracted_widget.value = measurement
        do_fitting_widget.value = True
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

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_x_um'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_um_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'], \
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
    plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_x_um'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_um_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'],\
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

    ax.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['gamma_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['I_Airy2_fit'])
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

# gaussian(x=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'], amp=1, cen=0, sigma=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_um_fit'])
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
    plt.scatter(df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['I_Airy2_fit'] , \
        df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['xi_um_fit'], \
            c=df0[(df0["timestamp_pulse_id"].isin(timestamp_pulse_ids))]['separation_um'],\
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

df_deconv_scany = pd.read_csv(Path.joinpath(scratch_dir, 'deconvmethod_steps', "sigma_y_F_gamma_um_guess_scan.csv"),
                              header=None, names=['ystep', 'sigma_y_F_gamma_um_guess', 'chi2distance'])
df_deconv_scany.plot('ystep', 'chi2distance')

# %% remove duplicates in df0
df0 = df0.drop_duplicates(subset=['timestamp_pulse_id'])
