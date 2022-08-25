# create a junction symbolic link for the data instead of copying it into the repo

import pyinputplus as pyip

# create_link = False
# create_link = input('Create link?') or False
prompt = 'Create link?\n'
create_link = pyip.inputYesNo(prompt)
print(create_link)
source = pyip.inputMenu(['Google Drive for Desktop', 'local'], numbered=True)
print(source)
download_and_extract_useful_data = False
download_and_extract_bgsubtracted_data = False
# if necessary, request access to coherenceanalysis_folder_url and download the following two folders locally:
coherenceanalysis_folder_url = 'https://drive.google.com/drive/folders/1-s3HWOnOqqi_Z5OYDH2k3n_GIKJFvY47?usp=sharing'
useful_folder_url = 'https://drive.google.com/drive/folders/1IS_ZeCzTBXbNlO8BqcNlQpaBBQKdIiBw?usp=sharing'
bgsubtracted_folder_url = 'https://drive.google.com/drive/folders/1h4M1rGUTMaq9XTlqQlz7DqyFCsoLqFoM?usp=sharing'

# create prompt here

from pathlib import Path

if source == 'Google Drive':
    data_dir_source = Path('g:/My Drive/coherenceanalysis/data/')

if source == 'local':
    downloads_dir = Path.joinpath(Path.home(), 'Downloads')
    data_dir_source = Path.joinpath(downloads_dir,'coherenceanalysis','data')
    prompt = 'Use this path to local folder (' + str(data_dir_source) + '\n'
    response = pyip.inputMenu([str(Path.joinpath(downloads_dir,'coherenceanalysis','data')),'Enter different path'], numbered=True)
    if response == 'Enter different path':
        data_dir_source = Path(input())
    print(str(data_dir_source))

# Directory containing the useful hdf5 files (cleaned)
useful_dir_source = Path.joinpath(data_dir_source,'useful')
useful_dir_target = Path('./data/useful/')

# Directory containing the background-subtracted hdf5 files
bgsubtracted_dir_source = Path.joinpath(data_dir_source,'bgsubtracted')
bgsubtracted_dir_target = Path('./data/bgsubtracted/')


# if os.path.isdir(bgsubtracted_dir_target) == False

if create_link == 'yes':

    import _winapi

    if os.path.isdir(data_dir_source) == True:
        print('Creating junction symbolic link from ' + str(useful_dir_source.absolute()) + ' --> ' + str(useful_dir_target.absolute()))
        _winapi.CreateJunction(str(useful_dir_source.absolute()), str(useful_dir_target.absolute()))

        print('Creating junction symbolic link from ' + str(bgsubtracted_dir_source.absolute()) + ' --> ' + str(bgsubtracted_dir_target.absolute()))
        _winapi.CreateJunction(str(bgsubtracted_dir_source.absolute()), str(bgsubtracted_dir_target.absolute()))
    else:
        if source == 'Google Drive':
            print('Access to Google Drive folder needs to be requested')
            if os.path.isdir(Path('g:/My Drive/')) == False:
                print('Google Drive for Desktop not installed!')
        if source == 'local':
            print('local data folder does not exists at ' + str(data_dir_source))