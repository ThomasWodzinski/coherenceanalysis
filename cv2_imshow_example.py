import cv2
%matplotlib inline

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

float_arr = pixis_image_norm

uint_img = np.array(float_arr*255).astype('uint8')

grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

cv2.startWindowThread()
cv2.namedWindow("preview")

# Using cv2.imshow() method 
# Displaying the image 
cv2.imshow("preview", grayImage)
  
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 