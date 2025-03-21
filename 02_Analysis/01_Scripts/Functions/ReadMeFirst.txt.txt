# Where used:
# Import the required functions from feature_creation.py in Functions directory
from Functions import feature_creation # no hastag in front, but still: not in use
from Functions import feature_creation2
from Functions import general_functions
#from Functions import post_processing , not in use # in front of the line
from Functions.general_functions import create_circular_mask

#example:
        if "conic_mean" not in data_frame.columns:
            print ("Calculating conic_mean...")
            count_conic_mean = feature_creation2.conic_mean(skyview, 6, 0.9894350171089172)
            data_frame["conic_mean"] = count_conic_mean.flatten()
            print("Added conic_mean to the DataFrame.")

These Functions were used with creating the sourcefiles for working, the data feature creating process. The data created from the npy files in folder:

01_Data/01_raw/features/features (npy files needed for working shared as external link)
