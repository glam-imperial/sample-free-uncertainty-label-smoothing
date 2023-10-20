# This script reads the PRAAT annotation, then reads the long WAV recordings and segments into 3 sec clips.
# The preprocessing is done (logMel-Spectrograms, Normalisation etc.) and everything is saved as a TF RECORD.
# TF RECORD files are serialised, and TF reads them quite fast -- very useful for training, and we avoid
# doing the segmentation, feature extraction, preprocessing at every run / epoch.
# The important thing is to run preprocess.py once, and then use the same TF RECORDS across all experiments.

import os
import itertools

import numpy as np

import osa.wavtools as wavtools
import osa.configuration as configuration
from common import normalise, tfrecord_creator

np.random.seed(0)
# TODO: Have not verified if the same clip segmentations happen every time exactly.
# Your positive clips may be slightly time shifted differently than the ones I used in the paper.
# However, the entire call is in all cases included in any positive clip version.
# Experimental performance should be almost 100% similar in any case.

for partition in ["train", "devel", "test"]:
    # This is where the positive and negative WAV clips, as well as the TF RECORDS will be stored.
    positive_folder = configuration.DATA_FOLDER + '/clipped-whinnies/' + partition + '/'
    negative_folder = configuration.DATA_FOLDER + '/clipped-negatives/' + partition + '/'
    tfrecords_folder = configuration.DATA_FOLDER + '/tfrecords/' + partition + '/'

    if not os.path.exists(positive_folder):
        os.makedirs(positive_folder)
    if not os.path.exists(negative_folder):
        os.makedirs(negative_folder)
    if not os.path.exists(tfrecords_folder):
        os.makedirs(tfrecords_folder)

positive_generator_list = list()
negative_generator_list = list()

########################################################################################################################
# Positives
########################################################################################################################
PRAAT_FILE_LIST = sorted([f for f in os.listdir(configuration.DATA_FOLDER + '/praat-files') if not f.startswith('.')])

folders = os.listdir(configuration.DATA_FOLDER + '/Positives/')
for f in folders:  # A folder here signifies data from a specific site.
    print(f)
    partition = None
    for k, v in configuration.PARTITIONS.items():
        if f in v:
            partition = k

    if partition is None:
        raise ValueError

    # Saves clipped positive samples in folder named clipped-whinnies.
    positive_extend = wavtools.clip_whinnies(praat_files=PRAAT_FILE_LIST,
                                             desired_duration_sec=3,
                                             unclipped_folder_location=configuration.DATA_FOLDER + '/Positives/' + f,
                                             clipped_folder_location=configuration.DATA_FOLDER + '/clipped-whinnies/' + partition + "/",
                                             number_of_versions=5,  # Offline random time shift versions.
                                             partition=partition)
    positive_generator_list.append(positive_extend)

    # Saves the rest of the negative recording in folder named sections-without-whinnies.
    negative_extend = wavtools.clip_noncall_sections(praat_files=PRAAT_FILE_LIST,
                                                     desired_duration_sec=3,
                                                     unclipped_folder_location=configuration.DATA_FOLDER + '/Positives/' + f,
                                                     clipped_folder_location=configuration.DATA_FOLDER + '/clipped-negatives/' + partition + '/',
                                                     partition=partition)
    negative_generator_list.append(negative_extend)

########################################################################################################################
# Make tfrecords.
########################################################################################################################
generator_list = list()
generator_list.extend(positive_generator_list)
generator_list.extend(negative_generator_list)

# Both Normaliser and TFRecordCreator take as input a generator (lazy evaluation -- per sample)
# Here I "concatenate" multiple generators, from each site.
# Each element of the generator to Normaliser is an instance of Sample -- see initialisation in wavtools.py
generator_list = itertools.chain.from_iterable(generator_list)

normaliser = normalise.Normaliser(sample_iterable=generator_list,
                                  normalisation_scope="sample")
normalised_sample_generator = normaliser.generate_normalised_samples()

tfrecord_creator = tfrecord_creator.TFRecordCreator(tf_records_folder=configuration.TFRECORDS_FOLDER,
                                                    sample_iterable=normalised_sample_generator,
                                                    are_test_labels_available=True,
                                                    is_continuous_time=False)
tfrecord_creator.create_tfrecords()
