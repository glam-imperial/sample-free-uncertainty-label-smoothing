import os
import itertools
import collections

import numpy as np
from numpy.random import default_rng
import librosa
from pydub import AudioSegment

from common import normalise
from common.tfrecord_creator import TFRecordCreator
from common.data_sample import Sample


def make_tfrecords(data_folder,
                   ann,
                   nann,
                   clip_folder,
                   usable_classes,
                   sorted_usable_classes,
                   random_generator=default_rng(),
                   verbose=False):
    make_folders(data_folder)
    positive_generator_list = list()
    negative_generator_list = list()

    unclipped_folder = clip_folder
    clipped_folder = data_folder + '/Positives'
    positive_extend = segmentcalls_fileindependent(unclipped_folder=unclipped_folder,
                                                   clipped_folder=clipped_folder,
                                                   anns=ann,
                                                   usable_classes=usable_classes,
                                                   sorted_usable_classes=sorted_usable_classes,
                                                   random_generator=random_generator,
                                                   verbose=verbose)
    positive_generator_list.append(positive_extend)
    negative_extend = negcalls(unclipped_folder=unclipped_folder,
                               clipped_folder=data_folder + '/Negatives',
                               anns=nann,
                               verbose=verbose)
    negative_generator_list.append(negative_extend)

    generator_list = list()
    generator_list.extend(positive_generator_list)
    generator_list.extend(negative_generator_list)
    generator_list = itertools.chain.from_iterable(generator_list)

    normaliser = normalise.Normaliser(sample_iterable=generator_list,
                                      normalisation_scope="sample")
    normalised_sample_generator = normaliser.generate_normalised_samples()

    creator = TFRecordCreator(tf_records_folder=data_folder + '/tfrecords',
                              sample_iterable=normalised_sample_generator,
                              are_test_labels_available=True,
                              is_continuous_time=False)
    creator.create_tfrecords()
#####################################################################################################################


# This section and the code in the Common folder was kindly provided by Georgios Rizos
####################################################################################################################
def get_features_and_stats(waveform, orig_sr=16000, verbose=False):
    waveform = librosa.core.resample(waveform, orig_sr=orig_sr, target_sr=16000)
    spectrogram = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=10 * 16)) ** 1.0
    logmel_spectrogram = librosa.power_to_db(
        librosa.feature.melspectrogram(y=waveform, sr=16000, S=spectrogram))
    mfcc = librosa.feature.mfcc(waveform,
                                sr=16000,
                                n_mfcc=80,
                                S=logmel_spectrogram)

    spectrogram = spectrogram.transpose()[:-1, :]
    logmel_spectrogram = logmel_spectrogram.transpose()[:-1, :]
    mfcc = mfcc.transpose()[:-1, :]

    # if verbose:
    #     if logmel_spectrogram.shape[0] != 300:
    #         print(logmel_spectrogram.shape)

    custom_stats = dict()

    x_dict = dict()
    x_dict["waveform"] = waveform
    x_dict["logmel_spectrogram"] = logmel_spectrogram
    x_dict["mfcc"] = mfcc

    for x_name, x in x_dict.items():
        custom_stats[x_name] = dict()
        custom_stats[x_name]["mean"] = np.mean(x, axis=0)
        custom_stats[x_name]["std"] = np.std(x, axis=0)
        custom_stats[x_name]["max_abs"] = np.max(np.abs(x))

    # Waveform needs padding here.

    if x_dict["waveform"].size % 640 != 0:
        x_dict["waveform"] = np.concatenate([x_dict["waveform"],
                                             x_dict["waveform"][-1] * np.ones((640 - (x_dict["waveform"].size % 640), ))])

    x_dict["waveform"] = x_dict["waveform"].reshape((-1, 640))

    return x_dict, custom_stats


def make_folders(data_folder):
    for partition in ["train", "devel", "test"]:
        # This is where the positive and negative WAV clips, as well as the TF RECORDS will be stored.
        positive_folder = data_folder + '/Positives/' + partition + '/'
        negative_folder = data_folder + '/Negatives/' + partition + '/'
        tfrecords_folder = data_folder + '/tfrecords/' + partition + '/'

        if not os.path.exists(positive_folder):
            os.makedirs(positive_folder)
        if not os.path.exists(negative_folder):
            os.makedirs(negative_folder)
        if not os.path.exists(tfrecords_folder):
            os.makedirs(tfrecords_folder)


def get_species(label, target):
    species = []
    for l,s in zip(label, target):
        if l == 1 or l == '1':
            species.append(s)
    return species


def decide_partition(train, dev, test, numinstance):
    partitions = ['train', 'devel','test']
    off_train = (numinstance*0.5 - train)/numinstance*0.5
    off_dev = (numinstance*0.25 - dev)/numinstance*0.25
    off_test = (numinstance*0.25 - test)/numinstance*0.25
    offs = [off_train, off_dev, off_test]
    return partitions[offs.index(max(offs))]


def segmentcalls_fileindependent(unclipped_folder,
                                 clipped_folder,
                                 anns,
                                 usable_classes,
                                 sorted_usable_classes,
                                 random_generator=default_rng(),
                                 verbose=False):
    files = np.unique(anns[:, 0])
    index = np.arange(len(files))
    random_generator.shuffle(index)
    start, end, labels = anns[:, 1], anns[:, 2], anns[:, 3:]
    processed_filenames = list()
    train, dev, test = np.array([]), np.array([]), np.array([])
    trfile, dfile, tefile = np.array([], int), np.array([], int), np.array([], int)
    trainnum, devnum, testnum = 0, 0, 0

    for i in index:
        wav_name = files[i]
        print('Now processing file ', wav_name)

        try:
            wavfile = AudioSegment.from_wav(unclipped_folder + '/' + str(wav_name) + '.wav')
        except IOError:
            continue

        waveform = np.array(wavfile.get_array_of_samples(), dtype=np.float32)
        _, custom_stats = get_features_and_stats(waveform, orig_sr=16000, verbose=verbose)

        start_times = start[anns[:, 0] == str(wav_name)]
        end_times = end[anns[:, 0] == str(wav_name)]
        species = labels[anns[:, 0] == str(wav_name)]

        species_infile = np.array([])
        for ic in range(species.shape[0]):
            species_infile = np.append(species_infile, get_species(species[ic], usable_classes))
        species_infile = np.unique(species_infile)

        for s in sorted_usable_classes:
            if s in species_infile:
                if len(train) > 0 and s not in train:
                    train = np.append(train, species_infile)
                    trfile = np.append(trfile, [wav_name] * len(species_infile))
                    trainnum += len(end_times)
                    partition = 'train'
                elif len(test) > 0 and s not in test:
                    test = np.append(test, species_infile)
                    tefile = np.append(tefile, [wav_name] * len(species_infile))
                    testnum += len(end_times)
                    partition = 'test'
                elif len(dev) > 0 and s not in dev:
                    dev = np.append(dev, species_infile)
                    dfile = np.append(dfile, [wav_name] * len(species_infile))
                    devnum += len(end_times)
                    partition = 'devel'
                else:
                    if len(train) == 0:
                        train = np.append(train, species_infile)
                        trfile = np.append(trfile, [wav_name] * len(species_infile))
                        trainnum += len(end_times)
                        partition = 'train'
                    elif len(test) == 0:
                        test = np.append(test, species_infile)
                        tefile = np.append(tefile, [wav_name] * len(species_infile))
                        testnum += len(end_times)
                        partition = 'test'
                    elif len(dev) == 0:
                        dev = np.append(dev, species_infile)
                        dfile = np.append(dfile, [wav_name] * len(species_infile))
                        devnum += len(end_times)
                        partition = 'devel'
                    else:
                        # all partition contain all species
                        # or this file not assigned for this species according to rarity
                        # check which partition needs more data
                        partition = decide_partition(trainnum, devnum, testnum, len(anns))
                        if partition == 'train':
                            train = np.append(train, species_infile)
                            trfile = np.append(trfile, [wav_name] * len(species_infile))
                            trainnum += len(end_times)
                        elif partition == 'devel':
                            dev = np.append(dev, species_infile)
                            dfile = np.append(dfile, [wav_name] * len(species_infile))
                            devnum += len(end_times)
                        else:
                            test = np.append(test, species_infile)
                            tefile = np.append(tefile, [wav_name] * len(species_infile))
                            testnum += len(end_times)
                break

        for idx, time in enumerate(end_times):
            clip = wavfile[float(start_times[idx]):float(time)]
            # Save clipped file to separate folder
            # name = wav_name + '_' + str(idx) + "_" + repr(version_id)
            name = str(wav_name) + '_' + str(idx)
            clip.export(clipped_folder + '/' + partition + '/' + name + '.WAV', format="wav")
            waveform, sr = librosa.core.load(clipped_folder + '/' + partition + '/' + name + '.WAV', sr=16000, duration=3.00)

            # waveform = np.array(clip.get_array_of_samples(), dtype=np.float32)
            x_dict, _ = get_features_and_stats(waveform)
            if (x_dict["logmel_spectrogram"].shape[0] != 300) or (x_dict["logmel_spectrogram"].shape[1] != 128):
                print(x_dict["logmel_spectrogram"].shape, "ignoring.")
                continue

            id_dict = collections.OrderedDict()
            id_dict["segment_id"] = idx
            # id_dict["version_id"] = version_id

            y_dict = dict()
            y_dict["single"] = np.zeros((2,), dtype=np.float32)
            y_dict["single"][1] = 1.0
            y_dict["label"] = np.array(species[idx], dtype=np.float32)

            length_milliseconds = float(time) - float(start_times[idx])

            continuous_size = x_dict["waveform"].shape[0] * x_dict["waveform"].shape[1]
            length_ratio = continuous_size / length_milliseconds

            start_store = int(np.floor(0 * length_ratio))
            if start_store < 0:
                start_store = 0

            end_store = int(np.ceil(length_milliseconds * length_ratio))
            if end_store > continuous_size:
                end_store = continuous_size

            continuous = np.zeros((continuous_size, 2), dtype=np.float32)

            # where on the time axis is the clip
            for t in range(continuous_size):
                if ((t >= start_store) and (t < end_store)):
                    continuous[t, 1] = 1.0
                else:
                    continuous[t, 0] = 1.0

            y_dict["continuous"] = continuous

            support = np.ones((continuous_size, 1), dtype=np.float32)

            # Make DataSample.
            sample = Sample(name="pos_" + name,
                            id_dict=id_dict,
                            partition=partition,
                            x_dict=x_dict,
                            y_dict=y_dict,
                            support=support,
                            is_time_continuous=False,
                            custom_stats=custom_stats)

            processed_filenames.append(unclipped_folder + '/' + str(wav_name) + '.WAV')
            yield sample
    if verbose:
        print(trainnum)
        print(devnum)
        print(testnum)
        print(len(np.unique(train)), np.unique(train))
        print(len(np.unique(dev)), np.unique(dev))
        print(len(np.unique(test)), np.unique(test))
        for s in sorted_usable_classes:
            print(s)
            print("Test: ", np.unique(tefile[test == s]))
            print("Dev: ", np.unique(dfile[dev == s]))
            print("Train: ", np.unique(trfile[train == s]))


def negcalls(unclipped_folder,
             clipped_folder,
             anns,
             verbose=False):
    print('Now processing ', unclipped_folder)
    files = np.unique(anns[:, 0])
    start, end, labels = anns[:, 1], anns[:, 2], anns[:, 3:]
    processed_filenames, unprocessed_filenames = list(), list()
    trainnum, devnum, testnum = 0, 0, 0

    for wav_name in files:
        print('Now processing file ', wav_name)
        try:
            wavfile = AudioSegment.from_wav(unclipped_folder + '/' + str(wav_name) + '.wav')
        except IOError:
            continue

        waveform = np.array(wavfile.get_array_of_samples(), dtype=np.float32)
        _, custom_stats = get_features_and_stats(waveform, orig_sr=16000, verbose=verbose)

        start_times = start[anns[:, 0] == wav_name]
        end_times = end[anns[:, 0] == wav_name]
        species = labels[anns[:, 0] == wav_name]

        partition = decide_partition(trainnum, devnum, testnum, len(anns))
        if partition == 'train':
            trainnum += len(end_times)
        elif partition == 'devel':
            devnum += len(end_times)
        else:
            testnum += len(end_times)

        for idx, time in enumerate(end_times):
            clip = wavfile[float(start_times[idx]):float(time)]
            # Save clipped file to separate folder
            name = str(wav_name) + '_' + str(idx)
            clip.export(clipped_folder + '/' + partition + '/' + name + '.WAV', format="wav")
            waveform, sr = librosa.core.load(clipped_folder + '/' + partition + '/' + name + '.WAV', sr=16000, duration=3.00)

            x_dict, _ = get_features_and_stats(waveform)
            if (x_dict["logmel_spectrogram"].shape[0] != 300) or (x_dict["logmel_spectrogram"].shape[1] != 128):
                print(x_dict["logmel_spectrogram"].shape, "ignoring.")
                continue

            id_dict = collections.OrderedDict()
            id_dict["segment_id"] = idx

            y_dict = dict()
            y_dict["single"] = np.zeros((2,), dtype=np.float32)
            y_dict["single"][0] = 1.0
            y_dict["label"] = np.array(species[idx], dtype=np.float32)

            length_milliseconds = float(time) - float(start_times[idx])

            continuous_size = x_dict["waveform"].shape[0] * x_dict["waveform"].shape[1]
            length_ratio = continuous_size / length_milliseconds

            start_store = int(np.floor(0 * length_ratio))
            if start_store < 0:
                start_store = 0

            end_store = int(np.ceil(length_milliseconds * length_ratio))
            if end_store > continuous_size:
                end_store = continuous_size

            continuous = np.zeros((continuous_size, 2), dtype=np.float32)

            for t in range(continuous_size):
                if ((t >= start_store) and (t < end_store)):
                    continuous[t, 1] = 1.0
                else:
                    continuous[t, 0] = 1.0

            y_dict["continuous"] = continuous

            support = np.ones((continuous_size, 1), dtype=np.float32)

            # Make DataSample.
            sample = Sample(name="neg_" + name,
                            id_dict=id_dict,
                            partition=partition,
                            x_dict=x_dict,
                            y_dict=y_dict,
                            support=support,
                            is_time_continuous=False,
                            custom_stats=custom_stats)

            processed_filenames.append(unclipped_folder + '/' + str(wav_name) + '.WAV')
            yield sample
    if verbose:
        print('Negatives')
        print('Training instances:', trainnum)
        print('Development instances', devnum)
        print('Test instances', testnum)


