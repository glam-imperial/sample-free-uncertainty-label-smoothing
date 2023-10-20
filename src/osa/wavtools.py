import os
import re
import collections

import numpy as np
import pydub
from pydub import AudioSegment
from pydub.utils import mediainfo
import librosa

import osa.configuration as configuration
from common.data_sample import Sample


def get_features_and_stats(waveform, orig_sr=48000, verbose=False):
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

    if verbose:
        if logmel_spectrogram.shape[0] != 300:
            print(logmel_spectrogram.shape)

    custom_stats = dict()

    x_dict = dict()
    # x_dict["waveform"] = waveform
    x_dict["logmel_spectrogram"] = logmel_spectrogram
    # x_dict["mfcc"] = mfcc

    for x_name, x in x_dict.items():
        custom_stats[x_name] = dict()
        custom_stats[x_name]["mean"] = np.mean(x, axis=0)
        custom_stats[x_name]["std"] = np.std(x, axis=0)
        custom_stats[x_name]["max_abs"] = np.max(np.abs(x))

    # Waveform needs padding here.

    # if x_dict["waveform"].size % 640 != 0:
    #     x_dict["waveform"] = np.concatenate([x_dict["waveform"],
    #                                          x_dict["waveform"][-1] * np.ones((640 - (x_dict["waveform"].size % 640), ))])
    #
    # x_dict["waveform"] = x_dict["waveform"].reshape((-1, 640))

    return x_dict, custom_stats


def clip_whinnies(praat_files,
                  desired_duration_sec,
                  unclipped_folder_location,
                  clipped_folder_location,
                  number_of_versions,
                  partition):
    print("Now processing:", unclipped_folder_location)

    processed_filenames = list()
    unprocessed_filenames = list()

    desired_duration_msec = desired_duration_sec * 1000

    for file in praat_files:
        start_times = whinny_starttimes_from_praatfile(configuration.DATA_FOLDER + '/praat-files/' + file)
        end_times = whinny_endtimes_from_praatfile(configuration.DATA_FOLDER + '/praat-files/' + file)
        wav_name = end_times[0]

        # following try-except accounts for praat files missing corresponding
        # audio files
        try:
            wavfile = AudioSegment.from_wav(unclipped_folder_location + '/' + wav_name + '.WAV')
        except IOError:
            # print("error: no wav file named " + wav_name + ".WAV at path " + unclipped_folder_location)
            continue

        if len(wavfile) < desired_duration_msec:
            unprocessed_filenames.append(unclipped_folder_location + '/' + wav_name + '.WAV')
            continue

        info = mediainfo(unclipped_folder_location + '/' + wav_name + '.WAV')
        if info["sample_rate"] != "48000":
            print(info["sample_rate"])

        waveform = np.array(wavfile.get_array_of_samples(), dtype=np.float32)
        _, custom_stats = get_features_and_stats(waveform, orig_sr=int(info["sample_rate"]))

        for idx, time in enumerate(end_times[1]):
            whinny_start = start_times[1][idx]
            whinny_end = time
            earliest_clip_start = whinny_end - desired_duration_msec + 1
            if earliest_clip_start < 0:
                earliest_clip_start = 0

            latest_clip_start = whinny_start
            if whinny_start + desired_duration_msec > len(wavfile):
                latest_clip_start = len(wavfile) - desired_duration_msec

            if latest_clip_start < earliest_clip_start:
                unprocessed_filenames.append(unclipped_folder_location + '/' + wav_name + '.WAV')
                continue

            for version_id in range(number_of_versions):
                actual_clip_start = np.random.uniform(earliest_clip_start, latest_clip_start)
                actual_clip_end = actual_clip_start + desired_duration_msec
                clip = wavfile[actual_clip_start:actual_clip_end]

                # Save clipped file to separate folder
                # name = wav_name + '_' + str(idx) + "_" + repr(version_id)
                name = wav_name + '_' + str(idx) + "_" + repr(version_id) + "_" + repr(int(actual_clip_start))
                clip.export(clipped_folder_location + '/' + name + '.WAV', format="wav")

                waveform, sr = librosa.core.load(clipped_folder_location + '/' + name + '.WAV',
                                                 sr=48000,
                                                 duration=3.00)

                # waveform = np.array(clip.get_array_of_samples(), dtype=np.float32)
                x_dict, _ = get_features_and_stats(waveform)

                id_dict = collections.OrderedDict()
                id_dict["segment_id"] = idx
                id_dict["version_id"] = version_id

                y_dict = dict()
                y_dict["whinny_single"] = np.zeros((1,), dtype=np.float32)
                y_dict["whinny_single"][0] = 1.0

                support = np.ones((x_dict["logmel_spectrogram"].shape[0], 1), dtype=np.float32)

                # Make DataSample.
                sample = Sample(name="pos_" + name,
                                id_dict=id_dict,
                                partition=partition,
                                x_dict=x_dict,
                                y_dict=y_dict,
                                support=support,
                                is_time_continuous=False,
                                custom_stats=custom_stats)

                processed_filenames.append(unclipped_folder_location + '/' + wav_name + '.WAV')
                yield sample


def clip_noncall_sections(praat_files, desired_duration_sec, unclipped_folder_location, clipped_folder_location, partition):
    processed_filenames = list()
    unprocessed_filenames = list()

    desired_duration_msec = desired_duration_sec * 1000

    for file in praat_files:

        start_times = non_starttimes_from_praatfile(configuration.DATA_FOLDER + '/praat-files/' + file)
        end_times = non_endtimes_from_praatfile(configuration.DATA_FOLDER + '/praat-files/' + file)
        wav_name = end_times[0]

        # Following try-except accounts for praat files missing corresponding
        # audio files
        try:
            wavfile = AudioSegment.from_wav(unclipped_folder_location + '/' + wav_name + '.WAV')

        except IOError:
            # print("error: no wav file named",wav_name,".WAV at path /home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies")
            continue

        info = mediainfo(unclipped_folder_location + '/' + wav_name + '.WAV')
        if info["sample_rate"] != "48000":
            print(info["sample_rate"])

        waveform = np.array(wavfile.get_array_of_samples(), dtype=np.float32)
        _, custom_stats = get_features_and_stats(waveform, orig_sr=int(info["sample_rate"]))

        for idx, time in enumerate(end_times[1]):

            segment_start = start_times[1][idx]
            segment_end = end_times[1][idx]

            segment = wavfile[segment_start:segment_end]

            # Save clipped file to separate folder
            # clip.export(clipped_folder_location + wav_name + '_' + str(idx) + '.WAV', format="wav")
            #
            # wavfile = AudioSegment.from_wav(clipped_folder_location + wav_name + '_' + str(idx) + '.WAV')

            if segment.duration_seconds < desired_duration_sec:
                # print(idx, file)
                unprocessed_filenames.append(unclipped_folder_location + '/' + wav_name + '.WAV')
                continue
            else:
                number_to_extract = int(segment.duration_seconds // desired_duration_sec)
                for i in range(number_to_extract):
                    pos = i * desired_duration_msec
                    if pos + desired_duration_msec > segment.duration_seconds * 1000:
                        unprocessed_filenames.append(unclipped_folder_location + '/' + wav_name + '.WAV')
                        continue
                    clip = segment[pos:pos + desired_duration_msec]
                    # Save clipped file to separate folder
                    # name = os.path.basename(os.path.splitext(file)[0])
                    name = wav_name + '_' + str(idx) + "_" + repr(i) + "_" + repr(int(segment_start + pos))

                    clip.export(clipped_folder_location + '/' + name + '.WAV',
                                format="wav")

                    waveform, sr = librosa.core.load(clipped_folder_location + '/' + name + '.WAV',
                                                     sr=48000,
                                                     duration=3.00)

                    # waveform = np.array(clip.get_array_of_samples(), dtype=np.float32)
                    x_dict, _ = get_features_and_stats(waveform)

                    id_dict = collections.OrderedDict()
                    id_dict["segment_id"] = idx
                    id_dict["version_id"] = i

                    y_dict = dict()
                    y_dict["whinny_single"] = np.zeros((1,), dtype=np.float32)
                    y_dict["whinny_single"][0] = 0.0

                    support = np.ones((x_dict["logmel_spectrogram"].shape[0], 1), dtype=np.float32)

                    # Make DataSample.
                    sample = Sample(name="neg_" + name,
                                    id_dict=id_dict,
                                    partition=partition,
                                    x_dict=x_dict,
                                    y_dict=y_dict,
                                    support=support,
                                    is_time_continuous=False,
                                    custom_stats=custom_stats)
                    processed_filenames.append(unclipped_folder_location + '/' + wav_name + '.WAV')
                    yield sample


def generate_negative_examples(noncall_files, desired_duration_sec, store_folder_location, partition):
    print("Now processing pure negative recordings.")

    processed_filenames = list()
    unprocessed_filenames = list()

    desired_duration_msec = desired_duration_sec * 1000

    for idx, file in enumerate(noncall_files):
        try:
            wavfile = AudioSegment.from_wav(file)

            wav_name = file.split("/")[-1][:-4]
        except pydub.exceptions.CouldntDecodeError as e:
            unprocessed_filenames.append(file)
            print('Could not decode:', file)
            continue
            # raise e
        except FileNotFoundError as e:
            unprocessed_filenames.append(file)
            print("File not found error.")
            continue
            # raise e
        if wavfile.duration_seconds < desired_duration_sec:
            # print(idx, file)
            continue
        else:
            number_to_extract = int(wavfile.duration_seconds // desired_duration_sec)
            for i in range(number_to_extract):
                pos = i * desired_duration_msec
                if pos+desired_duration_msec > wavfile.duration_seconds*1000:
                    unprocessed_filenames.append(file)
                    continue
                clip = wavfile[pos:pos+desired_duration_msec]
                # Save clipped file to separate folder
                # name = os.path.basename(os.path.splitext(file)[0])
                name = wav_name + '_' + str(idx) + "_" + repr(i) + "_" + repr(int(pos))

                clip.export(store_folder_location + '/' + name + '.WAV',
                            format="wav")

                waveform, sr = librosa.core.load(store_folder_location + '/' + name + '.WAV',
                                                 sr=48000,
                                                 duration=3.00)

                # waveform = np.array(clip.get_array_of_samples(), dtype=np.float32)
                x_dict, custom_stats = get_features_and_stats(waveform)

                id_dict = collections.OrderedDict()
                id_dict["segment_id"] = idx
                id_dict["version_id"] = i

                y_dict = dict()
                y_dict["whinny_single"] = np.zeros((1,), dtype=np.float32)
                y_dict["whinny_single"][0] = 0.0

                support = np.ones((x_dict["logmel_spectrogram"].shape[0], 1), dtype=np.float32)

                # Make DataSample.
                sample = Sample(name="neg_" + name,
                                id_dict=id_dict,
                                partition=partition,
                                x_dict=x_dict,
                                y_dict=y_dict,
                                support=support,
                                is_time_continuous=False,
                                custom_stats=custom_stats)

                processed_filenames.append(file)
                yield sample


def whinny_starttimes_from_praatfile(praat_file):
    """
    Extracts whinny start times (in milliseconds) from praat text file
    Output is tuple containing .wav file name and then all times (in ms)
    """
    start_times = []  # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()

    # - WAVNAME NOW FOUND USING NAME OF PRAAT FILE, NOT BY READING PRAAT FILE
    # line_with_wavname = praat_contents[10]
    # result = re.search('"(.*)"', line_with_wavname)
    # wav_name = result.group(1)

    wav_name = os.path.basename(os.path.splitext(praat_file)[0])

    for idx, line in enumerate(praat_contents):
        if "Whinny" in line and "intervals" in praat_contents[idx + 1]:
            time_line = praat_contents[idx - 2]
            start_times.extend(re.findall("(?<=xmin\s=\s)(\d+\.?\d*)(?=\s)", time_line))

    start_times = map(float, start_times)  # converts time to number, not
    # character string

    # Comment line below if time in seconds is wanted:
    start_times = [times * 1000 for times in start_times]

    return wav_name, start_times


def whinny_endtimes_from_praatfile(praat_file):
    """
    Extracts whinny end times (in milliseconds) from praat text file
    Output is tuple containing .wav file name and then all times in ms
    """
    end_times = []  # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()

    # - WAVNAME NOW FOUND USING NAME OF PRAAT FILE, NOT BY READING PRAAT FILE
    # line_with_wavname = praat_contents[10]
    # result = re.search('"(.*)"', line_with_wavname)
    # wav_name = result.group(1)

    wav_name = os.path.basename(os.path.splitext(praat_file)[0])

    for idx, line in enumerate(praat_contents):
        if "Whinny" in line and "intervals" in praat_contents[idx + 1]:
            time_line = praat_contents[idx - 1]
            end_times.extend(re.findall("\d+\.\d+", time_line))

    end_times = map(float, end_times)  # converts time to number, not
    # character string

    # Comment line below if time in seconds is wanted:
    end_times = [times * 1000 for times in end_times]

    return wav_name, end_times


def non_starttimes_from_praatfile(praat_file):
    """
    Extracts start time of region known to not contain a spider
    monkey whinny from praat text file (for generating negatives)
    Output is tuple containing .wav file name and then all times (in ms)
    """
    start_times = []  # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()

    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents):
        if "Non Call" in line:
            # if "Whinny" not in line:
            time_line = praat_contents[idx - 2]
            start_times.extend(re.findall("\d+\.\d+|\d", time_line))

    start_times = map(float, start_times)  # converts time to number, not
    # character string

    # Comment line below if time in seconds is wanted:
    start_times = [times * 1000 for times in start_times]

    return wav_name, start_times


def non_endtimes_from_praatfile(praat_file):
    """
    Extracts end time of region known to not contain a spider
    monkey whinny from praat text file (for generating negatives)
    Output is tuple containing .wav file name and then all times in ms
    """
    end_times = []  # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()

    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents):
        if "Non Call" in line:
            # if "Whinny" not in line:
            time_line = praat_contents[idx - 1]
            end_times.extend(re.findall("\d+\.\d+|\d{2}", time_line))

    end_times = map(float, end_times)  # converts time to number, not
    # character string

    # Comment line below if time in seconds is wanted:
    end_times = [times * 1000 for times in end_times]

    return wav_name, end_times
