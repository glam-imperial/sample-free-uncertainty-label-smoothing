import collections

import numpy as np


class Normaliser:
    def __init__(self,
                 sample_iterable,
                 normalisation_scope):
        self._sample_iterable = sample_iterable
        self._normalisation_scope = normalisation_scope

        if self._normalisation_scope not in ["sample",
                                             "custom",
                                             "partition"]:
            raise ValueError("Invalid normalisation scope.")

        self._dataset_stats = None

    def calculate_stats(self):
        print("Calculate train partition stats.")
        # TODO: This is not finished.

        # sum_value = 0.0
        # sum_squares_value = 0.0
        # for au in audio_frames:
        #     sum_value += np.sum(au)
        # mean_value = sum_value / (len(audio_frames) * 640)
        #
        # for au in audio_frames:
        #     sum_squares_value += np.sum(np.power(au - mean_value, 2.0))
        # standard_deviation = np.sqrt(sum_squares_value / (len(audio_frames) * 640))
        #
        # for i, au in enumerate(audio_frames):
        #     audio_frames[i] = (au - mean_value) / standard_deviation

        stats_temp = dict()
        stats = dict()
        for sample in self._sample_iterable:
            if sample.get_partition() == "train":
                x_dict = sample.get_x_dict()

                for x_name, x in x_dict.items():
                    if x_name not in stats_temp.keys():
                        stats_temp[x_name] = collections.defaultdict(int)
                    stats_temp[x_name]["sum_value"] += np.sum(x)
                    stats_temp[x_name]["num_elements"] += x.size
                    stats_temp[x_name]["max_abs"] = np.max(np.abs(x)) # TODO: Fix this.
        for x_name, x in x_dict.items():
            stats[x_name] = dict()
            stats[x_name]["mean"] = stats_temp[x_name]["sum_value"] / stats_temp[x_name]["num_elements"]
            stats[x_name]["max_abs"] = stats_temp[x_name]["max_abs"]

        for sample in self._sample_iterable:
            if sample.get_partition() == "train":
                x_dict = sample.get_x_dict()

                for x_name, x in x_dict.items():
                    stats_temp[x_name]["sum_squares_value"] += np.sum(np.power(x - stats[x_name]["mean"], 2.0))
        for x_name, x in x_dict.items():
            stats[x_name]["std"] = np.sqrt(stats_temp[x_name]["sum_squares_value"] / stats_temp[x_name]["num_elements"])

        print(stats)
        return stats

    def generate_normalised_samples(self):
        if self._normalisation_scope == "partition":
            if self._dataset_stats is None:
                self.calculate_stats()

        for sample in self._sample_iterable:
            x_dict = sample.get_x_dict()

            if self._normalisation_scope == "partition":
                stats = self._dataset_stats
            elif self._normalisation_scope == "sample":
                stats = self._get_sample_level_stats(x_dict)
            elif self._normalisation_scope == "custom":
                stats = sample.get_custom_stats()
            else:
                raise ValueError("Invalid normalisation scope.")

            new_x_dict = dict()
            for x_name, x in x_dict.items():
                if "waveform" in x_name:
                    # new_x_dict[x_name] = self._max_expand_norm(x, stats[x_name])
                    new_x_dict[x_name] = self._waveform_z_norm(x, stats[x_name])
                else:
                    new_x_dict[x_name] = self._z_norm(x, stats[x_name])

            sample.set_x_dict(new_x_dict)

            yield sample

    def generate_subsegmented(self):
        raise NotImplementedError

    def generate_padded(self):
        raise NotImplementedError

    def _get_sample_level_stats(self, x_dict):
        stats = dict()
        for x_name, x in x_dict.items():
            stats[x_name] = dict()
            if "waveform" in x_name:
                stats[x_name]["mean"] = np.mean(x)
                stats[x_name]["std"] = np.std(x)
                stats[x_name]["max_abs"] = np.max(np.abs(x))
            else:
                stats[x_name]["mean"] = np.mean(x, axis=0)
                stats[x_name]["std"] = np.std(x, axis=0)
                stats[x_name]["max_abs"] = np.max(np.abs(x))

        return stats

    def _waveform_z_norm(self, x, stats):
        std = stats["std"]

        if std == 0.0:
            std = 1.0

        new_x = (x - stats["mean"]) / std

        return new_x

    def _z_norm(self, x, stats):
        std = stats["std"]
        std[std == 0.0] = 1.0

        new_x = (x - stats["mean"]) / std

        return new_x

    def _max_expand_norm(self, x, stats):
        x = x * (0.7079 / stats["max_abs"])
        maxv = np.iinfo(np.int16).max
        x = (x * maxv).astype(np.float32)
        return x
