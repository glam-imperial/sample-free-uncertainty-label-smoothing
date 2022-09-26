class Sample:
    def __init__(self,
                 name,
                 id_dict,  # Should be an OrderedDict
                 partition,
                 x_dict,
                 y_dict,
                 support,
                 is_time_continuous,
                 custom_stats=None):
        self._name = name
        self._id_dict = id_dict
        self._partition = partition
        self._x_dict = x_dict
        self._y_dict = y_dict

        self._support = support
        self._is_time_continuous = is_time_continuous

        self._custom_stats = custom_stats

    def get_id_dict(self):
        return self._id_dict

    def get_partition(self):
        return self._partition

    def get_x_dict(self):
        return self._x_dict

    def get_y_dict(self):
        return self._y_dict

    def get_support(self):
        return self._support

    def get_number_of_steps(self):
        raise NotImplementedError

    def is_time_continuous(self):
        return self._is_time_continuous

    def get_custom_stats(self):
        return self._custom_stats

    def set_id_dict(self, id_dict):
        self._id_dict = id_dict

    def set_partition(self, partition):
        self._partition = partition

    def set_x_dict(self, x_dict):
        self._x_dict = x_dict

    def set_y_dict(self, y_dict):
        self._y_dict = y_dict

    def set_support(self, support):
        self._support = support

    def set_number_of_steps(self, number_of_steps):
        raise NotImplementedError

    def set_is_time_continuous(self, is_time_continuous):
        self._is_time_continuous = is_time_continuous

    def set_custom_stats(self, custom_stats):
        self._custom_stats = custom_stats

    def get_composite_name(self):
        composite_name = [self._partition,
                          self._name]

        for id_name, id_number in self._id_dict.items():
            composite_name.append(repr(id_number))

        composite_name = "_".join(composite_name)
        return composite_name

    def trim_sequences(self):
        # I used that in MuSe.
        min_len = None

        for x_name, x in self._x_dict.items():
            min_len = x.shape[0]
            break

        if self._is_time_continuous:
            for y_name, y in self._y_dict.items():
                if y.shape[0] < min_len:
                    min_len = y.shape[0]

        for x_name, x in self._x_dict.items():
            if x.shape[0] < min_len:
                min_len = x.shape[0]

        if self._is_time_continuous:
            self._y_dict = {k: v[:min_len] for k, v in self._y_dict.items()}
        self._x_dict = {k: v[:min_len] for k, v in self._x_dict.items()}
