from numpy.random import default_rng
from safe_msmt.annotations import make_annotation_text_files
from safe_msmt.data_helper import identify_classes_to_use
from safe_msmt.feature_partition import make_tfrecords


# seed_value = 68103
seed_value = 0
clip_length = 3
# threshold: to find species that appeared in at least 3 files
threshold = 3
verbose = True

rg = default_rng(seed_value)
data_folder = '/data/Data/XinWen'
clip_folder = data_folder + '/raw_data'
xlsx_filepath = data_folder + '/Species_Record_Sheet_SMitchell.xlsx'

usable_classes,\
sorted_usable_classes,\
negative_files,\
file_list,\
start_list,\
end_list,\
y_labels_list = identify_classes_to_use(xlsx_filepath=xlsx_filepath,
                                        threshold=threshold,
                                        verbose=verbose)

ann,\
nann = make_annotation_text_files(data_folder=data_folder,
                                  clip_length=clip_length,
                                  usable_classes=usable_classes,
                                  negative_files=negative_files,
                                  file_list=file_list,
                                  start_list=start_list,
                                  end_list=end_list,
                                  y_labels_list=y_labels_list,
                                  move_nontarget=True,
                                  random_generator=rg,
                                  verbose=verbose)
make_tfrecords(data_folder=data_folder,
               ann=ann,
               nann=nann,
               clip_folder=clip_folder,
               usable_classes=usable_classes,
               sorted_usable_classes=sorted_usable_classes,
               random_generator=rg,
               verbose=verbose)
