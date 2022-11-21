# ==================================================================
# import 
# ==================================================================
import load_datasets

# ================================
# set anatomy here
# ================================
# 'brain' / 'cardiac' / 'prostate'
# ================================
ANATOMIES = ['brain', 'brain', 'brain', 'brain']

# ================================
# set dataset here.
# ================================
# for brain: 'HCP_T1' / 'HCP_T2' / 'ABIDE_caltech' / 'ABIDE_stanford'
# for cardiac: 'ACDC' / 'RVSC'
# for prostate: 'NCI' / 'PIRAD_ERC'
# ================================
DATASETS = ['HCP_T1', 'HCP_T2', 'ABIDE_caltech', 'ABIDE_stanford']

# ================================
# read images and segmentation labels
# ================================
for data_split in ['train', 'test', 'validation']:
    for dataset, anatomy in zip(DATASETS, ANATOMIES):
        images, labels = load_datasets.load_dataset(
            anatomy=anatomy,
            dataset=dataset,
            train_test_validation=data_split,
            first_run = False  # <-- SET TO TRUE FOR THE FIRST RUN (enables preliminary preprocessing e.g. bias field correction)
)
