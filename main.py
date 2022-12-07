# ==================================================================
# import
# ==================================================================
import load_datasets

# ================================
# set anatomy here
# ================================
# 'brain' / 'cardiac' / 'prostate' / 'wmh'
# ================================
ANATOMIES = ['wmh', 'wmh', 'wmh', 'brain', 'brain', 'brain', 'brain']

# ================================
# set dataset here.
# ================================
# for brain: 'HCP_T1' / 'HCP_T2' / 'ABIDE_caltech' / 'ABIDE_stanford'
# for cardiac: 'ACDC' / 'RVSC'
# for prostate: 'NCI' / 'PIRAD_ERC'
# for WMH: 'NUHS' / 'UMC' / 'VU'
# ================================
DATASETS = ['NUHS', 'UMC', 'VU', 'HCP_T1', 'HCP_T2', 'ABIDE_caltech', 'ABIDE_stanford']

# ================================
# read images and segmentation labels
# ================================
for dataset, anatomy in zip(DATASETS, ANATOMIES):
    for data_split in ['train', 'test', 'validation']:
        images, labels = load_datasets.load_dataset(
            anatomy=anatomy,
            dataset=dataset,
            train_test_validation=data_split,
            save_original=True  # Whether original dataset should be saved as well
        )
