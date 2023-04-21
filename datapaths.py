import os

# ==================================================================
# SET THESE PATHS MANUALLY 
# ==================================================================

# ==================================================================
# PATHS for original data
# ==================================================================
bmic_data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/'
orig_dir_hcp = os.path.join(bmic_data_root,'HCP/3T_Structurals_Preprocessed/')
orig_dir_abide = os.path.join(bmic_data_root,'ABIDE/') 
orig_dir_acdc = os.path.join(bmic_data_root,'Challenge_Datasets/ACDC_challenge_new/')
orig_dir_rvsc = os.path.join(bmic_data_root,'Challenge_Datasets/Cardiac_RVSC/AllData/')
orig_dir_nci = os.path.join(bmic_data_root,'Challenge_Datasets/NCI_Prostate/')
orig_dir_pirad_erc = os.path.join(bmic_data_root,'USZ/Prostate/')
orig_dir_wmh_miccai = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Challenge_Datasets/WMH_MICCAI2017'

# ==================================================================
# PATHS for preprocessed data
# ==================================================================
preproc_dir_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/data_readers/preproc_data/'
preproc_dir_hcp = os.path.join(preproc_dir_root, 'hcp/')
preproc_dir_abide = os.path.join(preproc_dir_root, 'abide/')
preproc_dir_acdc = os.path.join(preproc_dir_root, 'acdc/')
preproc_dir_rvsc = os.path.join(preproc_dir_root, 'rvsc/')
preproc_dir_nci = os.path.join(preproc_dir_root, 'nci/')
preproc_dir_pirad_erc = os.path.join(preproc_dir_root, 'pirad_erc/')
preproc_dir_wmh_miccai = os.path.join(preproc_dir_root, 'wmh_miccai/')

# ==================================================================
# PATHS to atlas for registration
# ==================================================================
atlas_path_mni_152 = '/home/scnico/datasets/mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii'
