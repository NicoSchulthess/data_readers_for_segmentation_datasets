import os
import numpy as np
import logging
import gc
import h5py
import glob
import utils
from scipy.ndimage import shift
from skimage.transform import rescale
from shutil import copyfile
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# =============================================================================================
# Function for copying the images and labels to a local directory and carrying out bias field correction using N4
# =============================================================================================
def copy_files_to_local_directory_and_correct_bias_field(src_folder,
                                                         dst_folder,
                                                         list_of_patients_to_skip):

    src_folders_list = sorted(glob.glob(src_folder + '*/'))
    
    # set the destination for this site
    for patient_num in range(len(src_folders_list)):
        
        patient_name = src_folders_list[patient_num][:-1][src_folders_list[patient_num][:-1].rfind('/')+1:]                
        logging.info('Patient: %d out of %d' % (patient_num+1, len(src_folders_list)))

        if patient_name in list_of_patients_to_skip:
            continue        

        src_folder_this_patient = src_folder + patient_name
        dst_folder_this_patient = dst_folder + patient_name
        if not os.path.exists(dst_folder_this_patient):
            os.makedirs(dst_folder_this_patient)
            
        # copy image
        suffix = '/MPRAGE.nii'
        copyfile(src_folder_this_patient + suffix, dst_folder_this_patient + suffix) 
        
        # copy labels
        suffix = '/orig_labels_aligned_with_true_image.nii.gz'
        copyfile(src_folder_this_patient + suffix, dst_folder_this_patient + suffix) 

        # correct bias field for the image
        subprocess.call(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", dst_folder_this_patient + '/MPRAGE.nii', dst_folder_this_patient + '/MPRAGE_n4.nii'])
                        
# ===============================================================
# Helper function to get paths to the image and label file
# ===============================================================
def get_image_and_label_paths(filename,
                              protocol = '',
                              extraction_folder = ''):
        
    _patname = filename[filename[:-1].rfind('/') + 1 : -1]
    _imgpath = filename + 'MPRAGE.nii'
    _segpath = filename + 'orig_labels_aligned_with_true_image.nii.gz'
                            
    return _patname, _imgpath, _segpath
                            
# ===============================================================
# helper function to count number of slices perpendicular to the coronal slices (this is fixed to the 'depth' parameter for each image volume)
# ===============================================================
def count_slices(filenames,
                 idx_start,
                 idx_end,
                 protocol,
                 preprocessing_folder,
                 depth):

    num_slices = 0
    
    for idx in range(idx_start, idx_end):    
        # _, image_path, _ = get_image_and_label_paths(filenames[idx])        
        # image, _, _ = utils.load_nii(image_path)        
        # num_slices = num_slices + image.shape[1] # will append slices along axes 1
        num_slices = num_slices + depth # the number of slices along the append axis will be fixed to this number to crop out zeros
        
    return num_slices

# ===============================================================
# helper function to center images and labels in the coronal slices
# ===============================================================
def center_image_and_label(image, label):
    
    orig_image_size_x = image.shape[0]
    orig_image_size_y = image.shape[1]
    
    fg_coords = np.where(label > 0)
    x_fg_min = np.min(np.array(fg_coords)[0,:])
    x_fg_max = np.max(np.array(fg_coords)[0,:])
    y_fg_min = np.min(np.array(fg_coords)[1,:])
    y_fg_max = np.max(np.array(fg_coords)[1,:])
    
    border = 20
    x_min = np.maximum(x_fg_min - border, 0)
    x_max = np.minimum(x_fg_max + border, orig_image_size_x)
    y_min = np.maximum(y_fg_min - border, 0)
    y_max = np.minimum(y_fg_max + border, orig_image_size_y)
    
    image_cropped = image[x_min:x_max, y_min:y_max, :]
    label_cropped = label[x_min:x_max, y_min:y_max, :]
    
    return image_cropped, label_cropped

# ===============================================================
# This function carries out all the pre-processing steps
# ===============================================================
def prepare_data(
    input_folder,
    output_file,
    site_name,
    idx_start,
    idx_end,
    protocol,
    size,
    depth,
    target_resolution,
    preprocessing_folder,
    atlas_path,
    force_overwrite,
    ):

    # ========================    
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + site_name + '/nifti/*/'))
    logging.info('Number of images in the dataset: %s' % str(len(filenames)))

    # =======================
    # create a new hdf5 file
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # ===============================
    # Create datasets for images and labels
    # ===============================
    data = {}
    num_slices = count_slices(filenames,
                              idx_start,
                              idx_end,
                              protocol,
                              preprocessing_folder,
                              depth)
    
    # ===============================
    # the sizes of the image and label arrays are set as: [(number of coronal slices per subject*number of subjects), size of coronal slices]
    # ===============================
    data['images'] = hdf5_file.create_dataset("images", [num_slices] + list(size), dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", [num_slices] + list(size), dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================        
    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []
    
    # ===============================
    # initialize counters        
    # ===============================        
    write_buffer = 0
    counter_from = 0
    
    # ===============================
    # iterate through the requested indices
    # ===============================
    for idx in range(idx_start, idx_end):
        
        # ==================
        # get file paths
        # ==================
        patient_name, image_path, label_path = get_image_and_label_paths(filenames[idx])
        
        # ============
        # apply bias correction
        # ============
        image_path = utils.n4_bias_correction(
            patient_name,
            image_path,
            preprocessing_folder,
            force_overwrite,
        )

        # ============
        # register images to atlas
        # ============
        image_path, label_path = utils.register_to_mni152(
            patient_name,
            image_path,
            label_path,
            preprocessing_folder,
            atlas_path,
            verbose=True,
            force_overwrite=force_overwrite,
        )
            
        # ============
        # read the image
        # ============
        image, _, image_hdr = utils.load_nii(image_path)
        image = np.swapaxes(image, 1, 2) # swap axes 1 and 2 -> this makes the coronal slices be in the 0-1 plane
        
        # ==================
        # read the label file
        # ==================        
        label, _, _ = utils.load_nii(label_path)        
        label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this makes the coronal slices be in the 0-1 plane
        label = utils.group_segmentation_classes(label) # group the segmentation classes as required
        
        # ============
        # create a segmentation mask and use it to get rid of the skull in the image
        # ============
        label_mask = np.copy(label)
        label_mask[label > 0] = 1
        image = image * label_mask

        # ==================
        # collect some header info.
        # ==================
        
        # Order of z and y is inversed since axes 1 and 2 have been swapped.
        # this is important when dealing with pixel dimensions
        px, pz, py = image_hdr.get_zooms()
        nx, ny, nz = image.shape

        px_list.append(px)
        py_list.append(py)
        pz_list.append(pz)
        nx_list.append(nx)
        ny_list.append(ny)
        nz_list.append(nz)
        pat_names_list.append(patient_name)
        
        # ==================
        # normalize the image
        # ==================
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')
                        
        # ======================================================  
        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
        # ======================================================
        scale_vector = [
            px / target_resolution[0],
            py / target_resolution[1],
            pz / target_resolution[2],
        ]


        # ============
        # rescale the images and labels so that their orientation matches that of the nci dataset
        # ============            
        image_rescaled = rescale(
            image_normalized,
            scale_vector,
            order=1,
            preserve_range=True,
            mode = 'constant',
        )

        label_rescaled = rescale(
            label,
            scale_vector,
            order=0,
            preserve_range=True,
            mode='constant',
        )
        
        # ============            
        # crop or pad to make of the same size
        # ============            
        image_rescaled = utils.crop_or_pad_slice_to_size(image_rescaled, size[0], size[1])
        image_rescaled = utils.crop_or_pad_volume_to_size_along_z(image_rescaled, depth)
        label_rescaled = utils.crop_or_pad_slice_to_size(label_rescaled, size[0], size[1])
        label_rescaled = utils.crop_or_pad_volume_to_size_along_z(label_rescaled, depth)

        # ============   
        # append to list
        # ============   
        image_list += [np.squeeze(img) for img in np.split(image_rescaled, image_rescaled.shape[2], axis=2)]
        label_list += [np.squeeze(lab) for lab in np.split(label_rescaled, label_rescaled.shape[2], axis=2)]

        write_buffer += label_rescaled.shape[2]

        # ============   
        # Writing needs to happen inside the loop over the slices
        # ============   
        if write_buffer >= MAX_WRITE_BUFFER:

            counter_to = counter_from + write_buffer

            _write_range_to_hdf5(data,
                                    image_list,
                                    label_list,
                                    counter_from,
                                    counter_to)
            
            _release_tmp_memory(image_list,
                                label_list)

            # ============   
            # update counters 
            # ============   
            counter_from = counter_to
            write_buffer = 0
        
    # ============   
    # write leftover data
    # ============   
    logging.info('Writing remaining data')
    counter_to = counter_from + write_buffer
    _write_range_to_hdf5(data,
                         image_list,
                         label_list,
                         counter_from,
                         counter_to)
    _release_tmp_memory(image_list,
                        label_list)

    # ============   
    # Write the small datasets - image resolutions, sizes, patient ids
    # ============   
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S10"))
    
    # ============   
    # close the hdf5 file
    # ============   
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    lab_arr = np.asarray(mask_list, dtype=np.uint8)

    hdf5_data['images'][counter_from : counter_to, ...] = img_arr
    hdf5_data['labels'][counter_from : counter_to, ...] = lab_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list,
                        mask_list):

    img_list.clear()
    mask_list.clear()
    gc.collect()
    
# ===============================================================
# function to read a single subjects image and labels without any shape / resolution related pre-processing
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    site_name,
                                    idx,
                                    preprocessing_folder,
                                    depth=-1,
    ):
    
    # ========================    
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + site_name + '/nifti/*/'))

    # ==================
    # get file paths
    # ==================
    patient_name, image_path, label_path = get_image_and_label_paths(filenames[idx])

    # ============
    # apply bias correction
    # ============
    image_path = utils.n4_bias_correction(patient_name, image_path, preprocessing_folder)
    
    # ============
    # read the image
    # ============
    image, _, image_hdr = utils.load_nii(image_path)
    image = np.swapaxes(image, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
    
    # ==================
    # read the label file
    # ==================        
    label, _, _ = utils.load_nii(label_path)        
    label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
    label = utils.group_segmentation_classes(label) # group the segmentation classes as required
    
    # ============
    # create a segmentation mask and use it to get rid of the skull in the image
    # ============
    label_mask = np.copy(label)
    label_mask[label > 0] = 1
    image = image * label_mask

    # ==================
    # crop volume along z axis (as there are several zeros towards the ends)
    # ==================
    if depth != -1:
        image = utils.crop_or_pad_volume_to_size_along_z(image, depth)
        label = utils.crop_or_pad_volume_to_size_along_z(label, depth)     
    
    # ==================
    # normalize the image    
    # ==================
    image = utils.normalise_image(image, norm_type='div_by_max')
    
    return image, label

# ===============================================================
# Main function that preprocesses images and labels and returns a handle to a hdf5 file containing them.
# The images and labels are returned as [num_subjects*nz, nx, ny],
#   where nz is the number of coronal slices per subject ('depth')
#         nx, ny is the size of the coronal slices ('size')  
#         the resolution in the coronal slices is 'target_resolution'
#         the resolution perpendicular to coronal slices is unchanged.
    
# The segmentations labels are obtained by segmenting the 'MPRAGE.nii.gz' file using Freesurfer.
    # Freesurfer internally does some registration steps, so the segmentation 'aparc+aseg.mgz' is not aligned with the original image.
    # To counter this, the image 'orig.mgz' (generated as an intermediate image by FS) is registered with the original image 'MPRAGE.nii.gz'
    # and the transformation so obtained is applied to the labels 'aparc+aseg.mgz' in order to get 'orig_labels_aligned_with_true_image.nii.gz'
    # The segmentation labels are then grouped into 15 classes.    

# Each image undergoes the following pre-processing steps:
    # bias field correction using N4 (using the correct_bias_field() function).
    # Skull stripping by setting all background pixels to 0
    # cropping to center image in the coronal slices
    # fixing the number of coronal slices to 'depth'
    # normalization to [0-1].    
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                site_name,
                                idx_start,
                                idx_end,
                                protocol,
                                size,
                                depth,
                                target_resolution,
                                atlas_path,
                                force_overwrite = False):
    
    # ==============================================
    # Set the 'first_run' option to True for the first run. This copies the files from the bmicnas server
    # into a local directory (preprocessing folder) and carries out bias field correction using N4.
    # For the 1st run, copy the files to a local directory and run bias field correction on the images.
    # ==============================================
    # This is not required now. we have already done bias correction and saved the corresponding images in the input_folder.
    # ==============================================
    # if first_run == True:
    #    if site_name == 'caltech':
    #        list_of_patients_to_skip = ['A00033264', 'A00033493']
    #    else:
    #        list_of_patients_to_skip = ['A00033547']
    #    logging.info('Copying files to local directory and carrying out bias field correction...')
    #    copy_files_to_local_directory_and_correct_bias_field(src_folder = input_folder + site_name + '/nifti/',
    #                                                         dst_folder = preprocessing_folder + site_name + '/',
    #                                                         list_of_patients_to_skip = list_of_patients_to_skip)
    
    # ==============================================
    # now, pre-process the data
    # ==============================================
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    preprocessing_folder = preprocessing_folder + site_name + '/'
    data_file_name = 'data_%s_2d_size_%s_depth_%d_res_%s_from_%d_to_%d.hdf5' % (protocol, size_str, depth, res_str, idx_start, idx_end)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)
    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     site_name,
                     idx_start,
                     idx_end,
                     protocol,
                     size,
                     depth,
                     target_resolution,
                     preprocessing_folder,
                     atlas_path,
                     force_overwrite)
    
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# function to read images and labels without any pre-processing
#  returns aa handle to a hdf5 file containing them.
#  The images and labels are returned as [num_subjects, nz, nx, ny],
#   where nz is the number of coronal slices per subject ('depth')
#         nx, ny is the size of the coronal slices ('size')
#         the resolution in the coronal slices is 'target_resolution'
#         the resolution perpendicular to coronal slices is unchanged.
# The read images are the ones from after the 'PreFreeSurfer' preprocessing pipeline.
    # So, they have undergone several pre-processing steps (including skull stripping and bias field correction)
    # For details, see the comments in the get_image_and_label_paths function.
# Each image volume is normalized to [0-1].
# The segmentation labels are grouped into 15 classes.
# ===============================================================
def load_multiple_without_size_preprocessing(input_folder,
                                             preprocessing_folder,
                                             site_name,
                                             idx_start,
                                             idx_end,
                                             protocol,
                                             depth=-1):

    images_original = []
    labels_original = []
    nx = []
    ny = []
    nz = []

    preprocessing_folder = os.path.join(preprocessing_folder, site_name)

    for i in range(idx_start, idx_end):

        img_orig, lab_orig = load_without_size_preprocessing(
            input_folder=input_folder,
            site_name=site_name,
            idx=i,
            preprocessing_folder=preprocessing_folder,
            depth=depth,
        )

        images_original.append(img_orig)
        labels_original.append(lab_orig)

        assert img_orig.shape == lab_orig.shape

        nx.append(img_orig.shape[0])
        ny.append(img_orig.shape[1])
        nz.append(img_orig.shape[2])

    nx = np.array(nx)
    ny = np.array(ny)
    nz = np.array(nz)

    # =====================================
    # Padding all images to the same shape
    # =====================================
    max_nx = np.max(nx)
    max_ny = np.max(ny)
    max_nz = np.max(nz)

    for i in range(len(images_original)):

        pad_x = max_nx - nx[i]
        pad_y = max_ny - ny[i]
        pad_z = max_nz - nz[i]

        images_original[i] = np.pad(
            images_original[i], 
            [(0, pad_x), (0, pad_y), (0, pad_z)]
        )

        labels_original[i] = np.pad(
            labels_original[i], 
            [(0, pad_x), (0, pad_y), (0, pad_z)]
        )

    # =====================================
    # Convert from N x nx x ny x nz to N x nz x nx x ny
    # =====================================
    images_original = np.moveaxis(np.stack(images_original), -1, 1)
    labels_original = np.moveaxis(np.stack(labels_original), -1, 1)

    data_file_name = 'data_%s_original_depth_%d_from_%d_to_%d.hdf5' % (
        protocol, depth, idx_start, idx_end)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    with h5py.File(data_file_path, 'w') as f:
        f.create_dataset('images', data=images_original)
        f.create_dataset('labels', data=labels_original)
        f.create_dataset('nx',     data=nx)
        f.create_dataset('ny',     data=ny)
        f.create_dataset('nz',     data=nz)

    return h5py.File(data_file_path, 'r')
