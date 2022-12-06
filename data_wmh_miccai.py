import glob
import h5py
import logging
import numpy as np
import os
from skimage.transform import rescale

import utils



def prepare_data(
    input_folder,
    output_file,
    dataset,
    idx_start,
    idx_end,
    size,
    depth,
    target_resolution,
):

    # ========================
    # read the filepaths
    # ========================
    paths = sorted(glob.glob(os.path.join(input_folder, dataset, '*')))
    logging.info('Number of images in the dataset: %s' % str(len(paths)))

    # # =======================
    # # create a new hdf5 file
    # # =======================
    # hdf5_file = h5py.File(output_file, 'w')

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

    for idx in range(idx_start, idx_end):
        # ==================
        # get file paths
        # ==================
        patient_name = os.path.split(paths[idx])[1]
        label_path = os.path.join(paths[idx], 'wmh.nii.gz')
        image_path = os.path.join(paths[idx], 'pre', 'FLAIR.nii.gz')

        # ============
        # read the image
        # ============
        image, _, image_hdr = utils.load_nii(image_path)

        # ==================
        # read the label file
        # ==================        
        label, _, _ = utils.load_nii(label_path)
        label = (label == 1)  # Only uses white matter hyperintensity as foreground class (label == 2 would be 'other pathology')
    
        # ==================
        # crop volume along z axis (as there are several zeros towards the ends)
        # ==================
        if depth != -1:
            image = utils.crop_or_pad_volume_to_size_along_z(image, depth)
            label = utils.crop_or_pad_volume_to_size_along_z(label, depth)  

        # ==================
        # collect some header info.
        # ==================
        px_list.append(float(image_hdr.get_zooms()[0]))
        py_list.append(float(image_hdr.get_zooms()[1]))
        pz_list.append(float(image_hdr.get_zooms()[2]))
        nx_list.append(image.shape[0]) 
        ny_list.append(image.shape[1])
        nz_list.append(image.shape[2])
        pat_names_list.append(patient_name)

        # ==================
        # normalize the image
        # ==================
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')

        # ======================================================
        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
        # ======================================================
        scale_vector = [image_hdr.get_zooms()[0] / target_resolution[0],
                        image_hdr.get_zooms()[1] / target_resolution[1]]

        for zz in range(image.shape[2]):

            # ============
            # rescale the images and labels so that their orientation matches that of the nci dataset
            # ============
            image2d_rescaled = rescale(np.squeeze(image_normalized[:, :, zz]),
                                       scale_vector,
                                       order=1,
                                       preserve_range=True,
                                       mode='constant')

            label2d_rescaled = rescale(np.squeeze(label[:, :, zz]),
                                       scale_vector,
                                       order=0,
                                       preserve_range=True,
                                       mode='constant')

            # ============
            # crop or pad to make of the same size
            # ============
            image2d_rescaled_rotated_cropped = utils.crop_or_pad_slice_to_size(image2d_rescaled, size[0], size[1])
            label2d_rescaled_rotated_cropped = utils.crop_or_pad_slice_to_size(label2d_rescaled, size[0], size[1])

            # ============
            # append to list
            # ============
            image_list.append(image2d_rescaled_rotated_cropped)
            label_list.append(label2d_rescaled_rotated_cropped)

    # ============
    # write data to disk
    # ============
    logging.info('Writing data')

    with h5py.File(output_file, 'w') as hdf5_file:

        # ============   
        # Write the small datasets - image resolutions, sizes, patient ids
        # ============   
        hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
        hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
        hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
        hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
        hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
        hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
        hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S3"))

        hdf5_file.create_dataset('images', data=np.asarray(image_list, dtype=np.float32))
        hdf5_file.create_dataset('labels', data=np.asarray(label_list, dtype=np.uint8))
    


def load_and_maybe_process_data(
    input_folder,
    preprocessing_folder,
    dataset,
    idx_start,
    idx_end,
    size,
    depth,
    target_resolution,
    force_overwrite=False,
):
    
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    preprocessing_folder = os.path.join(preprocessing_folder, dataset.lower())
    data_file_name = f'data_{dataset.lower()}_2d_size_{size_str}_depth_{depth}_res_{res_str}_from_{idx_start}_to_{idx_end}.hdf5'
    data_file_path = os.path.join(preprocessing_folder, data_file_name)
    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     dataset,
                     idx_start,
                     idx_end,
                     size,
                     depth,
                     target_resolution)
    
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


# ===============================================================
# function to read a single subjects image and labels without any shape / resolution related pre-processing
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    dataset,
                                    idx,
                                    depth=-1):
    
    # ========================    
    # read the filepaths
    # ========================
    paths = sorted(glob.glob(os.path.join(input_folder, dataset, '*')))

    # ==================
    # get file paths
    # ==================
    patient_name = os.path.split(paths[idx])[1]
    label_path = os.path.join(paths[idx], 'wmh.nii.gz')
    image_path = os.path.join(paths[idx], 'pre', 'FLAIR.nii.gz')
    
    # ============
    # read the image
    # ============
    image, _, image_hdr = utils.load_nii(image_path)
    
    # ==================
    # read the label file
    # ==================        
    label, _, _ = utils.load_nii(label_path)        
    label = (label == 1)  # Only uses white matter hyperintensity as foreground class (label == 2 would be 'other pathology')
    
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


def load_multiple_without_size_preprocessing(input_folder,
                                             preprocessing_folder,
                                             dataset,
                                             idx_start,
                                             idx_end,
                                             depth=-1):

    images_original = []
    labels_original = []
    nx = []
    ny = []
    nz = []

    for i in range(idx_start, idx_end):

        img_orig, lab_orig = load_without_size_preprocessing(
            input_folder=input_folder,
            dataset=dataset,
            idx=i,
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

    data_file_name = f'data_{dataset.lower()}_original_depth_{depth}_from_{idx_start}_to_{idx_end}.hdf5'
    data_file_path = os.path.join(preprocessing_folder, dataset.lower(), data_file_name)

    with h5py.File(data_file_path, 'w') as f:
        f.create_dataset('images', data=images_original)
        f.create_dataset('labels', data=labels_original)
        f.create_dataset('nx',     data=nx)
        f.create_dataset('ny',     data=ny)
        f.create_dataset('nz',     data=nz)

    return h5py.File(data_file_path, 'r')
