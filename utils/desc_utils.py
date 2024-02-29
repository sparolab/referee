import os
import natsort

import numpy as np

def descriptor_path(descriptor_dir):
    descriptor_dir = os.path.join(descriptor_dir)
    descriptorfile_list = os.listdir(descriptor_dir)
    descriptorfile_list = natsort.natsorted(descriptorfile_list)
    descriptor_fullpaths = [os.path.join(descriptor_dir, name) for name in descriptorfile_list]
    num_descriptors = len(descriptorfile_list)
    return descriptor_fullpaths, num_descriptors

def get1DDescritor(descriptor_dir):
    descriptor_fullpaths, num_descriptors = descriptor_path(descriptor_dir)
    desc_test = np.load(descriptor_fullpaths[0])
    try:
        dimension, list = desc_test.shape
        descriptor = np.zeros((num_descriptors, dimension))
        for i in range(num_descriptors):
            descriptor[i, :] = np.load(descriptor_fullpaths[i])[:, 0]
    except ValueError:
        dimension = len(desc_test)
        descriptor = np.zeros((num_descriptors, dimension))
        for i in range(num_descriptors):
            descriptor[i, :] = np.load(descriptor_fullpaths[i])
    return descriptor
