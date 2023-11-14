
import h5py
import hist
import numpy as np

from narf import ioutils
from utilities import logging

import pdb


logger = logging.child_logger(__name__)

def get_fitresult(fitresult_filename):
    h5file = h5py.File(fitresult_filename, mode='r')
    h5results = ioutils.pickle_load_h5py(h5file["results"])
    return h5results
