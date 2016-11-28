"""This module provides debug helper functions."""
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


def debub_array(array, name):
    shape = array.get_shape().as_list()[1:]
    LOGGER.debug('%s shape: %s, dimension: %s', name, shape, np.prod(shape))
