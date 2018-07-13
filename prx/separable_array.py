# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Separable array class."""

import numpy as np

__all__ = ('separray',)


class separray(np.ndarray):
    """Separable array, for iterating over parts of an array.

    Iterating over this array iterates over views to the separate parts of the
    array as defined from initialization. Aside from that and the :attr:`parts`
    attribute, it is functionally equivalent to its base class, numpy's
    ndarray.

    Slices and ufuncs that change the size of the array will return a plain
    ndarray, while other operations will maintain the separray class.

    .. automethod:: __new__


    Attributes
    ----------

    parts : tuple
        Views to the separate parts of the array.

    """

    def __new__(cls, *arrays):
        """Create combined array with views to separate arrays.

        The provided arrays are flattened and concatenated in the order
        given to make the combined array. The array views accessible with the
        :attr:`parts` attribute and through iteration provide views into the
        combined array that correspond to the location of the original arrays.


        Parameters
        ----------

        arrays : iterable
            Individual arrays to combine.

        """
        dtype = np.result_type(*arrays)
        sizes = [arr.size for arr in arrays]
        size = np.sum(sizes)

        self = np.ndarray.__new__(cls, size, dtype=dtype)

        idxs = [0] + list(np.cumsum(sizes))
        self._slices = tuple(slice(idxs[k], idxs[k + 1])
                             for k in xrange(len(idxs) - 1))
        self._shapes = tuple(arr.shape for arr in arrays)

        # copy original arrays into corresponding views of the combined array
        for view, arr in zip(iter(self), arrays):
            view[...] = arr

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            # got here from ndarray's __new__ called from our __new__
            # everything will be initialized in __new__
            return
        # copy over slice and shape data for views, and create new views
        self._slices = obj._slices
        self._shapes = obj._shapes

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.shape != self.shape:
            # shape has changed, need to return ndarray and not separray
            out_arr = out_arr.view(np.ndarray)
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, key):
        # getting portions of separable array is likely to mess up our array
        # views, so we need to go back to the base class and return an ndarray
        plainself = self.view(np.ndarray)
        return np.ndarray.__getitem__(plainself, key)

    def __getslice__(self, i, j):
        # need to implement because overriding __getitem__
        plainself = self.view(np.ndarray)
        return np.ndarray.__getslice__(plainself, i, j)

    def __iter__(self):
        for slc, shape in zip(self._slices, self._shapes):
            yield self[slc].reshape(shape)

    @property
    def parts(self):
        """Views to the separate parts of the array."""
        return tuple(self)
