import collections
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

def stat(bnd, bnd_qa, int qa):
	cdef np.ndarray[np.uint8_t, ndim=2] _dat_qa = bnd_qa.data
	cdef np.ndarray[np.int16_t, ndim=2] _dat_va = bnd.data
	cdef int _rows = bnd_qa.height, _cols = bnd_qa.width
	cdef int _row, _col, _v_va, _v_qa

	_vs = collections.defaultdict(lambda: 0.0)
	for _row in xrange(_rows):
		for _col in xrange(_cols):
			_v_qa = _dat_qa[_row, _col]
			if _v_qa != qa:
				continue

			_v_va = _dat_va[_row, _col]
			_vs[_v_va] += 1.0

	return _vs

def bit_set(np.ndarray[np.uint8_t, ndim=2] dat, int bit, np.ndarray[np.uint8_t, ndim=2] dat_ref, int val):
	cdef int _rows = dat.shape[0], _cols = dat.shape[1]
	cdef int _row, _col
	cdef char _v_va, _v_rf

	for _row in xrange(_rows):
		for _col in xrange(_cols):
			_v_va = dat[_row, _col]
			if _v_va == 0:
				continue

			_v_rf = dat_ref[_row, _col]
			if _v_rf == val:
				_v_va = _v_va | (1 << bit)
			else:
				_v_va = _v_va & (~(1 << bit))
			dat[_row, _col] = _v_va

def bit(np.ndarray[np.uint8_t, ndim=2] dat, int bit, np.ndarray[np.uint8_t, ndim=2, cast=True] dat_idx):
	cdef int _rows = dat.shape[0], _cols = dat.shape[1]
	cdef int _row, _col
	cdef char _v_va, _v_rf

	for _row in xrange(_rows):
		for _col in xrange(_cols):
			_v_va = dat[_row, _col]
			if _v_va == 0:
				continue

			_v_rf = dat_idx[_row, _col]
			if _v_rf > 0:
				_v_va = _v_va | (1 << bit)
			else:
				_v_va = _v_va & (~(1 << bit))
			dat[_row, _col] = _v_va

