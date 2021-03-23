import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

def _log_weight(x):
    import math
    _y = (x + 0) * (math.e - 1) + 1
    return math.log(_y)

def match_prob(bnd, ref):
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = bnd.data
    cdef np.ndarray[np.uint8_t, ndim=2] _ref = ref.data

    cdef int _rows = bnd.height, _cols = bnd.width
    cdef int _row, _col, _v, _r, _n1 = bnd.nodata, _n2 = ref.nodata
    cdef float _a = 0.0, _z = 0.0, _x1 = 0.0, _x9 = 0.0, _t = 0.0, _y1 = 0.0, _y9 = 0.0
    cdef float _mba = 0.0, _n = 0.0, _rmse = 0.0

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v = _dat[_row, _col]
            _r = _ref[_row, _col]

            if _v > 220:
                continue

            if _r > 200:
                continue

            _z += 1.0

            if _v <= 200:
                _t += 1

            if _v > 100:
                _v = 0

            if _r > 100:
                _r = 0

            _mba += (_v - _r)
            _rmse += (_v - _r) ** 2
            _n += 1
    
    if _n <= 0.0:
        return -1.0, -1.0, -1.0

    _rmse = (_rmse / _n) ** 0.5
    _mba = _mba / _n

    # return _log_weight(_t / _z), _rmse, _mba
    return _t / _z, _rmse, _mba

def match(bnd, ref):
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = bnd.data
    cdef np.ndarray[np.uint8_t, ndim=2] _ref = ref.data

    cdef int _rows = bnd.height, _cols = bnd.width
    cdef int _row, _col, _v, _r, _n1 = bnd.nodata, _n2 = ref.nodata
    cdef float _a = 0.0, _z = 0.0, _x1 = 0.0, _x9 = 0.0, _t = 0.0, _y1 = 0.0, _y9 = 0.0

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v = _dat[_row, _col]
            _r = _ref[_row, _col]

            if _v == 0 or _v > 200:
                continue

            if _r == 0 or _r > 200:
                continue

            # if _r != 1 and _r != 4 and _r != 9:
            if _r not in (1, 4, 9):
                continue
            _z += 1.0

            if _v == 3:
                continue
            _t += 1

            if _r == 1:
                _y1 += 1

                if _v == 1:
                    _x1 += 1
            
            if _r == 9:
                _y9 += 1

                if _v == 9:
                    _x9 += 1
                if _v == 4:
                    _x9 += 1.0

            if _r == 4:
                _y9 += 1

                if _v == 4:
                    _x9 += 1.0

    if _z <= 0.0 or _t <= 0.0:
        return -1.0

    if _y1 + _y9 <= 0:
        return -1.0

    return _log_weight(_t / _z) * ((_x1 + _x9) / (_y1 + _y9))

def match_old(bnd, ref):
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = bnd.data
    cdef np.ndarray[np.uint8_t, ndim=2] _ref = ref.data

    cdef int _rows = bnd.height, _cols = bnd.width
    cdef int _row, _col, _v, _r, _n1 = bnd.nodata, _n2 = ref.nodata
    cdef float _a = 0.0, _z = 0.0, _x1 = 0.0, _x2 = 0.0

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v = _dat[_row, _col]
            _r = _ref[_row, _col]

            if _v == 0 or _v > 200:
                continue

            if _r == 0 or _r > 200:
                continue

            # if _r != 1 and _r != 4 and _r != 9:
            if _r not in (1, 9):
                continue

            # if _v != 1 and _v != 4 and _v != 9 and _v != 3:
            if _v not in (1, 3, 9):
                continue

            _x1 += 1
            if _v in (1, 9):
                _x2 += 1

            if _v ==  _r:
                _a += 1.0

            if _v == 3 and _r == 9:
                _a += 0.5

            _z += 1.0

    if _z <= 0.0 or _x1 <= 0.0:
        return -1.0

    if _x2 < 100.0:
        return -1.0

    # print _a, _z, _x2, _x1
    # print (_a / _z), (_x2 / _x1)

    # return (_a / _z) * (_x2 / _x1)
    return (_a / _z)

