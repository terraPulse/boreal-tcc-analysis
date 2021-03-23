
import scipy.stats
import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

def prob_v(v, e, t_forest):
    _a = (t_forest - v) / e
    _p = scipy.stats.norm.cdf(_a)
    return _p

def prob(bnd_dat, bnd_err, float t_forest):
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = bnd_dat.data
    cdef np.ndarray[np.float32_t, ndim=2] _err = bnd_err.data

    cdef int _rows = bnd_dat.height, _cols = bnd_dat.width
    cdef int _row, _col
    cdef int _v, _nv = bnd_dat.nodata
    cdef float _e, _ne = bnd_err.nodata
    cdef float _a, _p

    cdef np.ndarray[np.float32_t, ndim=2] _out = np.empty((_rows, _cols), dtype=np.float32)
    _out.fill(-9999)

    _cs = {}
    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v = _dat[_row, _col]
            if not (0 <= _v <= 100):
                continue

            _e = _err[_row, _col]
            if _e == _ne:
                continue

            if _e == 0:
                _e = 0.0001

            _a = (t_forest - _v) / _e

            if _a not in _cs:
                _p = scipy.stats.norm.cdf(_a)
                _cs[_a] = _p

            _out[_row, _col] = _cs[_a]

    return bnd_dat.from_grid(_out, nodata=-9999)

def estimate_type(float p1, float p2, float s_fn, float s_nf):
    cdef float _f1 = 1 - p1, _n1 = p1
    cdef float _f2 = 1 - p2, _n2 = p2

    cdef float _p_nn = _n1 * _n2
    cdef float _p_ff = _f1 * _f2
    cdef float _p_nf = _n1 * _f2
    cdef float _p_fn = _f1 * _n2

    _vs = [_p_ff, _p_nn, _p_fn, _p_nf]
    _ts = [11, 99, 19, 91]
    _ms = dict(zip(_ts, _vs))

    _max_v = max(_vs)
    _max_p = _vs.index(_max_v)
    _fc = _ts[_max_p]

    if _max_p < 2:
        return _fc, _max_v

    if _max_p == 2 and _max_v >= s_fn:
        return _fc, _max_v

    if _max_p == 3 and _max_v >= s_nf:
        return _fc, _max_v

    # cdef int _t1 = 1 if _f1 > _n1 else 9
    # cdef int _t2 = 1 if _f2 > _n2 else 9
    # cdef int _lc = _t1 * 10 + _t2

    _vs = [_f1, _f2, _n1, _n2]
    _ts = [11, 11, 99, 99]
    _id = _vs.index(max(_vs))

    return _ts[_id], _vs[_id]

def detect_change(bnd1, bnd2, float s_fn, float s_nf):
    assert(bnd1.width == bnd2.width and bnd1.height == bnd2.height)

    cdef np.ndarray[np.float32_t, ndim=2] _dat1 = bnd1.data
    cdef np.ndarray[np.float32_t, ndim=2] _dat2 = bnd2.data

    cdef int _rows = bnd1.height, _cols = bnd1.width
    cdef int _row, _col
    cdef float _v1, _v2
    cdef float _e1 = bnd1.nodata, _e2 = bnd2.nodata

    cdef np.ndarray[np.uint8_t, ndim=2] _out = np.empty((_rows, _cols), dtype=np.uint8)
    _out.fill(0)

    cdef np.ndarray[np.float32_t, ndim=2] _err = np.empty((_rows, _cols), dtype=np.float32)
    _err.fill(-9999)

    _cs = {}
    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v1 = _dat1[_row, _col]
            if _v1 == _e1:
                continue

            _v2 = _dat2[_row, _col]
            if _v2 == _e2:
                continue

            if _v1 not in _cs:
                _cs[_v1] = {}

            if _v2 not in _cs[_v1]:
                _cs[_v1][_v2] = estimate_type(_v1, _v2, s_fn, s_nf)

            _fc, _er = _cs[_v1][_v2]

            _out[_row, _col] = _fc
            _err[_row, _col] = _er

    return bnd1.from_grid(_out, nodata=0), bnd1.from_grid(_err, nodata=-9999)

def estimate_change_prob(int c1, float e1, int c2, float e2):
    if c1 == c2:
        return 

    _ee = max(0.0000001, (e1 ** 2 + e2 ** 2)) ** 0.5
    _cd = abs(c1 - c2)
    _cp = _cd / _ee

    _pp = scipy.stats.norm.cdf(_cp)

    return float(_pp)

def detect_tree_change(bnd1, err1, bnd2, err2, min_tcc=5, ftc=30, pb1=0.95, pb2=0.98):
    assert(bnd1.width == bnd2.width and bnd1.height == bnd2.height)
    assert(bnd1.width == err1.width and bnd1.height == err1.height)
    assert(err1.width == err2.width and err1.height == err2.height)

    cdef np.ndarray[np.uint8_t, ndim=2] _dat1 = bnd1.data
    cdef np.ndarray[np.uint8_t, ndim=2] _dat2 = bnd2.data

    cdef np.ndarray[np.float32_t, ndim=2] _err1 = err1.data
    cdef np.ndarray[np.float32_t, ndim=2] _err2 = err2.data

    cdef int _rows = bnd1.height, _cols = bnd1.width
    cdef int _row, _col
    cdef int _c1, _c2
    cdef float _e1, _e2

    cdef np.ndarray[np.uint8_t, ndim=2] _out = np.empty((_rows, _cols), dtype=np.uint8)
    _out.fill(0)

    cdef np.ndarray[np.float32_t, ndim=2] _err = np.empty((_rows, _cols), dtype=np.float32)
    _err.fill(-9999)

    _cs = {}
    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _c1 = _dat1[_row, _col]
            if _c1 > 100:
                continue

            _c2 = _dat2[_row, _col]
            if _c2 > 100:
                continue

            if _c1 == _c2:
                if _c1 >= 30:
                    _out[_row, _col] = 11
                else:
                    _out[_row, _col] = 99

                continue

            _e1 = _err1[_row, _col]
            if _e1 > 100.0 or _e1 < 0:
                continue

            _e2 = _err2[_row, _col]
            if _e2 > 100.0 or _e2 < 0:
                continue

            _pp = estimate_change_prob(_c1, _e1, _c2, _e2)

            if (_c1 > min_tcc) and (_c1 > _c2) and (_pp >= pb1):
                _out[_row, _col] = 19
            elif (_c2 > min_tcc) and (_c1 < _c2) and (_pp >= pb2):
                _out[_row, _col] = 91
            else:
                _cc = (_c1 + _c2) / 2
                if _cc >= ftc:
                    _out[_row, _col] = 11
                else:
                    _out[_row, _col] = 99

            # print _pp, _c1, _e1, _c2, _e2
            _err[_row, _col] = _pp

    return bnd1.from_grid(_out, nodata=0), bnd1.from_grid(_err, nodata=-9999)