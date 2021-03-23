
import collections
# import numpy as np
import math
import logging

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

def clean(bnd, float dis, int min_num, int val, int non):
    cdef int _rows = bnd.height, _cols = bnd.width
    cdef int _row, _col, _v, _vo
    cdef int _nodata = bnd.nodata
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = bnd.data
    cdef int _t = 0

    # import config
    # _only_water = config.cfg.getboolean('filter_noise', 'only_water')

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _vo = _dat[_row, _col]

            if _vo != val:
                continue

            # if _only_water and (_vo != 2):
            #     continue

            _v = _stat(_dat, _col, _row, dis, min_num, _nodata, val, non, 199)
            if _v != _vo:
                _t += 1
                _dat[_row, _col] = _v
    
    _dat[_dat == 199] = non
    bnd.data = _dat

    return _t

def init_zero():
    return 0

cdef stat_pixel(np.ndarray[np.uint8_t, ndim=2] dat, int col, int row, float dis, int min_num, int nodata):
    cdef int _row, _col
    cdef int _v, _vv

    _vv = dat[row, col]
    if _vv == nodata:
        return nodata

    cdef int _dis = int(dis)

    cdef int _row_s = max(0, row - _dis), _row_e = min(dat.shape[0], row + _dis + 1)
    cdef int _col_s = max(0, col - _dis), _col_e = min(dat.shape[1], col + _dis + 1)

    cdef int _num = 0

    _ss = collections.defaultdict(init_zero)
    for _row in xrange(_row_s, _row_e):
        for _col in xrange(_col_s, _col_e):
            _v = dat[_row, _col]
            if _v == nodata:
                continue

            _d = math.hypot(_row - row, _col - col)
            if _d > dis:
                continue

            if _v == _vv:
                _num += 1

                if _num >= min_num:
                    return _vv

            _ss[_v] += 1
    
    _v = max(_ss.values())
    for _k in _ss:
        if _ss[_k] == _v:
            return _k

    raise Exception('failed to find the dominated value')

cdef _stat(np.ndarray[np.uint8_t, ndim=2] dat, int col, int row, float dis, int min_num, int nodata, \
        int val_tar, int val_bak, int val_new):
    cdef int _row, _col
    cdef int _v, _vv

    _vv = dat[row, col]
    if _vv == nodata:
        return nodata

    cdef int _dis = int(dis)

    cdef int _row_s = max(0, row - _dis), _row_e = min(dat.shape[0], row + _dis + 1)
    cdef int _col_s = max(0, col - _dis), _col_e = min(dat.shape[1], col + _dis + 1)

    cdef int _num_wat = 0
    cdef int _num_non = 0

    for _row in xrange(_row_s, _row_e):
        for _col in xrange(_col_s, _col_e):
            _v = dat[_row, _col]
            if _v == nodata:
                continue

            # _d = math.hypot(_row - row, _col - col)
            # if _d > dis:
            #     continue

            if _v == val_tar:
                _num_wat += 1
                if _num_wat >= min_num:
                    return val_tar
            else:
                _num_non += 1
    
    if _num_non > _num_wat:
        return val_new

    return val_tar

def expand(np.ndarray[np.uint8_t, ndim=2] dat, np.ndarray[np.uint8_t, ndim=2, cast=True] ref, val, non):
    from gio import config

    cdef int _rows = dat.shape[0], _cols = dat.shape[1]
    cdef int _row, _col, _vw, _vr

    cdef int _t = 0

    cdef int _dis = config.cfg.getint('expand_forest', 'search_dist')
    cdef int _min_num = config.cfg.getint('expand_forest', 'min_num')

    logging.info('expand dis: %s, min num: %s' % (_dis, _min_num))

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _vw = dat[_row, _col]
            _vr = ref[_row, _col]

            if _vw != non:
                continue

            if _vr == 1 and _near(dat, _col, _row, _dis, _min_num, val) == 1:
                dat[_row, _col] = 199
                _t += 1
    
    dat[dat == 199] = val
    return _t

cdef _near(np.ndarray[np.uint8_t, ndim=2] dat, int col, int row, float dis, int num, int val):
    cdef int _row, _col
    cdef int _v, _vv = dat[row, col]

    cdef int _dis = int(dis)

    cdef int _row_s = max(0, row - _dis), _row_e = min(dat.shape[0], row + _dis + 1)
    cdef int _col_s = max(0, col - _dis), _col_e = min(dat.shape[1], col + _dis + 1)

    cdef int _num = 0
    for _row in xrange(_row_s, _row_e):
        for _col in xrange(_col_s, _col_e):
            _v = dat[_row, _col]

            if _v != val:
                continue

            # _d = math.hypot(_row - row, _col - col)
            # if _d > dis:
            #     continue

            _num += 1
            if _num >= num:
                return 1

    return 0

def clean_change(bnd, err):
    from gio import mod_filter as ff
    from gio import config
    
    cdef np.ndarray[np.uint8_t, ndim=2] _ref = bnd.data.copy()
    
    ff.filter_band_median(bnd, 1, 1)
    ff.filter_band_mmu(bnd, area=config.getfloat('conf', 'min_patch', 100 * 100))
    
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = bnd.data
    cdef np.ndarray[np.float32_t, ndim=2] _err = err.data.copy()
    cdef int _v, _r, _n = bnd.nodata
    cdef float _e
    
    _div = 3
    for _row in range(bnd.height):
        for _col in range(bnd.width):
            _v = _dat[_row, _col]
            _r = _ref[_row, _col]
            
            if _v == _r or _v == _n:
                continue
                
            _min_col = max(0, _col - _div)
            _max_col = min(bnd.width, _col + _div + 1)
            
            _min_row = max(0, _row - _div)
            _max_row = min(bnd.height, _row + _div + 1)
                
            _idx = _ref[_min_row: _max_row, _min_col: _max_col] == _v
            if _idx.sum() > 0:
                _e = _err[_min_row: _max_row, _min_col: _max_col][_idx].mean()
            else:
                _e = 0.1
                
            _err[_row, _col] = _e
            
    _err[_dat == _n] = err.nodata
    err.data = _err
