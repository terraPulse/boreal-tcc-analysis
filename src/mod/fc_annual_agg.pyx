'''
File: fc_annual_agg.pyx
Author: Min Feng
Version: 0.1
Create: 2018-04-18 18:35:03
Description:
'''

import datetime
import numpy as np
from scipy import stats
import numpy.ma as ma
import logging
import os
import re
import math
from gio import config
from gio import landsat
from gio import cache_mag
from gio import geo_raster as ge
from gio import geo_raster_ex as gx
from gio import geo_base as gb
from gio import progress_percentage
from . import fc_time_series
from . import fc_layer_mag

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

def update(bbs, bs1, bs2, idx):
    if None in bs2:
        return

    if type(None) in [type(_b.data) for _b in bs2]:
        return

    _b1_bnd = bs1[0]
    # _b1_err = bs1[1]

    _b2_bnd = bs2[0]
    _b2_err = bs2[1]

    cdef int _rows = _b1_bnd.height, _cols = _b1_bnd.width
    cdef int _row, _col
    cdef float _e
    cdef float _v1, _v2, _nodata = _b1_bnd.nodata

    cdef np.ndarray[np.float32_t, ndim=2] _dat1 = _b1_bnd.data
    cdef np.ndarray[np.float32_t, ndim=2] _dat2 = _b2_bnd.data

    # cdef np.ndarray[np.float32_t, ndim=2] _err1 = _b1_err.data
    cdef np.ndarray[np.float32_t, ndim=2] _err2 = _b2_err.data

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v1 = _dat1[_row, _col]
            if _v1 == _nodata:
                continue

            _v2 = _dat2[_row, _col]
            if _v2 == _nodata:
                continue

            _e = _err2[_row, _col]

            _i = _row * _cols + _col
            _m = bbs[_i]

            if _v2 <= 100:
                _vv = 0
            else:
                _vv = _v2

            if _vv not in _m:
                _m[_vv] = [1, _e, idx, _v2]
            else:
                _m[_vv][0] += 1
                # _m[_vv][1] += _e

                if _vv == 0:
                    if _e < _m[_vv][1]:
                        _m[_vv][1] = _e
                        _m[_vv][2] = idx
                        _m[_vv][3] = _v2

class pixel_value:

    def __init__(self, val, err, idx):
        self.val = val
        self.err = err
        self.idx = idx

    def __eq__(self, o):
        return self.val == o.val
        
    def __ne__(self, o):
        return self.val != o.val
        
    def __lt__(self, o):
        return self.val < o.val
        
    def __le__(self, o):
        return self.val <= o.val
        
    def __gt__(self, o):
        return self.val > o.val
        
    def __ge__(self, o):
        return self.val >= o.val

def _std(vs):
    if len(vs) <= 1:
        return 0

    _m = sum(vs) / len(vs)
    _a = 0.0

    for _v in vs:
        _a += (_v - _m) ** 2

    if _a <= 0:
        return 0

    return (_a / (len(vs) - 1)) ** 0.5

def update_date(bbs, msk, bs2, mods=None, v_max=5.0):
    if None in bs2:
        return 0

    if type(None) in [type(_b.data) for _b in bs2[1:3]]:
        return 0

    _b1_bnd = msk

    _ac_dat = bs2[0]
    _b2_bnd = bs2[1]
    _b2_err = bs2[2]
    _b2_inf = bs2[3]
    _wrs_row_north = _b2_inf.row <= 60

    _ac_year = _ac_dat.year

    # if (_b2_bnd.data <= 200).sum() <= ((_b2_bnd.width * _b2_bnd.height) / 8.0):
    if (_b2_bnd.data <= 200).sum() < 1:
        return 0

    if mods is not None:
        _mods = []

        # _v_max = max(3, _std([_mod.data.mean() for _mod in mods.values()]))
        # print 'STD modis', _v_max
        _v_max = v_max

        # for _x in [-1, 0, 1]:
        for _x in [0]:
            if (_ac_year + _x) not in mods:
                continue

            _mask = (_b2_bnd.data > 100) | (mods[_ac_year + _x].data > 100)
            _rat = (_mask.sum() * 1.0) / (_mask.shape[0] * _mask.shape[1])

            if _rat > 0.95:
                logging.info('skip scene %s, %s' % (bs2[0], _rat))
                return 0

            _tcc_avg = ma.masked_array(_b2_bnd.data, mask=_mask).mean()

            _mod_avg = ma.masked_array(mods[_ac_year + _x].data, mask=_mask).mean()
            _mods.append(abs(_mod_avg - _tcc_avg))

        if len(_mods):
            _min_v = min(_mods)
            if _min_v > _v_max:
                logging.info('skip scene %s, %s, %s' % (bs2[0], _min_v, _v_max))
                return 0

            logging.info('passed scene %s, %s, %s' % (bs2[0], _min_v, _v_max))

    cdef int _rows = _b1_bnd.height, _cols = _b1_bnd.width
    cdef int _row, _col
    cdef int _v1, _v2, _nodata = _b1_bnd.nodata, _nodata2 = _b2_bnd.nodata
    cdef float _e

    cdef np.ndarray[np.uint8_t, ndim=2] _dat1 = _b1_bnd.data
    cdef np.ndarray[np.uint8_t, ndim=2] _dat2 = _b2_bnd.data
    # cdef np.ndarray[np.uint8_t, ndim=2] _dat3
    cdef np.ndarray[np.float32_t, ndim=2] _err2 = _b2_err.data

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v1 = _dat1[_row, _col]
            if _v1 == _nodata:
                continue

            _v2 = _dat2[_row, _col]
            if _v2 == _nodata2:
                continue

            if _v2 > 100 or _v2 < 0:
                continue

            # if mods is not None:
            # 	_mods = []

            # 	for _x in [-1, 0, 1]:
            # 		if (_ac_year + _x) not in mods:
            # 			continue

            # 		_dat3 = mods[_ac_year + _x].data
            # 		_mods.append(abs(_dat3[_row, _col] - _v2))

            # 	if len(_mods):
            # 		_min_v = min(_mods)
            # 		if _min_v > 30:
            # 			continue

            _e = _err2[_row, _col]
            if _e < 0:
                continue

            _i = _row * _cols + _col
            if _i not in bbs:
                bbs[_i] = []

            bbs[_i].append(fc_time_series.fc_obj(_ac_dat, _v2, _e, _wrs_row_north))

    return 1

def _list_err_files(d, t):
    for _f in os.listdir(d):
        if _f.endswith('_err.tif') and t in _f:
            return os.path.join(d, _f)
    return None

def _err_file(f):
    if not os.path.exists(f):
        return None

    _f_err = f.replace('_tcc.tif', '_err.tif').replace('_dat.tif', '_err.tif')

    if f == _f_err:
        _m = re.search('(p\d{3}r\d{3})_TCC_(\d{8})\.tif', f)
        if _m:
            _f_err = os.path.join(os.path.dirname(f), 'vcf_%s_%s_vcf_err.tif' % (_m.group(1), _m.group(2)))
            if not os.path.exists(_f_err):
                _f_err = _list_err_files(os.path.dirname(f), '%s_%s' % (_m.group(1), _m.group(2)))

    if f == _f_err:
        raise Exception('failed to find error file (%s)' % f)

    if not os.path.exists(_f_err):
        return None

    return _f_err

def _ac_date(f):
    _f = os.path.basename(f)

    _m = re.search('\D(\d{8})\D', _f)
    if _m is not None:
        _d = _m.group(1)
        return datetime.datetime.strptime(_d, '%Y%m%d')

    _m = re.search('\D(\d{7})\D', _f)
    if _m is not None:
        _d = _m.group(1)
        return datetime.datetime.strptime(_d, '%Y%j')

    return None

def load_img(f, bnd=None, cache=None):
    _inf = landsat.parse(f)

    _r = None
    try:
        if f.endswith('_pob.tif') or f.endswith('_prob.tif'):
            _r = _load_img_noerr(f, bnd, cache)
            _r.append(_inf)
            return _r

        if '_tcc' in f:
            _r = _load_img_noerr(f, bnd, cache, 16.0)
            _r.append(_inf)
            return _r

        _r = _load_img(f, bnd, cache)
    except KeyboardInterrupt, _err:
        print('\n\n* User stopped the program')
        raise _err
    except Exception, err:
        logging.error('failed to load %s' % f)

    if _r is None:
        return None

    _d = _r[0]
    _dat = _r[1]
    _err = _r[2]

    if _dat is None or _err is None:
        return None

    _ddd = np.empty((_dat.height, _dat.width), dtype=np.uint8)
    _ddd.fill(255)

    _idx = _dat.data == 1
    _ddd[_idx] = 100.0 * _err.data[_idx]

    _idx = _dat.data == 9
    _ddd[_idx] = (1.0 - _err.data[_idx]) * 100.0

    if (_ddd <= 100).sum() <= 0:
        return None

    _idx = _dat.data == 4
    _ddd[_idx] = 200

    _idx = _dat.data == 5
    _ddd[_idx] = 220

    _idx = _dat.data == 2
    _ddd[_idx] = 210

    _idx = _dat.data == 3
    _ddd[_idx] = 211

    _idx = (_ddd < 200) & (_err.data >= 0)
    _err.data[_idx] = (1.0 - _err.data[_idx]) * 100.0

    return [_d, _dat.from_grid(_ddd, nodata=255), _err, _inf]

def _load_img(f, bnd=None, cache=None):
    _d = _ac_date(f)
    if _d is None:
        return None

    _f_inp = f
    _f_err = _err_file(_f_inp)

    if _f_err is None:
        return None

    if cache:
        _f = cache.put(_f_inp)
        _f_err = cache.put(_f_err)

    if bnd == None:
        return [_d, ge.open(_f_inp).get_band().cache(), \
                ge.open(_f_err).get_band().cache()]

    return [_d, ge.open(_f_inp).get_band().read_block(bnd), \
            ge.open(_f_err).get_band().read_block(bnd)]

def _add_err(d, bnd, v=20.0):
    import numpy as np
    from gio import config

    if config.getboolean('conf', 'compose_tcc', False) and config.getboolean('conf', 'calibrate_tcc', False):
        _dat = bnd.data
        _dat[_dat <= 100] = np.minimum(100, np.maximum(0, ((_dat.astype(np.float32)[_dat <= 100] - 12) * 30 / 18.0)))

    _err = None
    if bnd:
        _dat = np.empty((bnd.height, bnd.width), dtype=np.float32)
        _dat.fill(v)
        _err = bnd.from_grid(_dat, nodata=-9999)

    return [d, bnd, _err]

def _load_img_noerr(f, bnd=None, cache=None, e=20.0):
    _d = _ac_date(f)
    if _d is None:
        return None

    _f_inp = f

    if cache:
        _f = cache.put(_f_inp)

    if bnd is None:
        return _add_err(_d, ge.open(_f_inp).get_band().cache(), e)

    return _add_err(_d, ge.open(_f_inp).get_band().read_block(bnd), e)

def load_color(f):
    return ge.load_colortable(f)

def output_dates(msk, bbs, yys, chs=None, avg=None):
    cdef int _rows = msk.height, _cols = msk.width
    cdef int _row, _col, _d

    cdef float _v, _nodata = msk.nodata
    cdef float _e

    cdef int _y_v, _y_n
    cdef float _y_e

    cdef np.ndarray[np.uint8_t, ndim=2] _msk = msk.data

    cdef np.ndarray[np.uint8_t, ndim=2] _tcc
    cdef np.ndarray[np.float32_t, ndim=2] _err
    cdef np.ndarray[np.int16_t, ndim=2] _tum

    cdef np.ndarray[np.uint8_t, ndim=2] _num = None
    cdef np.ndarray[np.float32_t, ndim=2] _dif = None
    cdef np.ndarray[np.float32_t, ndim=2] _pro = None

    if chs is not None:
        _num = chs[0].data
        _dif = chs[1].data
        _pro = chs[2].data

    _ys = sorted(yys.keys()) if yys is not None else []

    # from gio import progress_percentage
    # _ppp = progress_percentage.progress_percentage(_rows)

    _compose_max = config.getboolean('conf', 'compose_max', False)
    logging.info('compose max: %s' % _compose_max)

    _ts1 = 0
    _ts2 = 0

    _pt_test = None
    # _pt_test = [53.13166569, 123.40479063]
    _fc_filter = config.getboolean('conf', 'apply_fc_filter', True)

    for _row in xrange(_rows):
        # _ppp.next()
        for _col in xrange(_cols):
            if _pt_test is not None:
                _x, _y = msk.to_location(_col, _row)
                _d = math.hypot(_y - _pt_test[0], _x - _pt_test[1])
                if _d > msk.cell_size * 2:
                    continue

            _v = _msk[_row, _col]
            if _v == _nodata:
                _ts1 += 1
                continue

            _i = _row * msk.width + _col
            if _i not in bbs:
                _ts2 += 1
                continue

            _m = bbs[_i]

            if len(_m) == 0:
                continue

            if _fc_filter:
                fc_time_series.fc_filtering(_m)

            if len(_m) == 0:
                continue

            # for _z in _m:
            # 	if _z.skip > 0:
            # 		continue

            # 	print '%s,%s,%s,%s,%s' % (_z.d.strftime('%Y-%m-%d'), _z.tcc, _z.err, _z.ch, _z.prob)

            if avg is not None:
                _a_vs, _a_es, _a_ns = fc_time_series.predict_mean(_m)
                _tcc = avg[0].data
                _y_v = _a_vs
                _tcc[_row, _col] = _y_v

                _err = avg[1].data
                _y_e = _a_es
                _err[_row, _col] = _y_e

                _tum = avg[2].data
                _y_n = _a_ns
                _tum[_row, _col] = _y_n

            if yys is not None:
                if _compose_max:
                    _a_vs, _a_es, _a_ns = fc_time_series.predict_years_max(_m, _ys)
                else:
                    _a_vs, _a_es, _a_ns = fc_time_series.predict_years_med(_m, _ys)

                # for _y in _ys:
                # 	print '%s-07-01,%s,%s' % (_y, _a_vs[_y], _a_es[_y])

                for _y in _ys:
                    if _a_vs[_y] >= 0:
                        _tcc = yys[_y][0].data
                        _y_v = _a_vs[_y]
                        _tcc[_row, _col] = _y_v

                        _err = yys[_y][1].data
                        _err[_row, _col] = _y_e

                        _tum = yys[_y][2].data
                        _y_n = _a_ns[_y]
                        _tum[_row, _col] = _y_n

            # _d_ch = tcc_time_series.detect_change(_m)
            # if _d_ch is None:
            # 	continue

            # _num[_row, _col] = _d_ch.d.year - 1990
            # _pro[_row, _col] = _d_ch.prob

            if chs is not None:
                _d_forest, _d_ch_loss, _d_ch_gain = fc_time_series.detect_change_annual(_ys, _a_vs, _a_es)

                if _d_ch_loss is not None:
                    _num[_row, _col] = _d_ch_loss[0] - 1970
                    _dif[_row, _col] = _d_ch_loss[1][0]
                    _pro[_row, _col] = _d_ch_loss[1][1]

    logging.info('skipped %s, %s' % (_ts1, _ts2))
    # _ppp.done()

def identify_p(ys, vs, es):
    _v_low = 40
    _v_top = 60

    _vs = [_v for _v in vs.values() if _v <= 100]
    if len(_vs) <= 1:
        return 0

    if min(_vs) > _v_top:
        return 1

    if max(_vs) < _v_low:
        return -1

    return 0

    # _y_min = min(ys)
    # _y_max = max(ys)

    # _v_min = vs[_y_min]
    # _v_max = vs[_y_max]
    # _v_med = sorted(vs.values())

    # if _v_min < _v_low and _v_max < _v_low and _v_med < _v_low:
    #     return -1

    # if _v_min > _v_top and _v_max > _v_top and _v_med > _v_top:
    #     return 1

    # return 0

def _load_layer(s, t, bnd):
    _f = config.get(s, t)
    if not _f:
        return None
        
    _bnd = gx.geo_band_stack_zip.from_shapefile(_f, \
            cache=None, extent=bnd)
            
    if _bnd is None:
        return None
        
    _bnd = _bnd.read_block(bnd)
    return _bnd

def init_grid(bs, rows, cols):
    _b_num = np.empty((rows, cols), dtype=np.uint8)
    _b_num.fill(255)

    _b_dif = np.empty((rows, cols), dtype=np.float32)
    _b_dif.fill(-9999)

    bs.append((_b_num, _b_dif))
    
def _linear_params(xs, ys):
    _mm = stats.linregress(xs, ys)
    return _mm.slope, _mm.pvalue #, _mm.rvalue

def produce_change_band(msk, bnds, errs, nums, b_reg, ys):
    cdef np.ndarray[np.uint8_t, ndim=2] _tcc
    cdef np.ndarray[np.float32_t, ndim=2] _err
    cdef np.ndarray[np.uint8_t, ndim=2] _wat
    cdef np.ndarray[np.int16_t, ndim=2] _nnn
    
    # slope and p-value
    cdef np.ndarray[np.float32_t, ndim=2] _slp
    cdef np.ndarray[np.float32_t, ndim=2] _pvl

    cdef int _rows = msk.height, _cols = msk.width
    cdef int _row, _col

    cdef float _v
    cdef float _e
    cdef int _y_n

    # forest prob
    cdef np.ndarray[np.uint8_t, ndim=2] _pro_age = np.empty((_rows, _cols), dtype=np.uint8)
    _pro_age.fill(255)

    _max_change_layers= config.getint('conf', 'max_change_layers', 3)
    cdef np.ndarray[np.uint8_t, ndim=2] _num_loss = np.empty((_rows, _cols), dtype=np.uint8)

    _gs_loss = []
    _gs_gain = []
    # _gs_age = []

    for _l in xrange(_max_change_layers):
        init_grid(_gs_loss, _rows, _cols)
        init_grid(_gs_gain, _rows, _cols)
        # init_grid(_gs_age, _rows, _cols)
        
    if b_reg:
        _slp = np.empty((_rows, _cols), dtype=np.float32)
        _pvl = np.empty((_rows, _cols), dtype=np.float32)
        
        _slp.fill(-9999)
        _pvl.fill(-9999)

    _bnd = gx.read_block(config.cfg.get('conf', 'land'), msk)

    if _bnd is None:
        return None

    _wfr = _load_layer('conf', 'water', msk)
    
    _lnd = (_bnd.data == 1)
    if _wfr is not None:
        _lnd = _lnd & (_wfr.data <= 30)

    _wat = (_lnd == False).astype(np.uint8)

    _ys = sorted(ys, reverse=True)

    _ppp = progress_percentage.progress_percentage(_rows)

    _debug = config.getboolean('conf', 'debug')
    # _bnd_bas = _load_layer('conf', 'tcc_data', msk)
            
    for _row in xrange(_rows):
        _ppp.next()

        for _col in xrange(_cols):
            _w = _wat[_row, _col]
            if _w == 1:
                continue

            _a_vs = {}
            _a_es = {}
            _a_ns = {}

            _a_ys = []

            for _y in _ys:
                _tcc = bnds[_y].data
                _a_v = _tcc[_row, _col]

                if _a_v > 100:
                    continue

                _a_ys.append(_y)
                _a_vs[_y] = _a_v

                _err = errs[_y].data
                _nnn = nums[_y].data

                _y_e = _err[_row, _col]
                _y_n = _nnn[_row, _col]
                # if 0 <= _y_e < 100:
                #     _y_e = 25 # min(20, math.hypot(_y_e, 16.83))

                _a_es[_y] = _y_e
                _a_ns[_y] = _y_n

                # print _y, _a_vs[_y], _a_es[_y], _a_ns[_y]

            # _v_ch = identify_p(_ys, _a_vs, _a_es, _a_ns)
            # if _v_ch != 0:
            #     continue
            if len(_a_ys) < 3:
                continue
            
            if b_reg:
                _v_slp, _v_pvl = _linear_params(_a_ys, [_a_vs[_y] for _y in _a_ys])
                
                _slp[_row, _col] = _v_slp
                _pvl[_row, _col] = _v_pvl

            _ts = fc_time_series.detect_change_annual(_a_ys, _a_vs, _a_es, _a_ns, _max_change_layers)
            if _ts is None:
                continue

            _d_forest, _d_ch_loss, _d_ch_gain = _ts
            _pro_age[_row, _col] = _d_forest

            if _d_forest > 100:
                continue

            if _debug:
                print('detected', _ts)

            for _i in xrange(_max_change_layers):
                # if _d_ch_loss is None or len(_d_ch_loss) == 0 or _i >= len(_d_ch_loss):
                #     break
                
                _d_loss = None
                _d_gain = None

                if _d_ch_loss is not None and _i < len(_d_ch_loss):
                    _d_loss = _d_ch_loss[_i]

                if _d_ch_gain is not None and _i < len(_d_ch_gain):
                    _d_gain = _d_ch_gain[_i]
                    
                gether_pixels(_d_forest, _row, _col, _d_loss, _d_gain, \
                        _gs_loss[_i], _gs_gain[_i])
                        # _gs_loss[_i], _gs_gain[_i], _gs_age[_i], _bnd_bas)

    _ppp.done()

    _bs_loss = to_band(msk, _gs_loss)
    _bs_gain = to_band(msk, _gs_gain)
    # _bs_esta = to_band(msk, _gs_age)
    
    _bs_regs = None
    if b_reg:
        _bs_regs = [msk.from_grid(_slp, nodata=-9999), msk.from_grid(_pvl, nodata=-9999)]

    return _bs_loss, _bs_gain, msk.from_grid(_pro_age, nodata=255), _bs_regs

def to_band(msk, gs):
    _bs = []
    for _g_dat, _g_dif in gs:
        _bs.append((msk.from_grid(_g_dat, nodata=255), \
                msk.from_grid(_g_dif, nodata=-9999)))
    return _bs

# def gether_pixels(d_forest, row, col, d_ch_loss, d_ch_gain, bs_loss, bs_gain, bs_age, bnd_bas):
def gether_pixels(d_forest, row, col, d_ch_loss, d_ch_gain, bs_loss, bs_gain):
    cdef int _row = row, _col = col
    cdef np.ndarray[np.uint8_t, ndim=2] _num_loss
    cdef np.ndarray[np.float32_t, ndim=2] _dif_loss

    cdef np.ndarray[np.uint8_t, ndim=2] _num_gain
    cdef np.ndarray[np.float32_t, ndim=2] _dif_gain

    cdef np.ndarray[np.uint8_t, ndim=2] _num_age
    cdef np.ndarray[np.float32_t, ndim=2] _dif_age

    cdef np.ndarray[np.uint8_t, ndim=2] _dat_bas

    _d_ch_loss = d_ch_loss
    _d_ch_gain = d_ch_gain

    _num_loss, _dif_loss = bs_loss
    _num_gain, _dif_gain = bs_gain
    # _num_age, _dif_age = bs_age

    _d_forest = d_forest

    if _d_ch_loss is not None:
        _num_loss[_row, _col] = _d_ch_loss[0] - 1970
        _dif_loss[_row, _col] = _d_ch_loss[1][0]
        # _pro_loss[_row, _col] = _d_ch_loss[1][1]

    if _d_ch_gain is not None:
        _num_gain[_row, _col] = _d_ch_gain[0] - 1970
        _dif_gain[_row, _col] = _d_ch_gain[1][0]
        # _pro_gain[_row, _col] = _d_ch_gain[1][1]

    return

    # _regrowth = 0
    # if _d_ch_gain is not None:
    #     if _d_ch_loss is not None and _d_ch_loss[0] >= _d_ch_gain[0]:
    #         pass
    #     else:
    #         _regrowth = 1
            
    # if _regrowth:
    #     _num_age[_row, _col] = _d_ch_gain[0] - 1970
    #     _dif_age[_row, _col] = _d_ch_gain[1][0]
    # else:
    #     if _d_forest >= 50:
    #         if _d_ch_loss is None:
    #             _num_age[_row, _col] = 0
    #             _dif_age[_row, _col] = min(100, abs(_d_forest - 50) * 2)
    #         else:
    #             _num_age[_row, _col] = _d_ch_loss[0] - 1970
    #             _dif_age[_row, _col] = _d_ch_loss[1][0]
    #     else:
    #         _num_age[_row, _col] = 100
    #         _dif_age[_row, _col] = min(100, abs(_d_forest - 50) * 2)

    # if bnd_bas is None:
    #     return
    
    # _dat_bas = bnd_bas.data
    # _v_bas = _dat_bas[_row, _col]
    
    # if _v_bas > 100:
    #     return
    
    # from gio import config
    # _tcc_bas = config.getint('conf', 'min_tcc', 30)
    # if _v_bas < _tcc_bas:
    #     if _num_age[_row, _col] < 100:
    #         _num_age[_row, _col] = 100
    #         _dif_age[_row, _col] = min(100, abs(_d_forest - 50) * 2)
            
    #     if _num_gain[_row, _col] < 100:
    #         if _num_gain[_row, _col] < _num_loss[_row, _col] < 100:
    #             pass
    #         else:
    #             _num_gain[_row, _col] = 255
        
    # else:
    #     _num_age[_row, _col] = 0
    
def val_medium(vs):
    if len(vs) == 0:
        raise Exception('no value found')

    _vs = sorted(vs)
    return _vs[len(_vs) / 2]

def _fill_sub(b_col, b_row, block_size, yys, chs, avg, tag, bnd, fs, ys, v_max, met, fzip):
    _b_width = min(bnd.width - b_col, block_size)
    _b_height = min(bnd.height - b_row, block_size)

    _b_bnd = bnd.subset(b_col, b_row, \
            _b_width, _b_height)

    # _pt_test = [53.13166569, 123.40479063]
    # from gio import geo_base as gb
    # if not _b_bnd.extent().contains(gb.geo_point(_pt_test[1], _pt_test[0], gb.proj_from_epsg())):
    #    return

    _rs = _combine_bnd_sub(tag, _b_bnd, fs, ys, v_max, met, fzip)
    
    if _rs is None:
        return
    
    _b_yys, _b_chs, _b_avg, _fs = _rs
    
    if yys is not None and _b_yys is not None:
        for _y in _b_yys:
            yys[_y][0].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_yys[_y][0].data
            yys[_y][1].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_yys[_y][1].data
            yys[_y][2].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_yys[_y][2].data

    if chs is not None and _b_chs is not None:
        chs[0].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_chs[0].data
        chs[1].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_chs[1].data
        chs[2].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_chs[2].data

    if avg is not None and _b_avg is not None:
        avg[0].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_avg[0].data
        avg[1].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_avg[1].data
        avg[2].data[b_row: b_row+_b_height, b_col: b_col + _b_width] = _b_avg[2].data

    del _b_bnd
    del _b_yys
    del _b_chs

def _combine_bnd_sub(tag, bnd, fs, ys, v_max, met, fzip):
    _met = met
    _cache = None

    _cache_tag = config.get('conf', 'cache_tag', None)
    if _cache_tag:
        _cache = cache_mag.cache_mag(_cache_tag)
        logging.warning('cache is enabled (%s)' % _cache_tag)

    _bnd = gx.read_block(config.get('conf', 'land'), bnd)
    if _bnd is None:
        logging.info('found no land')
        return None

    _lnd = (_bnd.data == 1)
    
    _wfr = gx.read_block(config.get('conf', 'water', None), bnd)
    if _wfr is not None:
        _wfr = _wfr.read_block(bnd)
        if _wfr is not None:
            _lnd = _lnd & (_wfr.data <= 30)

    _wat = (_lnd == False)

    _bnd.nodata = 0
    _bnd.data[_lnd] = 254
    _bnd.data[_wat] = 0

    _avg = None

    if config.getboolean('conf', 'predict_mean', True):
        _tcc = np.empty([_bnd.height, _bnd.width], dtype=np.uint8)
        _tcc.fill(255)
        _tcc[_wat] = 200

        _der = np.empty([_bnd.height, _bnd.width], dtype=np.float32)
        _der.fill(-9999)
        _der[_wat] = 0.0

        _num = np.empty([_bnd.height, _bnd.width], dtype=np.int16)
        _num.fill(-9999)
        _num[_wat] = 0.0

        _avg = (_bnd.from_grid(_tcc, nodata=255), \
                _bnd.from_grid(_der, nodata=-9999), \
                _bnd.from_grid(_num, nodata=-9999))

    _yys = None

    if config.getboolean('conf', 'predict_years', False):
        _yys = {}
        for _y in ys:
            _tcc = np.empty([_bnd.height, _bnd.width], dtype=np.uint8)
            _tcc.fill(255)
            _tcc[_wat] = 200

            _der = np.empty([_bnd.height, _bnd.width], dtype=np.float32)
            _der.fill(-9999)
            _der[_wat] = 0.0

            _num = np.empty([_bnd.height, _bnd.width], dtype=np.int16)
            _num.fill(-9999)
            _num[_wat] = 0.0

            _yys[_y] = (_bnd.from_grid(_tcc, nodata=255), \
                    _bnd.from_grid(_der, nodata=-9999), \
                    _bnd.from_grid(_num, nodata=-9999))

    _chs = None

    if False: #config.getboolean('conf', 'detect_change', False):
        _dnu = np.empty([_bnd.height, _bnd.width], dtype=np.uint8)
        _dnu.fill(255)
        _dnu[_wat] = 0
        _num = _bnd.from_grid(_dnu, nodata=255)

        _der = np.empty([_bnd.height, _bnd.width], dtype=np.float32)
        _der.fill(-9999)
        _der[_wat] = 1.0
        _err = _bnd.from_grid(_der, nodata=-9999)

        _chs = [_num, _err]

    _fs = []
    if _lnd.sum() > 0 and len(fs) > 0:
        _bbs = {}
        _mods = None

        if v_max > 0:
            _bbb = fc_layer_mag.load_refer_forest(_bnd, 0)
            _mods = {}
            for _y in ys:
                _mods[_y] = _bbb

        _ls = []

        for _i in xrange(0, len(fs)):
            _ffs = load_img(fs[_i], _bnd, _cache)
            if _ffs is None:
                logging.debug('skip %s' % fs[_i])
                continue

            if update_date(_bbs, _bnd, _ffs, _mods, v_max):
                _ls.append('%s=%s' % (_i, fs[_i]))
                _fs.append(fs[_i])
            else:
                logging.debug('failed %s' % fs[_i])

        logging.debug('found %s images' % len(_ls))

        if len(_fs) > 0:
            output_dates(_bnd, _bbs, _yys, _chs, _avg)

        if _mods:
            del _mods
        del _bbs
        del _bnd

    return _yys, _chs, _avg, _fs

