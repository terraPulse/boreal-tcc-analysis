import numpy as np
import logging
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

class scene_weight:

    def __init__(self):
        self.ws = None
        self._load_ws()

    def _load_ws(self):
        from gio import config
        import os

        _f_w = config.get('conf', 'scene_weight')

        if not _f_w:
            return 

        _ws = {}

        if os.path.isfile(_f_w):
            self._load_weight_file(_f_w, _ws)
        else:
            for _root, _dirs, _files in os.walk(_f_w):
                for _file in _files:
                    if os.path.splitext(_file)[1] in ['.txt', '.csv']:
                        self._load_weight_file(os.path.join(_root, _file), _ws)

        self.ws = _ws

    def _load_weight_file(self, f, ws):
        from gio import csv_util

        logging.debug('load weight file %s' % f)
        for _r in csv_util.open(f):
            ws[_r.get('id')] = float(_r.get('rate'))

    def get(self, p):
        if self.ws and str(p) in self.ws:
            return self.ws[p]

        return None

class mss_image:

    def __init__(self, f, ws):
        from gio import landsat
        self.f = f
        self.p = landsat.parse(f)
        self.dif = 100.0
        self.dif_c = 100.0
        self.avg = -1
        self.ref_avg = -1
        self.ws = ws

    def load(self, ref):
        _mmm = self.ws.get(self.p)

        if _mmm is None:
            if ref is None:
                return 100.0

            import numpy.ma as ma
            from gio import config
            from gio import geo_raster as ge

            _bnd = ge.open(self.f).get_band().read_block(ref)

            if _bnd is None:
                return 100.0

            from libtm import mod_grid
            _mmm = mod_grid.match(_bnd, ref)

        self.avg = -1
        self.ref_avg = -1

        self.dif = 1- _mmm
        self.dif_c = 1 - _mmm

        return self.dif

    def __eq__(self, o):
        return self.dif_c == o.dif_c
        
    def __ne__(self, o):
        return self.dif_c != o.dif_c
        
    def __lt__(self, o):
        return self.dif_c < o.dif_c
        
    def __le__(self, o):
        return self.dif_c <= o.dif_c
        
    def __gt__(self, o):
        return self.dif_c > o.dif_c
        
    def __ge__(self, o):
        return self.dif_c >= o.dif_c

def update(bbs, bs1, bs2, wet, ref, idx):
    import params

    # bs1[1].data[bs1[0].data == params.VAL_WATER] = 0.50
    # bs2[1].data[bs2[0].data == params.VAL_WATER] = 0.50

    # bs1[1].data[bs1[0].data == params.VAL_CLOUD] = 0.50
    if None in bs2:
        return

    if type(None) in [type(_b.data) for _b in bs2[1:]]:
        return

    if ref is not None:
        import numpy.ma as ma

        _dat = bs2[1].data
        _ref = ref.data

        _mask1 = _dat == bs2[1].nodata
        _mask2 = ((_ref != 1) & (_ref != 9))
        _mask = _mask1 | _mask2

        _tcc_avg = ma.masked_array((_dat == 1).astype(np.float32), mask=_mask).mean()
        _v_max = 0.4
        _v_min = -0.2

        _mod_avg = ma.masked_array((_ref == 1).astype(np.float32), mask=_mask).mean()
        _dif = _tcc_avg - _mod_avg

        logging.info('%s, %s = %s - %s' % (bs2[0], _dif, _tcc_avg, _mod_avg))
        if _dif > _v_max or _dif < _v_min:
            logging.info('skip %s, %s' % (bs2[0], _dif))
            return

    from gio import config

    _w_cloud = config.getfloat('conf', 'weight_cloud', 0.25)
    _w_water = config.getfloat('conf', 'weight_water', 0.75)

    bs2[2].data[bs2[1].data == params.VAL_CLOUD] = _w_cloud
    bs2[2].data[bs2[1].data == params.VAL_WATER] = _w_water

    # bs1[0].data[bs1[0].data == params.VAL_CLOUD] = params.VAL_NONFOREST
    # bs2[0].data[bs2[0].data == params.VAL_CLOUD] = params.VAL_NONFOREST

    # import mod_grid
    # _dat = (bs2[1].data > bs1[1].data) & (bs1[0].data != bs1[0].nodata)
    # mod_grid.update(bs1[0].data, _dat, bs2[0].data)
    # mod_grid.update(bs1[1].data, _dat, bs2[1].data)

    _b1_bnd = bs1[1]
    # _b1_err = bs1[2]

    _b2_bnd = bs2[1]
    _b2_err = bs2[2]

    cdef int _rows = _b1_bnd.height, _cols = _b1_bnd.width
    cdef int _row, _col, _v1, _v2, _nodata = _b1_bnd.nodata
    cdef float _e

    cdef np.ndarray[np.uint8_t, ndim=2] _dat1 = _b1_bnd.data
    cdef np.ndarray[np.uint8_t, ndim=2] _dat2 = _b2_bnd.data

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

            if _v2 not in _m:
                _m[_v2] = [1, _e, _e * wet, idx]
            else:
                _m[_v2][0] += 1
                _m[_v2][1] += _e
                _m[_v2][2] += _e * wet

def output(bbs, bs1):
    import params
    from gio import config

    _b_bnd = bs1[0]
    _b_err = bs1[1]
    _b_idx = bs1[2]
    _b_per = bs1[3]
    _b_num = bs1[4]

    cdef int _rows = _b_bnd.height, _cols = _b_bnd.width
    cdef int _row, _col, _v, _d, _nodata = _b_bnd.nodata, _v_mak
    cdef float _e
    cdef np.ndarray[np.uint8_t, ndim=2] _dat = _b_bnd.data
    cdef np.ndarray[np.uint8_t, ndim=2] _idx = _b_idx.data
    cdef np.ndarray[np.uint8_t, ndim=2] _per= _b_per.data
    cdef np.ndarray[np.uint8_t, ndim=2] _num= _b_num.data
    cdef np.ndarray[np.float32_t, ndim=2] _err = _b_err.data
    cdef np.ndarray[np.uint8_t, ndim=2] _mak = np.zeros((_b_bnd.height, _b_bnd.width), dtype=np.uint8)

    _f_wat = config.get('conf', 'water_freq_mask')
    if _f_wat:
        from gio import geo_raster_ex as gx
        _b_wat = gx.geo_band_stack_zip.from_shapefile(_f_wat).read_block(_b_bnd)
        if _b_wat:
            _mak[(_b_wat.data > 50) & (_b_wat.data <= 200)] = 200

    for _row in xrange(_rows):
        for _col in xrange(_cols):
            _v_mak = _mak[_row, _col]
            if _v_mak > 0:
                _dat[_row, _col] = _v_mak
                _per[_row, _col] = _v_mak
                continue

            _v = _dat[_row, _col]
            if _v == _nodata:
                continue

            _i = _row * _b_bnd.width + _col
            if _i not in bbs:
                continue

            _m = bbs[_i]

            _t = 0
            _e = 0
            _d = _b_idx.nodata

            _n = -100

            _va = 0.0
            _vf = 0.0
            _vw = 0.0

            _vn = 0
            for _k, _w in _m.items():
                # if (_w[1] / _w[0]) > _n:
                if _w[2] > _n:
                    _t = _k
                    _n = _w[2]
                    _e = _w[1] / _w[0]
                    _d = _w[3]


                if _k == 1:
                    _vf += _w[2]
                if _k == 4:
                    _vw += _w[2]
                _va += _w[2]
                _vn += _w[0]

            _dat[_row, _col] = _t
            _err[_row, _col] = _e
            _idx[_row, _col] = _d

            _v_per = 255
            if _va > 0:
                _v_per = _vf / _va
                _v_per = min(1.0 ,_v_per) * 100.0

                _v_wat = _vw * 100 / _va

                if _v_per < 30 and _v_wat > 30:
                    _v_per = 200
                
            _per[_row, _col] = int(_v_per)
            _num[_row, _col] = min(100, _vn)

def _load_ext(tile):
    from osgeo import ogr
    from gio import geo_base as gb
    from gio import config

    _shp = ogr.Open(config.cfg.get('conf', 'wrs'))
    _lyr = _shp.GetLayer()

    for _f in _lyr:
        _t = _f['PATHROW'].strip()
        if _t == tile:
            return gb.geo_polygon(_f.geometry().Clone())

    return None

def _make_ext(prj, cell, tile, fzip):
    from gio import config

    _ext = _load_ext(tile)
    if _ext == None:
        return None

    _ext = _ext.project_to(prj)

    _dis = config.cfg.getfloat('conf', 'buffer')
    if _dis > 0:
        _ext.poly = _ext.poly.Buffer(_dis, 1)

    _box = _ext.extent()

    from gio import geo_raster as ge
    _bnd = ge.geo_band_info([_box.minx, cell, 0, _box.maxy, 0, -cell], \
            int(_box.width()/cell), int(_box.height()/cell),  prj)

    from gio import rasterize_band
    _out = fzip.generate_file('', '.img')

    rasterize_band.rasterize_polygon(_bnd, _ext, _out, fzip.generate_file('', '.shp'))
    _bnd = ge.open(_out).get_band().cache()
    _bnd.nodata = 0
    _bnd.data[_bnd.data == 1] = 255

    return _bnd

def load_img(f, bnd=None):
    from gio import geo_raster as ge
    import os

    if not (f and os.path.exists(f)):
        return None, None

    _f_err = f.replace('_dat.tif', '_err.tif')

    if not os.path.exists(_f_err):
        return None, None

    if bnd == None:
        return f, ge.open(f).get_band().cache(), \
                ge.open(_f_err).get_band().cache()
    else:
        return f, ge.open(f).get_band().read_block(bnd), \
            ge.open(_f_err).get_band().read_block(bnd)

def load_color(f):
    from gio import geo_raster as ge
    return ge.load_colortable(f)

    # _bnd = ge.open(f).get_band()
    # return _bnd.color_table

def combine_bnd(tag, bnd, fs, f_clr, d_out):
    from gio import file_unzip

    with file_unzip.file_unzip() as _zip:
        return _combine_bnd(tag, bnd, fs, f_clr, d_out, _zip)

def _load_tcc_mean(f, bnd, fzip):
    if not f: return None

    from gio import geo_raster_ex as gx

    logging.info('loading data: %s' % f)
    _bnd = gx.geo_band_stack_zip.from_shapefile(f, file_unzip=fzip)

    from gio import agg_band
    _bbb = _bnd.read_block(bnd.scale(bnd.geo_transform[1] / 30))
    _out = agg_band.mean(_bbb, bnd, 0, 100)

    import numpy as np
    _ooo = np.zeros((_out.height, _out.width), dtype=np.uint8)

    _ooo[_out.data < 30] = 9
    _ooo[(_out.data <= 101) & (_out.data >= 30)] = 1

    _msk = agg_band.dominated(_bbb, bnd, True)
    _ooo[_msk.data == 200] = 4

    return _out.from_grid(_ooo, nodata=0)

def _load_forest_layer(bnd, fzip):
    from gio import file_unzip

    _f_tcc = '/data/glcf-nx-003/fengm/tcc/global/v05/comp/2015/comp_etm_oli/list/tcc_global_2015.shp'

    with file_unzip.file_unzip() as _zip:
        from gio import geo_raster_ex as gx
        _bnd = _load_tcc_mean(_f_tcc, bnd, fzip)

        return _bnd

def _combine_bnd(tag, bnd, fs, f_clr, d_out, fzip):
    if len(fs) == 0:
        return

    import os
    from gio import metadata

    _met = metadata.metadata()

    from gio import geo_raster_ex as gx
    from gio import config
    from gio import landsat

    _bnd = gx.geo_band_stack_zip.from_shapefile(config.cfg.get('conf', 'land'), file_unzip=fzip).read_block(bnd)
    if _bnd == None:
        return

    _bnd.nodata = 0
    _bnd.data[_bnd.data == 1] = 255

    import numpy as np

    _der = np.empty([_bnd.height, _bnd.width], dtype=np.float32)
    _der.fill(-9999)
    _err = _bnd.from_grid(_der, nodata=-9999)

    _idx = np.empty([_bnd.height, _bnd.width], dtype=np.uint8)
    _idx.fill(255)
    _ibn = _bnd.from_grid(_idx, nodata=255)

    _per = np.empty([_bnd.height, _bnd.width], dtype=np.uint8)
    _per.fill(255)
    _pen = _bnd.from_grid(_per, nodata=255)

    _num = np.empty([_bnd.height, _bnd.width], dtype=np.uint8)
    _num.fill(255)
    _nun = _bnd.from_grid(_num, nodata=255)

    from gio import progress_percentage

    import collections
    _bbs = collections.defaultdict(lambda: {})

    _ref = _load_forest_layer(_bnd, fzip)
    _v_max = config.getfloat('conf', 'diff_max', 0.3)
    _v_min = config.getfloat('conf', 'diff_min', -0.25)

    _ppp = progress_percentage.progress_percentage(len(fs))

    _ccc_img = []
    _res = []

    _msk = _bnd.scale(0.3)
    _rrr = _ref.read_block(_msk)

    _wss = scene_weight()
    for _f in fs:
        _ppp.next()

        _ip = landsat.parse(_f)

        _mss = mss_image(_f, _wss)
        _dif = _mss.load(_rrr)

        _met['input']['check'][(str(_ip))] = {'file': _f, 'diff': _dif}

        if _dif > _v_max:
            continue

        # if (_v_min > _dif) or (_mss.dif > _v_max):
        #     continue

        # if _mss.avg < 0:
        #     continue

        # _ref_avg = max(0.01, _mss.ref_avg)
        # if (_ref_avg * _v_min) <= _mss.dif <= (_v_max * _ref_avg):

        _ccc_img.append({'file': _f, 'perc': _mss.avg, 'perc_ref': _mss.ref_avg, 'diff': _mss.dif})
        _res.append(_mss)
    
    _met['input']['loaded'] = _ccc_img
    _ppp.done()

    _n_max = config.getint('conf', 'max_file_num', 25)

    _tss = {}
    for _r in sorted(_res):
        _ip = landsat.parse(_r.f)
        if _ip.tile not in _tss:
            _tss[_ip.tile] = []

        if len(_tss[_ip.tile]) >= _n_max:
            continue
        _tss[_ip.tile].append(_r)

    _fs = []
    for _k in _tss.keys():
        _fs.extend(_tss[_k])

    logging.info('selected %s files from %s' % (len(_fs), len(fs)))

    _ppp = progress_percentage.progress_percentage(len(_fs))
    import math

    _ls = []
    for _i in xrange(0, len(_fs)):
        _ppp.next()
        
        _wet = math.cos(_fs[_i].dif_c * math.pi / 2)
        if _wet <= 0:
            continue

        update(_bbs, (None, _bnd, None), load_img(_fs[_i].f, _bnd), _wet, None, _i)
        _ls.append('%s=%s' % (_i, fs[_i]))

        _mss = _fs[_i]
        _met['input']['selected'][_i] = {'file': _mss.f, 'weight': _wet, \
                'perc': _mss.avg, 'perc_ref': _mss.ref_avg, 'diff': _mss.dif}

    _ppp.done()

    output(_bbs, (_bnd, _err, _ibn, _pen, _nun))

    os.path.exists(d_out) or os.makedirs(d_out)

    _f_out = os.path.join(d_out, '%s_com_dat.tif' % tag)
    _f_err = os.path.join(d_out, '%s_com_err.tif' % tag)
    _f_idx = os.path.join(d_out, '%s_com_idx.tif' % tag)
    _f_txt = os.path.join(d_out, '%s_com_idx.txt' % tag)
    _f_met = os.path.join(d_out, '%s_com_met.txt' % tag)
    _f_ref = os.path.join(d_out, '%s_com_ref.tif' % tag)
    _f_pen = os.path.join(d_out, '%s_com_pen.tif' % tag)
    _f_nun = os.path.join(d_out, '%s_com_num.tif' % tag)

    _met['output'] = _f_out

    _bnd.save(_f_out, color_table=load_color(f_clr))
    _ibn.save(_f_idx)
    _err.save(_f_err)
    _ref.save(_f_ref, color_table=load_color(f_clr))

    f_clr_per = '/data/glcf-st-004/data/workspace/fengm/serv/map/tcc_2015_utah_61/color.txt'
    _pen.save(_f_pen, color_table=load_color(f_clr_per))

    f_clr_per = '/data/glcf-st-004/data/workspace/fengm/serv/map/tcc_2015_utah_61/color.txt'
    # _nun.save(_f_nun)
    _nun.save(_f_nun, color_table=load_color(f_clr_per))

    _f_old_src = config.get('conf', 'old_version', None)
    if _f_old_src:
        _f_old = os.path.join(d_out, '%s_com_old.tif' % tag)
        gx.geo_band_stack_zip.from_shapefile(_f_old_src, file_unzip=fzip).read_block(_bnd).save(_f_old, \
                color_table=load_color(f_clr))

    _met.save(_f_met)

    with open(_f_txt, 'w') as _fo:
        _fo.write('\n'.join(_ls))

