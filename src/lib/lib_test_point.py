'''
File: lib_test_point.py
Author: Min Feng
Description: provide the function for testing one single point
'''

import logging

def _grid_v(msk, v, t, nodata=None):
    import numpy as np
    
    _nnn = np.empty((msk.height, msk.width), dtype=t)
    _nnn.fill(v)
    
    _bnd = msk.from_grid(_nnn)
    if nodata is not None:
        _bnd.nodata = nodata
        
    return _bnd
    
def pixel_band(pt):
    from gio import geo_base as gb
    from gio import geo_raster as ge
    
    _proj = gb.modis_projection()
    
    _pt = pt.project_to(_proj)
    _geo = [_pt.x, 30, 0, _pt.y, 0, -30]
    
    return ge.geo_band_info(_geo, 1, 1, _proj)
    
def check_land(f, bnd):
    from gio import geo_raster_ex as gx

    _bnd = gx.geo_band_stack_zip.from_shapefile(f).read_block(bnd)
    if (_bnd.data == 1).sum() > 0:
        return True

    return False

def file_exists(f):
    from gio import file_mag
    return file_mag.get(f).exists()

def check_exists(d, tag, ys, ch, check_years=False):
    import os

    if file_exists(os.path.join(d, '%s_met.txt' % tag)):
        return []

    return ys

def filter_years(fs, ys):
    from gio import landsat
    from gio import file_mag

    _ls = []
    _ys = []

    for _y in ys:
        for _i in range(-1, 2):
            _n = _y + _i
            if _n not in _ys:
                _ys.append(_n)

    for _f in fs:
        _p = landsat.parse(_f)
        if not _p:
            continue

        if not file_mag.get(_f).exists():
            continue

        if _p.ac_date_obj.year not in _ys:
            continue

        _ls.append(_f)

    return _ls


def process_pt(pt, f_inp, ys, sys, daily_records, detect_change, met, fzip):
    from gio import config
    from gio import geo_base as gb
    from gio import geo_raster as ge
    from gio import landsat
    
    _tag = 'p001r001'
    _msk = pixel_band(pt)

    if not check_land(config.get('conf', 'land'), _msk):
        logging.info('skip tile (%s) because no land is identified' % _tag)
        return
    
    _fs = [_a['FILE'] for _, _a in gb.load_shp(f_inp, _msk.extent().to_polygon())]
    
    _ts = {}
    
    if daily_records:
        for _f in _fs:
            _p = landsat.parse(_f)
            _b = ge.open(_f).get_band().read_block(_msk)

            if config.getboolean('conf', 'compose_tcc', False) and config.getboolean('conf', 'calibrate_tcc', False):
                import numpy as np
                _dat = _b.data
                _dat[_dat <= 100] = np.minimum(100, np.maximum(0, ((_dat.astype(np.float32)[_dat <= 100] - 12) * 30 / 18.0)))

            _v = _b.read_cell(0, 0)
            if _v <= 100:
                _ts[_p.ac_date_obj] = _v
    
    met.num_loaded_images = len(_fs)
    if len(_fs) == 0:
        logging.warning('skip %s because no images were found' % _tag)
        return

    _sy = list(ys)
    if sys:
        for _y in sys:
            if _y not in _sy:
                _sy.append(_y)

    logging.info('found %s images for tile %s (years %s)' % (len(_fs), _tag, ', '.join(map(str, _sy))))

    from libtm import fc_composite
    from gio import file_unzip
    
    _yys, _chs, _avg = fc_composite._combine_bnd(_tag, _msk, _fs, ys, -1, met, fzip)
    
    _pob_bnds = {}
    _err_bnds = {}
    _num_bnds = {}
    
    _pob_vs = {}
    _loss_vs = []
    _gain_vs = []
    
    from libtm import mod_prob
    import numpy as np
    
    for _y, _b in _yys.items():
        _err_bnd = _grid_v(_msk, config.getfloat('conf', 'avg_tcc_err', 35.0), np.float32, -9999)
        _pro = mod_prob.prob(_b[0], _err_bnd, config.getint('conf', 'min_tcc', 30)).read_cell(0, 0)
        if _pro < 0:
            _pob_bnds[_y] = _grid_v(_msk, 255, np.uint8)
            _err_bnds[_y] = _grid_v(_msk, -9999, np.float32)
        else:
            _pob = (1.0 - _pro) * 100.0
            _pob_vs[_y] = _pob
            _pob_bnds[_y] = _grid_v(_msk, _pob, np.uint8)
            _err_bnds[_y] = _err_bnd
            
        _num_bnds[_y] = _grid_v(_msk, 8, np.int16)
        
    # if config.getboolean('conf', 'detect_change', True):
    if detect_change:
        from libtm import fc_annual_agg
        _loss, _gain, _prob, _stat = fc_annual_agg.produce_change_band(_msk, _pob_bnds, _err_bnds, \
                _num_bnds, False, sorted(_pob_bnds.keys()))
    
        if _loss:
            for _i in range(len(_loss)):
                _loss_v = _loss[_i][0].read_cell(0, 0)
                if _loss_v < 100:
                    _loss_vs.append((_loss_v, _loss[_i][1].read_cell(0, 0)))
                
        if _gain:
            for _i in range(len(_gain)):
                _gain_v = _gain[_i][0].read_cell(0, 0)
                if _gain_v < 100:
                    _gain_vs.append((_gain_v, _gain[_i][1].read_cell(0, 0)))
    
    _vs = {}
    for _y, _b in _yys.items():
        _vs[_y] = (_b[0].read_cell(0, 0), round(_b[1].read_cell(0, 0), 2), _b[2].read_cell(0, 0))
        
    return _ts, _vs, _pob_vs, _loss_vs, _gain_vs

def pt(pt, f_inp, ys, sys, daily_records=True, detect_change=True):
    import os
    from gio import file_unzip
    from gio import config
    from gio import file_mag
    from gio import metadata

    _met = metadata.metadata()
    with file_unzip.zip() as _zip:
        return process_pt(pt, f_inp, ys, sys, daily_records, detect_change, _met, _zip)
        