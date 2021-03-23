

def _default_err(bnd, dval):
    from gio import geo_raster as ge
    import numpy as np

    _def = dval
    _dat = np.empty((bnd.height, bnd.width), dtype=np.float32)
    _dat.fill(_def)

    _err = bnd.from_grid(_dat, nodata=-9999)
    _err.pixel_type = ge.pixel_type('float')

    return _err

def _load_tree_cover(f_dat, msk, bnd):
    from gio import geo_raster_ex as gx
    from gio import geo_raster as ge
    from gio import config
    import numpy as np
    from gio import file_mag
    from gio import agg_band

    _def = config.getfloat('conf', 'avg_tcc_err', 16.0)

    _lyr = gx.geo_band_stack_zip.from_shapefile(file_mag.get(f_dat).get(), extent=bnd)

    _aaa = _lyr.read_block(msk)
    _ttt = _lyr.read_block(bnd)
    _out = agg_band.mean(_aaa, bnd)
    _idx = (_out.data <= 100) & (_ttt.data > 100)
    _out.data[_idx] = _ttt.data[_idx]

    _tcc = _out
    _err = _default_err(bnd, _def)

    from libtm import mod_prob
    _bnd = mod_prob.prob(_tcc, _err, config.getint('conf', 'min_tcc', 30))

    _idx = (_bnd.data <= 1.0) & (_bnd.data >= 0.0)
    _bnd.data[_idx] = (1.0 - _bnd.data[_idx]) * 100.0
    _bnd.data[_idx == False] = 255
    _bnd.data[_tcc.data >= 200] = _tcc.data[_tcc.data >= 200]

    _bnd.pixel_type = ge.pixel_type()
    _bnd.nodata = 255
    _bnd.data = _bnd.data.astype(np.uint8)

    return _bnd

def _load_forest_prob(f):
    if not f.endswith('pob.tif'):
        raise Exception('only support prob file type')

    from gio import geo_raster as ge
    _bnd = ge.open(f).get_band()

    from gio import agg_band
    _msk = _bnd.scale(_bnd.cell_size / 500.0)
    _ttt = _bnd.read_block(_msk)
    _out = agg_band.mean(_bnd.cache(), _msk)
    _idx = (_out.data <= 100) & (_ttt.data > 100)
    _out.data[_idx] = _ttt.data[_idx]

    return _bnd, _out

def match(f, f_ref, f_img=None):
    from gio import geo_raster as ge
    from gio import agg_band

    _org, _bnd = _load_forest_prob(f)
    if _bnd is None:
        return f, -1
        
    _ref = _load_tree_cover(f_ref, _org, _bnd)
    if _ref is None:
        return None

    from libtm import mod_grid
    _mmm = mod_grid.match_prob(_bnd, _ref)

    if f_img:
        import os

        _tag = os.path.basename(f)[:-4]
        _d_out = os.path.join(f_img, _tag)

        from gio import config

        _f_clr = config.get('conf', 'forest_prob_color')
        if _f_clr:
            _clr = ge.load_colortable(_f_clr)
            
            _bnd.color_table = _clr
            _ref.color_table = _clr

        from gio import file_unzip
        with file_unzip.file_unzip() as _zip:
            _zip.save(_bnd, os.path.join(_d_out, _tag + '_dat.tif'))
            _zip.save(_ref, os.path.join(_d_out, _tag + '_ref.tif'))
            _zip.save(str(_mmm), os.path.join(_d_out, _tag + '_met.txt'))

    _log = mod_grid._log_weight
    _nlog = lambda x: (1.0 - mod_grid._log_weight(1.0 - x))

    from gio import config
    _max_diff = config.getfloat('conf', 'max_diff', 80.0)
    _r = _log(_mmm[0]) * (1.0 - _log(min(_max_diff, abs(_mmm[2])) / _max_diff))
    
    _ss = dict(zip('rate,perc,rmse,mae'.split(','), list(map(lambda x: round(x, 3), [_r] + list(_mmm)))))
    return _ss

def _load_prob(f):
    from gio import geo_raster as ge

    _bnd = ge.open(f).get_band().cache()
    if not f.endswith('pob.tif'):
        return _bnd

    _dat = _bnd.data

    _rat = float((_dat < 200).sum()) / float((_dat < 255).sum())
    if _rat < 0.15:
        return None

    import numpy as np
    _ddd = np.empty((_bnd.height, _bnd.width), dtype=np.uint8)

    _ddd[_dat < _bnd.nodata] = 3
    _ddd[_dat <= 50] = 9
    _ddd[_dat == 200] = 4
    _ddd[(_dat > 50) & (_dat <= 100)] = 1

    return _bnd.from_grid(_ddd)

def _load_tcc_mean(f, bnd):
    import logging

    if not f: return None

    from gio import geo_raster_ex as gx

    logging.info('loading data: %s' % f)
    _bnd = gx.geo_band_stack_zip.from_shapefile(f, extent=bnd)

    from gio import agg_band
    _bbb = _bnd.read_block(bnd.scale(bnd.geo_transform[1] / 30))

    logging.info('loaded refer data')
    _out = agg_band.mean(_bbb, bnd, 0, 100)
    if _out is None:
        return None

    import numpy as np
    _ooo = np.zeros((_out.height, _out.width), dtype=np.uint8)

    from gio import config
    _tcc = config.getint('conf', 'min_tcc', 30)

    _ooo[_out.data < (_tcc - 0)] = 9
    _ooo[(_out.data <= 100) & (_out.data >= (_tcc + 0))] = 1

    _msk = agg_band.dominated(_bbb, bnd, True)
    _ooo[_msk.data == 200] = 4

    logging.info('converted refer data')
    return _out.from_grid(_ooo, nodata=0)

