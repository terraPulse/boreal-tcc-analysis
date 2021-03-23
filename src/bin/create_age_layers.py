'''
File: detect_forest_change.py
Author: Min Feng
Version: 0.1
Create: 2018-04-20 15:42:37
Description: detect forest changes from foest probility layers and tree cover layers
'''

import logging

def _load_tcc(f_tcc, msk):
    from gio import geo_raster_ex as gx
    from gio import config
    import numpy as np

    _bnd = gx.read_block(f_tcc, msk)
    if _bnd is None:
        return None
        
    _dat = np.zeros(msk.data.shape, dtype=np.uint8)

    _m_tcc = config.getfloat('conf', 'min_tcc')
    _idx = _bnd.data >= _m_tcc
    _dat[_idx] = 100

    _idx = _bnd.data > 100
    _dat[_idx] = _bnd.data[_idx]

    return msk.from_grid(_dat, nodata=255)

def _task(tile, d_out, d_ref, opts):
    from gio import file_unzip
    from gio import config
    from gio import file_mag
    from gio import metadata
    from gio import geo_raster as ge
    from gio import geo_raster_ex as gx
    from gio import mod_filter
    import numpy as np
    import os
    import re

    _tag = tile.tag

    _ttt = config.get('conf', 'test_tile')
    if _ttt and _tag not in _ttt.replace(' ', '').split(','):
        return

    _m = re.match(r'(h\d+)(v\d+)', _tag)
    _h = _m.group(1)
    _v = _m.group(2)
    
    _d_out = os.path.join(d_out, _h, _v, _tag)
    _d_ref = os.path.join(d_ref, _h, _v, _tag)
    _f_met = os.path.join(_d_out, '%s_met.txt' % _tag)
    
    _fname = lambda t: os.path.join(_d_out, '%s_%s.tif' % (_tag, t))
    _fname_m1 = lambda t, a='_m1': _fname('%s_n0%s' % (t, a))

    # if not file_mag.get(_f_met).exists():
    #     logging.info('skip non-existing result for %s' % _tag)
    #     return

    if not file_mag.get(_fname_m1('esta_year')).exists():
        logging.info('skip non-existing result for %s' % _tag)
        return
    
    if (not _ttt) and file_mag.get(_fname_m1('age_year')).exists() and \
            (not config.getboolean('conf', 'over_write', False)):
        logging.info('skip processed esta result for %s' % _tag)
        return
    
    _b_esta_year = ge.open(_fname_m1('esta_year')).get_band().cache()
    # _b_esta_prob = ge.open(_fname_m1('esta_prob')).get_band().cache()

    _latest_year = config.getint('conf', 'latest_year')
    
    _est = _b_esta_year.data
    _dat = np.zeros((_b_esta_year.height, _b_esta_year.width), dtype=np.uint8)
    _idx = _est < 100
    _dat[_idx] = (_latest_year - 1970 - _est[_idx])
    
    _b_age = _b_esta_year.from_grid(_dat, nodata=255)
    
    _f_lnd = 's3://geo-dataset/data/land/list.shp'
    _b_lnd = gx.read_block(_f_lnd, _b_esta_year)
    _b_age.data[_b_lnd.data != 1] = _b_age.nodata
    _b_age.color_table = ge.load_colortable(config.get('conf', 'color'))
    
    with file_unzip.file_unzip() as _zip:
        _zip.save(_b_age, _fname_m1('age_year'))
    
    return True

def main(opts):
    import logging
    from gio import config
    from gio import file_mag
    from gio import global_task
    import os
    
    _d_inp = config.get('conf', 'input')
    _d_ref = config.get('conf', 'refer', _d_inp)
    
    _f_mak = file_mag.get(os.path.join(_d_inp, 'tasks.txt'))
    _ts = global_task.load(_f_mak)

    from gio import multi_task
    _rs = multi_task.run(_task, [(_t, os.path.join(_d_inp, 'data'), os.path.join(_d_ref, 'data'), opts) for _t in multi_task.load(_ts, opts)], opts)
    print('processed', len([_r for _r in _rs if _r]), 'tiles')

def usage():
    _p = environ_mag.usage(True)

    _p.add_argument('-i', '--input', dest='input')
    _p.add_argument('-y', '--latest-year', dest='latest_year', type=int, required=True)
    _p.add_argument('-w', '--over-write', dest='over_write', type='bool')
    _p.add_argument('--test-tile', dest='test_tile')

    return _p

if __name__ == '__main__':
    from gio import environ_mag
    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())])
