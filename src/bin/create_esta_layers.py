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
    _fname_ref = lambda t: os.path.join(_d_ref, '%s_%s.tif' % (_tag, t))
    _fname_m1 = lambda t, a='_m1': _fname('%s_n0%s' % (t, a))

    # if not file_mag.get(_f_met).exists():
    #     logging.info('skip non-existing result for %s' % _tag)
    #     return

    if not file_mag.get(_fname_m1('loss_year')).exists():
        logging.info('skip non-existing result for %s' % _tag)
        return
    
    if (not _ttt) and file_mag.get(_fname_m1('esta_year')).exists() and \
            (not config.getboolean('conf', 'over_write', False)):
        logging.info('skip processed esta result for %s' % _tag)
        return
    
    _b_loss_year = ge.open(_fname_m1('loss_year')).get_band().cache()
    _b_gain_year = ge.open(_fname_m1('gain_year')).get_band().cache()
    
    _b_loss_prob = ge.open(_fname_m1('loss_prob')).get_band().cache()
    _b_gain_prob = ge.open(_fname_m1('gain_prob')).get_band().cache()

    _f_tcc = config.get('conf', 'latest_tcc')
    _b_prob = _load_tcc(_f_tcc, _b_loss_year) if _f_tcc else ge.open(_fname_ref('age_prob')).get_band().cache()
    if _b_prob is None:
        logging.info('forced to use age_prob layer %s' % _fname_ref('age_prob'))
        _b_prob = ge.open(_fname_ref('age_prob')).get_band().cache()

    _d_forest_prob = _b_prob.data
    _d_loss = _b_loss_year.data
    _d_gain = _b_gain_year.data

    _d_esta = np.zeros(_d_forest_prob.shape, dtype=np.uint8)
    
    _d_prob = np.empty(_d_forest_prob.shape, dtype=np.float32)
    _d_prob.fill(100)
    _d_prob[_b_prob.data == _b_prob.nodata] = -9999
    
    _b_esta = _b_loss_year.from_grid(_d_esta, nodata=255)
    _b_esta.color_table = ge.load_colortable(config.get('conf', 'color'))

    _d_esta[_d_forest_prob > 100] = _d_forest_prob[_d_forest_prob > 100]
    
    for _y in range(1970, 2021):
        _y = _y - 1970
        
        _idx = _d_loss == _y
        _d_esta[_idx] = 100
        _d_prob[_idx] = _b_loss_prob.data[_idx]
        
        _idx = _d_gain == _y
        _d_esta[_idx] = _y
        _d_prob[_idx] = _b_gain_prob.data[_idx]
    
    _d_esta[_d_forest_prob < 50] = 100
    
    _d_test = (_d_esta < 100).astype(np.uint8)
    _d_test[(_d_esta < 100) & (_d_esta > 0)] = 1
    _b_test = _b_esta.from_grid(_d_test, nodata=255)
    mod_filter.filter_band_mmu(_b_test, area=config.getfloat('conf', 'min_patch'))
    _d_esta[(_d_esta == 100) & (_b_test.data == 1)] = 0
    
    _d_test = ((_d_esta > 0) & (_d_esta <= 100)).astype(np.uint8)
    _d_test[(_d_esta < 100) & (_d_esta > 0)] = 1
    _b_test = _b_esta.from_grid(_d_test, nodata=255)
    mod_filter.filter_band_mmu(_b_test, area=config.getfloat('conf', 'min_patch'))
    _d_esta[(_d_esta == 0) & (_b_test.data == 1)] = 100
    
    with file_unzip.file_unzip() as _zip:
        _zip.save(_b_esta, _fname_m1('esta_year'))
        _zip.save(_b_esta.from_grid(_d_prob, nodata=-9999), _fname_m1('esta_prob'))
    
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
    _p.add_argument('-r', '--refer', dest='refer')
    _p.add_argument('--latest-tcc', dest='latest_tcc')
    _p.add_argument('-w', '--over-write', dest='over_write', type='bool')
    _p.add_argument('--min-tcc', dest='min_tcc', type=int, default=30)
    _p.add_argument('-m', '--min-patch', dest='min_patch', type=float, default=100 * 100)
    _p.add_argument('--test-tile', dest='test_tile')

    return _p

if __name__ == '__main__':
    from gio import environ_mag
    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())])
