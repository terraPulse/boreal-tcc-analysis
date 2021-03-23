'''
File: fc_layer_mag.py
Author: Min Feng
Version: 0.1
Create: 2018-05-30 10:03:10
Description:
'''

def _find_err_file(f):
    import os

    _f_err = f.replace('_dat.shp', '_err.shp')
    if f != _f_err and os.path.exists(_f_err):
        return _f_err

    _f_err = f.replace('_tcc.shp', '_err.shp')
    if f != _f_err and os.path.exists(_f_err):
        return _f_err

    return None

def _find_num_file(f):
    import os
    from gio import file_mag

    _f_err = f.replace('_dat.shp', '_num.shp')
    if f != _f_err and file_mag.get(_f_err).exists():
        return _f_err

    _f_err = f.replace('_tcc.shp', '_num.shp')
    if f != _f_err and file_mag.get(_f_err).exists():
        return _f_err

    return None

def _default_err(bnd, dval):
    from gio import geo_raster as ge
    import numpy as np

    _def = dval
    _dat = np.empty((bnd.height, bnd.width), dtype=np.float32)
    _dat.fill(_def)

    _err = bnd.from_grid(_dat, nodata=-9999)
    _err.pixel_type = ge.pixel_type('float')

    return _err
    
def _expand_grid(bnd, size=1):
    from gio import geo_raster as ge
    
    _ggg = bnd.geo_transform
    _cel = _ggg[1]
    
    _geo = [_ggg[0] - (_cel * size), _cel, 0, _ggg[3] + (_cel * size), 0, -_cel]
    _bnd = ge.geo_raster_info(_geo, bnd.width + size * 2, bnd.height + size * 2, bnd.proj)

    return _bnd
    
def _load_median(f_dat, bnd, size=0):
    import logging
    from gio import geo_raster_ex as gx
    from gio import file_mag
    
    if size < 0:
        raise Exception('wrong size parameter %s' % size)
    
    if size == 0:
        _msk = bnd
    else:
        _msk = _expand_grid(bnd, size)
        
    logging.debug('loading %s' % f_dat)
    _bnd = gx.read_block(f_dat, _msk)
    
    if _bnd is None:
        return None
    
    if size == 0:
        return _bnd
    
    import scipy.ndimage
    _bnd.data = scipy.ndimage.median_filter(_bnd.data, size=(size * 2) + 1)
    
    return _bnd.read_block(bnd)

def load_forest_prob(f_dat, bnd, forest_only=False):
    from gio import geo_raster_ex as gx
    from gio import geo_raster as ge
    from gio import config
    from gio import file_mag
    import numpy as np
    import logging

    _def = config.getfloat('conf', 'avg_forest_err', 35.0)
    _bnd = _load_median(f_dat, bnd, config.getint('conf', 'forest_prob_median_size', 0))

    if _bnd is None:
        return None, None, None
        
    if forest_only:
        return _bnd, None, None
        
    # print f_dat, _bnd.data[0, 0]

    # _f_err = _find_err_file(f_dat)
    _f_err = None
    if _f_err:
        _err = gx.read_block(_f_err, bnd)
    else:
        _err = _default_err(bnd, _def)

    _num = None
    _f_num = _find_num_file(f_dat)
    
    if _f_num:
        _num = gx.read_block(_f_num, bnd)
        # if _num:
        #     _err.data[_num.data < 6] = 100.0

        _num_fix_rate = config.getfloat('conf', 'num_fix_rate_forest', 1.0)
        if _num_fix_rate < 1.0:
            _num.data = (_num.data * _num_fix_rate).astype(np.int16)
   
    if _num is None:
        logging.info('use default num: %s' % 8)
        
        _nnn = np.empty((bnd.height, bnd.width), dtype=np.int16)
        _nnn.fill(config.getint('conf', 'default_num', 8))
        _num = _bnd.from_grid(_nnn)
        
        # raise Exception('failed to load num band %s' % f_dat)

    # _err.data[(_err.data < _def) & (_err.data >= 0)] = _def
    if _bnd:
        _bnd.pixel_type = ge.pixel_type()
        _bnd.data = _bnd.data.astype(np.uint8)

    # print f_dat, _bnd.data[0, 0], _err.data[0, 0], _num.data[0, 0]
    return _bnd, _err, _num

def load_tcc_num(fs):
    if len(fs) == 0:
        return 0

    return _load_tcc_num(fs[0].band_file.file)

def _load_tcc_num(f):
    import re
    import os

    _m = re.search('(h\d+v\d+)_y(\d{4})_dat.tif', os.path.basename(f))
    if not _m:
        return 20
        # raise Exception('failed to parse the TCC file name')

    _t = _m.group(1)
    _y = int(_m.group(2))

    _f_met = os.path.join(os.path.dirname(f), '%s_met.txt' % _t)

    with open(_f_met) as _fi:
        import json
        from gio import landsat

        _n = 0
        for _l in list(json.load(_fi)['input'].values()):
            _p = landsat.parse(_l)

            if _p.ac_date_obj.year != _y:
                continue

            _n += 1

        # print 'tcc year', _t, _y, _n
        # print f
        return _n

def load_tree_cover(f_dat, bnd, forest_only=False):
    from gio import geo_raster_ex as gx
    from gio import geo_raster as ge
    from gio import config
    import numpy as np
    from gio import file_mag

    _def = config.getfloat('conf', 'avg_tcc_err', 35.0)

    _tcc = gx.read_block(f_dat, bnd)
    if _tcc is None:
        return None, None, None
        
    if forest_only:
        return _bnd, None, None

    # _f_err = _find_err_file(f_dat)
    _f_err = None
    if _f_err:
        _err = gx.read_block(_f_err, bnd)
    else:
        _err = _default_err(bnd, _def)

    from libtm import mod_prob
    _bnd = mod_prob.prob(_tcc, _err, config.getint('conf', 'min_tcc', 30))

    _idx = (_bnd.data <= 1.0) & (_bnd.data >= 0.0)
    _bnd.data[_idx] = (1.0 - _bnd.data[_idx]) * 100.0
    _bnd.data[_idx == False] = 255

    _bnd.pixel_type = ge.pixel_type()
    _bnd.nodata = 255
    _bnd.data = _bnd.data.astype(np.uint8)

    # _dat = _err.data

    # _min_num = config.getint('conf', 'min_num')
    # if load_tcc_num(_lyr.get_bands_pts([bnd.extent().get_center()])) < _min_num:
    #     _dat[(_dat < 100) & (_dat >= 0)] = 100
    # else:
    #     _dat[(_dat < _def) & (_dat >= 0)] = _def

    _nnn = np.empty((bnd.height, bnd.width), dtype=np.int16)
    _nnn.fill(config.getint('conf', 'default_num', 8))
    _num = _bnd.from_grid(_nnn)

    return _bnd, _err, _num

def load_forest_prob_pt(f_dat, pt):
    from gio import geo_raster_ex as gx

    _val = gx.geo_band_stack_zip.from_shapefile(f_dat, extent=pt).read(pt)

    _f_err = _find_err_file(f_dat)
    if _f_err:
        _err = gx.geo_band_stack_zip.from_shapefile(_f_err, extent=pt).read(pt)
    else:
        _err = _default_err(pt, -9999)

    return int(_val), _err


def load_tree_cover_pt(f_dat, pt):
    from gio import geo_raster_ex as gx
    from libtm import mod_prob

    _tcc = gx.geo_band_stack_zip.from_shapefile(f_dat, extent=pt).read(pt)

    _f_err = _find_err_file(f_dat)
    if _f_err:
        _err = gx.geo_band_stack_zip.from_shapefile(_f_err, extent=pt).read(pt)
    else:
        _err = _default_err(pt, -9999)

    _pro = mod_prob.prob(_tcc, _err, 30)

    if _pro < 0:
        return 255, -9999

    return int((1.0 - _pro) * 100.0), _err


def load_refer_forest(bnd, y):
    from gio import config
    import numpy as np

    _f_ref = config.get('conf', 'refer_tcc')
    _f_err = config.get('conf', 'refer_err')

    from gio import geo_raster_ex as gx

    _tcc = gx.geo_band_stack_zip.from_shapefile(_f_ref, extent=bnd).read_block(bnd)
    _err = gx.geo_band_stack_zip.from_shapefile(_f_err, extent=bnd).read_block(bnd)

    from libtm import mod_prob
    _bnd = mod_prob.prob(_tcc, _err, 30)
    # _bnd.data[_bnd.data <= 1.0] = _bnd.data[_bnd.data <= 1.0] * 100.0

    _idx = (_bnd.data <= 1.0) & (_bnd.data >= 0.0)
    _bnd.data[_idx] = (1.0 - _bnd.data[_idx]) * 100.0
    _bnd.data[_idx == False] = 255
    _bnd.nodata = 255

    _bnd.data = _bnd.data.astype(np.uint8)

    return _bnd

def load_tile_list(d, tile, fs, tag='forest'):
    if d is None or len(d) == 0:
        return None

    _fs = fs
    for _d in d:
        _load_tile_list(_d, tile, _fs, tag)
        
    return _fs

def _load_tile_list(d, tile, fs, tag='forest'):
    import os
    import re
    from gio import config
    from gio import file_mag
    from gio import global_task
    
    if not d:
        return None
        
    if not file_mag.get(os.path.join(d, 'tasks.txt')).exists():
        raise Exception('the input folder (%s) does not exist' % d)
    
    if not file_mag.get(os.path.join(d, 'data', tile.h, tile.v, tile.tag, '%s_met.txt' % (tile.tag, ))).exists():
        return None
    
    _ys = global_task.loads(file_mag.get(os.path.join(d, 'tasks.txt')).get())['params']['years']
    
    _yy = config.getjson('conf', 'year')
    if _yy is not None:
        _ys = [_y for _y in _ys if _y in [int(_y) for _y in _yy]]
        
    _fs = fs
    for _y in _ys:
        _y = str(_y)
        _fs[str(_y)] = os.path.join(d, 'data', tile.h, tile.v, tile.tag, '%s_y%s_dat.tif' % (tile.tag, _y))
        config.set(tag, _y, _fs[_y])
            
    return _fs
    
def load_shp_list(d, fs, tag='forest'):
    if d is None or len(d) == 0:
        return None

    _fs = fs
    for _d in d:
        _rs = _load_shp_list(_d, _fs, tag)

        for _r in _rs:
            _fs[_r] = _rs[_r]

    return _fs

def _load_shp_list(d, fs, tag='forest'):
    import os
    import re
    from gio import config
    from gio import file_mag
    
    if not d:
        return None
    
    if not file_mag.get(os.path.join(d, 'tasks.txt')).exists():
        raise Exception('the input folder (%s) does not exist' % d)
    
    if config.get('conf', 'prepare_input', False) or \
            len(file_mag.get(os.path.join(d, 'list/')).list()) == 0:
        prepare_input_lt(d)
    
    _fs = fs
    for _f in file_mag.get(os.path.join(d, 'list/')).list():
        _f = str(_f)
        
        _m = re.match('forest_(\d+)_dat.shp', os.path.basename(_f))
        if not _m:
            continue
        
        _fs[_m.group(1)] = str(_f)
            
    _ys = config.getjson('conf', 'year')
    for _y in _fs:
        if _ys is not None:
            if str(_y) not in _ys:
                continue
            
        config.set(tag, _y, _fs[_y])
            
    return _fs
    
def prepare_input_lt(d):
    import os
    import logging
    from gio import config
    from gio import file_mag
    from gio import global_task
    from gio import progress_percentage
    from gio import run_commands
    
    if not d.startswith('s3://'):
        d = os.path.abspath(d)
    
    _d_inp = file_mag.get(os.path.join(d, 'data/'))
    _d_out = os.path.join(d, 'list')
    
    _ys = global_task.loads(file_mag.get(os.path.join(d, 'tasks.txt')).get())['params']['years']
    
    _yy = config.getjson('conf', 'year')
    if _yy is not None:
        _ys = [_y for _y in _ys if _y in [int(_y) for _y in _yy]]

    _ts = []
    for _c in ['dat', 'num']:
        for _y in _ys:
            _f_out = os.path.join(_d_out, 'forest_%s_%s.txt' % (_y, _c))
            _f_shp = _f_out[:-4] + '.shp'
            
            if file_mag.get(_f_shp).exists():
                continue

            
            _e = os.path.join(d, 'data', '%(col)s/%(row)s/%(col)s%(row)s/%(col)s%(row)s_' + \
                    ('y%s_%s.tif' % (_y, _c)))
            _cmd = 'generate_tiles_extent.py -i %s -e "%s" -o %s -ts %s' % \
                    (d, _e, _f_shp, config.get('conf', 'task_num', 1))
                    
            _ts.append((_c, _y, _f_out, _f_shp, _cmd))
    
    if not _ts:
        return
    
    print('generate forest layer indexes')
    print('years: %s' % str(_ys))

    _ppp = progress_percentage.progress_percentage(len(_ts))
    for _c, _y, _f_out, _f_shp, _cmd in _ts:
        _ppp.next()

        logging.info('process %s, %s' % (_c, _y))
        run_commands.run(_cmd)

    _ppp.done()

def prepare_input(d):
    import os
    import re
    from gio import config
    from gio import file_mag
    
    _d_inp = file_mag.get(os.path.join(d, 'data/'))
    
    # if len(_d_inp.list()) == 0:
    #     raise Exception('unsupported input parameter')

    print('generate forest layer indexes')
    
    _fs = {}
    for _file in _d_inp.list(recursive=True):
        _file = str(_file)
        
        _m = re.search('(h\d+v\d+)_y(\d{4})_(.+)\.tif', _file)
        if not _m:
            continue
        
        _t = _m.group(1)
        _y = _m.group(2)
        _c = _m.group(3)
        
        # if _c not in ['dat', 'err', 'num']:
        if _c not in ['dat', 'num']:
            continue
        
        if _c not in _fs:
            _fs[_c] = {}
            
        if _y not in _fs[_c]:
            _fs[_c][_y] = []
            
        _fs[_c][_y].append(_file)
        
    import logging
    logging.info('found items %s, %s' % (len(list(_fs.keys())), len(list(_fs['dat'].values()))))
    
    _d_out = os.path.join(d, 'list')
    
    from gio import run_commands
    from gio import file_unzip
    import random
    
    with file_unzip.file_unzip() as _zip:
        for _c in _fs:
            print('process', _c)
    
            from gio import progress_percentage
            _ppp = progress_percentage.progress_percentage(len(_fs[_c]))
            
            _ys = _fs[_c]
            _ys = random.sample(list(_ys), len(_ys))
            
            for _y in _ys:
                _ppp.next()
                
                _f_out = os.path.join(_d_out, 'forest_%s_%s.txt' % (_y, _c))
                _f_shp = _f_out[:-4] + '.shp'
                
                if file_mag.get(_f_shp).exists():
                    continue
                
                _cmd = 'generate_tiles_extent.py -i %s -e %s -o %s -ts %s' % \
                        (d, _ys[_y][0], _f_shp, config.get('conf', 'task_num', 1))
                run_commands.run(_cmd)
                
                # _d_tmp = _zip.generate_file()
                # os.makedirs(_d_tmp)
                
                # _f_tmp = os.path.join(_d_tmp, os.path.basename(_f_out))
                # with open(_f_tmp, 'w') as _fo:
                #     _fo.write('\n'.join(_fs[_c][_y]))
                    
                # file_unzip.compress_folder(_d_tmp, os.path.dirname(_f_out), [])
                
                # _cmd = 'raster_extent2shp.py -i %s -ts %s' % (_f_out, config.get('conf', 'task_num', 1))
                # run_commands.run(_cmd)
                
            _ppp.done()
