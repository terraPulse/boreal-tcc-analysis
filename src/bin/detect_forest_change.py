'''
File: detect_forest_change.py
Author: Min Feng
Version: 0.1
Create: 2018-04-20 15:42:37
Description: detect forest changes from foest probility layers and tree cover layers
'''

import logging

def _task(tile, d_out, opts):
    from gio import file_unzip
    from gio import config
    from gio import file_mag
    from gio import metadata
    from libtm import fc_layer_mag
    import os
    import re

    _tag = tile.tag

    _ttt = config.get('conf', 'test_tile')
    if _ttt and _tag not in _ttt.replace(' ', '').split(','):
        return

    _m = re.match(r'(h\d+)(v\d+)', _tag)
    _d_out = os.path.join(d_out, _m.group(1), _m.group(2), _tag)

    _f_met = os.path.join(_d_out, '%s_met.txt' % _tag)
    if (not _ttt) and file_mag.get(_f_met).exists():
        logging.info('skip existing result for %s' % _tag)
        return
    
    _met = metadata.metadata()
    
    _met.tile = _tag
    _met.output = _d_out

    _fs = {}
    fc_layer_mag.load_shp_list(config.getjson('conf', 'input'), _fs, 'forest')
    fc_layer_mag.load_shp_list(config.getjson('conf', 'tcc'), _fs, 'tcc')
    
    _met.inputs = _fs

    _tcc_bnds = {}
    _err_bnds = {}
    _num_bnds = {}

    _mak = tile.extent()

    _year_min = config.getint('conf', 'year_min')
    _year_max = config.getint('conf', 'year_max')
    
    logging.info('year range: %s - %s' % (_year_min, _year_max))
    
    for _y, _f in config.cfg.items('forest'):
        if not re.match(r'\d{4}', _y):
            continue

        _y = int(_y)
        
        if _year_min is not None and _y < _year_min:
            continue
        if _year_max is not None and _y > _year_max:
            continue
        
        logging.info('loading forest layer %s' % _y)
        _bnd, _err, _num = fc_layer_mag.load_forest_prob(_f, _mak)

        if _bnd is None or _err is None or _num is None:
            continue

        _met.forests[_y] = _f
        _tcc_bnds[_y] = _bnd
        _err_bnds[_y] = _err
        _num_bnds[_y] = _num

    for _y, _f in config.cfg.items('tcc'):
        if not re.match(r'\d{4}', _y):
            continue

        _y = int(_y)
        
        if _year_min is not None and _y < _year_min:
            continue
        if _year_max is not None and _y > _year_max:
            continue

        logging.info('loading TCC layer %s' % _y)
        _bnd, _err, _num = fc_layer_mag.load_tree_cover(_f, _mak)

        if _bnd is None or _err is None or _num is None:
            continue

        _met.tcc[_y] = _f
        _tcc_bnds[_y] = _bnd
        _err_bnds[_y] = _err
        _num_bnds[_y] = _num

    with file_unzip.file_unzip() as _zip:
        _d_tmp = _zip.generate_file()
        os.makedirs(_d_tmp)

        if len(list(_tcc_bnds.keys())) > 0:
            from libtm import lib_forest_change
            lib_forest_change.detect(_tag, _tcc_bnds, _err_bnds, _num_bnds, _d_tmp)
        else:
            logging.warning('no input data found')
        
        _met.save(os.path.join(_d_tmp, os.path.basename(_f_met)))
        file_unzip.compress_folder(_d_tmp, _d_out, [])

def main(opts):
    import logging
    from gio import config
    from gio import file_mag
    import os
    
    _d_out = config.get('conf', 'output')
    
    _f_mak = file_mag.get(os.path.join(_d_out, 'tasks.txt'))
    _f_shp = file_mag.get(os.path.join(_d_out, 'tasks.shp'))

    from gio import global_task
    if config.getboolean('conf', 'prepare_input', False) or (not _f_mak.exists()):
        from libtm import fc_layer_mag

        _fs = {}
        fc_layer_mag.load_shp_list(config.getjson('conf', 'input'), _fs, 'forest')
        fc_layer_mag.load_shp_list(config.getjson('conf', 'tcc'), _fs, 'tcc')

        _f_inp = config.get('conf', 'region')
        if not _f_inp:
            _f_inp = list(_fs.values())[0]
            logging.info('use %s as extent' % _f_inp)

        _proj = None
        _cell = config.getfloat('conf', 'cell_size', 30)
        
        if opts.geog == True:
            from gio import geo_base as gb
            _proj = gb.proj_from_epsg()
            _cell = _cell / 120000.0
            print('use geog projection (%s)' % _cell)
            
        _ts = global_task.make(_f_inp, image_size=config.getint('conf', 'image_size', 2000), \
                cell_size=_cell, edge=config.getint('conf', 'image_edge', 1), f_shp=_f_shp, proj=_proj)
        global_task.save(_ts, _f_mak)
        return
    
    _ts = global_task.load(_f_mak)

    from gio import multi_task
    multi_task.run(_task, [(_r, os.path.join(_d_out, 'data'), opts) for _r in multi_task.load(_ts, opts)], opts)

def usage():
    _p = environ_mag.usage(True)

    _p.add_argument('-f', '--forest', dest='input', nargs='+')
    _p.add_argument('-y', '--year', dest='year', nargs='+')
    _p.add_argument('--tcc', dest='tcc', nargs='+')
    _p.add_argument('-r', '--region', dest='region')
    _p.add_argument('-c', '--color', dest='color')
    _p.add_argument('-o', '--output', dest='output', required=True)
    
    _p.add_argument('--geog', dest='geog', type='bool', default=True)
    _p.add_argument('--image-size', dest='image_size', type=int)
    _p.add_argument('--cell-size', dest='cell_size', type=int)
    _p.add_argument('-p', '--prepare-input', dest='prepare_input', type='bool', \
                default=False, help='only generate the input files forcefully')
                
    _p.add_argument('--year-min', dest='year_min', type=int)
    _p.add_argument('--year-max', dest='year_max', type=int, \
                help='year min/max can be used to limit the years for detecting')
                
    _p.add_argument('-al', '--adjust-loss-year', dest='adjust_loss_year', type='bool', default=True)
    _p.add_argument('-ag', '--adjust-gain-year', dest='adjust_gain_year', type='bool', default=True)
    
    _p.add_argument('-n', '--max-change-layers', dest='max_change_layers', type=int, default=3)
    _p.add_argument('-t', '--min-tcc', dest='min_tcc', type=int, default=30)
    _p.add_argument('-d', '--tcc-data', dest='tcc_data')
    _p.add_argument('--test-tile', dest='test_tile')

    return _p

if __name__ == '__main__':
    from gio import environ_mag
    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())])
