'''
File: calculate_layer_stats.py
Author: Min Feng
Description: calcualte the stats of the a layer and save the results in a binary file
'''

def task(f):
    from gio import geo_raster as ge
    from gio import geo_base as gb
    import re
    
    _tag = re.search('h\d+v\d+', f).group()
    _bnd = ge.open(f).get_band().cache()
    
    _dat = _bnd.data
    _dat[:, -1] = 255
    _dat[-1, :] = 255
    
    from gio import stat_band
    _sss = stat_band.stat(_bnd)
    
    _ext = _bnd.extent()
    _loc = _ext.get_center().project_to(ge.proj_from_epsg())
    
    _pol = _ext.to_polygon()
    _are = _pol.segment_ratio(10).project_to(gb.modis_projection()).area() / ((_bnd.width * _bnd.height) * 1000000.0)
    
    _ooo = {}
    for _k, _v in _sss.items():
        _ooo[_k] = _v * _are
    
    return _tag, round(_loc.x, 2), round(_loc.y, 2), _ooo
    
def _merge_dict(d1, d2):
    for _k, _v in d2.items():
        if _k > 100:
            continue
        
        d1[_k] = d1.get(_k, 0.0) + float(_v)
    return d1
    
def _output_years(tag, rs, f_out):
    _as = list(range(20, 50))
    
    _ls = [','.join([tag] + ['y%s' % (1970 + _y) for _y in _as])]
    
    _frm = lambda x: '%.2f' % x 
    
    _rs = sorted(rs.keys())
    for _r in _rs:
        _ls.append(','.join(['%.2f' % _r] + [_frm(rs[_r].get(_y, 0)) for _y in _as]))
        
    from gio import file_unzip
    with file_unzip.zip() as _zip:
        _zip.save('\n'.join(_ls), f_out)

def main(opts):
    from gio import geo_base as gb
    from gio import file_mag as fm
    _ps = []
    
    for _g, _i in gb.load_shp(fm.get(opts.input).get()):
        _fe = _i['FILE']
        
        # for _n in range(opts.layer_num):
        #     _ff = _fe.replace('_n0', '_n%s' % _n)
        #     _ps.append((_ff, ))
            
        _ps.append((_fe, ))
        # if len(_ps) > 10:
        #     break
        
    print(len(_ps))
    
    from gio import multi_task
    _rs = multi_task.run(task, _ps, opts)
    
    import json
    from gio import file_unzip
    with file_unzip.zip() as _zip:
        _zip.save(json.dumps(_rs, indent=2), opts.output)
        
    # import pickle
    # with open(opts.output, 'wb') as _fo:
    #     _json.saves(_rs, intend=2)
        # pickle.dump(_rs, _fo)
        
def usage():
    _p = environ_mag.usage(True)

    _p.add_argument('-i', '--input', dest='input', required=True)
    _p.add_argument('-o', '--output', dest='output')
    
    return _p

if __name__ == '__main__':
    from gio import environ_mag
    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())]) 