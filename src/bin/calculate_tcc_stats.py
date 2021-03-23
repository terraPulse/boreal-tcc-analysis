
def task(f, f_reg, f_out):
    from gio import run_commands as run
    from gio import file_mag

    if file_mag.get(f_out).exists():
        return

    from gio import geo_raster as ge

    _bnd = ge.open(f).get_band().cache()
    _dat = _bnd.data

    _

def main(opts):
    import os
    import re
    from gio import file_mag
    from gio import config
    from libtm import fc_layer_mag

    _d_inp = config.getjson('conf', 'input')

    _fs = {}
    fc_layer_mag.load_shp_list(_d_inp, _fs, 'forest')

    from gio import file_unzip
    with file_unzip.zip() as _zip:
        from gio import geo_base as gb
        from gio import rasterize_band as rb

        _ce = 30.0
        _gs = [_g for _g, _ in gb.load_shp(opts.region)]

        if len(_gs) == 0:
            raise Exception('no valid region provided')

        if _gs[0].proj.IsGeographic():
            _ce = _ce / 120000

        _msk = rb.to_mask(rb.to_raster(_gs, _ce), _gs)
        _f_msk = _zip.generate_file('', '.tif')

        _zip.save(_msk, _f_msk)

        _d_out = opts.output if opts.output else os.path.join(_d_inp[0], 'maps', opts.tag, opts.version)

        _ps = []
        for _y in sorted(_fs.keys()):
            _f_out = os.path.join(_d_out, 'tcc', '%s_%s_tcc_%s_dat.tif' % (opts.tag, opts.version, _y))
            _ps.append((_fs[_y], _f_msk, _f_out))

        from gio import multi_task
        multi_task.run(task, multi_task.load(_ps, opts), opts)

def usage():
    _p = environ_mag.usage(True)

    _p.add_argument('-i', '--input', dest='input', required=True, nargs='+')
    _p.add_argument('-r', '--region', dest='region', required=True)
    _p.add_argument('-t', '--tag', dest='tag', required=True)
    _p.add_argument('-v', '--version', dest='version', required=True)
    _p.add_argument('-o', '--output', dest='output')
    _p.add_argument('-c', '--calibrate', dest='calibrate', type='bool')
    
    return _p

if __name__ == '__main__':
    from gio import environ_mag
    environ_mag.init_path()
    environ_mag.run(main, [environ_mag.config(usage())]) 