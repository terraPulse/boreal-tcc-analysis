'''
File: mod_forest_change
Author: Min Feng
Version: 0.1
Create: 2018-04-20 15:42:37
Description: detect forest changes from foest probility layers and tree cover layers
'''

from gio import config
import logging

import os
from libtm import fc_annual_agg
from gio import geo_raster as ge
from gio import mod_filter
from gio import color_table

def detect(tag, tcc_bnds, err_bnds, num_bnds, d_tmp):
    _max_change_layers= config.getint('conf', 'max_change_layers', 1)
    
    _b_reg = config.getboolean('conf', 'output_linear_reg', True)
    _res = fc_annual_agg.produce_change_band(list(tcc_bnds.values())[0], tcc_bnds, err_bnds, num_bnds, _b_reg, \
            sorted(tcc_bnds.keys()))
            
    if _res is None:
        logging.warning('failed to process the change band')
        return None
    
    _loss, _gain, _prob, _regs = _res

    _d_tmp = d_tmp
    _tag = tag
    
    _clr = ge.load_colortable(config.get('conf', 'color'))
    _clr_age = ge.load_colortable(config.get('conf', 'color_age'))
    _clr_pro = ge.load_colortable(config.get('conf', 'color_pro'))

    _f_pro_age = os.path.join(_d_tmp, '%s_age_prob.tif' % _tag)
    _prob.save(_f_pro_age, color_table=_clr_pro)
    
    if _b_reg:
        _f_reg_slp = os.path.join(_d_tmp, '%s_reg_slp.tif' % _tag)
        _regs[0].save(_f_reg_slp)
        
        _f_reg_pvl = os.path.join(_d_tmp, '%s_reg_pvl.tif' % _tag)
        _regs[1].save(_f_reg_pvl)

    _bs = [_loss, _gain]
    _ns = ['loss', 'gain']

    # _ns = ['loss', 'gain', 'esta']
    # _bs = [_loss, _gain, _age]
    
    for _n in range(len(_ns)):
        for _i in range(len(_loss)):
            _t = _ns[_n]
            
            _f_clr = config.get('color', _t)
            if _f_clr:
                _clr = color_table.load(_f_clr)

            _f_num = os.path.join(_d_tmp, '%s_%s_year_n%d.tif' % (_tag, _t, _i))
            _f_dif = os.path.join(_d_tmp, '%s_%s_prob_n%s.tif' % (_tag, _t, _i))
            
            _bs[_n][_i][0].save(_f_num, color_table=_clr)
            _bs[_n][_i][1].save(_f_dif)

            mod_filter.filter_band_median(_bs[_n][_i][0], 1, 1)
            mod_filter.filter_band_mmu(_bs[_n][_i][0], area=100*100)

            _f_num = os.path.join(_d_tmp, '%s_%s_year_n%d_m1.tif' % (_tag, _t, _i))
            _bs[_n][_i][0].save(_f_num, color_table=_clr)
