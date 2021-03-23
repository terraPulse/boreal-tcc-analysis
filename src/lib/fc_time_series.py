'''
File: fc_time_series.py
Author: Min Feng
Version: 0.1
Create: 2018-04-18 20:43:47
Description:
'''

import sys
import datetime
from gio import config
from . import lib_time_op
from . import lib_detect_loss05 as lib_detect_loss
from . import lib_detect_gain04 as lib_detect_gain

class fc_obj:

    def __init__(self, d, t, err, f_tcc=None, f_err=None, wrs_row_north=True):
        self.d = d
        # self.t = t
        # self.f = -1
        self.f = t

        # if t == 1:
        #     self.f = err
        # if t == 9:
        #     self.f = (1 - err)

        # if self.f > 0:
        #     self.f *= 100.0

        self.err = err
        self.ch = 0
        self.prob = 0.0
        self.skip = 0 if 0 <= self.f <= 100 and (err >= 0) else 1
        self.wrs_row_north = wrs_row_north
        # self.f_tcc = f_tcc
        # self.f_err = f_err

    def change(self, ch, prob):
        self.ch = ch
        self.prob = prob

    def __str__(self):
        return ', '.join(map(str, [self.d, self.f, self.err, self.ch, self.prob, self.skip, self.wrs_row_north]))

def filter_noise(ds):
    _ps = []
    _nu = 0

    for _i in range(len(ds)):
        _d = ds[_i]

        if _d.skip:
            continue

        if len(_ps) == 0:
            _d.change(0, 0)
        else:
            _p1 = _ps[-1]
            _t, _p = lib_time_op.estimate_change(_p1.f, _p1.err, _d.f, _d.err)
            _d.change(_t, _p)

        _ps.append(_d)

    if len(_ps) > 0 and _ps[-1].ch != 0:
        _ps[-1].skip = 1

    return _nu

def smooth(ds):
    for _i in range(1, len(ds)):
        _dd = [ds[_i].f, ds[_i-1].f, ds[_i-2].f if _i > 1 else ds[_i-1].f]
        _dd.sort()

        ds[_i-1].f = _dd[1]

def filter_noise2(ds):
    _ps = []
    _nu = 0

    for _i in range(len(ds)):
        _d = ds[_i]

        if _d.skip:
            continue

        if _d.ch > 0 and len(_ps) > 0:
            if len(_ps) == 1:
                _nu += 1
                _ps[-1].skip = 1
                continue

            for _j in range(min(3, len(_ps))):
                _idx = -1 * _j

                _nu += 1
                _ps[_idx].skip = 1

                if _ps[_idx].ch < 0:
                    _p1 = _ps[_idx - 1]
                    _t, _p = lib_time_op.estimate_change(_p1.f, _p1.err, _d.f, _d.err)
                    _d.change(_t, _p)
                    break

        if _d.ch < 0 and len(_ps) > 0:
            if len(_ps) == 1:
                _nu += 1
                _ps[-1].skip = 1
                continue

            for _j in range(min(3, len(_ps))):
                _idx = -1 * _j

                _nu += 1
                _ps[_idx].skip = 1

                if _ps[_idx].ch > 0:
                    _p1 = _ps[_idx - 1]
                    _t, _p = lib_time_op.estimate_change(_p1.f, _p1.err, _d.f, _d.err)
                    _d.change(_t, _p)
                    break

        _ps.append(_d)

    return _nu

def average_vals(vs, es, dd):
    if len(vs) == 0:
        return -1, -1

    if len(vs) == 1:
        return vs[0], es[0]

    _tw = sum(dd)

    _vv = 0.0
    _ee = 0.0

    for _i in range(len(vs)):
        _w = dd[_i] / _tw

        _vv += _w * vs[_i]
        _ee += _w * es[_i]

    return _vv, _ee


def max_estimate(os, dd):
    _len = len(os)

    if _len == 0:
        return None

    if _len <= 2:
        return os[0]

    _os = sorted(os, reverse=True, key=lambda x: x.obj.f)
    if _len <= 3:
        return _os[0]

    if _len <= 6:
        return _os[1]

    return _os[max(2, (_len / 4))]


def median_estimate(os, dd, reverse):
    _len = len(os)

    if _len == 0:
        return None

    _os = sorted(os, reverse=reverse, key=lambda x: x.obj.f)
    if _len <= 2:
        return _os[0]

    return _os[int(len(_os) / 3)]


def average_annual(ds, y):
    _max_days = 365 * 3

    # _ds = datetime.datetime(y - 3, 1, 1)
    _dc = datetime.datetime(y, 7, 1)
    # _de = datetime.datetime(y + 4, 1, 1)

    _vs1 = []
    _es1 = []
    _dd1 = []

    _vs2 = []
    _es2 = []
    _dd2 = []

    for _d in ds:
        if _d.skip != 0:
            continue

        _ss = float((_d.d - _dc).days)
        if _ss == 0.0:
            _ss = 0.01

        if abs(_ss) > _max_days:
            continue

        if _ss < 0:
            _dd1.append(1.0 / (_ss ** 2))
            # _dd1.append(1.0 / (_ss))

            _vs1.append(_d.f)
            _es1.append(_d.err)
        else:
            _dd2.append(1.0 / (_ss ** 2))
            # _dd2.append(1.0 / (_ss))

            _vs2.append(_d.f)
            _es2.append(_d.err)

    _v1, _e1 = average_vals(_vs1, _es1, _dd1)
    _v2, _e2 = average_vals(_vs2, _es2, _dd2)

    if _v1 < 0:
        return _v2, _e2, len(_vs2)

    if _v2 < 0:
        return _v1, _e1, len(_vs1)

    # return (_v1 + _v2) / 2, (_e1 + _e2) / 2
    return (_v1 + _v2) / 2, min(_e1, _e2), min(len(_vs1), len(_vs2))

def average_annual_med(ds, y, ys):
    _v, _e, _n = _average_annual_med(ds, y, ys)

    if _v >= 0:
        return _v, _e, _n

    return -1, 0, 0
    
def average_annual_max(ds, y, ys):
    _v, _e, _n = _average_annual_max(ds, y, ys)

    if _v >= 0:
        return _v, _e, _n

    return -1, 0, 0

def average_annual_max_v0(ds, y, ys):
    for _y in range(3, 12):
        _v, _e, _n = _average_annual_max_v0(ds, y, 365 * _y)

        if _v >= 0:
            return _v, _e, _n

    return -1, 0, 0

class record:

    def __init__(self, o):
        self.obj = o

def _search_year_range(y, ys):
    _yi = ys.index(y)

    _yl = ys[_yi-1] if _yi > 0 else 1900
    _yu = ys[_yi+1] if _yi < len(ys) - 1 else 2100

    _ys = y
    _ye = y

    if _yl < _yi - 1:
        _ys = int(round((_yl + y) / 2.0))

    if _yu > _yi + 2:
        _ye = int((_yu + y - 1) / 2.0)

    return _ys, _ye

def _average_annual_med(ds, y, ys):
    _dm = 6

    _ss1 = []
    _dd1 = []
    _dn1, _dn2 = 0, 0

    _max_num = config.getint('conf', 'max_records_aggregation', 15)
    _ys, _ye = _search_year_range(y, ys)

    for _d in ds:
        if _d.skip != 0 or _d.f < 0 or _d.f > 100 or _d.err < 0:
            continue

        if not (_ys <= _d.d.year <= _ye):
            continue

        # _ss = float((_d.d - _dc).days)
        # _ss = float(_d.d.month - _dm) / 6.0

        # _rr = abs(_ss)
        # _rr = 1.0 / (max(0.01, _rr if _d.wrs_row_north else 1.0 - _rr) ** 2)

        # _dd1.append(_rr)
        _ss1.append(record(_d))
        
        _dn1 += 1
        
        if _dn1 >= _max_num:
            break

    _s1 = median_estimate(_ss1, _dd1, True)

    _v1, _e1 = -1, -1
    if _s1 is not None:
        _v1, _e1 = _s1.obj.f, _s1.obj.err

    return _v1, _e1, _dn1
    
def _average_annual_max(ds, y, ys):
    _dm = 6

    _ss1 = []
    _dd1 = []

    _ss2 = []
    _dd2 = []
    
    _max_num = config.getint('conf', 'max_records_aggregation', 3)
    _dn1, _dn2 = 0, 0
    _ys, _ye = _search_year_range(y, ys)

    for _d in ds:
        if _d.skip != 0 or _d.f < 0 or _d.f > 100 or _d.err < 0:
            continue

        if not (_ys <= _d.d.year <= _ye):
            continue

        # _ss = float((_d.d - _dc).days)
        _ss = float(_d.d.month - _dm) / 6.0

        _rr = abs(_ss)
        _rr = 1.0 / (max(0.01, _rr if _d.wrs_row_north else 1.0 - _rr) ** 2)

        if _ss < 0:
            if _dn1 >= _max_num:
                continue
            
            _dd1.append(_rr)
            # _dd1.append(1.0 / (_ss))

            _ss1.append(record(_d))
            _dn1 += 1
        else:
            if _dn2 >= _max_num:
                continue
            
            _dd2.append(_rr)
            # _dd2.append(1.0 / (_ss))

            _ss2.append(record(_d))
            _dn2 += 1

    _s1 = median_estimate(_ss1, _dd1, True)
    _s2 = median_estimate(_ss2, _dd2, False)

    _v1, _e1 = -1, -1
    _v2, _e2 = -1, -1

    if _s1 is not None:
        _v1, _e1 = _s1.obj.f, _s1.obj.err

    if _s2 is not None:
        _v2, _e2 = _s2.obj.f, _s2.obj.err

    if _v1 < 0:
        return _v2, _e2, len(_ss2)

    if _v2 < 0:
        return _v1, _e1, len(_ss1)

    # if _v2 == 0 or _v1 == 0:
    #     return _v1, _e1

    # if _e1 / _v1 < _e2 / _v2:
    #     return _v1, _e1

    # return _v2, _e2
    # return (_v1 + _v2 * 2) / 3, (_e1 + _e2 * 2) / 3, min(len(_ss1), len(_ss2))
    return (_v1 + _v2) / 2, (_e1 + _e2) / 2, min(len(_ss1), len(_ss2))
    # return (_v1 + _v2 * 2) / 3, min(_e1, _e2)

def _average_annual_max_v0(ds, y, days=365):
    _dc = datetime.datetime(y, 7, 1)

    _ss1 = []
    _dd1 = []

    _ss2 = []
    _dd2 = []
    
    _max_num = config.getint('conf', 'max_records_aggregation', 1)
    _dn1, _dn2 = 0, 0

    for _d in ds:
        if _d.skip != 0 or _d.f < 0 or _d.f > 100 or _d.err < 0:
            continue

        _ss = float((_d.d - _dc).days)
        if _ss == 0.0:
            _ss = 0.01

        if not ((days / -2.0) < _ss < (days / 2.0)):
            continue

        if _ss < 0:
            if _dn1 >= _max_num:
                continue
            
            _dd1.append(1.0 / (_ss ** 2))
            # _dd1.append(1.0 / (_ss))

            _ss1.append(record(_d))
            _dn1 += 1
        else:
            if _dn2 >= _max_num:
                continue
            
            _dd2.append(1.0 / (_ss ** 2))
            # _dd2.append(1.0 / (_ss))

            _ss2.append(record(_d))
            _dn2 += 1

    _s1 = median_estimate(_ss1, _dd1, True)
    _s2 = median_estimate(_ss2, _dd2, False)

    _v1, _e1 = -1, -1
    _v2, _e2 = -1, -1

    if _s1 is not None:
        _v1, _e1 = _s1.obj.f, _s1.obj.err

    if _s2 is not None:
        _v2, _e2 = _s2.obj.f, _s2.obj.err

    if _v1 < 0:
        return _v2, _e2, len(_ss2)

    if _v2 < 0:
        return _v1, _e1, len(_ss1)

    # if _v2 == 0 or _v1 == 0:
    #     return _v1, _e1

    # if _e1 / _v1 < _e2 / _v2:
    #     return _v1, _e1

    # return _v2, _e2
    # return (_v1 + _v2 * 2) / 3, (_e1 + _e2 * 2) / 3, min(len(_ss1), len(_ss2))
    return (_v1 + _v2) / 2, (_e1 + _e2) / 2, min(len(_ss1), len(_ss2))
    # return (_v1 + _v2 * 2) / 3, min(_e1, _e2)

def predict_mean(ds):
    _vs = []

    for _d in ds:
        if _d.skip != 0:
            continue

        if not (0 <= _d.f <= 100):
            continue

        _vs.append(_d.f)

    sys.stdout.flush()

    if len(_vs) <= 0:
        return -1, 0, 0

    if len(_vs) == 1:
        return _vs[0], 100.0, 1

    _av = sum(_vs) / float(len(_vs))
    _ta = sum([(_v - _av) ** 2 for _v in _vs]) / (len(_vs) - 1)
    if _ta > 0.0001:
        _ta = _ta ** 0.5

    return _av, _ta, len(_vs)

def predict_years(ds, ys):
    _vs = {}
    _es = {}
    _ns = {}

    for _y in ys:
        _v, _e, _n = average_annual(ds, _y)

        _vs[_y] = _v
        _es[_y] = _e
        _ns[_y] = _n

    return _vs, _es, _ns

def predict_years_med(ds, ys):
    _vs = {}
    _es = {}
    _ns = {}

    for _y in sorted(ys):
        _v, _e, _n = average_annual_med(ds, _y, ys)

        _vs[_y] = _v
        _es[_y] = _e
        _ns[_y] = _n
        
    _max_diff = config.getfloat('conf', 'median_filter_max_diff', -1)
    if _max_diff > 0:
        for _y, _v in filter_median(_vs, _max_diff).items():
            _vs[_y] = _v
        
    return _vs, _es, _ns

def predict_years_max(ds, ys):
    _vs = {}
    _es = {}
    _ns = {}

    for _y in sorted(ys):
        _v, _e, _n = average_annual_max(ds, _y, ys)

        _vs[_y] = _v
        _es[_y] = _e
        _ns[_y] = _n

    _max_diff = config.getfloat('conf', 'median_filter_max_diff', -1)
    if _max_diff > 0:
        for _y, _v in filter_median(_vs, _max_diff).items():
            _vs[_y] = _v
        
    return _vs, _es, _ns

def detect_change(ds):
    _cp = None

    for _d in ds:
        if _d.skip != 0:
            continue

        if _d.ch >= 0:
            continue

        if _cp is None:
            _cp = _d
        else:
            if _d.prob > _cp.prob:
                _cp = _d

    return _cp

def _determine_year(ys, last_year=-1):
    if config.getboolean('conf', 'debug'):
        print('change years:', ys)

    if len(ys) > 0:
        for _y in ys:
            if last_year > 0 and _y[0] >= last_year:
                continue

            return _y

    return None

    # _min_v = None
    #
    # for _i in xrange(len(rs)):
    #     if _min_v is None or _min_v[1] > rs[_i][1]:
    #         _min_v = rs[_i]
    #
    # return _min_v
    
def filter_median(vs, max_diff):
    _max_it = config.getint('conf', 'median_filter_max_iteration', 3)
    
    _vs = vs
    for _ in range(_max_it):
        _vs, _nu = _filter_median(_vs, max_diff)
        if _nu == 0:
            break
    
    return _vs
    
def _filter_median(vs, max_diff):
    _vs = {_k: _v for _k, _v in vs.items()}
    _ys = sorted(_vs.keys())
    _nu = 0
    
    for _i in range(len(_ys)):
        if _i <= 0 or _i >= len(_ys) - 1:
            continue
        
        if not (0 <= _vs[_ys[_i]] <= 100):
            continue

        _vv = []

        for _d in range(3):
            _y = _i - _d - 1
            if _y < 0:
                break
            _v = _vs[_ys[_y]]
            if 0 <= _v <= 100:
                _vv.append(_v)
                break

        _vv.append(_vs[_ys[_i]])
        
        for _d in range(3):
            _y = _i + _d + 1
            if _y >= len(_ys):
                break
            _v = _vs[_ys[_y]]
            if 0 <= _v <= 100:
                _vv.append(_v)
                break

        # _vv = [_vs[_y] for _y in _ys[_i-1:_i+2] if 0 <= _vs[_y] <= 100]
        if len(_vv) < 3:
            continue

        _vv.sort()
        _vm = _vv[1]
        
        _vt = vs[_ys[_i]]
        
        if _vm == _vt:
            continue
        
        if abs(_vt - _vm) > max_diff:
            _vs[_ys[_i]] = int((sum(_vv) - _vt) / 2)
            _nu += 1
            
    return _vs, _nu

def _filter_median_2(vs, max_diff):
    _vs = {_k: _v for _k, _v in vs.items()}
    _ys = sorted(_vs.keys())
    _nu = 0
    
    for _i in range(len(_ys)):
        if _i <= 0 or _i >= len(_ys) - 1:
            continue

        if not (0 <= _vs[_ys[_i]] <= 100):
            continue
        
        _vv = [_vs[_y] for _y in _ys[_i-1:_i+2] if 0 <= _vs[_y] <= 100]
        if len(_vv) < 3:
            continue

        _vv.sort()
        _vm = _vv[1]
        
        _vt = _vs[_ys[_i]]
        
        if _vm == _vt:
            continue
        
        if abs(_vt - _vm) > max_diff:
            _vs[_ys[_i]] = int((sum(_vv) - _vt) / 2)
            _nu += 1
            
    return _vs, _nu
    
def detect_change_annual(ys, vs, es, ns, max_change_layers):
    _vs = {_y: _v for _y, _v in vs.items() if _v <= 100}
    if len(_vs.keys()) < 2:
        return None
        
    _max_diff = config.getfloat('conf', 'pob_max_diff', 20)
    _vs = filter_median(_vs, _max_diff)
    _ys = sorted(_vs.keys(), reverse=True)

    _ly, _lv = select_values(_vs, _ys, 2)

    if (max(_vs) - min(_vs)) < config.getfloat('conf', 'min_var', 20):
        return (_lv, None, None)

    if max(_vs) < config.getfloat('conf', 'min_forest', 55):
        return (_lv, None, None)

    # _ax = 0
    # _px = 0

    _rs_loss = []
    _rs_gain = []
    
    _nu_loss = 0
    _nu_gain = 0

    _debug = config.getboolean('conf', 'debug')
    _flag_gain = False
    _flag_loss = True

    _min_diff_loss = config.getfloat('conf', 'min_diff_loss', 10)
    _min_diff_gain = config.getfloat('conf', 'min_diff_gain', 5)
    _min_forest = config.getfloat('conf', 'min_forest', 47)
    _last_year = 1000000

    for _i in range(len(_ys)):
        _y = _ys[_i]

        if _y > _last_year:
            continue

        if _debug:
            print('year', _y, _vs[_y])

        if _vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
            continue
        
        _v_base = max([_vs[_ys[_z]] for _z in range(_i, min(_i+2, len(_ys)))])
        if _debug:
            print('loss base', _v_base, _min_forest)

        if _v_base > _min_forest:
            _yp = _ys[max(0, _i - 1)]
            if _vs[_yp] < (_v_base - _min_diff_loss):
                if _debug:
                    print('---- loss', _y)
                
                _a, _t, _p = lib_detect_loss.detect(_ys, _vs, es, ns, _y, _flag_gain)
    
                if _a > 0:
                    if config.getboolean('conf', 'adjust_loss_year', True):
                        _a = lib_detect_loss.adjust(_ys, _vs, _a)
                        
                    _rs_loss.append((_a, _p))
                    _nu_loss += 1

                    _last_year = min(_last_year, _a)
                    
                    _flag_gain = False
                    _flag_loss = True

        _v_base = max([_vs[_ys[_z]] for _z in range(max(0, _i - 1), _i + 1)])
        if _debug:
            print('gain base', _v_base, _min_forest)

        if _v_base > _min_forest - 10:
            _yp = _ys[min(len(_ys) - 1, _i + 1)]
            if _vs[_yp] < (_vs[_y] - _min_diff_gain):
                if _debug:
                    print('---- gain', _y)
    
                _a, _t, _p = lib_detect_gain.detect(_ys, _vs, es, ns, _y)
    
                if _a > 0:
                    if config.getboolean('conf', 'adjust_gain_year', True):
                        _a = lib_detect_gain.adjust(_ys, _vs, _a)
                    _rs_gain.append((_a, _p))
                    _nu_gain += 1

                    _last_year = min(_last_year, _a)
                    
                    _flag_gain = True
                    _flag_loss = False
            
        if _debug:
            print('loss %s gain %s, %s' % (_nu_loss, _nu_gain, max_change_layers))
            
        if min(_nu_loss, _nu_gain) >= max_change_layers:
            break


        # if _nu_loss != _nu_gain or min(_nu_loss, _nu_gain) >= max_change_layers:
        #     break

    _ys_loss = []
    _ys_gain = []
    _last_year = -1

    while True:
        _y_loss, _y_gain = _decide_years(_ly, _lv, _vs, _ys, _rs_loss, _rs_gain, _last_year)

        if _y_loss is None and _y_gain is None:
            break
        else:
            if len(_ys_gain) > 0 and _ys_gain[-1] is None and _y_loss is not None:
                _ys_gain[-1] = (_y_loss[0], (_y_loss[1][0] / 3.0, _y_loss[1][1]))

            _ys_loss.append(_y_loss)
            _ys_gain.append(_y_gain)
            
            _last_year = min([_y for _y in [_y_loss, _y_gain] if _y is not None])[0]
            if _last_year > 0:
                _last_year -= 3

    return _lv, _ys_loss, _ys_gain

def _decide_years(ly, lv, vs, ys, rs_loss, rs_gain, last_year):
    _y_loss, _y_gain = _determine_year(rs_loss, last_year), _determine_year(rs_gain, last_year)

    if _y_loss is not None:
        if _y_gain is not None and _y_gain[0] > _y_loss[0]:
            return _y_loss, _y_gain

        if lv > 50 and ly > _y_loss[0]:
            _yy = search_for_forest(vs, ys, _y_loss[0])
            if _yy is not None:
                _yn = _yy[0]
                if config.getboolean('conf', 'adjust_gain_year', True):
                    _yn = lib_detect_gain.adjust(ys, vs, _yn)
                _y_gain = (_yn, (_yy[1], 0.0))

    return _y_loss, _y_gain

def search_for_forest(vs, ys, s):
    _ys = sorted(ys)

    for _i in range(_ys.index(s), len(_ys)):
        if 50 < vs[_ys[_i]] <= 100:
            _y = _ys[_i]
            return _y, abs(vs[_y] - vs[s])

    return None

def select_values(vs, ys, num=2):
    _ts = []

    for _y in ys:
        _v = vs[_y]
        if 0 <= _v <= 100:
            _ts.append(_v)

        if len(_ts) >= num:
            return _y, sum(_ts) / len(_ts)

    return 0, None

def fc_filtering(ds):
    ds.sort(key=lambda x: x.d)

    smooth(ds)
    return

    for i in range(1):
        if filter_noise(ds) == 0:
            break

    for i in range(1):
        if filter_noise2(ds) == 0:
            break

