'''
File: lib_detect_loss04.py
Author: Min Feng
Version: 0.1
Create: 2019-12-16 12:55:32
Description: changed the forest prob difference checking rule before variation checkiong to identify fast regrowth forests in Myanmar
'''

import math
from gio import config
from . import lib_time_op

def _last_val_value(vs, ys, i):
    for _i in range(i + 1, len(ys)):
        _v = vs[ys[_i]]

        if _v > 100:
            continue

        return _v

    return 255

def adjust(ys, vs, y):
    _ys = sorted(ys, reverse=True)
    _id = _ys.index(y)

    if _id >= len(_ys) - 1:
        return y

    _cy = [y]
    _cv = vs[y]
    _min_diff = 5.0

    for _i in range(_id + 1, len(_ys)):
        _y = _ys[_i]

        if _y > y + 2:
            break

        _v = vs[_y]
        if _v > 100:
            continue

        _v2 = vs[_ys[_i + 1]] if _i + 1 < len(_ys) else 0
        if _v2 > 100:
            continue

        _d = _min_diff if _v < 60 else _min_diff * 2.0
        if _v >= _cv and _v < _v2 - _d:
            _cy.append(_y)
            _cv = _v
            continue

        break

    if len(_cy) == 1:
        return _cy[0]

    return _cy[-1]

def detect(ys, vs, es, ns, y, flag_gain):
    _debug = config.getboolean('conf', 'debug')

    # _min_prob = config.getfloat('loss', 'min_prob', 0.5)
    _min_forest = config.getfloat('loss', 'min_forest', 50)
    _min_forest_change = config.getfloat('loss', 'min_forest_change', 65)
    _min_num = config.getfloat('loss', 'min_num', 5)
    
    _min_num_forest = _min_num #config.getfloat('loss', 'min_num_forest', 3)
    _min_num_nonforest = _min_num #config.getfloat('loss', 'min_num_nonforest', 3)
    
    # _min_ob_num = config.getfloat('loss', 'min_ob_num', 12)
    # _min_rate = config.getfloat('loss', 'min_rate', 0.8)

    _ys = sorted(ys)
    _id = _ys.index(y)

    if _id == len(_ys) - 1:
        return -1, 0, (-9999, -9999)

    _cs1 = []
    _cs2 = []

    _as1 = []
    _as2 = []

    _na1 = 0.0
    _na2 = 0.0

    _nu1 = 0.0
    _nu2 = 0.0

    _cy = -1

    _val_forest = config.getfloat('loss', 'val_forest')
    _val_others = config.getfloat('loss', 'val_others')
    _min_years = config.getint('loss', 'min_years', 3)
    _min_years_valid = config.getint('loss', 'min_years_valid', 1)
    _max_year_span = config.getfloat('loss', 'max_year_span', 5)
    _min_change_prob = config.getfloat('loss', 'min_change_prob', 0.07)
    _use_simulate_vals = config.getboolean('conf', 'use_simulate_vals', False)

    if _id >= 0:
        _yy = _ys[:_id]
        _yy.reverse()

        # _yy = _ys[_id: max(-1, _id-25): -1] if _id > 0 else [_ys[_id]]
        for _y in _yy:
            if _na1 > 0 and (_nu1 >= _min_num_forest) and (abs(_y - y) > _max_year_span):
                continue

            if vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
                continue

            _vv = max(0, _val_forest - vs[_y])
            if _use_simulate_vals:
                ns[_y] = min(5, max(2, ns[_y]))
                for _x in range(ns[_y]):
                    _cs1.append(lib_time_op.simulate_val(_vv, es[_y]))
            else:
                _cs1.append(_vv)

            _as1.append(vs[_y])

            _nu1 += ns[_y]
            _na1 += 1

            # if _na1 >= 3 or _nu1 > _min_num:
            if _na1 >= _min_years and (_nu1 >= _min_num_forest):
                break

        _y = y
        if _na1 > 0 and vs[_y] < _min_forest_change:
            _cy = _y
        else:
            _vv = _val_forest - vs[_y]
            if _use_simulate_vals:
                for _x in range(ns[_y]):
                    _cs1.append(lib_time_op.simulate_val(_vv, es[_y]))
            else:
                _cs1.append(_vv)

            _as1.append(vs[_y])
            _na1 += 1

            ns[_y] = min(3, max(1, ns[_y]))
            _nu1 += ns[_y]

        _yy = _ys[min(len(vs), _id+1): min(len(vs), _id + 25)] if \
            _id + 1 < len(vs) else [_ys[_id+1]]

        for _y in _yy:
            if _na2 > 0 and (_nu2 >= _min_num_nonforest) and (abs(_y - y) > _max_year_span):
                break

            if vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
                continue

            if _cy < 0:
                _cy = _y

            _vv = max(0, vs[_y] - _val_others)
            if _use_simulate_vals:
                ns[_y] = min(5, max(2, ns[_y]))
                for _x in range(ns[_y]):
                    _cs2.append(lib_time_op.simulate_val(_vv, es[_y]))
            else:
                _cs2.append(_vv)

            _as2.append(vs[_y])
            _nu2 += ns[_y]
            _na2 += 1

            # if _cy >= 0 and (_na2 >= 3 or _nu2 > _min_ob_num):
            if _cy >= 0 and (_na2 >= _min_years) and (_nu2 >= _min_num_nonforest):
                break

    if _debug:
        print('min years: %s, %d/%d 1) num(%d/%d, %d/%d) min_num: %s, %s' % \
                (_min_years, y, _cy, _nu1, _na1, _nu2, _na2, _min_num_forest, _min_num_nonforest))
        print('# values 1:', [_val_forest - _c for _c in _cs1])
        print('# values 2:', [_val_others + _c for _c in _cs2])

    if _nu1 < _min_num_forest or _nu2 < _min_num_nonforest:
        if _debug:
            print('*1', _nu1, _min_num_forest, _nu2, _min_num_nonforest)
        return -1, 0, (-9999, -9999)

    if _na1 < _min_years_valid or _na2 < _min_years_valid:
        if _debug:
            print('*2', _na1, _min_years_valid, _na2, _min_years_valid)
        return -1, 0, (-9999, -9999)

    _mm1, _pp1 = lib_time_op.rmse(_cs1)
    _mm2, _pp2 = lib_time_op.rmse(_cs2)
    _mma, _ppa = lib_time_op.rmse(_cs1 + _cs2)
    _mmz, _ppz = lib_time_op.rmse([_a - _mm1 for _a in _cs1] +
                                  [_a - _mm2 for _a in _cs2])

    _mmt, _ppt = lib_time_op.rmse(list(vs.values()))

    if None in (_mm1, _pp1, _mm2, _pp2, _mma, _ppa, _mmz, _ppz, _mmt, _ppt):
        return -1, 0, (-9999, -9999)

    _min_dif = config.getfloat('loss', 'min_dif', 30.0)
    _mmm_dif = config.getfloat('loss', 'max_dif_nochange', 15.0)
    _min_dif_adjust = config.getfloat('loss', 'min_dif_adjust', 0.6)
    
    if flag_gain:
        _min_dif *= _min_dif_adjust
        
    # _max_dif = config.getfloat('loss', 'max_dif', 10.0)
    # _max_val = config.getfloat('loss', 'max_val', 35.0)
    _min_std1 = config.getfloat('loss', 'min_std1', 5.0)
    _min_std2 = config.getfloat('loss', 'min_std2', 5.0)
    _min_std = config.getfloat('loss', 'min_std', 5.0)

    _val_dif = (_val_forest - max(0, _mm1)) - (max(0, _mm2) + _val_others)
    
    _ppu = min(30, math.hypot(_ppt, config.getfloat('conf', 'avg_forest_err')))

    _z = lib_time_op.z_test(_val_forest - _mm1, _mm2 + _val_others,
                            0, _ppu, _ppu, _nu1, _nu2)
    # _z = lib_time_op.z_test(_val_forest - _mm1, _mm2 + _val_others,
    #                         0, _ppu, _ppu, _na1, _na2)

    if _debug:
        print('*', _ppz)
        print(('%d/%d 2) num(%d/%d, %d/%d) p1: (%0.2f, %0.2f) ' + \
              'p2: (%0.2f, %0.2f) dif: %.2f val: %.2f (%.2f) Z: %s') % \
              (y, _cy, _nu1, _na1, _nu2, _na2, _val_forest - _mm1, _pp1,
               _mm2 + _val_others, _pp2, _val_dif, _mmz, _ppt, str(_z)))

    _e = _val_dif * 100.0 / (_val_forest - _val_others)

    if _val_dif < 0:
        return -1, 0, (_e, _z[1])
        
    if _debug:
        print('val dif: %s, %s' % (_val_dif, _min_dif))

    # if (_val_forest - _mm1 < 45) or (_mm2 + _val_others > 60):
    # if _na2 > 1 and (_mm2 + _val_others) > config.getint('conf', 'min_nonforest', 55):
    if (_mm2 + _val_others) > config.getint('loss', 'min_nonforest', 60):
        if _debug:
            print('$ -1')
        return -1, 0, (_e, _z[1])

    if _z[1] > _min_change_prob:
        if _debug:
            print('$ -2', _z[1], _min_change_prob)
        return -1, 0, (_e, _z[1])

    if _val_dif > _min_dif:
        if _debug:
            print('$ +1, %s, %s' % (_val_dif, _min_dif))
        return _cy, -1, (_e, _z[1])
        
    if _val_dif < _mmm_dif:
        if _debug:
            print('$ -2.1, %s, %s' % (_val_dif, _mmm_dif))
        return -1, 0, (_e, _z[1])

    # if (_pp1 > _min_std1) or (_pp2 > _min_std2 and _ppz > _min_std):
    # if (y >= 1984 and _pp1 > _min_std1) or (_pp2 > _min_std2 and _ppz > _min_std):
    if (y >= 1984 and _pp1 > _min_std1) or (_pp2 > _min_std2 and _ppz > _min_std):
        if _debug:
            print('$ -3', _pp1, _min_std1, _pp2, _min_std2, _ppz, _min_std)
        return -1, 0, (_e, _z[1])

    if _debug:
        from scipy import stats
        print(_as1, _as2)
        print('**', stats.linregress(list(range(len(_as1 + _as2))),
                                     [_a - _as1[-1] for _a in _as1] +
                                     [_a - _as2[0] for _a in _as2]))

    _min_z = config.getfloat('loss', 'min_z', 0.05)

    if min(_na1, _na2) > 1 and _z[1] < _min_z:
        if _debug:
            print('$ +2, %s, %s, %s, %s' % (_na1, _na2, _cy, _z[1]))
        return _cy, -1, (_e, _z[1])

    if _debug:
        print('$ -4')

    return -1, 0, (_e, _z[1])

