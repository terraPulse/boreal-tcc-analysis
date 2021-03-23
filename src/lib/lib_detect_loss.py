'''
File: lib_detect_loss.py
Author: Min Feng
Version: 0.1
Create: 2018-08-16 15:58:34
Description:
'''


def detect_loss(ys, vs, es, ns, y):
    from gio import config
    from . import lib_time_op

    # _min_prob = config.getfloat('conf', 'min_prob', 0.5)
    _min_forest = config.getfloat('conf', 'min_forest', 50)
    _min_forest_change = config.getfloat('conf', 'min_forest_change', 65)
    _min_num = config.getfloat('conf', 'min_num', 5)
    # _min_ob_num = config.getfloat('conf', 'min_ob_num', 12)
    # _min_rate = config.getfloat('conf', 'min_rate', 0.8)

    _ys = sorted(ys)
    _id = _ys.index(y)

    if _id == len(_ys) - 1:
        return -1, 0, 0

    # _nc = 0.0
    # _yy = 0

    if vs[y] > 100:
        return -1, 0, 0

    if vs[y] > 100 or vs[y] < _min_forest or es[y] >= 100 or es[y] < 0:
        return -1, 0, 0

    _cs1 = []
    _cs2 = []

    _na1 = 0.0
    _na2 = 0.0

    _nu1 = 0.0
    _nu2 = 0.0

    _cy = -1

    _val_forest = config.getfloat('conf', 'val_forest')
    _val_others = config.getfloat('conf', 'val_others')
    _min_years = config.getint('conf', 'min_years', 3)

    if _id >= 0:
        _yy = _ys[:_id]
        _yy.reverse()

        # _yy = _ys[_id: max(-1, _id-25): -1] if _id > 0 else [_ys[_id]]
        for _y in _yy:
            if _na1 > 0 and (abs(_y - y) > 25):
                continue

            if vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
                continue

            _vv = max(0, _val_forest - vs[_y])
            for _x in range(ns[_y]):
                _cs1.append(_vv)

            _nu1 += ns[_y]
            _na1 += 1

            # if _na1 >= 3 or _nu1 > _min_num:
            if _na1 >= _min_years:
                break

        _y = y
        if _na1 > 0 and vs[_y] < _min_forest_change:
            _cy = _y
        else:
            _vv = _val_forest - vs[_y]
            for _x in range(ns[_y]):
                _cs1.append(_vv)

            _na1 += 1
            _nu1 += ns[_y]

        _yy = _ys[min(len(vs), _id+1): min(len(vs), _id + 25)] if \
            _id + 1 < len(vs) else [_ys[_id+1]]

        for _y in _yy:
            if _na2 > 0 and (abs(_y - y) > 25):
                break

            if vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
                continue

            if _cy < 0:
                _cy = _y

            if ns[_y] == 0:
                ns[_y] = 5

            _vv = max(0, vs[_y] - _val_others)
            for _x in range(ns[_y]):
                _cs2.append(_vv)

            _nu2 += ns[_y]
            _na2 += 1

            # if _cy >= 0 and (_na2 >= 3 or _nu2 > _min_ob_num):
            if _cy >= 0 and (_na2 >= _min_years):
                break

    # if config.getboolean('conf', 'debug'):
    #     print '%d/%d 1) num(%d/%d, %d/%d) ' % \
    #             (y, _cy, _nu1, _na1, _nu2, _na2)

    if config.getboolean('conf', 'debug'):
        print('%d/%d 1) num(%d/%d, %d/%d) min_num: %s' % \
                (y, _cy, _nu1, _na1, _nu2, _na2, _min_num))

    if _nu1 < _min_num or _nu2 < _min_num:
        return -1, 0, 0

    if _na1 <= 0 or _na2 <= 0:
        return -1, 0, 0

    _mm1, _pp1 = lib_time_op.rmse(_cs1)
    _mm2, _pp2 = lib_time_op.rmse(_cs2)
    _mma, _ppa = lib_time_op.rmse(_cs1 + _cs2)
    _mmt, _ppt = lib_time_op.rmse(list(vs.values()))

    _min_dif = config.getfloat('conf', 'min_dif', 30.0)
    # _max_dif = config.getfloat('conf', 'max_dif', 10.0)
    _max_val = config.getfloat('conf', 'max_val', 20.0)
    _min_std = config.getfloat('conf', 'min_std', 5.0)

    _val_dif = (_val_forest - max(0, _mm1)) - (max(0, _mm2) + _val_others)
    # _z = z_test(_val_forest - _mm1, _mm2 + _val_others, 0,
    #             _ppt, _ppt, _nu1, _nu2)

    _z = lib_time_op.z_test(_val_forest - _mm1, _mm2 + _val_others,
                            0, _ppt, _ppt, _na1, _na2)

    if config.getboolean('conf', 'debug'):
        print(('%d/%d 2) num(%d/%d, %d/%d) p1: (%0.2f, %0.2f) p2: ' + \
              '(%0.2f, %0.2f) dif: %.2f val: %.2f (%.2f) Z: %s') % \
                (y, _cy, _nu1, _na1, _nu2, _na2, _mm1, _pp1, _mm2,
                 _pp2, _val_dif, _mma, _ppa, str(_z)))

    if _val_dif > _min_dif:
        return _cy, -1, _ppa

    _min_z = config.getfloat('conf', 'min_z', 0.05)
    if _val_dif > 0 or _z[1] < _min_z:
        return _cy, -1, _ppa

    if _val_dif < 0 or _z[1] > _min_z:
        return -1, 0, 0

    if max(_mm1, _mm2) > _max_val:
        return -1, 0, 0

    if max(_pp1, _pp2) > _min_std:
        return -1, 0, 0

    # if _rmse > _min_rmse:
    #     return -1, 0, 0
    #
    # if _mm > _min_diff or _pa < _min_prob:
    #     return -1, 0, 0

    if config.getboolean('conf', 'debug'):
        print('*', y, _cy, _ppa)

    return _cy, -1, _ppa

def detect_loss_2(ys, vs, es, ns, y):
    from gio import config
    from . import lib_time_op

    # _min_prob = config.getfloat('conf', 'min_prob', 0.5)
    _min_forest = config.getfloat('conf', 'min_forest', 50)
    _min_forest_change = config.getfloat('conf', 'min_forest_change', 65)
    _min_num = config.getfloat('conf', 'min_num', 5)
    # _min_ob_num = config.getfloat('conf', 'min_ob_num', 12)
    # _min_rate = config.getfloat('conf', 'min_rate', 0.8)

    _ys = sorted(ys)
    _id = _ys.index(y)

    if _id == len(_ys) - 1:
        return -1, 0, 0

    # _nc = 0.0
    # _yy = 0

    if vs[y] > 100:
        return -1, 0, 0

    # print 'forest', y, vs[y], _min_forest
    if vs[y] > 100 or vs[y] < _min_forest or es[y] >= 100 or es[y] < 0:
        return -1, 0, 0

    if _id < len(_ys) - 2 and vs[_ys[_id]] <= vs[_ys[_id+1]]:
        return -1, 0, 0

    _cs1 = []
    _cs2 = []

    _as1 = []
    _as2 = []

    _na1 = 0.0
    _na2 = 0.0

    _nu1 = 0.0
    _nu2 = 0.0

    _cy = -1

    _val_forest = config.getfloat('conf', 'val_forest')
    _val_others = config.getfloat('conf', 'val_others')
    _min_years = config.getint('conf', 'min_years', 3)
    _max_year_span = config.getfloat('conf', 'max_year_span', 5)

    if _id >= 0:
        _yy = _ys[:_id]
        _yy.reverse()

        # _yy = _ys[_id: max(-1, _id-25): -1] if _id > 0 else [_ys[_id]]
        for _y in _yy:
            if _na1 > 0 and (abs(_y - y) > _max_year_span):
                continue

            if vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
                continue

            _vv = max(0, _val_forest - vs[_y])
            for _x in range(ns[_y]):
                _cs1.append(lib_time_op.simulate_val(_vv, es[_y]))

            _as1.append(vs[_y])
            _nu1 += ns[_y]
            _na1 += 1

            # if _na1 >= 3 or _nu1 > _min_num:
            if _na1 >= _min_years + 1:
                break

        _y = y
        if _na1 > 0 and vs[_y] < _min_forest_change:
            _cy = _y
        else:
            _vv = _val_forest - vs[_y]
            for _x in range(ns[_y]):
                _cs1.append(lib_time_op.simulate_val(_vv, es[_y]))

            _as1.append(vs[_y])
            _na1 += 1
            _nu1 += ns[_y]

        _yy = _ys[min(len(vs), _id+1): min(len(vs), _id + 25)] if \
            _id + 1 < len(vs) else [_ys[_id+1]]

        for _y in _yy:
            if _na2 > 0 and (abs(_y - y) > _max_year_span):
                break

            if vs[_y] > 100 or es[_y] >= 100 or es[_y] < 0:
                continue

            if _cy < 0:
                _cy = _y

            if ns[_y] == 0:
                ns[_y] = 5

            _vv = max(0, vs[_y] - _val_others)
            for _x in range(ns[_y]):
                _cs2.append(lib_time_op.simulate_val(_vv, es[_y]))

            _as2.append(vs[_y])
            _nu2 += ns[_y]
            _na2 += 1

            # if _cy >= 0 and (_na2 >= 3 or _nu2 > _min_ob_num):
            if _cy >= 0 and (_na2 >= _min_years):
                break

    # if config.getboolean('conf', 'debug'):
    #     print '%d/%d 1) num(%d/%d, %d/%d) ' % \
    #             (y, _cy, _nu1, _na1, _nu2, _na2)

    if config.getboolean('conf', 'debug'):
        print('%d/%d 1) num(%d/%d, %d/%d) min_num: %s' % \
                (y, _cy, _nu1, _na1, _nu2, _na2, _min_num))

    if _nu1 < _min_num or _nu2 < _min_num:
        return -1, 0, 0

    if _na1 <= 0 or _na2 <= 0:
        return -1, 0, 0

    _mm1, _pp1 = lib_time_op.rmse(_cs1)
    _mm2, _pp2 = lib_time_op.rmse(_cs2)
    _mma, _ppa = lib_time_op.rmse(_cs1 + _cs2)
    _mmz, _ppz = lib_time_op.rmse([_a - _mm1 for _a in _cs1] +
                                  [_a - _mm2 for _a in _cs2])

    _mmt, _ppt = lib_time_op.rmse(list(vs.values()))

    _min_dif = config.getfloat('conf', 'min_dif', 30.0)
    # _max_dif = config.getfloat('conf', 'max_dif', 10.0)
    # _max_val = config.getfloat('conf', 'max_val', 35.0)
    _min_std1 = config.getfloat('conf', 'min_std1', 5.0)
    _min_std2 = config.getfloat('conf', 'min_std2', 5.0)
    _min_std = config.getfloat('conf', 'min_std', 5.0)

    _val_dif = (_val_forest - max(0, _mm1)) - (max(0, _mm2) + _val_others)

    import math
    _ppu = math.hypot(_ppt, config.getfloat('conf', 'avg_forest_err'))

    _z = lib_time_op.z_test(_val_forest - _mm1, _mm2 + _val_others,
                            0, _ppu, _ppu, _nu1, _nu2)

    if config.getboolean('conf', 'debug'):
        print('*', _ppz)
        print(('%d/%d 2) num(%d/%d, %d/%d) p1: (%0.2f, %0.2f) ' + \
              'p2: (%0.2f, %0.2f) dif: %.2f val: %.2f (%.2f) Z: %s') % \
              (y, _cy, _nu1, _na1, _nu2, _na2, _val_forest - _mm1, _pp1,
               _mm2 + _val_others, _pp2, _val_dif, _mma, _ppa, str(_z)))

    _e = 100 - _val_dif

    if _val_dif < 0:
        return -1, 0, _e

    # if (_val_forest - _mm1 < 45) or (_mm2 + _val_others > 60):
    if _na2 > 1 and (_mm2 + _val_others) > config.getint('conf', 'min_nonforest', 55):
        return -1, 0, _e

    if _z[1] > 0.08:
        return -1, 0, _e

    if (_pp1 > _min_std1) or (_pp2 > _min_std2 and _ppz > _min_std):
        return -1, 0, _e

    if _val_dif > _min_dif:
        return _cy, -1, _e

    # _mean = lambda x: sum(x) / len(x)
    if config.getboolean('conf', 'debug'):
        from scipy import stats
        print(_as1, _as2)
        print('**', stats.linregress(list(range(len(_as1 + _as2))),
                                     [_a - _as1[-1] for _a in _as1] +
                                     [_a - _as2[0] for _a in _as2]))

    _min_z = config.getfloat('conf', 'min_z', 0.05)

    # if max(_pp1, _pp2) > _min_std:
    # if max(_mm1, _mm2) > _max_val:
    # if _mma > _max_val:
    #     return -1, 0, 0

    if _z[1] < _min_z:
        return _cy, -1, _e

    # if _rmse > _min_rmse:
    #     return -1, 0, 0
    #
    # if _mm > _min_diff or _pa < _min_prob:
    #     return -1, 0, 0

    # if config.getboolean('conf', 'debug'):
    #     print '*', y, _cy, _ppa

    return -1, 0, _e


# @deprecated
def _detect_change_annual(ys, vs, es, ns, y):
    # print '####', y, vs[y], es[y], ns[y]
    from gio import config
    from . import lib_time_op

    _min_prob = config.getfloat('conf', 'min_prob', 0.5)
    _min_forest = config.getfloat('conf', 'min_forest', 75)
    _min_num = config.getfloat('conf', 'min_num', 5)
    _min_rate = config.getfloat('conf', 'min_rate', 0.8)

    _ys = sorted(ys)
    _id = _ys.index(y)

    _cs = []

    _na = 0.0
    _nc = 0.0
    # _yy = 0

    if vs[y] > 100:
        return -1, 0, 0

    if vs[y] < _min_forest or es[y] >= 100 or es[y] < 0 or ns[y] < _min_num:
        return -1, 0, 0

    # _t, _p = estimate_change(vs[y], es[y], 50, 0, 0.80, 0.80)
    # if _t >= 0:
    #     return -1, 0, 0

    # for _y in _ys[min(len(_ys), _id+1): min(len(_ys), _id+5)]:

    _na_0 = 0.0
    _nc_0 = 0.0
    _cs_0 = []

    if _id > 0:
        for _y in _ys[max(0, _id-2): max(0, _id)]:
            if _na_0 > 0 and (abs(_y - y) > 15):
                continue

            # print es[_y] >= 100 or es[_y] < 0 or ns[_y] < _min_num
            if es[_y] >= 100 or es[_y] < 0 or ns[_y] < _min_num:
                continue

            _na_0 += 1.0
            # _yy = _y

            # if vs[_y] > _min_forest:
            #     continue

            _t, _p = lib_time_op.estimate_change(vs[y], es[y], vs[_y], es[_y],
                                     _min_prob, _min_prob, ns[y], ns[_y])
            # print vs[y], es[y], ns[y], vs[_y], es[_y], ns[_y], _t, _p
            # if y == 1975:
            # print '#', y, _y, _t, _p, vs[y], es[y], vs[_y], es[_y]

            _cs_0.append((_y, _t, _p))

            if _t == 0:
                _nc_0 += 1.0

            if _na_0 >= 10:
                break

    # print '++', y, _nc_0, _na_0
    if _na_0 < 1 or _nc_0 / _na_0 < _min_rate:
        return -1, 0, 0

    for _y in _ys[min(len(_ys), _id+1): min(len(_ys), _id+10)]:
        # if _na > 0 and (abs(_y - _yy) > 5 or (_y - y) > 15):
        if _na > 0 and ((_y - y) > 15):
            continue

        if es[_y] >= 100 or es[_y] < 0 or ns[_y] < _min_num:
            continue

        _na += 1.0
        # _yy = _y

        if vs[_y] > _min_forest:
            continue

        # _t, _p = estimate_change(vs[y], es[y], vs[_y], es[_y], 0.85, 0.85)
        _t, _p = lib_time_op.estimate_change(vs[y], es[y], vs[_y], es[_y], _min_prob,
                                 _min_prob, ns[y], ns[_y])
        # print vs[y], es[y], ns[y], vs[_y], es[_y], ns[_y], _t, _p
        # if y == 1975:
        # print '#', y, _y, _t, _p, vs[y], es[y], vs[_y], es[_y]

        _cs.append((_y, _t, _p))

        if _t < 0:
            _nc += 1.0

        if _na >= 10:
            break

    # print '+', y, _na, _nc
    # if (_na > 0.0) and (_na <= (1 + _nc)) and (_nc / _na) > 0.5:
    # print '#', _nc, _na

    if _na <= 1:
        return -1, 0, 0

    _r = (_nc / _na)

    if (_na > 0.0) and _r > _min_rate:
        for _i in range(len(_cs)):
            if _cs[_i][1] < 0:
                # print _cs[_i]
                _p = _r * _cs[_i][2]
                if _p > _min_prob:
                    return _cs[_i]

    return -1, 0, 0

