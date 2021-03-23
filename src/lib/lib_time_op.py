'''
File: lib_time_op.py
Author: Min Feng
Version: 0.1
Create: 2018-08-16 15:58:00
Description:
'''

import numpy as np
from numpy import sqrt, abs
from scipy.stats import norm
from gio import geo_raster as ge
from . import lib_time_op

def z_test(x1, x2, mudiff, sd1, sd2, n1, n2):
    pooled_se = sqrt(sd1**2/n1 + sd2**2/n2)

    if pooled_se == 0:
        pooled_se = 0.0000001

    z = ((x1 - x2) - mudiff) / pooled_se
    pval = 2 * norm.sf(abs(z))

    return z, pval

def mean(vs):
    if len(vs) <= 0:
        return None

    return sum(vs) / len(vs)

def std(vs, mm):
    if mm is None:
        return None

    _sz = len(vs)

    if _sz <= 1:
        return None

    _ts = sum([(_v - mm) ** 2 for _v in vs]) / (_sz - 1)
    if _ts <= 0:
        return None

    return _ts ** 0.5

def rmse(vs):
    if len(vs) == 0:
        return None, 0

    if len(vs) < 2:
        return vs[0], 0
    
    _aa = np.array(vs)

    _mm = _aa.mean()
    _pp = _aa.std()

    # _rmse = (sum([_a ** 2 for _a in _cs]) / len(_cs)) ** 0.5

    # _mm = mean(vs)
    # _pp = std(vs, _mm)

    return _mm, _pp


def simulate_val(m, s):
    return m

    # import random
    #
    # _v = random.gauss(m, s)
    # if _v < 0:
    #     return 0
    #
    # if _v > 100:
    #     return 100
    #
    # return _v


def read_pt(f, pt):
    _bnd = ge.open(f).get_band()
    _pt = pt.project_to(_bnd.proj)

    return _bnd.read_location(_pt.x, _pt.y)


def estimate_change_prob(c1, e1, c2, e2, n1, n2):
    if c1 == c2:
        return 0.0

    _z, _p = lib_time_op.z_test(c1, c2, 0, e1, e2, n1, n2)

    return 1 - _p

    # import scipy.stats

    # _ee = max(0.0000001, (e1 ** 2 + e2 ** 2)) ** 0.5
    # _cd = abs(c1 - c2)
    # _cp = _cd / _ee

    # _pp = scipy.stats.norm.cdf(_cp)

    # return float(_pp)


def estimate_change(c1, e1, c2, e2, pb1=0.68, pb2=0.68, n1=10, n2=10):
    # if min(c1, c2) > 20:
    #     return 0, 0.0

    if c1 < 0 or c2 < 0 or e1 < 0 or e2 < 0:
        return 0, 0.0

    if c1 > 100 or c2 > 100:
        return 0, 0.0

    # _min_dif = max(10, e1, e2)
    # if abs(c1 - c2) < _min_dif:
    #     return 0, 0.0

    # _p = estimate_change_prob(c1, e1, c2, e2, n1, n2)

    # if (c1 > c2) and (_p >= pb1):
    #     return -1, _p

    # if (c1 < c2) and (_p >= pb2):
    #     return 1, _p

    # return 0, _p

    # print cmp(c1, 50), cmp(50, c2)
    _cmp = lambda x, y: 0 if x == y else (1 if x > y else -1)
    if _cmp(c1, 50) != _cmp(50, c2):
        _p = estimate_change_prob(c1, e1, c2, e2, n1, n2)
        return 0, _p

    _p1 = estimate_change_prob(c1, e1, 50, e1, n1, n1)
    _p2 = estimate_change_prob(50, e2, c2, e2, n2, n2)

    # import math
    _p = (_p1 + _p2) / 2.0

    if (c1 > c2) and (_p >= pb1):
        return -1, _p

    if (c1 < c2) and (_p >= pb2):
        return 1, _p

    return 0, _p

