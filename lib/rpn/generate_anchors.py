# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import math

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=8, ratios=[0.5, 1, 2],  # change base_size to use as radius
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([base_size, base_size, base_size]) - 1  # change to use circle
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):  # change implementation
    """
    Return radius, x center, and y center for an anchor (window).
    """

    r = anchor[2]
    x_ctr = anchor[0]
    y_ctr = anchor[1]
    return r, x_ctr, y_ctr

def _mkanchors(rs, x_ctr, y_ctr):  # change implementation
    """
    Given a vector of radiuses (rs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    rs = rs[:, np.newaxis]
    anchors = np.hstack((x_ctr, y_ctr, rs))
    return anchors

def _ratio_enum(anchor, ratios):  # change implementation
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    r, x_ctr, y_ctr = _whctrs(anchor)
    size = math.pi * r**2
    size_ratios = size / ratios
    rs = np.round(np.sqrt(size_ratios) / math.pi)
    anchors = _mkanchors(rs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):  # change implementation
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    r, x_ctr, y_ctr = _whctrs(anchor)
    rs = r * scales
    anchors = _mkanchors(rs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
