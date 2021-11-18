from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os

from skimage.measure import regionprops
from csbdeep.utils import _raise

from ..utils import path_absolute, _normalize_grid
from ..matching import _check_label_array
from ..lib.stardist3d import c_star_dist3d, c_polyhedron_to_label, c_dist_to_volume, c_dist_to_centroid



def _cpp_star_dist3D(lbl, rays, grid=(1,1,1)):
    dz, dy, dx = rays.vertices.T
    grid = _normalize_grid(grid,3)

    return c_star_dist3d(lbl.astype(np.uint16, copy=False),
                         dz.astype(np.float32, copy=False),
                         dy.astype(np.float32, copy=False),
                         dx.astype(np.float32, copy=False),
                         int(len(rays)), *tuple(int(a) for a in grid))


def _py_star_dist3D(img, rays, grid=(1,1,1)):
    grid = _normalize_grid(grid,3)
    img = img.astype(np.uint16, copy=False)
    dst_shape = tuple(s // a for a, s in zip(grid, img.shape)) + (len(rays),)
    dst = np.empty(dst_shape, np.float32)

    dzs, dys, dxs = rays.vertices.T

    for i in range(dst_shape[0]):
        for j in range(dst_shape[1]):
            for k in range(dst_shape[2]):
                value = img[i * grid[0], j * grid[1], k * grid[2]]
                if value == 0:
                    dst[i, j, k] = 0
                else:

                    for n, (dz, dy, dx) in enumerate(zip(dzs, dys, dxs)):
                        x, y, z = np.float32(0), np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            z += dz
                            ii = int(round(i * grid[0] + z))
                            jj = int(round(j * grid[1] + y))
                            kk = int(round(k * grid[2] + x))
                            if (ii < 0 or ii >= img.shape[0] or
                                        jj < 0 or jj >= img.shape[1] or
                                        kk < 0 or kk >= img.shape[2] or
                                        value != img[ii, jj, kk]):
                                dist = np.sqrt(x * x + y * y + z * z)
                                dst[i, j, k, n] = dist
                                break

    return dst


def _ocl_star_dist3D(lbl, rays, grid=(1,1,1)):
    from gputools import OCLProgram, OCLArray, OCLImage

    grid = _normalize_grid(grid,3)

    # if not all(g==1 for g in grid):
    #     raise NotImplementedError("grid not yet implemented for OpenCL version of star_dist3D()...")

    res_shape = tuple(s//g for s, g in zip(lbl.shape, grid))

    lbl_g = OCLImage.from_array(lbl.astype(np.uint16, copy=False))
    dist_g = OCLArray.empty(res_shape + (len(rays),), dtype=np.float32)
    rays_g = OCLArray.from_array(rays.vertices.astype(np.float32, copy=False))

    program = OCLProgram(path_absolute("kernels/stardist3d.cl"), build_options=['-D', 'N_RAYS=%d' % len(rays)])
    program.run_kernel('stardist3d', res_shape[::-1], None,
                       lbl_g, rays_g.data, dist_g.data,
                       np.int32(grid[0]),np.int32(grid[1]),np.int32(grid[2]))

    return dist_g.get()


def star_dist3D(lbl, rays, grid=(1,1,1), mode='cpp'):
    """lbl assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    grid = _normalize_grid(grid,3)
    if mode == 'python':
        return _py_star_dist3D(lbl, rays, grid=grid)
    elif mode == 'cpp':
        return _cpp_star_dist3D(lbl, rays, grid=grid)
    elif mode == 'opencl':
        return _ocl_star_dist3D(lbl, rays, grid=grid)
    else:
        _raise(ValueError("Unknown mode %s" % mode))


def polyhedron_to_label(dist, points, rays, shape, prob=None, thr=-np.inf, labels=None, mode="full", verbose=True):
    """
    creates labeled image from stardist representations

    mode can be "full", "kernel", or "bbox"

    :param dist: (n_points,n_rays)
    :param points: (n_points, 3)
    :param rays: RaysSphere objects with vertices and faces
    :param prob: (n_points,)
    :shape :


    :param thr:
    :return:
    """
    if len(points) == 0:
        if verbose:
            print("warning: empty list of points (returning background-only image)")
        return np.zeros(shape, np.uint16)

    dist = np.asanyarray(dist)
    points = np.asanyarray(points)

    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if labels is None:
        labels = np.arange(1, len(points) + 1)

    if np.amin(dist) <= 0:
        raise ValueError("distance array should be positive!")

    prob = np.ones(len(points)) if prob is None else np.asanyarray(prob)

    if dist.ndim != 2:
        raise ValueError("dist should be 2 dimensional but has shape %s" % str(dist.shape))

    if dist.shape[1] != len(rays):
        raise ValueError("inconsistent number of rays!")

    if len(prob) != len(points):
        raise ValueError("len(prob) != len(points)")

    if len(labels) != len(points):
        raise ValueError("len(labels) != len(points)")

    modes = {"full": 0, "kernel": 1, "hull": 2, "bbox": 3, "debug": 4}

    if not mode in modes:
        raise KeyError("Unknown render mode '%s' , allowed:  %s" % (mode, tuple(modes.keys())))

    lbl = np.zeros(shape, np.uint16)

    # filter points
    ind = np.where(prob >= thr)[0]
    if len(ind) == 0:
        if verbose:
            print("warning: no points found with probability>= {thr:.4f} (returning background-only image)".format(thr=thr))
        return lbl

    prob = prob[ind]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    # sort points with decreasing probability
    ind = np.argsort(prob)[::-1]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_polyhedron_to_label(_prep(dist, np.float32),
                                 _prep(points, np.float32),
                                 _prep(rays.vertices, np.float32),
                                 _prep(rays.faces, np.int32),
                                 _prep(labels, np.int32),
                                 np.int(modes[mode]),
                                 np.int(verbose),
                                 shape
                                 )


def relabel_image_stardist3D(lbl, rays, verbose=False, **kwargs):
    """relabel each label region in `lbl` with its star representation"""
    _check_label_array(lbl, "lbl")

    dist_all = star_dist3D(lbl, rays, **kwargs)

    regs = regionprops(lbl)

    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regs))
    labs = np.array(tuple(r.label for r in regs))
    dist = np.array(tuple(dist_all[p[0], p[1], p[2]] for p in points))
    dist = np.maximum(dist, 1e-3)

    lbl_new = polyhedron_to_label(dist, points, rays, shape=lbl.shape, labels=labs, verbose=verbose)
    return lbl_new


def dist_to_volume(dist, rays):
    """ returns areas of polyhedra
        dist.shape = (nz,ny,nx,nrays)
    """
    dist = np.asanyarray(dist)
    dist.ndim == 4 or _raise(ValueError("dist.ndim = {dist.ndim} but should be 4".format(dist = dist)))
    dist.shape[-1]== len(rays) or _raise(ValueError("dist.shape[-1] = {d} but should be {l}".format(d = dist.shape[-1], l = len(rays))))

    dist = np.ascontiguousarray(dist.astype(np.float32, copy=False))

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_dist_to_volume(_prep(dist, np.float32),
                          _prep(rays.vertices, np.float32),
                          _prep(rays.faces, np.int32))


def dist_to_centroid(dist, rays, mode = 'absolute'):
    """ returns centroids of polyhedra

        dist.shape = (nz,ny,nx,nrays)
        mode = 'absolute' or 'relative'

    """
    dist.ndim == 4 or _raise(ValueError("dist.ndim = {dist.ndim} but should be 4".format(dist = dist)))
    dist.shape[-1]== len(rays) or _raise(ValueError("dist.shape[-1] = {d} but should be {l}".format(d = dist.shape[-1], l = len(rays))))
    dist = np.ascontiguousarray(dist.astype(np.float32, copy=False))

    mode in ('absolute', 'relative') or _raise(ValueError("mode should be either 'absolute' or 'relative'"))

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_dist_to_centroid(_prep(dist, np.float32),
                          _prep(rays.vertices, np.float32),
                          _prep(rays.faces, np.int32),
                              np.int32(mode=='absolute'))
