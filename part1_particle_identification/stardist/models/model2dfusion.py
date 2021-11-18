from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import math
from tqdm import tqdm

from distutils.version import LooseVersion
import keras
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model

from csbdeep.models import BaseConfig
from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import CARETensorBoard
from csbdeep.data import sample_patches_from_multiple_stacks
from skimage.segmentation import clear_border

from .base import StarDistBase, StarDistDataBase
from ..utils import edt_prob, _normalize_grid
from ..geometry import star_dist, dist_to_coord, polygons_to_label
from ..nms import non_maximum_suppression



class StarDistData2D(StarDistDataBase):

    def __init__(self, X, Y, batch_size, n_rays, patch_size=(256,256), b=32, grid=(1,1), shape_completion=False, augmenter=None, **kwargs):

        super().__init__(X=X, Y=Y, n_rays=n_rays, grid=grid,
                         batch_size=batch_size, patch_size=patch_size,
                         augmenter=augmenter, **kwargs)

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            self.b = slice(b,-b),slice(b,-b)
        else:
            self.b = slice(None),slice(None)

        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'


    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = list(self.perm[idx])

        arrays = [sample_patches_from_multiple_stacks((self.Y[k],) + self.channels_as_tuple(self.X[k]),
                                                      patch_size=self.patch_size, n_samples=1,
                                                      patch_filter=self.no_background_patches_cached(k)) for k in idx]
        if self.n_channel is None:
            X, Y = list(zip(*[(x[0][self.b],y[0]) for y,x in arrays]))
        else:
            X, Y = list(zip(*[(np.stack([_x[0] for _x in x],axis=-1)[self.b], y[0]) for y,*x in arrays]))

        X, Y = self.augmenter(X, Y)

        prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            dist      = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode)[self.b+(slice(None),)] for lbl in Y_cleared])
            dist_mask = np.stack([edt_prob(lbl[self.b]) for lbl in Y_cleared])
        else:
            dist      = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode) for lbl in Y])
            dist_mask = prob

        X = np.stack(X)
        if X.ndim == 3: # input image has no channel axis
            X = np.expand_dims(X,-1)
        prob = np.expand_dims(prob,-1)
        dist_mask = np.expand_dims(dist_mask,-1)

        # subsample wth given grid
        dist_mask = dist_mask[self.ss_grid]
        prob      = prob[self.ss_grid]
        dist      = dist[self.ss_grid]

        return [X,dist_mask], [prob,dist]



class Config2D(BaseConfig):
    """Configuration for a :class:`StarDist2D` model.

    Parameters
    ----------
    axes : str or None
        Axes of the input images.
    n_rays : int
        Number of radial directions for the star-convex polygon.
        Recommended to use a power of 2 (default: 32).
    n_channel_in : int
        Number of channels of given input image (default: 1).
    grid : (int,int)
        Subsampling factors (must be powers of 2) for each of the axes.
        Model will predict on a subsampled grid for increased efficiency and larger field of view.
    backbone : str
        Name of the neural network architecture to be used as backbone.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    unet_pool : (int,int)
        Maxpooling size for all (U-Net) convolution layers.
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    unet_* : *
        Additional parameters for U-net backbone.
    train_shape_completion : bool
        Train model to predict complete shapes for partially visible objects at image boundary.
    train_completion_crop : int
        If 'train_shape_completion' is set to True, specify number of pixels to crop at boundary of training patches.
        Should be chosen based on (largest) object sizes.
    train_patch_size : (int,int)
        Size of patches to be cropped from provided training images.
    train_background_reg : float
        Regularizer to encourage distance predictions on background regions to be 0.
    train_dist_loss : str
        Training loss for star-convex polygon distances ('mse' or 'mae').
    train_loss_weights : tuple of float
        Weights for losses relating to (probability, distance)
    train_epochs : int
        Number of training epochs.
    train_steps_per_epoch : int
        Number of parameter update steps per epoch.
    train_learning_rate : float
        Learning rate for training.
    train_batch_size : int
        Batch size for training.
    train_n_val_patches : int
        Number of patches to be extracted from validation images (``None`` = one patch per image).
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress.
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.
    use_gpu : bool
        Indicate that the data generator should use OpenCL to do computations on the GPU.

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes='YX', n_rays=32, n_channel_in=1, grid=(1,1), backbone='unet', **kwargs):
        """See class docstring."""

        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1+n_rays)

        # directly set by parameters
        self.n_rays                    = int(n_rays)
        self.grid                      = _normalize_grid(grid,2)
        self.backbone                  = str(backbone).lower()

        # default config (can be overwritten by kwargs below)
        if self.backbone == 'unet':
            self.unet_n_depth          = 3
            self.unet_kernel_size      = 3,3
            self.unet_n_filter_base    = 32
            self.unet_n_conv_per_depth = 2
            self.unet_pool             = 2,2
            self.unet_activation       = 'relu'
            self.unet_last_activation  = 'relu'
            self.unet_batch_norm       = False
            self.unet_dropout          = 0.0
            self.unet_prefix           = ''
            self.net_conv_after_unet   = 128
        else:
            # TODO: resnet backbone for 2D model?
            raise ValueError("backbone '%s' not supported." % self.backbone)

        if backend_channels_last():
            self.net_input_shape       = None,None,self.n_channel_in
            self.net_mask_shape        = None,None,1
        else:
            self.net_input_shape       = self.n_channel_in,None,None
            self.net_mask_shape        = 1,None,None

        self.train_shape_completion    = False
        self.train_completion_crop     = 32
        self.train_patch_size          = 256,256
        self.train_background_reg      = 1e-4

        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = 1,0.2
        self.train_epochs              = 400
        self.train_steps_per_epoch     = 100
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 4
        self.train_n_val_patches       = None
        self.train_tensorboard         = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'epsilon' if LooseVersion(keras.__version__)<=LooseVersion('2.1.5') else 'min_delta'
        self.train_reduce_lr           = {'factor': 0.5, 'patience': 40, min_delta_key: 0}

        self.use_gpu                   = False

        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)



class StarDist2D(StarDistBase):
    """StarDist2D model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config=Config2D(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        self.config.backbone == 'unet' or _raise(NotImplementedError())

        input_img  = Input(self.config.net_input_shape, name='input')
        if backend_channels_last():
            grid_shape = tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[:-1])) + (1,)
        else:
            grid_shape = (1,) + tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[1:]))
        input_mask = Input(grid_shape, name='dist_mask')

        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        # maxpool input image to grid size
        pooled = np.array([1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet        = unet_block(**unet_kwargs)(pooled_img)
        if self.config.net_conv_after_unet > 0:
            unet    = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                             name='features', padding='same', activation=self.config.unet_activation)(unet)

        output_prob  = Conv2D(1,                  (1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist  = Conv2D(self.config.n_rays, (1,1), name='dist', padding='same', activation='linear')(unet)
        return Model([input_img,input_mask], [output_prob,output_dist])


    def train(self, X, Y, validation_data, augmenter=None, seed=None, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of input images.
        Y : :class:`numpy.ndarray`
            Array of label masks.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of X,Y validation arrays.
        augmenter : None or callable
            Function with expected signature ``Xbt, Ybt = augmenter(Xb, Yb)``
            that takes in batch input/label images (Xb,Yb) and returns
            transformed images (Xbt, Ybt) for the purpose of data augmentation
            during training. Not applied to validation images.
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        validation_data is not None or _raise(ValueError())
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        b = self.config.train_completion_crop if self.config.train_shape_completion else 0
        div_by = self._axes_div_by(axes)
        [(p-2*b) % d == 0 or _raise(ValueError(
            "'train_patch_size' - 2*'train_completion_crop' must be divisible by {d} along axis '{a}'".format(a=a,d=d) if self.config.train_shape_completion else
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            n_rays           = self.config.n_rays,
            patch_size       = self.config.train_patch_size,
            grid             = self.config.grid,
            shape_completion = self.config.train_shape_completion,
            b                = self.config.train_completion_crop,
            use_gpu          = self.config.use_gpu,
        )

        # generate validation data and store in numpy arrays
        data_val = StarDistData2D(*validation_data, batch_size=1, **data_kwargs)
        n_data_val = len(data_val)
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        ids = tuple(np.random.choice(n_data_val, size=n_take, replace=(n_take > n_data_val)))
        Xv, Mv, Pv, Dv = [None]*n_take, [None]*n_take, [None]*n_take, [None]*n_take
        for i,k in enumerate(ids):
            (Xv[i],Mv[i]),(Pv[i],Dv[i]) = data_val[k]
        Xv, Mv, Pv, Dv = np.concatenate(Xv,axis=0), np.concatenate(Mv,axis=0), np.concatenate(Pv,axis=0), np.concatenate(Dv,axis=0)
        data_val = [[Xv,Mv],[Pv,Dv]]

        data_train = StarDistData2D(X, Y, batch_size=self.config.train_batch_size, augmenter=augmenter, **data_kwargs)

        for cb in self.callbacks:
            if isinstance(cb,CARETensorBoard):
                # show dist for three rays
                _n = min(3, self.config.n_rays)
                cb.output_slices = [[slice(None)]*4,[slice(None)]*4]
                cb.output_slices[1][1+axes_dict(self.config.axes)['C']] = slice(0,(self.config.n_rays//_n)*_n,self.config.n_rays//_n)

        history = self.keras_model.fit_generator(generator=data_train, validation_data=data_val,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)
        self._training_finished()

        return history


    def _instances_from_prediction(self, img_shape, prob, dist, prob_thresh=None, nms_thresh=None, **nms_kwargs):
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms

        coord = dist_to_coord(dist, grid=self.config.grid)
        points = non_maximum_suppression(coord, prob, grid=self.config.grid,
                                         prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
        labels = polygons_to_label(coord, prob, points, shape=img_shape)
        return prob, labels, dict(coord=coord[points[:,0],points[:,1]], points=points, prob=prob[points[:,0],points[:,1]])


    def _axes_div_by(self, query_axes):
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        query_axes = axes_check_and_normalize(query_axes)
        assert len(self.config.unet_pool) == len(self.config.grid)
        div_by = dict(zip(
            self.config.axes.replace('C',''),
            tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
        ))
        return tuple(div_by.get(a,1) for a in query_axes)


    # def _axes_tile_overlap(self, query_axes):
    #     self.config.backbone == 'unet' or _raise(NotImplementedError())
    #     query_axes = axes_check_and_normalize(query_axes)
    #     assert len(self.config.unet_pool) == len(self.config.grid) == len(self.config.unet_kernel_size)
    #     # TODO: compute this properly when any value of grid > 1
    #     # all(g==1 for g in self.config.grid) or warnings.warn('FIXME')
    #     overlap = dict(zip(
    #         self.config.axes.replace('C',''),
    #         tuple(tile_overlap(self.config.unet_n_depth + int(np.log2(g)), k, p)
    #               for p,k,g in zip(self.config.unet_pool,self.config.unet_kernel_size,self.config.grid))
    #     ))
    #     return tuple(overlap.get(a,0) for a in query_axes)


    @property
    def _config_class(self):
        return Config2D
