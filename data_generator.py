import numpy as np
import random
import tensorflow.keras
from tensorflow.keras.utils import Sequence
import imgaug as ia
import imgaug.augmenters as iaa
from utils import get_image
import cv2
import os
import pandas as pd
import re


class BaseBrainDataset:
    """
        Base class for generators of brain images.
    """
    def __init__(self, base_folder, reshape_to=None):
        """
        Loop over base folder and create a DataFrame `df` 
        with metadata of found images.

        :param base_folder:
        root folder of data set.

        """
        assert os.path.isdir(base_folder)
        self.base_folder = base_folder
        self.target_shape = reshape_to
        self.df = None
        self._set_data_frame()

    def _set_data_frame(self):
        """
        Loop over folders and saves file info on DataFrame.
        :return:
        """
        meta = {'atlas': [], 'coordinate': [], 'plane': [], 'is_mask': [],
               'image_type': [], 'path': []}
        for current_folder, sub_folders, files in os.walk(self.base_folder):
            for file in files:
                path = os.path.join(current_folder, file)
                if os.path.splitext(path)[-1] not in ['.md']: # ignore markdown files
                    # extract and pre-process meta
                    path = os.path.normpath(path)  # normalize path (eg remove douple seps)
                    *stuff, atlas, plane, is_mask, name = path.split(os.sep)
                    image_type = os.path.splitext(name)[-1]
                    assert plane in ['cor', 'sag', 'hor']
                    assert is_mask in ['images', 'masks']
                    is_mask = is_mask == 'masks'
                    # extract and pp coordinate
                    m = re.match("\w*x(\w*).\w*", name)
                    if m is None:
                        coordinate = np.nan
                    else:
                        assert len(m.groups()) == 1  # multiple x in name?
                        coordinate = int(m.groups()[0])
                    # assign
                    meta['atlas'].append(atlas)
                    meta['plane'].append(plane)
                    meta['coordinate'].append(coordinate)
                    meta['is_mask'].append(is_mask)
                    meta['image_type'].append(image_type)
                    meta['path'].append(path)
        self.df = pd.DataFrame(meta)

    def get_image_u8_from_path(self, path):
        """
        Load image from `path` in uint8 BGR mode. If `self.target_shape`
        is defined, reshape the image to target shape.
        :param path:
        :return:
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # IMREAD_COLOR always convert to BGR
        assert img.dtype == 'uint8'  # before augmentation type must be uint8
        if self.target_shape is not None or self.target_shape is not False:
            img = cv2.resize(img, self.target_shape)  # resize image
        return img


class DataGenerator(BaseBrainDataset, Sequence):
    """
        Data generator with augmentation for one shot atlas.
    """
    
    def __init__(self, base_folder, reshape_to=(512, 512), batch_size=128):
        super().__init__(base_folder=base_folder, reshape_to=reshape_to)
        self.batch_size = batch_size
        
        # fetch coronal whose posterior is multiple of 10
        mask_1 = self.df['plane'] == 'cor'
        mask_2 = self.df['atlas'] != 'brainmaps'
        mask_3 = self.df.coordinate % 10 == 0
        mask_4 = self.df.is_mask == False
        df_subset = self.df[mask_1 & mask_2 & mask_3 & mask_4].copy()
        
        # Perform tensor product to get pairs
        df_subset['dummy_key'] = 1
        df_1 = df_subset.copy()
        df_2 = df_subset.copy()
        self.df_pairs = df_1.merge(df_2, on='dummy_key')
        self.df_pairs.drop('dummy_key', axis=1, inplace=True)
        self.n_pairs = len(self.df_pairs)
        
        # shuffle pairs
        self.on_epoch_end()
        
    def __len__(self):
        """
            Get number of batches in an epoch (arbitrary).
        """
        return (132 ** 2) // self.batch_size
#         return self.n_pairs // self.batch_size
    
    def on_epoch_end(self):
        # shuffle couples
        self.df_pairs = self.df_pairs.sample(frac=1)
    
    def __getitem__(self, index):
        """
            Get batch of images/labels, both float32
            and normalized in between [0, 1].
        """
        batch_ixs = list(range(0, self.n_pairs, self.batch_size))
        batch_ix = batch_ixs[index]
        batch_df = self.df_pairs[batch_ix:batch_ix+self.batch_size]
        assert len(batch_df) == self.batch_size

        # create uint8 batch and labels
        batch_shape = (self.batch_size, 2,) + self.target_shape + (3,)
        x_batch_u = np.zeros(shape=batch_shape, dtype='u1')
        y_batch = []
        for entry, image_pair in zip(batch_df.iterrows(), x_batch_u):
            image_pair[0] = self.get_image_u8_from_path(entry[1].path_x)
            image_pair[1] = self.get_image_u8_from_path(entry[1].path_y)
            posterior_1 = entry[1].coordinate_x
            posterior_2 = entry[1].coordinate_y
            y = np.abs((posterior_1 - posterior_2) / 1320.)
            y_batch.append(y)
        y_batch = np.array(y_batch, dtype='f')
        
        #augment
        x_batch_u[:, 0, :, :, :] = self.augmentator(x_batch_u[:, 0, :, :, :])
        x_batch_u[:, 1, :, :, :] = self.augmentator(x_batch_u[:, 1, :, :, :])
        
        # create float32 batch
        x_batch_f = x_batch_u.astype('float32') / 255.
        return x_batch_f, y_batch
    
    @staticmethod
    def augmentator(images):
        """Apply data augmentation"""
        augmenter = iaa.Sequential([
            # Invert pixel values on 25% images
            iaa.Invert(0.25, per_channel=0.5),
            # Blur 30% of images
            iaa.Sometimes(.3,
                          iaa.OneOf([
                              iaa.GaussianBlur(sigma=(0.0, 3.0)),
                              iaa.AverageBlur(k=(2, 2)),
                              iaa.MedianBlur(k=(1, 3)),
                          ]),
                          ),
            # Do embossing or sharpening
            iaa.OneOf([
                iaa.Sometimes(.2, iaa.Emboss(alpha=(0.0, .3), strength=(.2, .8))),
                iaa.Sometimes(.2, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),
            ]),
            # Add one noise (or none)
            iaa.OneOf([
                iaa.Dropout((0, 0.01)),
                iaa.AdditiveGaussianNoise(scale=0.01 * 255),
                iaa.SaltAndPepper(0.01),
                iaa.Noop(),
            ]),
            # Convert to grayscale
            iaa.Sometimes(.2, iaa.Grayscale(alpha=(0.0, 1.0))),
            iaa.Sometimes(.4, iaa.LinearContrast((0.5, 1.5), per_channel=0.5)),
            #iaa.PiecewiseAffine(scale=(0.005, 0.05)),
        ])
        images = augmenter(images=images)
        return images
