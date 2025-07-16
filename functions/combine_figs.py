#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:43:07 2024

@author: evagnegy
"""

from PIL import Image

# puts the two colorbars together and then adds it to the main table. I found these functions online
im1 = Image.open('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/colorbars/wspd_climo.png')
im2 = Image.open('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/colorbars/relwspd_bias.png')

def get_concat_v_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
    if im1.width == im2.width:
        _im1 = im1
        _im2 = im2
    elif (((im1.width > im2.width) and resize_big_image) or
          ((im1.width < im2.width) and not resize_big_image)):
        _im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width)), resample=resample)
    dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (0, _im1.height))
    return dst

def get_concat_h_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst

get_concat_h_resize(im1, im2, resize_big_image=False).save('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/colorbars/relwspd_combined.png')


im2 = Image.open('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/colorbars/relwspd_combined.png')
im1 = Image.open('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/bias_climo/ann_relwspd_d03.png')


get_concat_v_resize(im1, im2, resize_big_image=False).save('/Users/evagnegy/Desktop/CanESM2_WRF_Eval/figures/bias_climo/ann_relwspd_d03_w_colorbar.png')
