# What-is / How-to use this

- Brain image database to be used for our projects.
- To use it see Python data generator which provides a baseclass that reads the folder structures and build a DataFrame with list of images and their meta.
- `brain_images` is the root directory.
- anatomical plane folders must be either `cor`, `sag` or `hor`.

#### Folder structure convention

_brain_images &rightarrow; \<atlas_name\> &rightarrow; \<anatomical_plane\> &rightarrow; \<images or masks\>_ 

#### Example of folder tree
```
brain_images
├── ccfv2
│   ├── cor
│   │   ├── images
│   │   └── masks
│   │       ├── fgbg
│   │       └── main_4
│   ├── hor
│   └── sag
├── ccfv3
```

### Images and masks naming convention

- Images are named `<id>[_xn].suff` where `id` is non-important (typically the original name of the image) and the part `_xn`, if present, contains the image coordinate in PIR system. `suff` defines the image type (_eg_ `jpg`). Some examples:
 
    - `ccfv2/cor/images/blabla_x10.jpg`: coronal slices from atlas CCFv2 at 10um posterior.
    - `ccfv2/cor/images/blabla.jpg`: coronal slices from atlas CCFv2 at unknown posterior.
    
    
- Masks bear the same name of the corresponding images but with an additional `_mask` suffix. They are located into the `mask` folder and in one the sub-folders that defines the masking experiment (see below). Examples:
    - `ccfv2/cor/images/mask/fgbg/blabla_mask.jpg`: mask for foreground/background of coronal slices from atlas CCFv2 at unknown posterior.
    - `ccfv2/cor/images/mask/fgbg/blabla.jpg`: not a valid mask file (misses the `_mask`).


### Masking groups

- `fgbg`: two labels, foreground (1) and background (0)
- `main_4`: 4 structures in Allen onthology: cerebellum, cortex, white matter, other grey and background

