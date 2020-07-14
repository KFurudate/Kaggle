# -*- coding: utf-8 -*-
"""Data_resize.ipynb
Used for resizing image dataset by Kaggle notebooks.
"""

from glob import glob
from joblib import Parallel, delayed

import os

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_image(image_path, output_folder, resize):
	base_name = os.path.basename(image_path)
	outpath = os.path.join(output_folder, base_name)
	img = Image.open(image_path)
	img = img.resize(
			(resize[1], resize[0]), resample=Image.BILINEAR
			)
	img.save(outpath)

"""# Reseize train images"""

SIZE=512

# Set path
in_f_train = "../input/siim-isic-melanoma-classification/jpeg/train"
in_f_test = "../input/siim-isic-melanoma-classification/jpeg/test"

out_f_train="/kaggle/working/train_512"
out_f_test="/kaggle/working/test_512"

! mkdir -p $out_f_train
! mkdir -p $out_f_test
! ls

# Resize train images
images = glob(os.path.join(in_f_train, "*.jpg"))
Parallel(n_jobs=12)(
            delayed(resize_image)(i, out_f_train, (SIZE, SIZE))
            for i in tqdm(images)
            )

# Resize test images
images = glob(os.path.join(in_f_test, "*.jpg"))
Parallel(n_jobs=12)(
            delayed(resize_image)(i, out_f_test, (SIZE, SIZE))
            for i in tqdm(images)
            )

