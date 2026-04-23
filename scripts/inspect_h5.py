import glob
import h5py
import numpy as np

paths = glob.glob('data/*.h5')[:5]
print('sample count:', len(paths))
for p in paths:
    with h5py.File(p, 'r') as f:
        image = f['image'][()]
        mask = f['mask'][()]
    print(p)
    print(' image shape:', image.shape, 'dtype:', image.dtype)
    print(' mask shape:', mask.shape, 'dtype:', mask.dtype)
    print(' min/max image:', np.min(image), np.max(image))
    print()