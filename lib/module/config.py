from easydict import EasyDict as edict
import numpy as np


__C = edict()

cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# learning rate
__C.TRAIN.LEARNING_RATE = 0.01



#
# Test options
#
__C.TEST = edict()


# input image mean
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])