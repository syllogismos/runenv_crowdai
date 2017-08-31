from .running_stat import RunningStat
import numpy as np
import copy


class Composition(object):
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, x, update=True):
        for f in self.fs:
            x = f(x)
        return x

    def output_shape(self, input_space):
        out = input_space.shape
        for f in self.fs:
            out = f.output_shape(out)
        return out


class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class RelativeFilter(object):
    """
    get relative x of other body parts wrt pelvis


    observations contains relative position of head wrt left and right toe
    and relative position of all x coordinates wrt pelvis
    """
    def __init__(self, clip=10.0):
        self.clip = clip

    def __call__(self, x):
        # compute relative x's of other body parts wrt pelvis
        dx = copy.copy(x)
        pelvis_x = x[1]
        head_x = x[22]
# removed pelvix_x observations completely
        dx[1] = head_x - x[28]  # relative position of head and left toe
        dx[24] = head_x - x[30]  # relative position of head and right toe
        dx[18] = pelvis_x - x[18]  # relative center of mass
        dx[22] = pelvis_x - x[22]  # relative head x
        dx[26] = pelvis_x - x[26]  # relative torso x
        dx[28] = pelvis_x - x[28]  # relative left toe x
        dx[30] = pelvis_x - x[30]  # relative right toe x
        dx[32] = pelvis_x - x[32]  # relative left talus x
        dx[34] = pelvis_x - x[34]  # relative right talus x

        if self.clip != 0:
            dx = np.clip(dx, -self.clip, self.clip)
        return np.array(dx)

    def output_shape(self, input_space):
        return input_space.shape


class Flatten(object):
    def __call__(self, x, update=True):
        return x.ravel()

    def output_shape(self, input_space):
        return (int(np.prod(input_space.shape)),)


class Ind2OneHot(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, x, update=True):
        out = np.zeros(self.n)
        out[x] = 1
        return out

    def output_shape(self, input_space):
        return (input_space.n,)