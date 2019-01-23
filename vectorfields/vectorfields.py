#    MIT License
#
#    Copyright (c) 2019 Olaf Haag
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np


class VectorField(ABC):
    """ Creates a vector field in three dimensional cartesian space. """
    def __init__(self, size=None, resolution=None):
        self.resolution = self._get_param_as_array(resolution)
        self.size = self._get_param_as_array(size)
        self.grid_x, self.grid_y, self.grid_z = self._generate_grid()
        self._set_uvw()
        self.vectors = self._get_vector_table()
    
    @staticmethod
    def _get_param_as_array(param, absolute=True, dtype=int):
        arr = (np.ones(3) * 2).astype(dtype)
        if not param:
            return arr
        else:
            try:
                arr[: len(param)] = param[:3]
            except TypeError:
                arr = np.array(param).repeat(3)
            except ValueError:
                print("ERROR: Parameters resolution and size need to be numbers or iterables of 3 numbers.\n"
                      "       Setting default values.")
                return arr
        
        if absolute:
            return np.abs(arr).astype(dtype)
        else:
            return arr.astype(dtype)

    def _generate_grid(self):
        # Center bounding volume.
        min = -self.size * 0.5
        max = self.size * 0.5
    
        # generate grid
        x = np.linspace(min[0], max[0], self.resolution[0])
        y = np.linspace(min[1], max[1], self.resolution[1])
        z = np.linspace(min[2], max[2], self.resolution[2])
        x, y, z = np.meshgrid(x, y, z)
        return x, y, z

    @abstractmethod
    def _set_uvw(self):
        """ set self.u, self.v and self.w. """
        # Todo: simple example
        self.u = None
        self.v = None
        self.w = None
    
    def _get_vector_table(self):
        """ This composes a 3*m matrix of all the row vectors in preparation for FGA export. """
        vectors = np.vstack((self.u.flat, self.v.flat, self.w.flat)).T
        return vectors
        
    def save_fga(self, filename):
        np.savetxt(filename, self.vectors, delimiter=',', newline=',\n', fmt='%3.5f')
        
        prependix = "{0},{1},{2},\n-{3},-{4},-{5},\n{3},{4},{5},".format(self.resolution[0],
                                                                         self.resolution[1],
                                                                         self.resolution[2],
                                                                         self.size[0]*0.5,
                                                                         self.size[1]*0.5,
                                                                         self.size[2]*0.5)
        try:
            with open(filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(prependix + '\n' + content)
        except (OSError, IOError) as e:
            print("ERROR {}: Failed to save file {}. {}".format(e.errno, filename, e.strerror))
    
    def plot(self, filename=None):
        pass
    
    def _plot_save_or_show(self, filename=None):
        if not filename:
            plt.show()
        else:
            try:
                plt.savefig(filename, transparent=False, dpi=80, bbox_inches="tight")
            except (OSError, IOError) as e:
                print("ERROR {}: Failed to save file {}.".format(e.errno, filename))


class VectorField2D(VectorField):
    """ Creates VectorField only in the XY Plane.
        This is still an abstract base class.
    """
    def __init__(self, size=None, resolution=None):
        if not size:
            size = 4
        if not resolution:
            resolution = [32, 32, 1]
        super(VectorField2D, self).__init__(size, resolution)
        
    def plot(self, filename=None):
        # plot vector field
        plt.figure(figsize=(6, 6))
        plt.quiver(self.grid_x, self.grid_y, self.u, self.v, pivot='middle', headwidth=4, headlength=6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('image')
        self._plot_save_or_show(filename)
    
    
class Vortex2D(VectorField2D):
    
    def __init__(self, size=None, resolution=None):
        super(Vortex2D, self).__init__(size, resolution)

    def _set_uvw(self):
        # Calculate vector field.
        sq_sum = self.grid_x ** 2 + self.grid_y ** 2
        divisor = np.sqrt(sq_sum)
        factor = np.exp(-sq_sum)
        self.u = factor * -self.grid_y / divisor
        self.v = factor *  self.grid_x / divisor
        self.w = np.zeros(self.resolution)
        

class Convolution2D(VectorField2D):
    
    def __init__(self, size=None, resolution=None):
        super(Convolution2D, self).__init__(size, resolution)
        
    def _set_uvw(self):
        # Calculate vector field.
        term = np.exp(-(self.grid_x ** 2 + self.grid_y ** 2))
        self.u = -2 * self.grid_x * term
        self.v = -2 * self.grid_y * term
        self.w = np.zeros(self.resolution)


class ElectricDipole2D(VectorField2D):
    
    def __init__(self, size=None, resolution=None, normalize=False):
        super(ElectricDipole2D, self).__init__(size, resolution)
        if normalize:
            self.normalize()

    def _E(self, q, a):
        return q * (self.grid_x - a[0]) / ((self.grid_x - a[0]) ** 2 + (self.grid_y - a[1]) ** 2) ** 1.5, \
               q * (self.grid_y - a[1]) / ((self.grid_x - a[0]) ** 2 + (self.grid_y - a[1]) ** 2) ** 1.5
    
    def _set_uvw(self):
        # Calculate vector field.
        u1, v1 = self._E(1, [-1, 0])
        u2, v2 = self._E(-1, [1, 0])
        self.u = u1 + u2
        self.v = v1 + v2
        self.w = np.zeros(self.resolution)
    
    def normalize(self):
        """ Normalization loses all information about field strength! """
        self.u = self.u / np.sqrt(self.u ** 2 + self.v ** 2)
        self.v = self.v / np.sqrt(self.u ** 2 + self.v ** 2)
        self.vectors = self._get_vector_table()
        
    def clamp(self, E_max=10):
        """ Clamp field strength to E_max. """
        E = np.sqrt(self.u ** 2 + self.v ** 2)
        k = np.where(E.flat[:] > E_max)[0]
        self.u.flat[k] = self.u.flat[k] / E.flat[k] * E_max
        self.v.flat[k] = self.v.flat[k] / E.flat[k] * E_max
        self.vectors = self._get_vector_table()
