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
        """ Checks the input parameter and, if necessary, transforms it
        to an array of length 3 for x, y, and z values.
        
        :param absolute: Change sign to positive only.
        :param dtype: transform to int or float
        
        :rtype: numpy.ndarray
        :return: ndarray of length 3.
        """
        arr = (np.ones(3) * 2).astype(dtype)
        if param is None:
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
        """ Generate the grid coordinates on which the functions for u, v, and w will be evaluated. """
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
        """ Set functions for u, v, and w (xyz) vector components.
        Set self.u, self.v and self.w.
        """
        # Todo: simple example
        self.u = None
        self.v = None
        self.w = None
    
    def _get_vector_table(self):
        """ This composes a 3*m matrix of all the row vectors in preparation for FGA export. """
        vectors = np.vstack((self.u.flat, self.v.flat, self.w.flat)).T
        return vectors
        
    def save_fga(self, filename):
        """ Export the vector field as .FGA file (Fluid Grid ASCII) format to disk. """
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
        """ Plot the vector field as a preview. """
        # Todo: 3D plots get messy quickly. Any way to use plotly for interactive plots wihout the Jupyter overhead?
        raise NotImplementedError
    
    def _plot_save_or_show(self, filename=None):
        """ Helper method to decide whether to show the plot or save it do file. """
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
        """ Set default resolution and size. """
        if not size:
            size = 4
        if not resolution:
            resolution = [32, 32, 1]
        # For 2D fields make sure there is only 1 slice in Z direction.
        resolution = self._get_param_as_array(resolution)
        resolution[2] = 1
        super(VectorField2D, self).__init__(size, resolution)
        
    def plot(self, filename=None):
        """ Plot a top-down view on the XY plane of the vector field. """
        plt.figure(figsize=(6, 6))
        plt.quiver(np.squeeze(self.grid_x, axis=2),
                   np.squeeze(self.grid_y, axis=2),
                   np.squeeze(self.u, axis=2),
                   np.squeeze(self.v, axis=2),
                   pivot='middle', headwidth=4, headlength=6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('image')
        self._plot_save_or_show(filename)
    
    
class Vortex2D(VectorField2D):
    
    def _set_uvw(self):
        """ Calculate vector field. """
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
        """ Calculate vector field. """
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
        """ Calculate electric field. """
        return q * (self.grid_x - a[0]) / ((self.grid_x - a[0]) ** 2 + (self.grid_y - a[1]) ** 2) ** 1.5, \
               q * (self.grid_y - a[1]) / ((self.grid_x - a[0]) ** 2 + (self.grid_y - a[1]) ** 2) ** 1.5
    
    def _set_uvw(self):
        """ Calculate vector field. """
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


class CustomUV2D(VectorField2D):
    """ Provide custom functions for creating u and v vector components.
        These functions must take 2 parameters that will be substituted for grid_x and grid_y.
    """
    def __init__(self, u_func, v_func, size=None, resolution=None):
        self.u_func = u_func
        self.v_func = v_func
        super(CustomUV2D, self).__init__(size, resolution)
        
    def _set_uvw(self):
        """ Calculate vector field. """
        self.u = self.u_func(self.grid_x, self.grid_y)
        self.v = self.v_func(self.grid_x, self.grid_y)
        self.w = np.zeros(self.resolution)


class CustomUVW(VectorField):
    """ Provide custom functions for creating UVW vector components.
        These functions must take 3 parameters that will be substituted
        for grid_x, grid_y and grid_z.
    """
    
    def __init__(self, u_func, v_func, w_func, size=None, resolution=None):
        self.u_func = u_func
        self.v_func = v_func
        self.w_func = w_func
        super(CustomUVW, self).__init__(size, resolution)
    
    def _set_uvw(self):
        """ Calculate vector field. """
        self.u = self.u_func(self.grid_x, self.grid_y, self.grid_z)
        self.v = self.v_func(self.grid_x, self.grid_y, self.grid_z)
        self.w = self.w_func(self.grid_x, self.grid_y, self.grid_z)
