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

import sys
import os
import abc
import struct

import matplotlib.pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
#from mpl_toolkits.mplot3d import axes3d
import numpy as np

# Compatible with Python 2 *and* 3.
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class VectorField(ABC):
    """ Creates a vector field in three dimensional cartesian space. """
    def __init__(self, size=None, resolution=None):
        self._initialized = False
        self.size = size
        self.resolution = resolution
        self._initialized = True
        self._evaluate()
        
    def _evaluate(self):
        if self._initialized:
            self.grid_x, self.grid_y, self.grid_z = self._generate_grid()
            self._evaluate_vectors()
        
    def _evaluate_vectors(self):
        if self._initialized:
            self._set_uvw()
            self.vectors = self._get_vector_table()
    
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = self._get_param_as_array(value, dtype=float)
        self._evaluate()

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, value):
        self._resolution = self._get_param_as_array(value, dtype=int)
        self._evaluate()

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
            except (TypeError, IndexError):
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
        
    @abc.abstractmethod
    def _set_uvw(self):
        """ Set functions for u, v, and w (xyz) vector components.
        Set self.u, self.v and self.w.
        """
        self.u = self.grid_x
        self.v = self.grid_y
        self.w = self.grid_z
    
    def _get_vector_table(self):
        """ This composes a 3*m matrix of all the row vectors in preparation for FGA export. """
        vectors = np.vstack((self.u.flat, self.v.flat, self.w.flat)).T
        return vectors
        
    def save_fga(self, filename):
        """ Write the vector field as .FGA file (Fluid Grid ASCII) format to disk. """
        np.savetxt(filename, self.vectors, delimiter=',', newline=',\n', fmt='%3.5f')
        
        prependix = "{0},{1},{2},\n-{3},-{4},-{5},\n{3},{4},{5},".format(self.resolution[0],
                                                                         self.resolution[1],
                                                                         self.resolution[2],
                                                                         self.size[0]*0.5,
                                                                         self.size[1]*0.5,
                                                                         self.size[2]*0.5)
        try:
            with open(filename, 'r+') as file_handle:
                content = file_handle.read()
                file_handle.seek(0, 0)
                file_handle.write(prependix + '\n' + content)
        except (OSError, IOError) as e:
            print("ERROR {}: Failed to save file {}. {}".format(e.errno, filename, e.strerror))
    
    def save_vf(self, filename):
        """ Write the vector field as .vf file format to disk. """
        if not np.unique(self.resolution).size == 1:
            raise ValueError("Vectorfield resolution must be the same for X, Y, Z when exporting to Unity3D.")
    
        file_handle = open(filename, 'wb')
        for val in [b'V', b'F', b'_', b'V',
                    struct.pack('H', self.resolution[0]),
                    struct.pack('H', self.resolution[1]),
                    struct.pack('H', self.resolution[2])]:
            file_handle.write(val)
        
        # Layout data in required order.
        u_stream = self.u.flatten('F')
        v_stream = self.v.flatten('F')
        w_stream = self.w.flatten('F')
        for i in range(u_stream.size):
            file_handle.write(struct.pack('f', v_stream[i]))
            file_handle.write(struct.pack('f', u_stream[i]))
            file_handle.write(struct.pack('f', w_stream[i]))
        file_handle.close()
        
    def save(self, filename):
        """ Write the vector field to disk. """
        file_name, ext = os.path.splitext(filename)
        if ext.lower() == ".fga":
            self.save_fga(filename)
        elif ext.lower() == ".vf":
            self.save_vf(filename)
        else:
            print("ERROR: file format '{}' is not supported."
                  "\nSupported formats are: 'FGA'(Unreal Engine 4), 'VF'(Unity3D)".format(ext))
        
    def plot(self, filename=None):
        """ Plot the vector field as a preview. """
        # Todo: 3D plots get messy quickly. Any way to use plotly for interactive plots wihout the Jupyter overhead?
        raise NotImplementedError("Only implemented for 2D vector fields right now.")
    
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
            resolution = 32
        super(VectorField2D, self).__init__(size, resolution)

    @VectorField.resolution.setter
    def resolution(self, value):
        self._resolution = self._get_param_as_array(value, dtype=int)
        # For 2D fields make sure there is only 1 slice in Z direction.
        self._resolution[2] = 1
        self._evaluate()
        
    def to_3d(self):
        """ Returns a 3D vector field with resolution x=y=z. """
        res = self.resolution.max()
        func = lambda x, y, z: np.zeros(x.shape)
        vf3d = CustomUVW(func, func, func, size=self.size, resolution=res)
        # Copy the values to the new vector field. Fit into new resolution and leave non-overlapping as zeros.
        vf3d.u[:self.u.shape[0], :self.u.shape[1], :] = self.u.repeat(res, axis=2)
        vf3d.v[:self.v.shape[0], :self.v.shape[1], :] = self.v.repeat(res, axis=2)
        return vf3d
        
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
        
    def _refit(self, arr, out_range=(0, 255)):
        """ Fit the values of the array into new range.
            The bounds for the data will be -abs(max), +abs(max), not min and max!.
        """
        min_val, max_val = -np.abs(arr).max(), np.abs(arr).max()
        y = (arr - (max_val + min_val) * 0.5) / (max_val - min_val)
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) * 0.5
    
    def _get_flow_rgb(self):
        uv = np.dstack((self.u, self.v))
        # Convert to 0..255 for rgb values. And invert.
        rg = np.abs(self._refit(uv).astype(int)-255)
        # The blue channel is all zeros (unused).
        b = np.zeros((self.resolution[0], self.resolution[1])).astype(int)
        rgb = np.dstack((rg, b))
        return rgb
    
    def save_flowmap(self, filename):
        rgb = self._get_flow_rgb()
        try:
            plt.imsave(filename, rgb)
        except ValueError:
            file_name, ext = os.path.splitext(filename)
            print("ERROR: file format '{}' is not supported.".format(ext),
                  "\nSupported formats are: 'FGA'(Unreal Engine 4), 'VF'(Unity3D)",
                  "\nFlowmaps (Images): {}".format(", ".join(sorted(FigureCanvasBase.get_supported_filetypes()))))

    def save(self, filename):
        """ Write the vector field to disk. """
        file_name, ext = os.path.splitext(filename)
        if ext.lower() == ".fga":
            self.save_fga(filename)
        elif ext.lower() == ".vf":
            self.save_vf(filename)
        else:
            self.save_flowmap(filename)
            

class Vortex2D(VectorField2D):
    
    def __init__(self, radius=1.0, pull=0.5, size=None, resolution=None):
        self._initialized = False
        self.radius = radius
        self.pull = pull
        super(Vortex2D, self).__init__(size, resolution)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            print("Warning: Vortex radius should be positive.")
        if value == 0:
            print("Warning: Vortex radius can't be zero. Setting to 0.001.")
            value = 0.001
        self._radius = value
        self._evaluate_vectors()
        
    @property
    def pull(self):
        return self._pull

    @pull.setter
    def pull(self, value):
        self._pull = value
        self._evaluate_vectors()
    
    def _set_uvw(self):
        """ Calculate vector field. """
        sq_sum = np.square(self.grid_x) + np.square(self.grid_y)
        divisor = np.sqrt(sq_sum)
        factor = np.exp(-sq_sum / self._radius)
        self.u = factor * self.grid_y / divisor - self._pull * self.grid_x
        self.v = factor * -self.grid_x / divisor - self._pull * self.grid_y
        self.w = np.zeros(self.resolution)
        

class Convolution2D(VectorField2D):
    
    def __init__(self, size=None, resolution=None):
        super(Convolution2D, self).__init__(size, resolution)
        
    def _set_uvw(self):
        """ Calculate vector field. """
        term = np.exp(-(np.square(self.grid_x) + np.square(self.grid_y)))
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
        return q * (self.grid_x - a[0]) / (np.square(self.grid_x - a[0]) + np.square(self.grid_y - a[1])) ** 1.5, \
               q * (self.grid_y - a[1]) / (np.square(self.grid_x - a[0]) + np.square(self.grid_y - a[1])) ** 1.5
    
    def _set_uvw(self):
        """ Calculate vector field. """
        u1, v1 = self._E(1, [-1, 0])
        u2, v2 = self._E(-1, [1, 0])
        self.u = u1 + u2
        self.v = v1 + v2
        self.w = np.zeros(self.resolution)
    
    def normalize(self):
        """ Normalization loses all information about field strength! """
        self.u = self.u / np.sqrt(np.square(self.u) + np.square(self.v))
        self.v = self.v / np.sqrt(np.square(self.u) + np.square(self.v))
        self.vectors = self._get_vector_table()
        
    def clamp(self, E_max=10):
        """ Clamp field strength to E_max. """
        E = np.sqrt(np.square(self.u) + np.square(self.v))
        k = np.where(E.flat[:] > E_max)[0]
        self.u.flat[k] = self.u.flat[k] / E.flat[k] * E_max
        self.v.flat[k] = self.v.flat[k] / E.flat[k] * E_max
        self.vectors = self._get_vector_table()


class TwirlFlow2D(VectorField2D):
    
    def __init__(self, vert_flow=0.5, vert_freq=0.5, hori_freq=1.0, size=12, resolution=None):
        self._initialized = False
        self.vertical_flow = vert_flow
        self.vertical_freq = vert_freq
        self.horizontal_freq = hori_freq
        super(TwirlFlow2D, self).__init__(size, resolution)
    
    def _set_uvw(self):
        """ Calculate vector field. """
        self.u = np.cos(self.horizontal_freq * self.grid_x) + np.cos(self.grid_y * 2.5)
        self.v = np.sin(2.0 * self.grid_x) * np.cos(self.vertical_freq * self.grid_y + 1.5) - self.vertical_flow
        self.w = np.zeros(self.resolution)

    @property
    def vertical_flow(self):
        return self._vertical_flow
    
    @vertical_flow.setter
    def vertical_flow(self, value):
        self._vertical_flow = value
        self._evaluate_vectors()

    @property
    def vertical_freq(self):
        return self._vertical_freq

    @vertical_freq.setter
    def vertical_freq(self, value):
        self._vertical_freq = value
        self._evaluate_vectors()

    @property
    def horizontal_freq(self):
        return self._horizontal_freq

    @horizontal_freq.setter
    def horizontal_freq(self, value):
        self._horizontal_freq = value
        self._evaluate_vectors()


class Belt2D(VectorField2D):
    """ Creates rotating areas that merge together to create a stream. """
    def __init__(self, pulleys=None, size=12, resolution=None):
        self._initialized = False
        if pulleys is None:
            pulleys = ((-2.0, 1.75), (0.0, -1.75), (2., 1.75))
        self._pulleys = list()
        try:
            for pulley in pulleys:
                self.add_pulley(*pulley)
        except TypeError:
            raise TypeError("Argument pulleys must be an iterable of iterables!\n"
                            "Each item must have at least an x and y coordinate\n"
                            "and optionally a radius, thickness and speed.\n"
                            "e.g.: [(-2., 0.), (2., -0.5, 0.75, 0.5, -1.0)]")
        super(Belt2D, self).__init__(size, resolution)
        
    def add_pulley(self, x, y, radius=1.0, thickness=2.5, speed=1.0):
        self._pulleys.append([x, y, radius, speed, thickness])
        self._evaluate_vectors()
    
    def remove_pulley(self, idx):
        try:
            self._pulleys.pop(idx)
        except IndexError:
            raise IndexError("Pulley index '{}' out of range. There are {} pulleys.".format(idx, len(self._pulleys)))
        self._evaluate_vectors()
    
    def _field(self, points, radius=1.0, thickness=2.5, speed=1.0):
        # Sanity checks.
        thickness = max(thickness, 0.001)
        radius = max(radius, 0.0)
        
        d = np.linalg.norm(points, axis=2, keepdims=True) - radius
        # Negate Y values.
        points[:, :, 1] = -points[:, :, 1]
        # Switch Y and X.
        points = np.flip(points, axis=2)
        field = points * speed * np.exp(-np.square(d)/thickness)
        return field
    
    def _set_uvw(self):
        """ Calculate vector field. """
        grid = np.dstack((self.grid_x, self.grid_y))
        field = np.zeros(grid.shape)
        for pulley in self._pulleys:
            points = grid + [-pulley[0], -pulley[1]]
            field = np.add(field, self._field(points, pulley[2], pulley[3]))
        
        self.u = field[:, :, 0, np.newaxis]
        self.v = field[:, :, 1, np.newaxis]
        self.w = np.zeros(self.resolution)
    

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
