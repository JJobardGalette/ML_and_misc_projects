from numba import njit
import numpy as np
import time


@njit(parallel=True, nogil=True, fastmath=True)
def compute_mandelbrot_set(width, length, n_iter, x_position, y_position, zoom, previous_x, previous_y):
    fractals = np.zeros((width, length))
    for x in range(length):
        for y in range(width):
            p = np.sqrt((x-1/4)**2+y*y)
            if x>=p -2*p**2+1/4 and (x+1)**2+y**2>=16:
                x_r, y_r = transform_pyplot_to_r2(x, y, width, length, previous_x, previous_y, zoom)
                z = complex(x_r, y_r)
                z_0 = z
                j=0
                while abs(z) < 2 and j < n_iter:
                    z = z**2 + z_0
                    j+=1
                fractals[y, x] = j-1
    return fractals

@njit(nogil=True, fastmath=True)
def transform_pyplot_to_r2(x, y, width, length, offset_x, offset_y, zoom):
    """transform the pyplot coordinates into coordinates in the complex plane
    x(int): the pyplot horizontal coordinate
    y(int): the pyplot vertical coordinate
    width: width of the pyplot window
    length: length of the pyplot window
    offset_x: the corresponding 0 in R2
    offset_y: the corresponding 0 in R2"""
    x_r = offset_x + x/zoom/length # problems seem to occur here
    y_r = (length - y)/width/zoom + offset_y
    return x_r, y_r
