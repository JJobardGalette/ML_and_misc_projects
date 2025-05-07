
import numpy as np
from fractals import compute_mandelbrot_set, transform_pyplot_to_r2

import matplotlib.pyplot as plt
x_cursor = 0
y_cursor = 0

class FractalExplorer:
    def __init__(self, width, length, n_iter):
        self.x_cursor, self.y_cursor = 0, 0
        self.zoom = 1
        self.up_zoom = 1.2
        self.down_zoom = 1/self.up_zoom
        self.n_iter = n_iter
        self.width = width
        self.length = length
        self.fig, self.ax = plt.subplots()
        self.palette = np.zeros((self.n_iter, 3))
        self.set_palette(np.array([0, 100, 180]))
        self.previous_x = -2
        self.scroll_number = 0
        self.previous_y = -1/2 #in the coordinates of the frame
        # self.alpha = 0.16
        # self.palette = np.array(
        #     [
        #         (0.5 * np.cos(np.arange(self.n_iter) * self.alpha) + 0.5) * 255,  # R
        #         (0.5 * np.sin(np.arange(self.n_iter) * self.alpha + 0.987) + 0.5)
        #         * 255,  # G
        #         (0.5 * np.cos(np.arange(self.n_iter) * self.alpha + 4.188) + 0.5)
        #         * 255,  # B
        #     ],
        #     dtype=np.uint8,
        # ).transpose(1, 0)
        self.imgplot = self.ax.imshow(self.init_frame())
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
    def init_frame(self):
        fractal = compute_mandelbrot_set(self.width, self.length, self.n_iter, 
                                        self.x_cursor, self.y_cursor, self.zoom, self.previous_x, self.previous_y).astype(int)
        indices = fractal.astype(int)
        matched_elements = self.palette[indices]
        return matched_elements
    
    def set_palette(self, color_a, color_b = np.array([0, 0, 0])):
        self.palette[0] = color_b
        for i in range(1, self.n_iter):
            self.palette[i] = -1*color_b+(self.n_iter-i)/self.n_iter*color_a
        self.palette = self.palette.astype(int)
    
    def update_fractal(self):
        
        fractal = compute_mandelbrot_set(self.width, self.length, self.n_iter, self.x_cursor, self.y_cursor, self.zoom, self.previous_x, self.previous_y)
        self.set_new_offset(self.x_cursor, self.y_cursor)
        indices = fractal.astype(int)
        matched_elements = self.palette[indices]
        self.imgplot.set_data(matched_elements)
        self.fig.canvas.draw()

    def on_move(self, event):
        if event.inaxes:
            self.x_cursor=event.xdata
            self.y_cursor=event.ydata
            x_r, y_r = transform_pyplot_to_r2(self.x_cursor, self.y_cursor, self.width, self.length, self.previous_x, self.previous_y, self.zoom)

     
    def on_scroll(self, event):
        if event.button == 'up':
            self.zoom = self.up_zoom
            self.scroll_number +=1
            self.update_fractal()
        if event.button == 'down':
            self.zoom = self.down_zoom
            self.scroll_number -=1
            self.update_fractal()

    def set_new_offset(self, x_pos, y_pos):
        x_r, y_r = transform_pyplot_to_r2(x_pos, y_pos, self.width, self.length, self.previous_x, self.previous_y, self.zoom)
        print('previous_values', self.previous_x, self.previous_y)
        print('new, values', x_r, y_r)
        self.previous_x = (1-1/(self.zoom)**self.scroll_number) * (self.previous_x + x_r) 
        self.previous_y = (1 - 1/self.zoom**self.scroll_number) * (self.previous_y + y_r) 
if __name__ == '__main__':
    _ = FractalExplorer(1000, 1000, 60)
    plt.show()
