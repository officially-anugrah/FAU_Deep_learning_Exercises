import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be divisble by 2x tile_size")
        
        single_tile = np.array([[0, 1], [1, 0]]) #0 is black and 1 is white
        repeat_count = self.resolution // (2 * self.tile_size)

        tiles = np.tile(single_tile, (repeat_count, repeat_count))
        self.output = np.kron(tiles, np.ones((self.tile_size, self.tile_size)))

        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x = np.linspace(0, self.resolution - 1, self.resolution)
        y = np.linspace(0, self.resolution - 1, self.resolution)
        Y, X = np.meshgrid(x, y)

        distance_from_center = (X - self.position[1]) ** 2 + (Y - self.position[0]) ** 2
        self.output = (distance_from_center <= self.radius ** 2).astype(float)
        
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        r = np.linspace(0, 1, self.resolution).reshape(1, -1).repeat(self.resolution, axis=0)
        g = np.linspace(0, 1, self.resolution).reshape(-1, 1).repeat(self.resolution, axis=1)
        b = np.flip(r, axis=1)  # Flip the red channel horizontally for the blue channel

        self.output = np.dstack((r, g, b))
        
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output)
        plt.show()