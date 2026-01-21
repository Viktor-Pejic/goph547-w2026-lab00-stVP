from tkinter import Image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from goph547lab00.arrays import square_ones


def open_image():
    img = np.asarray(Image.open('rock_canyon.jpg'))
    print(f'image array: {img}')

    plt.imshow(img)
    plt.show()
    # image array shape is shape=(296, 474, 3)
    img_gray = np.asarray(Image.open('rock_canyon.jpg').convert('L'))
    plt.imshow(img_gray)
    plt.show()

    small_gray_scale = img_gray[150:250,100:150]
    plt.imshow(small_gray_scale)
    plt.show()

    # plotting subplots with mean RGB

    ny, nx, _ = img.shape

    mean_R_x = np.mean(img[:, :, 0], axis=0)
    mean_G_x = np.mean(img[:, :, 1], axis=0)
    mean_B_x = np.mean(img[:, :, 2], axis=0)
    mean_RGB_x = np.mean(img, axis=(0, 2))

    mean_R_y = np.mean(img[:, :, 0], axis=1)
    mean_G_y = np.mean(img[:, :, 1], axis=1)
    mean_B_y = np.mean(img[:, :, 2], axis=1)
    mean_RGB_y = np.mean(img, axis=(1, 2))

    x = np.arange(nx)
    y = np.arange(ny)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x, mean_R_x, color='red', label='Mean R')
    axs[0].plot(x, mean_G_x, color='green', label='Mean G')
    axs[0].plot(x, mean_B_x, color='blue', label='Mean B')
    axs[0].plot(x, mean_RGB_x, color='black', linewidth=2, label='Mean RGB')

    axs[0].set_xlabel('x-coordinate')
    axs[0].set_ylabel('Colour value')
    axs[0].set_title('Mean colour values vs x-coordinate')
    axs[0].legend()

    axs[1].plot(mean_R_y, y, color='red', label='Mean R')
    axs[1].plot(mean_G_y, y, color='green', label='Mean G')
    axs[1].plot(mean_B_y, y, color='blue', label='Mean B')
    axs[1].plot(mean_RGB_y, y, color='black', linewidth=2, label='Mean RGB')

    axs[1].set_xlabel('Colour value')
    axs[1].set_ylabel('y-coordinate')
    axs[1].set_title('Mean colour values vs y-coordinate')
    axs[1].invert_yaxis()
    axs[1].legend()

    plt.savefig('rock_canyon_RGB_summary.png')


def main():
    # test creating square array of ones
    A_np = np.ones((3,3))
    A = square_ones(3)

    array_1 = np.ones((3,5))
    array_2 = np.nan*np.ones((6,3))
    array_3 = np.arange(45,77,2).reshape(-1,1)
    sum_of_3 = np.sum(array_3)
    array_4 = np.array([[5,7,2],[1,-2,3],[4,4,4]])
    array_5 = np.identity(3)
    array_6 = array_4 * array_5
    dot_prod = np.dot(array_4, array_5)
    cross_prod = np.cross(array_4, array_5)


    print(f'A_np:\n{A_np}\n')
    print(f'A:\n{A}\n')
    print(f'array_1:\n{array_1}\n')
    print(f'array_2:\n{array_2}\n')
    print(f'array_3:\n{array_3}\n')
    print(f'sum_of_3:\n{sum_of_3}\n')
    print(f'array_4:\n{array_4}\n')
    print(f'array_5:\n{array_5}\n')
    print(f'array_6:\n{array_6}\n')
    print(f'dot_prod:\n{dot_prod}\n')
    print(f'cross_prod:\n{cross_prod}\n')

    open_image()

if __name__ == '__main__':
    main()