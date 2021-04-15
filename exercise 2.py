import cv2
import numpy as np
import matplotlib.pyplot as plt

# exercise a
img = cv2.imread("external files/exercise 2/lab2a.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# set of parameters
a = 50
b = 200
alpha = 0.3
beta = 2
gamma = 1
ya = 30
yb = 100

# contrast stretching

def const_stretch(img, a, alpha, ya, beta, gamma, b, yb):
    new_image = np.zeros(shape=(np.shape(img)[0], np.shape(img)[1]))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            current_pix = img[i,j]
            if current_pix < a:
                new_image[i,j] = alpha * current_pix
            elif current_pix <= b:
                new_image[i,j] = beta * (current_pix - a) + ya
            else:
                new_image[i,j] = gamma * (current_pix - a) + yb
    plt.imshow(new_image, cmap="gray")
    plt.show()

# const_stretch(gray, a, alpha, ya, beta, gamma, b, yb)


# clipping
def clipping(img, a, alpha, ya, beta, gamma, b, yb):
    new_image = np.zeros(shape=(np.shape(img)[0], np.shape(img)[1]))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            cur_pix = img[i,j]
            if cur_pix < a:
                new_image[i,j] = 0
            elif cur_pix < b:
                new_image[i,j] = beta * (cur_pix - a)
            else:
                new_image[i,j] = beta * (b-a)
    plt.imshow(new_image, cmap="gray")
    plt.show()

# clipping(gray, a, alpha, ya, beta, gamma, b, yb)



# Histogram equalization
img = cv2.imread("external files/exercise 2/Unequalized_H.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def histo_equal(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.subplot(221)
    plt.plot(cdf_normalized)
    plt.hist(img.flatten(), 256, [0, 256])
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.subplot(222)
    plt.imshow(img, cmap="gray")


    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    plt.subplot(223)
    plt.hist(img2.flatten(), 256, [0, 256])
    plt.subplot(224)
    plt.imshow(img2, cmap="gray")
    plt.show()

# histo_equal(img)


# creating a 9x9 blur filter
img = cv2.imread("external files/exercise 2/lab2b.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def blur(img):

    filter = np.repeat(1/81, 81)
    filter = filter.reshape((9,9))
    offset = 4

    new_img = np.zeros(shape=(np.shape(img)[0] + 8, np.shape(img)[1] + 8))

    for x in range(np.shape(img)[0]):
        for y in range(np.shape(img)[1]):
            img_pix = img[x,y]

            for a in range(x-offset, x+offset):
                for b in range(y-offset, y+offset):
                    new_img[a,b] = new_img[a,b] + 1/81 * img_pix

    plt.imshow(new_img, cmap="gray")
    plt.show()


# blur(img)


# creating a hp-filter
img = cv2.imread("external files/exercise 2/lab2b.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def hp_filter(img):
    filter = [0,-1,0,-1,4,-1,0,-1,0]
    filter = np.reshape(filter, (3,3))

    offset = np.shape(filter)[0] // 2

    new_image = np.zeros(shape=(np.shape(img)[0] + 2 * offset, np.shape(img)[1] + 2 * offset))

    for x in range(np.shape(img)[0]):
        for y in range(np.shape(img)[1]):
            cur_px = img[x,y]
            for a in range(np.shape(filter)[0]):
                for b in range(np.shape(filter)[1]):
                    xn = x - offset + a
                    yn = y - offset + b
                    new_image[xn,yn] += cur_px * (1/9 * filter[a,b])
    plt.imshow(new_image + 128, cmap="gray")
    plt.show()

# hp_filter(img)


# create the furrior transform
img = cv2.imread("external files/exercise 2/lab2b.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def fourier(img):
    plt.figure()
    plt.subplot(221)
    plt.title("original image")
    plt.imshow(img, cmap="gray")
    plt.subplot(222)
    plt.title("2D Fourier transform")
    img_ft = np.fft.fft2(img)
    magnetude_spect = 20* np.log(np.abs(img_ft))

    plt.imshow(magnetude_spect, cmap="gray")
    plt.subplot(223)
    plt.title("Fourier Transform shift")
    img_ft_shift = np.fft.fftshift(img_ft)
    magnetude_spect = 20* np.log(np.abs(img_ft_shift))
    plt.imshow(magnetude_spect, cmap="gray")

    plt.subplot(224)
    plt.title("spatial domain")
    img_saptial = np.fft.ifft2(img_ft_shift)
    magnetude_spect = 20* np.log(np.abs(img_saptial))
    plt.imshow(magnetude_spect, cmap="gray")


    plt.show()

fourier(img)
