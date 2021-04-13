import cv2
import numpy as np
import matplotlib.pyplot as plt

# exercise b
a = np.random.rand(100)
b = a**2
A = np.broadcast_to(b, (b.shape[0], b.shape[0]))
plt.imshow(A)
plt.close()

# exercise c
a = cv2.imread("external files/lab1a.png")
b = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
plt.imshow(b)
plt.close()

# exercise d
a = cv2.imread("external files/lab1a.png")
b = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
plt.imshow(b, cmap='gray')
plt.close()

# exercise e
fig = plt.figure()
for i in range(1, 4):
    img = b[:, :, i - 1]
    fig.add_subplot(3, 1, i)
    plt.imshow(img)
# plt.show()
plt.close()

# exercise f
fig = plt.figure()
for i in [0,50,100,150,200 ,255]:
    ret, bw_img = cv2.threshold(b, i, 255, cv2.THRESH_BINARY)
    fig.add_subplot(3,2,[0,50,100,150,200 ,255].index(i)+1)
    plt.imshow(bw_img)
# plt.show()
plt.close()

# exercise g
fig = plt.figure()
img = b
crop_image = img[350:425, 260:400]
for i in [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA]:
    resized_image = cv2.resize(crop_image, None, fx=0.75, fy=0.75, interpolation = i)
    fig.add_subplot(2,2,[cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA].index(i)+1)
    # fig.set_title(str(i))
    plt.imshow(resized_image)
plt.show()
