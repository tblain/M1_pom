import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import copy

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
from keras import layers
import keras
from keras import Model
from tensorflow.keras.optimizers import RMSprop

# ========== Rectangle ========================================================


class Rectangle:

    """Docstring for Rectangle. """

    def __init__(self, img, hau, lar, x, y):
        # image de base qu'on guarde intacte
        self.img = img

        self.hau = hau
        self.lar = lar
        self.x = x
        self.y = y

    def draw(self, w_img):
        # dessine la figure sur l'image
        w_img[
            self.x: self.x + self.lar,
            self.y: self.y + self.hau,
            0
        ] = 1

        return w_img

    def move(self, dirx, diry):
        self.x += dirx
        self.y += diry

# ========== Model ============================================================


img_input = layers.Input(shape=(100, 120, 2))

x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(10, activation='relu')(x)
x = layers.Dropout(0.3)(x)
# Create output layer with a single node and sigmoid activation
output = layers.Dense(2, activation=keras.activations.elu)(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully
# connected layer + sigmoid output layer
model = Model(img_input, output)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

# ========== MAIN =============================================================

if __name__ == "__main__":
    # img_bgr = cv.imread(sys.argv[1])
    # img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    img = np.zeros((100, 120, 1))
    w_img2 = copy.deepcopy(img)
    print(img.shape)

    rect1 = Rectangle(img, 20, 30, 10, 15)
    rect1.draw(w_img2)

    for _ in range(10):
        w_img1 = copy.deepcopy(w_img2)

        rect1.move(5, 0)
        w_img2 = copy.deepcopy(img)

        w_img2 = rect1.draw(w_img2)

        flow = cv.calcOpticalFlowFarneback(w_img1, w_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print("flow shape: ", flow.shape)

        # plt.imshow(flow)
        # plt.show()

        input = np.hstack((w_img1, w_img2))
        # [[predx, predy]] = model.predict(np.array([input]))
        [[predx, predy]] = model.predict(np.array([flow]))
        print(predx, predy)

        model.fit(np.array([flow]), np.array([[5, 0]]), verbose=0)

