import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import copy
import scipy
import random
import pandas as pd
from tqdm import tqdm
import os
import math

import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

print()
print("============================================")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available()
print("============================================")
print()


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
        ] = 255

        return w_img

    def move(self, dirx, diry):
        self.x += dirx
        self.y += diry

# ========== Model ============================================================


class ModelOrientation:
    """
    le but est de prendre une image
    creer un cadre qu'on va deplacer dans l'image aleatoirement

    a chaque deplacement on recupere le morceau de l'image se trouvant dans le cadre
    On cree le flot optique entre le morceau actuel et le precedant
    On stoque le flot ainsi que le deplacement effectue

    Au bout d'un certain nombre d'etapes, on entraine le model a trouver le deplacement en fonction du flot
    """
    def __init__(self, cadre_hau, cadre_lar, cadre_x, cadre_y):
        self.hau = cadre_hau
        self.lar = cadre_lar
        self.cx = cadre_x
        self.cy = cadre_y
        self.save_name = 'weights-test2.4.hdf5'

        self.create_model()

    def create_model(self):
        img_input = layers.Input(shape=(self.hau, self.lar, 2))

        x = layers.Conv2D(32, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)

        # x = layers.Conv2D(64, 3, activation='relu')(img_input)
        # x = layers.Conv2D(128, 3, activation='relu')(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = layers.Dropout(0.5)(x)

        # x = layers.Conv2D(64, 3, activation='relu')(img_input)
        # x = layers.Conv2D(64, 3, activation='relu')(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = layers.Dropout(0.5)(x)

        # x = layers.Conv2D(64, 3, activation='relu')(img_input)
        # x = layers.Conv2D(128, 3, activation='relu')(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = layers.Dropout(0.5)(x)

        # Flatten feature map to a 1-dim tensor so we can add fully connected layers
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation
        # x = layers.Dense(200, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(200, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(200, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(100, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(50, activation='relu')(x)
        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(2, activation='linear')(x)
        # output = keras.layers.Linear(x)

        # Create model:
        self.model = Model(img_input, output)

        self.model.compile(loss='mean_squared_error',
                           optimizer='adam'
                          )

        if os.path.isfile(self.save_name):
            self.model.load_weights(self.save_name)
            pass

        print(self.model.summary())

    def fit(self, img, steps, size_of_training):
        self.img = img
        self.next = np.zeros((self.hau, self.lar))

        # arrays pour contenir les donnees pour le train
        self.features = np.zeros((size_of_training, self.hau * self.lar * 2), dtype='float32')
        self.targets = np.zeros((size_of_training, 2))

        for i in tqdm(range(steps)):
            if i % (size_of_training / 10) == 0 and i > 0:
                # on entraine le model sur les donnees crees
                self.train(i, size_of_training)
            elif i % 100 == 0 and i > 0:
                # on fait un grand bon dans l'image car sinon le cadre
                # a tendance a rester dans la meme region
                longx, longy = self.get_new_coords(10, 10, 1000, 1000)
                self.move(longx, longy)
                _, self.next = self.draw()

            self.create_data(i, size_of_training)

    def get_new_coords(self, minx, miny, maxx, maxy):
        # de combien de case va on deplacer le rectangle
        longx = random.randint(minx, maxx)
        longy = random.randint(miny, maxy)

        longx = min(self.img.shape[0] // 2 - self.lar, longx)
        longy = min(self.img.shape[1] // 2 - self.lar, longy)
        # print(longx, longy)

        # # pour choisir l'axe sur lequel on deplace
        # bool_choix1 = random.randint(0, 1)

        # pour choisir la direction sur lequel on deplace
        dirx = random.choice([-1, 1])
        diry = random.choice([-1, 1])

        # si on peut pas deplacer le rectangle dans ce sens car il sortirait de l'image
        if self.cx + longx * dirx < 0 or self.cx + longx * dirx + self.lar > self.img.shape[0]:
            dirx *= -1

        longx *= dirx

        # si on peut pas deplacer le rectangle dans ce sens car il sortirait de l'image
        if self.cy + longy * diry < 0 or self.cy + longy * diry + self.hau > self.img.shape[1]:
            diry *= -1

        longy *= diry

        return longx, longy

    def create_data(self, i, size_of_training):
        """
        cree des donnees et remplace les plus vielles au passage
        """

        longx, longy = self.get_new_coords(0, 0, 3, 3)

        self.move(longx, longy)

        prvs, next = self.draw()
        # print(" shape: ", prvs.shape, next.shape)

        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.features[i % size_of_training, :] = flow.flatten()
        self.targets[i % size_of_training, :] = [longx, longy]

        # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # hsv = np.zeros((self.hau, self.lar, self.img.shape[2]), dtype='uint8')

        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 1] = 255
        # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # cv.imshow('frame2', next)
        # k = cv.waitKey(30) & 0xff

        # if k == 27:
            # break
        # elif k == ord('s'):
            # cv.imwrite('opticalfb.png', prvs)
            # cv.imwrite('opticalhsv.png', bgr)

        self.prvs = next

    def move(self, longx, longy):
        """
        deplace le cadre dans l'image
        """
        self.cx += longx
        self.cy += longy

    def draw(self):
        """
        recupere le morceau de l'image ou se trouve le cadre
        """
        prvs = copy.deepcopy(self.next)
        w_img2 = np.zeros((self.hau, self.lar, 3), dtype='uint8')

        a = self.img[
            self.cx:self.cx+self.lar,
            self.cy:self.cy+self.hau,
            :
        ]

        w_img2[:, :, :] = a
        next = cv.cvtColor(w_img2, cv.COLOR_BGR2GRAY)

        self.next = next

        return prvs, next

    def train(self, i, size_of_training):
        print("==========================================================")
        print("Step: ", i)

        self.features = np.reshape(self.features, (size_of_training, self.hau, self.lar, 2))
        print("----------------")
        print("Train")

        self.model_checkpoint = ModelCheckpoint(self.save_name)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


        if i < size_of_training:
            self.model.fit(self.features[:i], self.targets[:i], shuffle=True, callbacks=[self.model_checkpoint])
        else:
            self.model.fit(self.features, self.targets, shuffle=True, callbacks=[self.model_checkpoint])

        print("----------------")
        print("Evaluate on test data")

        indexes = np.random.randint(0, min(i, size_of_training), (100,))
        x_test = self.features[:100]
        y_test = self.targets[:100]
        results = self.model.test_on_batch(x_test, y_test)

        print("test loss", results)


        # for j in indexes:
            # [[predx, predy]] = self.model.predict(np.array([self.features[j]]))
            # predx = round(predx)
            # predy = round(predy)

            # loss = math.sqrt((predx - self.targets[j, 0])**2 + (predy - self.targets[j, 1])**2)
            # print(loss, "  / Pred: ", predx, predy, " / Diff: ", self.targets[j, 0] - predx, self.targets[j, 1] - predy, " / Target: ", self.targets[j])

        # on reset les arrays

        self.features = np.reshape(self.features, (size_of_training, self.hau * self.hau * 2))

        print()

# ========== MAIN =============================================================

if __name__ == "__main__":
    img = cv.imread(sys.argv[1])
    # print("shape: ", img.shape)
    # grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('frame2', grey)
    # cv.waitKey(0) # waits until a key is pressed
    # cv.destroyAllWindows() # destroys the window showing image

    # cap = cv.VideoCapture(cv.samples.findFile("./Optical_flow_demo.mkv"))
    # ret, frame1 = cap.read()
    # prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    model = ModelOrientation(200, 200, 20, 20)
    model.fit(img, 10000000000000, 10000)
