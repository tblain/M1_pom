import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import copy
import scipy
import random
import pandas as pd
from tqdm import tqdm

import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
from keras import layers
from keras import Model
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import os

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
    Le model doit trouver le deplacement effectue par le rectangle dans l'image vide
    Pour cela il dispose du flot optique entre l'image precedente et celle actuelle
    """
    def __init__(self, img_hau, img_lar, rect_x, rect_y, rect_hau, rect_lar):
        self.hau = img_hau
        self.lar = img_lar
        self.img = np.zeros((img_hau, img_lar, 3), dtype="uint8")

        self.w_img2 = copy.deepcopy(self.img)

        self.rect = Rectangle(self.img, rect_x, rect_y, rect_hau, rect_lar)
        self.w_img2 = self.rect.draw(self.w_img2)
        self.create_model()

    def create_model(self):
        img_input = layers.Input(shape=(self.hau, self.lar, 2))

        x = layers.Conv2D(16, 3, activation='relu')(img_input)
        x = layers.Conv2D(32, 3, activation='relu')(img_input)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)

        # x = layers.Conv2D(64, 3, activation='relu')(x)
        # x = layers.Conv2D(128, 3, activation='relu')(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = layers.Dropout(0.5)(x)

        # Flatten feature map to a 1-dim tensor so we can add fully connected layers
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(50, activation='relu')(x)
        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(2, activation='linear')(x)
        # output = keras.layers.Linear(x)

        # Create model:
        self.model = Model(img_input, output)

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        if os.path.isfile("./weights.hdf5"):
            self.model.load_weights('./weights.hdf5')

        print(self.model.summary())

    def fit(self, steps, size_of_training):
        self.prvs = cv.cvtColor(self.w_img2, cv.COLOR_BGR2GRAY)

        # arrays pour contenir les donnees pour le train
        self.features = np.zeros((size_of_training, self.hau * self.lar * 2))
        self.targets = np.zeros((size_of_training, 2))

        for i in tqdm(range(steps)):

            if i % (size_of_training / 10) == 0 and i > 0:
                # on entraine le model sur les donnees crees
                self.train(i, size_of_training)

            self.create_data(i, size_of_training)

    def create_data(self, i, size_of_training):
        """
        cree des donnees et remplace les plus vielles au passage
        """
        # de combien de case va on deplacer le rectangle
        longx = random.randint(0, 3)
        longy = random.randint(0, 3)

        # # pour choisir l'axe sur lequel on deplace
        # bool_choix1 = random.randint(0, 1)

        # pour choisir la direction sur lequel on deplace
        dirx = random.choice([-1, 1])
        diry = random.choice([-1, 1])

        # si on peut pas deplacer le rectangle dans ce sens car il sortirait de l'image
        if self.rect.x + longx * dirx < 0 or self.rect.x + longx * dirx + self.rect.lar > self.img.shape[0]:
            dirx *= -1

        longx *= dirx

        # si on peut pas deplacer le rectangle dans ce sens car il sortirait de l'image
        if self.rect.y + longy * diry < 0 or self.rect.y + longy * diry + self.rect.hau > self.img.shape[1]:
            diry *= -1

        longy *= diry

        self.rect.move(longx, longy)

        w_img2 = copy.deepcopy(self.img)
        w_img2 = self.rect.draw(w_img2)

        next = cv.cvtColor(w_img2, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(self.prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.features[i % size_of_training, :] = flow.flatten()
        self.targets[i % size_of_training, :] = [longx, longy]

        # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # hsv = np.zeros((self.img.shape[0], self.img.shape[1], self.img.shape[2]), dtype='uint8')

        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 1] = 255
        # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # cv.imshow('frame2', bgr)
        # k = cv.waitKey(30) & 0xff

        # if k == 27:
            # break
        # elif k == ord('s'):
            # cv.imwrite('opticalfb.png', w_img2)
            # cv.imwrite('opticalhsv.png', bgr)

        self.prvs = next

    def train(self, i, size_of_training):
        print("==========================================================")
        print("Step: ", i)

        self.features = np.reshape(self.features, (size_of_training, self.hau, self.lar, 2))
        print("----------------")
        print("Train")

        self.model_checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)

        if i < size_of_training:
            self.model.fit(self.features[:i], self.targets[:i], shuffle=True, callbacks=[self.model_checkpoint])
        else:
            self.model.fit(self.features, self.targets, shuffle=True, callbacks=[self.model_checkpoint])

        print("----------------")
        print("Test")

        for j in range(10):
            [[predx, predy]] = self.model.predict(np.array([self.features[j, :, :]]))
            predx = round(predx)
            predy = round(predy)

            print("Loss: ", np.linalg.norm([predx - self.targets[j, 0], predy - self.targets[j, 1]]), "  / Pred: ", predx, predy, " / Diff: ", self.targets[j, 0] - predx, self.targets[j, 1] - predy)

        # on reset les arrays

        self.features = np.reshape(self.features, (size_of_training, self.hau * self.lar * 2))

        print()

# ========== MAIN =============================================================

if __name__ == "__main__":
    # img_bgr = cv.imread(sys.argv[1])
    # img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # cap = cv.VideoCapture(cv.samples.findFile("./Optical_flow_demo.mkv"))
    # ret, frame1 = cap.read()
    # prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    model = ModelOrientation(120, 100, 20, 20, 35, 42)
    model.fit(100000000000, 10000)
