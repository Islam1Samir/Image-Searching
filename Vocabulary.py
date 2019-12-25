import numpy as np
import cv2
from scipy.cluster.vq import *


class Vocabulary(object):

    def __init__(self,name):
        self.name = name
        self.centroids = []
        self.idf = []
        self.ndx = []
        self.nbr_words = 0
        self.stop_words = False

    def train(self, paths, k=100, subsampling=1,stop_words = False):


        self.stop_words = stop_words
        orb = cv2.ORB_create()
        nbr_images = len(paths)
        descr = []
        img = cv2.imread(paths[0], 0)
        kp1, des = orb.detectAndCompute(img, None)
        descr.append(des)
        descriptors = descr[0]
        for i in np.arange(1, nbr_images):
            img = cv2.imread(paths[i], 0)
            kp1, des = orb.detectAndCompute(img, None)
            descr.append(des)
            descriptors = np.vstack((descriptors, descr[i]))

        descriptors = descriptors * 1.0
        self.centroids, _ = kmeans(descriptors[::subsampling,:], k, 1)
        self.nbr_words = self.centroids.shape[0]
        imwords = np.zeros((nbr_images, self.nbr_words))

        for i in range(nbr_images):
            imwords[i] = self.project(descr[i])

        nbr_occurences = np.sum((imwords > 0) * 1, axis=0)

        self.idf = np.log((1.0 * nbr_images) / (1.0 * nbr_occurences + 1))
        if self.stop_words ==True:
           self.ndx = np.argsort(self.idf)[:300]
           self.idf = np.delete(self.idf, self.ndx)


    def project(self, descriptors):
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descriptors, self.centroids)
        for w in words:
            imhist[w] += 1

        if self.stop_words == True:
            imhist = np.delete(imhist, self.ndx)

        return imhist

    def project_tfidf(self,iw):
        return iw * self.idf

    def project_stop(self, descriptors):
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descriptors, self.centroids)
        for w in words:
            imhist[w] += 1


        imhist = np.delete(imhist,self.ndx)

        return imhist