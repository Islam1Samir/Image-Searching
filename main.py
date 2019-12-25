import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import Vocabulary
import DbSqlite
from sqlite3 import dbapi2 as sqlite
import Searcher
import pickle
from PIL import Image
import os


files = glob.glob('H:\Image Searching\Data\*')
nbr_images = len(files)
if os.path.isfile('vocab'):
    voc = pickle.load(open('vocab', 'rb'))
    print('Load Vocabulary class')
else:
    voc = Vocabulary.Vocabulary('ukbenchtest')
    voc.train(files,1500,10,True)
    pickle.dump(voc, open('vocab', 'wb'))


orb = cv2.ORB_create()
if not os.path.isfile('vocab'):
    indx = DbSqlite.Dbsqlite('test.db', voc)
    indx.create_database()
    indx.db_commit()

    for i in range(nbr_images):
        img = cv2.imread(files[i], 0)
        kp1, des = orb.detectAndCompute(img, None)
        indx.add_image(files[i], des)
        indx.db_commit()




src = Searcher.Searcher('test.db',voc)
q_ind = [21,137,257,376,426,474,767,958]
nbr_results = 30
plt.figure()
nbr_q = len(q_ind)

for index in range(nbr_q):

    res_reg = [w[1] for w in src.query(files[q_ind[index]])[:nbr_results]]

    img1 = cv2.imread(files[q_ind[index]], 0)
    kp1, des1 = orb.detectAndCompute(img1, None)
    rank = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in res_reg:
        img = cv2.imread(files[i - 1], 0)
        kp2, des2 = orb.detectAndCompute(img, None)
        matches = bf.match(des1, des2)
        fp = np.array([kp1[j.queryIdx].pt for j in matches])
        tp = np.array([kp2[j.trainIdx].pt for j in matches])
        H, mask = cv2.findHomography(fp, tp, cv2.RANSAC)
        rank[i] = sum(mask == 1)

    sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
    res_geom = [s[0] for s in sorted_rank]
    print('top matches (homography):', res_geom)
    plt.subplot(nbr_q, 10, index*10+1)
    plt.imshow(np.array(Image.open(files[q_ind[index]])))
    plt.axis('off')

    for i in range(9):
        imname = src.get_filename(res_geom[i])
        plt.subplot(nbr_q, 10, index*10+i + 2)
        plt.imshow(np.array(Image.open(imname)))
        plt.axis('off')

plt.savefig('img5.png')
plt.show()



