import pickle
from sqlite3 import dbapi2 as sqlite
import numpy as np
import cv2


class Searcher(object):
    def __init__(self,db,voc):
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()


    def candidates_from_word(self,imword):
        im_ids = self.con.execute("select distinct imid from imwords where wordid=%d" % imword).fetchall()

        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        words = imwords*self.voc.idf
        words = np.argsort(words)[::-1][0:5]
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates +=c

        tmp = [(w, candidates.count(w)) for w in set(candidates)]
        tmp.sort(key= lambda x:x[1],reverse=True)
        tmp.sort()
        tmp.reverse()

        return [w[0] for w in tmp]

    def get_imhistogram(self, imname):
        im_id = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()

        s = self.con.execute("select histogram from imhistograms where rowid='%d'" % im_id).fetchone()
        return pickle.loads(s[0])

    def query(self, imname,nbr_results=15):
        orb = cv2.ORB_create()
        img = cv2.imread(imname, 0)
        kp1, des = orb.detectAndCompute(img, None)
        iw = self.voc.project(des)
        iw_copy = iw.copy()
        iw_copy = iw_copy / np.sum(iw_copy)
        iw_copy = self.voc.idf*iw_copy

        ndx = np.argsort(iw_copy)[::-1][0:50]
        iw_copy = iw_copy[ndx]

        candidates = self.candidates_from_histogram(iw)
        matchscores = []
        for imid in candidates:



            cand_name = self.con.execute("select filename from imlist where rowid=%d" % imid).fetchone()
            cand_h = self.get_imhistogram(cand_name)
            cand_h = cand_h/np.sum(cand_h)
            cand_h =  self.voc.idf*cand_h
            cand_h = cand_h[ndx]
            cand_dist = np.sqrt(sum((iw_copy - cand_h) ** 2))
            matchscores.append((cand_dist, imid))

        matchscores.sort()
        return matchscores

    def get_filename(self, imid):
        s = self.con.execute("select filename from imlist where rowid='%d'" % imid).fetchone()
        return s[0]



