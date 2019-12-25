import pickle
from sqlite3 import dbapi2 as sqlite
import numpy as np

class Dbsqlite(object):
    def __init__(self,db,voc):
        self.con =sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()


    def create_database(self):
        self.con.execute('create table imlist(filename)')
        self.con.execute('create table imwords(imid,wordid)')
        self.con.execute('create table imhistograms(imid,histogram)')
        self.con.execute('create index im_idx on imlist(filename)')
        self.con.execute('create index wordid_idx on imwords(wordid)')
        self.con.execute('create index imid_idx on imwords(imid)')
        self.con.execute('create index imidhist_idx on imhistograms(imid)')
        self.db_commit()

    def add_image(self,imname,descriptor):
        if self.is_indexed(imname): return
        print('Adding', imname)

        imid = self.get_id(imname)
        imwords = self.voc.project(descriptor)
        words = imwords.nonzero()[0]

        for word in words:
            self.con.execute("insert into imwords(imid,wordid) values (?,?)",(imid,int(word)))



        self.con.execute("insert into imhistograms(imid,histogram) values (?,?)",(imid, pickle.dumps(imwords)))

    def is_indexed(self, imname):
        im = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()

        return im != None

    def get_id(self, imname):

        cur = self.con.execute("select rowid from imlist where filename='%s'" % imname)
        res = cur.fetchone()

        if res == None:
            cur = self.con.execute("insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid
        else:
            return res[0]

