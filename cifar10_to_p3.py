#!/usr/bin/python
import numpy as np
import cPickle as pickle
import sys
import glob
import os.path

if __name__ == '__main__':
	folder = sys.argv[1]
	paths = [_ for _ in glob.glob(os.path.join(folder, "data_batch_*"))]
	for p in paths:
		print "Converting", p
		with open(p, "rb") as fo:
			d = pickle.load(fo)["data"]
			d = 0.299 * d[:,:1024] + 0.587 * d[:,1024:2048] + 0.114 * d[:,2048:3072]
			out = p + ".npy"
			print "Saving data in", p, "to", out
			np.save(out, d)
	print "Done"
