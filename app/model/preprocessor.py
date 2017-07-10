import pickle
import numpy as np
import os.path
import base64
from PIL import Image
import sys

class Preprocessor:
	def __init__(self, fn="dataset.txt"):
		self.datafile = fn
		self.target_shape = [20,20]
		self.splitchar = "&"

		if os.path.isfile(self.datafile):
			with open(fn, 'r') as f:
				self.vocab = eval(f.readline())
				assert type(self.vocab) is list, "The first line of the file was not a vocab list."
		else:
			self.vocab = None

	def add_sample(self, data, index, vocab):
		assert self.vocab == vocab or self.vocab is None, "Vocab list has changed. You cannot append to this dataset"
		if self.vocab is None:
			with open(self.datafile, 'w') as f:
				v = str(vocab).replace("\n","")
				f.write(v)
		self.vocab = vocab
		n = self.preprocess(data)
		with open(self.datafile, "a") as f:
			n = str(n.tolist()).replace("\n","")
			f.write("\n" + str(self.vocab[index]) + self.splitchar + n)

	def load_sample(self, line_i = 0):
		X = None ; y = None
		if os.path.isfile(self.datafile):
			with open(self.datafile, 'r') as f:
				vocab = eval(f.readline())
				assert vocab == self.vocab, "The vocab for this datafile does not match the current vocab"

			with open(self.datafile, "r") as f:
				lines = f.readlines()
				X = np.asarray(eval(lines[line_i + 1].split(self.splitchar)[1]))
				y = lines[line_i + 1].split(self.splitchar)[0]
			return [X, y]

		else:
			raise ValueError("Could not find the datafile")

	def load_all(self):
		X = None ; y = None
		if os.path.isfile(self.datafile):
			with open(self.datafile, 'r') as f:
				vocab = eval(f.readline())
				assert vocab == self.vocab, "The vocab for this datafile does not match the current vocab"

			with open(self.datafile, "r") as f:
				lines = f.readlines()

				X = np.zeros((len(lines) - 1, self.target_shape[0]*self.target_shape[1]))
				y = []

				for i in range (len(lines)-1):
					X[i,:] = np.asarray(eval(lines[i + 1].split(self.splitchar)[1]))
					y.append(lines[i + 1].split(self.splitchar)[0])
				y = np.asarray(y)
			return [X, y]

		else:
			raise ValueError("Could not find the datafile")

	def preprocess(self, jpgtxt):
		# data = base64.decodestring(data)
		data = jpgtxt.split(',')[-1]
		data = base64.b64decode(data.encode('ascii'))

		g = open("temp.jpg", "wb")
		g.write(data)
		g.close()

		pic = Image.open("temp.jpg")
		M = np.array(pic) #now we have image data in numpy

		M = self.rgb2gray(M)
		M = self.squareTrim(M,threshold=0)
		M = self.naiveInterp2D(M,self.target_shape[0],self.target_shape[0])
		[N, mean, sigma] = self.normalize(M)
		n = N.reshape(-1)
		if np.isnan(np.sum(n)):
			n = np.zeros(n.shape)
		return n

	def dataset_length(self):
		with open(self.datafile) as f:
			for i, l in enumerate(f):
				pass
		return i + 1

	@staticmethod
	def squareTrim(M, min_side=20, threshold=0):
		assert M.shape[0]==M.shape[1],"Input matrix must be a square"
		wsum = np.sum(M,axis=0)
		nonzero = np.where(wsum > threshold*M.shape[1])[0]
		if len(nonzero) >=1:
			wstart = nonzero[0]
			wend = nonzero[-1]
		else:
			wstart=0 ; wend = 0

		hsum = np.sum(M,axis=1)
		nonzero = np.where(hsum > threshold*M.shape[0])[0]
		if len(nonzero) >=1:
			hstart = nonzero[0]
			hend = nonzero[-1]
		else:
			hstart=0 ; hend = 0

		diff = abs((wend-wstart) - (hend-hstart))
		if (wend-wstart > hend-hstart):
			side = max(wend-wstart+1, min_side)
			m = np.zeros((side, side))
			cropped = M[hstart:hend+1,wstart:wend+1]
			shift = diff/2
			m[shift:cropped.shape[0]+shift,:cropped.shape[1]] = cropped
		else:
			side = max(hend-hstart+1, min_side)
			m = np.zeros((side, side))
			cropped = M[hstart:hend+1,wstart:wend+1]
			shift=diff/2
			m[:cropped.shape[0],shift:cropped.shape[1]+shift] = cropped
		return m

	@staticmethod
	def rgb2gray(rgb):
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray

	@staticmethod
	def naiveInterp2D(M, newx, newy):
		result = np.zeros((newx,newy))
		for i in range(M.shape[0]):
			for j in range(M.shape[1]):
				indx = i*newx / M.shape[0]
				indy = j*newy / M.shape[1]
				result[indx,indy] +=M[i,j]
		return result

	@staticmethod
	def normalize(M):
		sigma = np.std(M)
		mean = np.mean(M)
		return [(M-mean)/sigma, mean, sigma]
