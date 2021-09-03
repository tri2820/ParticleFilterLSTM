# import math
# import numpy as np
# from scipy.stats import norm

# class KLDResampling(object):
# 	def __init__(self, bin_size, delta=0.01, epsilon=0.05, Nmax=100):
# 		# initialize first state
# 		self.k = 0
# 		self.bin_size = bin_size
# 		self.num_samples = 0
# 		self.delta = delta
# 		self.epsilon = epsilon
# 		self.bin_size = bin_size

# 		self.Nmax = Nmax

# 	def _is_occupied(self, bin_map, curr_sample):
# 		curr_bin_index = tuple(np.floor(curr_sample).astype(int))
# 		is_occupied  = bin_map[curr_bin_index]
# 		# Now this bin is occupied
# 		bin_map[curr_bin_index] = True
# 		return is_occupied
	
# 	def resample(self, particles, weights, room_size):
# 		i = 0
# 		N = 1 
# 		nbin_along_dimension = np.ceil(np.array(room_size)/np.array(self.bin_size)).astype(int)
# 		bin_map = np.zeros(nbin_along_dimension, dtype=bool)
# 		resample_particles = []
# 		while i<=N and i<=self.Nmax:
# 			idx = np.random.multinomial(1, weights).argmax()
# 			curr_sample = particles[idx][:]
# 			resample_particles.append(curr_sample)
# 			if not self._is_occupied(bin_map, curr_sample):
# 				self.k += 1
# 				if self.k >= 2:
# 					zvalue = norm.cdf(1-self.delta)
# 					N = math.ceil((self.k - 1) / (2*self.epsilon) * pow(1 - 2/(9*(self.k - 1)) + math.sqrt(2/(9 * (self.k - 1))) * zvalue, 3))
# 			i += 1
# 		return np.array(resample_particles)

# if __name__ == "__main__":
# 	kld = KLDResampling([0.01,0.01], Nmax=10) 
# 	particles = np.array([[1,2],[2,2],[3,3]])
# 	weights = np.array([.25, .25, .5]) 
# 	room_size = [5,5]
# 	print(kld.resample(particles, weights, room_size))


import math
import numpy as np
from scipy.stats import norm

class KLDResampling(object):
	Nmax = 10 # max number of particles
	def __init__(self, bin_size, delta=0.01, epsilon=0.05):
		# initialize first state
		self.num_resample_bin = 0
		self.zvalue = 0.
		self.bin_size = bin_size
		self.num_samples = 0
		self.bins = []
		self.curr_sample = []
		self.delta = delta
		self.epsilon = epsilon
		self.bin_size = bin_size

	def in_empty_bin(self):
		curr_bin = []
		for i in range(len(self.curr_sample)):
			curr_bin.append(math.floor(self.curr_sample[i]/self.bin_size[i]))
		if curr_bin in self.bins:
			return False
		self.bins.append(curr_bin)
		return True

	"""
	resample: Resampling using KLD
	----------------------------------
	Args:
		particles: array of shape (N, d), N is number of particles, d is dimension
		weights: array of (1, N), should be in the same order with particles
		# true position is not necessary in resampling process
	Returns:
		resample_particles: list of resampled particles
	"""
	def resample(self, particles, weights):
		i = 0
		N = 1 
		resample_particles = []
		while i<=N and i<=KLDResampling.Nmax:
			idx = np.random.multinomial(1, weights).argmax()
			self.curr_sample = particles[idx][:]
			resample_particles.append(particles[idx][:])
			if self.in_empty_bin():
				self.num_resample_bin += 1
				if self.num_resample_bin >= 2:
					zvalue = norm.cdf(1-self.delta)
					N = math.ceil((self.num_resample_bin - 1) / (2*self.epsilon) * pow(1 - 2/(9*(self.num_resample_bin - 1)) + math.sqrt(2/(9 * (self.num_resample_bin - 1))) * zvalue, 3))
			i += 1
		return np.array(resample_particles)

# Example 
# # bin size should has the shape of (d,) where d is the dimension of particles
# # here, all particles has the dimension of 1, therefore, bin size should only have one dimension
kld = KLDResampling([0.01]) 
particles = np.array([[1],[2],[3]]) # each particles has only one dimension
weights = np.array([.25, .25, .5]) # weights should be normalized (this weight is normalized, so its sum equals to 1)
print(kld.resample(particles, weights))

# Usage: 
# Each time the resample process takes place, initialize KLDResampling object, and invoke resample methods
# It should be as follows:
# ...
# kld = KLDResampling(<bin_size>)
# kld.resample(<particles>, <weights>)
# ...