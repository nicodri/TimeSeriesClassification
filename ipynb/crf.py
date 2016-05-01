import marshal
import numpy as np
from scipy import misc,optimize
import copy

def convert_ll(y):
	tag = y[0]
	y_ind = []
	y_cur = []
	for i in range(len(y)):
	    if y[i] != tag:
	        y_ind.append(np.array(copy.copy(y_cur)))
	        y_cur = [i]
	    else:
	        y_cur.append(i)
	    tag = y[i]
	y_ind.append(np.array(copy.copy(y_cur)))
	return y_ind

def break_even_ll(y, size = 100):
	y_ind = []
	y_cur = []
	for i in range(len(y)):
	    if len(y_cur) >= size:
	        y_ind.append(np.array(copy.copy(y_cur)))
	        y_cur = []
	    y_cur.append(i)
	y_ind.append(np.array(copy.copy(y_cur)))
	return y_ind

def log_dot_vm(loga,logM):
	return misc.logsumexp(loga.reshape(loga.shape+(1,))+logM,axis=0)
def log_dot_mv(logM,logb):
	return misc.logsumexp(logM+logb.reshape((1,)+logb.shape),axis=1)

class CRF:
	def __init__(self,sigma=10):
		self.v = sigma ** 2
		self.v2 = self.v * 2
		self.C = 6 # number of classes
		self.D = 567 # number of features
		self.theta  = np.random.randn(self.D)

	def regulariser(self,w):
		return np.sum(w ** 2) /self.v2

	def regulariser_deriv(self,w):
		return np.sum(w) / self.v

	def all_features(self, x):
		"""
		Axes:
		0 - T or time or sequence index
		1 - y' or previous label
		2 - y  or current  label
		3 - f(y',y,x,i) for i s
		"""
		result = np.zeros((len(x), self.C, self.C, self.D))
		for i in range(len(x)):
			for j,yp in enumerate(range(self.C)):
				feature_vector = x[i]
				feature_vector[:self.C] = 0
				feature_vector[j] = 1
				result[i,j,:,:] = np.tile(feature_vector, (self.C, 1))

		return result

	def forward(self,M,start=0):
		alphas = np.NINF * np.ones((M.shape[0],M.shape[1]))
		alpha  = alphas[0]
		alpha[start] = 0
		for i in range(M.shape[0]-1):
			alpha = alphas[i+1] = log_dot_vm(alpha,M[i])
		alpha = log_dot_vm(alpha,M[-1])
		return (alphas,alpha)

	def backward(self,M,end=-1):
		betas = np.zeros((M.shape[0],M.shape[1]))
		beta  = betas[-1]
		beta[end] = 0
		for i in reversed(range(M.shape[0]-1)):
			beta = betas[i] = log_dot_mv(M[i+1],beta)
		beta = log_dot_mv(M[0],beta)
		return (betas,beta)

	def create_vector_list(self,x_seq,y_seq):
		observations = [ self.all_features(x) for x in x_seq ]
		labels = len(y_seq) * [None]
		for i in range(len(y_seq)):
			# start from end of previous sequence
			start = y_seq[i-1][-1] if i > 0 else y_seq[i][0]
			labels[i] = np.array([start] + list(y_seq[i]))
		return (observations, labels)

	def neg_likelihood_and_deriv(self,x_vec_list,y_vec_list,theta,debug=False):
		likelihood = 0
		derivative = np.zeros(len(self.theta))
		for x_vec,y_vec in zip(x_vec_list,y_vec_list):
			"""
			all_features:	len(x_vec) + 1 x Y x Y x K
			M:				len(x_vec) + 1 x Y x Y
			alphas:			len(x_vec) + 1 x Y
			betas:			len(x_vec) + 1 x Y
			log_probs:		len(x_vec) + 1 x Y x Y  (Y is the size of the state space)
			`unnormalised` value here is alpha * M * beta, an unnormalised probability
			"""
			all_features    = x_vec
			length 			= x_vec.shape[0]
			yp_vec_ids      = y_vec[:-1]
			y_vec_ids       = y_vec[1:]
			log_M           = np.dot(all_features,theta)
			log_alphas,last = self.forward(log_M)
			log_betas, zero = self.backward(log_M)
			time,state      = log_alphas.shape

			# reshape
			log_alphas1 = log_alphas.reshape(time,state,1)
			log_betas1  = log_betas.reshape(time,1,state)
			log_Z       = misc.logsumexp(last)
			log_probs   = log_alphas1 + log_M + log_betas1 - log_Z
			log_probs   = log_probs.reshape(log_probs.shape+(1,))
			
			if debug:
				print '**********'
				print 'length'
				print length
				print 'len(yp_vec_ids)'
				print len(yp_vec_ids)
				print 'len(y_vec_ids)'
				print len(y_vec_ids)
				print 'all_features.shape'
				print all_features.shape
				print 'yp_vec_ids'
				print yp_vec_ids
				print 'y_vec_ids'
				print y_vec_ids
				print '**********'

			"""
			Expected and empirical values
			"""
			exp_features = np.sum( np.exp(log_probs) * all_features, axis= (0,1,2) )
			emp_features = np.sum( all_features[range(length),yp_vec_ids,y_vec_ids], axis = 0 )

			likelihood += np.sum(log_M[range(length),yp_vec_ids,y_vec_ids]) - log_Z
			derivative += emp_features - exp_features
		
		print 'likelihood = {}'.format(likelihood - self.regulariser(theta))
		return (
			- ( likelihood - self.regulariser(theta)), 
			- ( derivative - self.regulariser_deriv(theta))
		)
	
	def predict(self,x_vec, debug=False):
		# small overhead, no copying is done
		"""
		all_features:	len(x_vec+1) x Y' x Y x K
		log_potential:	len(x_vec+1) x Y' x Y
		argmaxes:		len(x_vec+1) x Y'
		"""
		all_features  = self.all_features(x_vec)
		log_potential = np.dot(all_features,self.theta)
		return self.viterbi_bp(log_potential,len(x_vec),self.C)
	
	def viterbi_bp(self,log_potential,N,K,debug=False):
		g0 = log_potential[0,0]
		g  = log_potential[1:]

		B = np.ones((N,K), dtype=np.int32) * -1
		# compute max-marginals and backtrace matrix
		V = g0
		for t in xrange(1,N):
			U = np.empty(K)
			for y in xrange(K):
				w = V + g[t-1,:,y]
				B[t,y] = b = w.argmax()
				U[y] = w[b]
			V = U
		# extract the best path by brack-tracking
		y = V.argmax()
		trace = []
		for t in reversed(xrange(N)):
			trace.append(y)
			y = B[t, y]
		trace.reverse()
		return trace

	def train(self,debug=False):
		X,Y = self.create_vector_list(train_x_seq,train_y_seq)
		l = lambda theta: self.neg_likelihood_and_deriv(X,Y,theta)
		val = optimize.fmin_l_bfgs_b(l,self.theta)
		if debug: print val
		self.theta,_,_  = val
		return self.theta


if __name__ == '__main__':
	x_train_file = open('../HAR/train/X_train.txt', 'r')
	y_train_file = open('../HAR/train/y_train.txt', 'r')
	s_train_file = open('../HAR/train/subject_train.txt', 'r')

	x_test_file = open('../HAR/test/X_test.txt', 'r')
	y_test_file = open('../HAR/test/y_test.txt', 'r')
	s_test_file = open('../HAR/test/subject_test.txt', 'r')

	# Create empty lists
	x_train = []
	y_train = []
	s_train = []

	x_test = []
	y_test = []
	s_test = []

	# Loop through datasets
	for x in x_train_file:
	    x_train.append([float(ts) for ts in x.split()])
	    
	for y in y_train_file:
	    y_train.append(int(y.rstrip('\n')))
	    
	for s in s_train_file:
	    s_train.append(int(s.rstrip('\n')))
	    
	for x in x_test_file:
	    x_test.append([float(ts) for ts in x.split()])
	    
	for y in y_test_file:
	    y_test.append(int(y.rstrip('\n')))

	for s in s_test_file:
	    s_test.append(int(s.rstrip('\n')))
	    
	# Convert to numpy for efficiency
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	s_train = np.array(s_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	s_test = np.array(s_test)

	x_train_with_past = np.zeros((x_train.shape[0]-1,x_train.shape[1]+6))
	for i in range(x_train.shape[0]-1):
	    tofill = np.zeros(x_train.shape[1]+6)
	    tofill[y_train[i]-1] = 1
	    tofill[6:] = x_train[i+1,:]
	    x_train_with_past[i,:] = tofill[:]
	y_train_with_past = y_train[1:] - 1 # convert to 0-based

	x_test_with_past = np.zeros((x_test.shape[0]-1,x_test.shape[1]+6))
	for i in range(x_test.shape[0]-1):
	    tofill = np.zeros(x_test.shape[1]+6)
	    tofill[y_test[i]-1] = 1
	    tofill[6:] = x_test[i+1]
	    x_test_with_past[i,:] = tofill[:]
	y_test_with_past = y_test[1:] - 1 # convert to 0-based

	# convert y_train into list of lists
	y_ind_train = convert_ll(y_train_with_past)
	train_x_seq = []
	train_y_seq = []
	for i in range(len(y_ind_train)):
		indices = y_ind_train[i]
		train_x_seq.append(x_train_with_past[indices])
		train_y_seq.append(y_train_with_past[indices])

	y_ind_test = convert_ll(y_test_with_past)
	test_x_seq = []
	test_y_seq = []
	for i in range(len(y_ind_test)):
		indices = y_ind_test[i]
		test_x_seq.append(x_test_with_past[indices])
		test_y_seq.append(y_test_with_past[indices])

	crf = CRF()
	crf.train()

	# total_correct = 0
	# total_count = 0
	# for i in range(len(test_x_seq)):
	# 	y_hat = crf.predict(test_x_seq[i])
	# 	total_correct += np.sum(y_hat == test_y_seq[i])
	# 	total_count += len(test_y_seq[i])
	# print 'accuracy = {}'.format(float(total_correct) / total_count)
