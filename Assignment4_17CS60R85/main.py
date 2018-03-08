#________________________________________________________
# Ami Ladani (17CS60R85)
# Assignment 4 : Image classification using LSTM and GRU
#________________________________________________________


from data_loader import DataLoader
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import argparse
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer

tf.set_random_seed(42)


class _train_LSTM_Cell(rnn.RNNCell):

	def __init__(self, no_hidden_units, input_size):
		self.no_hidden = no_hidden_units
		self.size_vec  = input_size


	@property
	def input_size(self):
		return self.size_vec

	@property
	def output_size(self):
		return self.no_hidden

	@property
	def state_size(self):
		return 2 * self.no_hidden


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_C, prev_h = tf.split(state, 2, 1)
			I_Wx = tf.get_variable("I_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			I_Uh = tf.get_variable("I_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))
			I_b  = tf.get_variable("I_b", [1, self.no_hidden], initializer=tf.ones_initializer())

			F_Wx = tf.get_variable("F_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			F_Uh = tf.get_variable("F_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))
			F_b  = tf.get_variable("F_b", [1, self.no_hidden], initializer=tf.ones_initializer())

			T_Wx = tf.get_variable("T_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			T_Uh = tf.get_variable("T_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))
			T_b  = tf.get_variable("T_b", [1, self.no_hidden], initializer=tf.ones_initializer())

			O_Wx = tf.get_variable("O_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			O_Uh = tf.get_variable("O_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))
			O_b  = tf.get_variable("O_b", [1, self.no_hidden], initializer=tf.ones_initializer())

			i_gate  = tf.sigmoid(tf.add( tf.add( tf.matmul(x, I_Wx), tf.matmul(prev_h, I_Uh) ) , I_b))
			f_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, F_Wx), tf.matmul(prev_h, F_Uh) ) , F_b))
			o_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, O_Wx), tf.matmul(prev_h, O_Uh) ) , O_b))
			t_output = tf.tanh(tf.add( tf.add( tf.matmul(x, T_Wx), tf.matmul(prev_h, T_Uh) ) , T_b))
			C_t = tf.add(tf.multiply(f_gate, prev_C), tf.multiply(i_gate, t_output))
			y_t = tf.multiply(o_gate, tf.tanh(C_t))
			return y_t, tf.concat([C_t, y_t], 1)


class _test_LSTM_Cell(rnn.RNNCell):
	def __init__(self, no_hidden_units, input_size, parameters):
		self.no_hidden = no_hidden_units
		self.size_vec  = input_size
		self.paras = parameters
	
	@property
	def input_size(self):
		return self.size_vec

	@property
	def output_size(self):
		return self.no_hidden

	@property
	def state_size(self):
		return 2 * self.no_hidden

	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_C, prev_h = tf.split(state, 2, 1)
			I_Wx = self.paras['I_Wx']
			I_Uh = self.paras['I_Uh']
			I_b = self.paras['I_b']

			F_Wx = self.paras['F_Wx']
			F_Uh = self.paras['F_Uh']
			F_b = self.paras['F_b']

			T_Wx = self.paras['T_Wx']
			T_Uh = self.paras['T_Uh']
			T_b = self.paras['T_b']

			O_Wx = self.paras['O_Wx']
			O_Uh = self.paras['O_Uh']
			O_b = self.paras['O_b']
	
			i_gate  = tf.sigmoid(tf.add( tf.add( tf.matmul(x, I_Wx), tf.matmul(prev_h, I_Uh) ) , I_b))
			o_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, O_Wx), tf.matmul(prev_h, O_Uh) ) , O_b))
			f_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, F_Wx), tf.matmul(prev_h, F_Uh) ) , F_b))
			t_output = tf.tanh(tf.add( tf.add( tf.matmul(x, T_Wx), tf.matmul(prev_h, T_Uh) ) , T_b))
			C_t = tf.add(tf.multiply(f_gate, prev_C), tf.multiply(i_gate, t_output))
			y_t = tf.multiply(o_gate, tf.tanh(C_t))
			return y_t, tf.concat([C_t, y_t], 1)


class _test_GRU_Cell(rnn.RNNCell):
	def __init__(self, no_hidden_units, input_size, parameters):
		self.no_hidden = no_hidden_units
		self.size_vec  = input_size
		self.paras = parameters
	
	@property
	def input_size(self):
		return self.size_vec

	@property
	def output_size(self):
		return self.no_hidden

	@property
	def state_size(self):
		return self.no_hidden


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_h = state
			Z_Wx = self.paras['Z_Wx']
			Z_Uh = self.paras['Z_Uh']

			R_Wx = self.paras['R_Wx']
			R_Uh = self.paras['R_Uh']

			T_Wx = self.paras['T_Wx']
			T_Uh = self.paras['T_Uh']
			T_b = self.paras['T_b']
	
			u_gate  = tf.sigmoid(tf.add( tf.matmul(x, Z_Wx), tf.matmul(prev_h, Z_Uh) ))
			r_gate = tf.sigmoid(tf.add( tf.matmul(x, R_Wx), tf.matmul(prev_h, R_Uh) ) )
			t_output = tf.tanh( tf.add (tf.add( tf.matmul(x, T_Wx), tf.matmul(tf.multiply(prev_h, r_gate), T_Uh) ), T_b))
			y_t =  tf.add ( tf.multiply((1.0-u_gate),prev_h) , tf.multiply(u_gate, t_output) )			
			return y_t, y_t


class _train_GRU_Cell(rnn.RNNCell):
	def __init__(self, no_hidden_units, input_size):
		self.no_hidden = no_hidden_units
		self.size_vec  = input_size

	@property
	def input_size(self):
		return self.size_vec

	@property
	def output_size(self):
		return self.no_hidden

	@property
	def state_size(self):
		return self.no_hidden


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_h = state
			Z_Wx = tf.get_variable("Z_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			Z_Uh = tf.get_variable("Z_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))

			R_Wx = tf.get_variable("R_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			R_Uh = tf.get_variable("R_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))

			T_Wx = tf.get_variable("T_Wx", [self.size_vec, self.no_hidden], initializer=xavier_initializer(seed=42))
			T_Uh = tf.get_variable("T_Uh", [self.no_hidden, self.no_hidden], initializer=xavier_initializer(seed=42))
			T_b  = tf.get_variable("T_b", [1, self.no_hidden], initializer=tf.ones_initializer())

			u_gate  = tf.sigmoid(tf.add( tf.matmul(x, Z_Wx), tf.matmul(prev_h, Z_Uh) ))
			r_gate = tf.sigmoid(tf.add( tf.matmul(x, R_Wx), tf.matmul(prev_h, R_Uh) ) )
			t_output = tf.tanh( tf.add (tf.add( tf.matmul(x, T_Wx), tf.matmul(tf.multiply(prev_h, r_gate), T_Uh) ), T_b))
			y_t =  tf.add ( tf.multiply((1.0-u_gate),prev_h) , tf.multiply(u_gate, t_output) )			
			return y_t, y_t


def _train_model(vector_size=28, no_vectors = 28,no_classes=10, l_rate=0.0001,  epochs=100, batch_size=32):
	x = tf.unstack(X , vector_size, 1)
	if args.model == 'lstm':
		# train lstm-cell		
		train_lstm_cell = _train_LSTM_Cell(no_hidden_units, no_vectors)
		outputs, states	= tf.nn.static_rnn(train_lstm_cell, x, dtype=tf.float32) 

	if args.model == 'gru': 
		# train gru-cell
		train_gru_cell = _train_GRU_Cell(no_hidden_units, no_vectors)
		outputs, states	= tf.nn.static_rnn(train_gru_cell, x, dtype=tf.float32)  

	# fully-connected layer
	FC_W = tf.get_variable("FC_W", [no_hidden_units, no_classes], initializer = tf.contrib.layers.xavier_initializer(seed=42))
	FC_b = tf.get_variable("FC_b", [no_classes], initializer = tf.zeros_initializer())
	L = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)

	#optimization
	loss = _cross_entropy_loss(L, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate = l_rate)
	training_op = optimizer.minimize(loss)
	actual_pred = tf.equal(tf.argmax(L, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(actual_pred, tf.float32))

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	weight_filepath = "./weights/"+str(args.model)+"/hidden_unit" + str(no_hidden_units)+ "/model.ckpt"
	with tf.Session() as sess:
		tf.set_random_seed(42)
		init.run()
		#training
		for epoch in range(epochs):
			for X_batch, Y_batch in itertools.izip(x_batch, y_batch):
				X_batch	= X_batch.reshape((-1,	vector_size, no_vectors))
				sess.run(training_op, feed_dict={X:	X_batch, Y: Y_batch})	
			acc_train = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})
			print("Train accuracy after %s epochs: %s" %( str(epoch+1), str(acc_train*100) ))	
		saver.save(sess, weight_filepath)

def test_model(vector_size=28, no_vectors = 28,no_classes=10):
	with tf.Session() as sess:
		if args.model == 'lstm':
			model_cell = '_train_LSTM_Cell'

		elif args.model == 'gru':
			model_cell = '_train_GRU_Cell'

		
		weight_filepath = "./weights/"+str(args.model)+"/hidden_unit" + str(no_hidden_units)
		new_saver = tf.train.import_meta_graph(weight_filepath + "/model.ckpt.meta")
		new_saver.restore(sess, tf.train.latest_checkpoint(weight_filepath))

		FC_W = sess.run('FC_W:0')
		FC_b = sess.run('FC_b:0') 
		
		if args.model == 'lstm':
			I_Wx = sess.run('rnn/'+str(model_cell)+'/I_Wx:0')
			I_Uh = sess.run('rnn/'+str(model_cell)+'/I_Uh:0')
			I_b = sess.run('rnn/'+str(model_cell)+'/I_b:0')
			F_Wx = sess.run('rnn/'+str(model_cell)+'/F_Wx:0')
			F_Uh = sess.run('rnn/'+str(model_cell)+'/F_Uh:0')
			F_b = sess.run('rnn/'+str(model_cell)+'/F_b:0')
			O_Wx = sess.run('rnn/'+str(model_cell)+'/O_Wx:0')
			O_Uh = sess.run('rnn/'+str(model_cell)+'/O_Uh:0')
			O_b = sess.run('rnn/'+str(model_cell)+'/O_b:0')
			T_Wx = sess.run('rnn/'+str(model_cell)+'/T_Wx:0')
			T_Uh = sess.run('rnn/'+str(model_cell)+'/T_Uh:0')
			T_b = sess.run('rnn/'+str(model_cell)+'/T_b:0')

			parameters = {
				"F_Wx" : F_Wx,
				"F_Uh" : F_Uh,
				"F_b"  : F_b,
				"I_Wx" : I_Wx,
				"I_Uh" : I_Uh,
				"I_b"  : I_b,
				"O_Wx" : O_Wx,
				"O_Uh" : O_Uh,
				"O_b"  : O_b,				
				"T_Wx" : T_Wx,
				"T_Uh" : T_Uh,
				"T_b" : T_b				
			}

		elif args.model == 'gru':
			Z_Wx   = sess.run('rnn/'+str(model_cell)+'/Z_Wx:0')
			Z_Uh   = sess.run('rnn/'+str(model_cell)+'/Z_Uh:0')
			R_Wx   = sess.run('rnn/'+str(model_cell)+'/R_Wx:0')
			R_Uh   = sess.run('rnn/'+str(model_cell)+'/R_Uh:0')
			T_Wx = sess.run('rnn/'+str(model_cell)+'/T_Wx:0')
			T_Uh = sess.run('rnn/'+str(model_cell)+'/T_Uh:0')
			T_b  = sess.run('rnn/'+str(model_cell)+'/T_b:0')
		
			parameters = {
				"R_Wx" : R_Wx,
				"R_Uh" : R_Uh,
				"Z_Wx" : Z_Wx,
				"Z_Uh" : Z_Uh,				
				"T_Wx" : T_Wx,
				"T_Uh" : T_Uh,
				"T_b" : T_b
			}	
		# lstm-cell
		x = tf.unstack(X, vector_size, 1)

		if args.model == 'lstm':
			lstm_cell_test = _test_LSTM_Cell(no_hidden_units, no_vectors, parameters)
			outputs, states	= tf.nn.static_rnn(lstm_cell_test, x, dtype=tf.float32) 

		elif args.model == 'gru':
			gru_cell_test = _test_GRU_Cell(no_hidden_units, no_vectors, parameters)
			outputs, states	= tf.nn.static_rnn(gru_cell_test, x, dtype=tf.float32) 

		# fully-connected layer
		L = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
		actual_pred = tf.equal(tf.argmax(L, 1), tf.argmax(Y, 1))
		p_accuracy = tf.reduce_mean(tf.cast(actual_pred, tf.float32))

		init = tf.global_variables_initializer()
		test_accuracy = p_accuracy.eval(feed_dict={X: x_test, Y: y_test})				
		print "Test accuracy:  ", test_accuracy * 100 

def _cross_entropy_loss(logits, labels):
	tf.set_random_seed(42)
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

vector_size = 28
no_vectors = 28
no_classes = 10
batch_size = 32
l_rate =0.001
epochs = 10

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--test' , action='store_true',help='To test the data')
parser.add_argument('--train', action='store_true',help='To train the data')
parser.add_argument('--hidden_unit', action='store',help='No of hidden units',type=int)
parser.add_argument('--model', action='store',help='which model will be used - LSTM or GRU')
args = parser.parse_args()

tf.set_random_seed(100)
np.random.seed(60)

no_hidden_units = args.hidden_unit
X = tf.placeholder(tf.float32, [None, vector_size, no_vectors])
Y = tf.placeholder(tf.int32, [None, no_classes])

# data-loading
data_loader = DataLoader()
x_train, y_train = data_loader.load_data()
y_train = np.eye(10)[np.asarray(y_train, dtype=np.int32)]
x_batch, y_batch = data_loader.create_batches(x_train, y_train, batch_size)	
x_test,y_test = data_loader.load_data(mode='test')
y_test = np.eye(10)[np.asarray(y_test, dtype=np.int32)]
x_test = x_test.reshape((-1, vector_size, no_vectors))


if args.train:		
	_train_model(vector_size, no_vectors, no_classes, l_rate, epochs, batch_size)
   
elif args.test:		
	test_model(vector_size, no_vectors, no_classes)
