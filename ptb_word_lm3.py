from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import math
import reader
import util
import argparse

"""
To obtain data:
mkdir data && cd data && wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz && tar xvf simple-examples.tgz
"""

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='small',
                    choices=['small', 'medium', 'large', 'test'])
parser.add_argument('-data_path', type=str, default='./data/simple-examples/data')
parser.add_argument('-save_path', type=str, default='./model/saved')
parser.add_argument('-prior_pi', type=float, default=0.25)
parser.add_argument('-log_sigma1', type=float, default=-1.0)
parser.add_argument('-log_sigma2', type=float, default=-8.0)
parser.add_argument('-inference_mode', type=str, default='sample')
parser.add_argument('-b_stochastic', type=int, default=1)
FLAGS = parser.parse_args()
FLAGS.b_stochastic = bool(FLAGS.b_stochastic)


def get_config():
	"""Get model config."""
	config = None
	if FLAGS.model == "small":
		config = SmallConfig()
	elif FLAGS.model == "medium":
		config = MediumConfig()
	elif FLAGS.model == "large":
		config = LargeConfig()
	elif FLAGS.model == "test":
		config = TestConfig()
	else:
		raise ValueError("Invalid model: %s", FLAGS.model)

	# Set BBB params
	config.prior_pi = FLAGS.prior_pi
	config.log_sigma1 = FLAGS.log_sigma1
	config.log_sigma2 = FLAGS.log_sigma2
	config.b_stochastic = FLAGS.b_stochastic

	return config


def get_bbb_variable(shape, name, prior, is_training):
	"""gets a bbb_variable.

	It assumes Gaussian posterior and it creates two variables: name +'_mean',
	which corresponds to the mean of the gaussian; and name+ '_rho' which
	corresponds to the std of the gaussian (sigma = tf.nn.softplus(rho) + 1e-5).

	Args:
	  shape: shape of variable
	  name: string with variable name
	  prior: belongs to class Prior
	  kl: if True will compute(approx) kl between prior and current variable and
		  add it to a collection called "KL_layers"
	  reuse: either to reuse variable or not

	Returns:
	  output: sample from posterior Normal(mean, sigma)
	"""

	# No initializer specified -> will use the U(-scale, scale) init from main()
	with tf.variable_scope('BBB', reuse=not is_training):
		mu = tf.get_variable(name + '_mean', shape, dtype=tf.float32)

	rho_max_init = math.log(math.exp(prior.sigma_mix / 1.0) - 1.0)
	rho_min_init = math.log(math.exp(prior.sigma_mix / 2.0) - 1.0)
	init = tf.random_uniform_initializer(rho_min_init,
	                                     rho_max_init)

	with tf.variable_scope('BBB', reuse=not is_training):
		rho = tf.get_variable(
			name + '_rho', shape, dtype=tf.float32, initializer=init)

	if is_training or FLAGS.inference_mode == 'sample':
		epsilon = tf.contrib.distributions.Normal(0.0, 1.0).sample(shape)
		sigma = tf.nn.softplus(rho) + 1e-5
		output = mu + sigma * epsilon
	else:
		output = mu
	if not is_training:
		return output

	sample = output
	kl = compute_kl(shape, mu, sigma, prior, sample)
	# kl = compute_kl(shape, tf.reshape(mu, [-1]), tf.reshape(sigma, [-1]), prior, sample)
	tf.add_to_collection('KL_layers', kl)
	return output


class Prior(object):
	def __init__(self, pi, log_sigma1, log_sigma2):
		self.pi = pi
		self.log_sigma1 = log_sigma1
		self.log_sigma2 = log_sigma2
		self.sigma_mix = pi * math.exp(log_sigma1) + (1. - pi) * math.exp(log_sigma2)

	def get_logp(self, sample):
		var1, var2 = tf.exp(2 * self.log_sigma1), tf.exp(2 * self.log_sigma2)
		return tf.log(normal_mix(sample, self.pi, 0., 0., var1, var2))


def compute_kl(shape, mu, sigma, prior, sample):
	logp = prior.get_logp(sample)
	logq = tf.log((1.0 / tf.sqrt(2.0 * tf.square(sigma) * math.pi)) * tf.exp(
		-tf.square(sample - mu) / (2.0 * tf.square(sigma))))

	return tf.reduce_sum(logq - logp)


def normal_mix(samples, pi, mean1, mean2, var1, var2):
	"""
	Compute p(\theta) = pi * N(0, std1^{2}) + (1-pi) * N(0, std2^{2}).
	"""
	gaussian1 = (1.0 / tf.sqrt(2.0 * var1 * math.pi)) * tf.exp(
		- tf.square(samples - mean1) / (2.0 * var1))
	gaussian2 = (1.0 / tf.sqrt(2.0 * var2 * math.pi)) * tf.exp(
		- tf.square(samples - mean2) / (2.0 * var2))
	mixture = (pi * gaussian1) + ((1. - pi) * gaussian2)
	return mixture


class BayesianLSTM(tf.contrib.rnn.BasicLSTMCell):
	"""
	Pass weights in init.
	"""

	def __init__(self, num_units, theta, b, forget_bias=1.0, state_is_tuple=True,
	             activation=tf.tanh,
	             reuse=None, name=None):
		super(BayesianLSTM, self).__init__(num_units, forget_bias, state_is_tuple, activation,
		                                   reuse=reuse)

		self.theta = theta
		self.b = b
		self.name = name

	def _output(self, inputs, h):
		"""
		Forward pass in the LSTM.
		"""
		xh = tf.concat([inputs, h], 1)
		out = tf.matmul(xh, self.theta) + self.b
		return out

	def call(self, inputs, state):
		if self._state_is_tuple:
			c, h = state
		else:
			c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

		concat = self._output(inputs, h)
		i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

		new_c = (
			c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
		new_h = self._activation(new_c) * tf.sigmoid(o)

		if self._state_is_tuple:
			new_state = tf.contrib.rnn.LSTMStateTuple(c=new_c, h=new_h)
		else:
			new_state = tf.concat(values=[new_c, new_h], axis=1)

		return new_h, new_state


class PTBInput(object):
	"""The input data."""

	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(
			data, batch_size, num_steps, name=name)


class PTBModel(object):
	def __init__(self, is_training, config, input_):
		self._is_training = is_training
		self._input = input_
		self._rnn_params = None
		self._cell = None
		self.batch_size = input_.batch_size
		self.num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		with tf.device("/cpu:0"):
			embedding = tf.get_variable(
				"embedding", [vocab_size, size], tf.float32)
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

		# Construct prior
		prior = Prior(config.prior_pi, config.log_sigma1, config.log_sigma2)

		# Build the BBB LSTM cells
		cells = []
		theta_shape = (size + config.hidden_size, 4 * config.hidden_size)
		b_shape = (4 * config.hidden_size)
		for i in range(config.num_layers):
			theta = get_bbb_variable(theta_shape, 'theta_{}'.format(i), prior, is_training)
			if config.b_stochastic:
				b = get_bbb_variable(b_shape, 'b_{}'.format(i), prior, is_training)
			else:
				# Note: Reuse is passed down from variable scope in main()
				b = tf.get_variable('b_{}'.format(i), b_shape, tf.float32,
				                    tf.constant_initializer(0.))
			cells.append(BayesianLSTM(config.hidden_size, theta, b, forget_bias=0.0))

		cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		self._initial_state = cell.zero_state(config.batch_size, tf.float32)
		state = self._initial_state

		# Forward pass for the truncated mini-batch
		with tf.variable_scope("RNN"):
			inputs = tf.unstack(inputs, num=config.num_steps, axis=1)
			outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
			                                           initial_state=state)

			output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

		# Softmax BBB output projection layer
		softmax_w_shape = (size, vocab_size)
		softmax_b_shape = (vocab_size,)
		softmax_w = get_bbb_variable(softmax_w_shape, 'softmax_w', prior, is_training)

		if config.b_stochastic:
			softmax_b = get_bbb_variable(softmax_b_shape, 'softmax_b', prior, is_training)
		else:
			softmax_b = tf.get_variable('softmax_b', softmax_b_shape, tf.float32,
			                            tf.constant_initializer(0.))

		logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

		# Reshape logits to be a 3-D tensor for sequence loss
		logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

		# Use the contrib sequence loss and average over the batches
		loss = tf.contrib.seq2seq.sequence_loss(
			logits,
			input_.targets,
			tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True)

		# Update the cost
		self._cost = tf.reduce_sum(loss)
		self._final_state = state

		if not is_training:
			return

		# Compute KL divergence for each cell and the projection layer
		# KL is scaled by 1./(B*C) as in the paper
		C = self._input.epoch_size
		B = self.batch_size
		scaling = 1. / (B * C)

		kl_div = tf.add_n(tf.get_collection('KL_layers'), 'kl_divergence')
		kl_div *= scaling

		# ELBO
		self._total_loss = self._cost + kl_div

		# Learning rate & optimization
		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self._total_loss, tvars),
		                                  config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())

		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	def export_ops(self, name):
		"""Exports ops to collections."""
		self._name = name
		ops = {util.with_prefix(self._name, "cost"): self._cost}
		if self._is_training:
			ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
			if self._rnn_params:
				ops.update(rnn_params=self._rnn_params)
		for name, op in ops.items():
			tf.add_to_collection(name, op)
		self._initial_state_name = util.with_prefix(self._name, "initial")
		self._final_state_name = util.with_prefix(self._name, "final")
		util.export_state_tuples(self._initial_state, self._initial_state_name)
		util.export_state_tuples(self._final_state, self._final_state_name)

	def import_ops(self):
		"""Imports ops from collections."""
		if self._is_training:
			self._train_op = tf.get_collection_ref("train_op")[0]
			self._lr = tf.get_collection_ref("lr")[0]
			self._new_lr = tf.get_collection_ref("new_lr")[0]
			self._lr_update = tf.get_collection_ref("lr_update")[0]
			rnn_params = tf.get_collection_ref("rnn_params")
			if self._cell and rnn_params:
				params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
					self._cell,
					self._cell.params_to_canonical,
					self._cell.canonical_to_params,
					rnn_params,
					base_variable_scope="Model/RNN")
				tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
		self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
		num_replicas = 1
		self._initial_state = util.import_state_tuples(
			self._initial_state, self._initial_state_name, num_replicas)
		self._final_state = util.import_state_tuples(
			self._final_state, self._final_state_name, num_replicas)

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def initial_state_name(self):
		return self._initial_state_name

	@property
	def final_state_name(self):
		return self._final_state_name


class SmallConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000


# class MediumConfig(object):
# 	"""Medium config."""
# 	init_scale = 0.05
# 	learning_rate = 1.0
# 	max_grad_norm = 5
# 	num_layers = 2
# 	num_steps = 35
# 	hidden_size = 650
# 	max_epoch = 6
# 	max_max_epoch = 39
# 	keep_prob = 0.5
# 	lr_decay = 0.8
# 	batch_size = 20
# 	vocab_size = 10000

class MediumConfig(object):
	"""Medium config."""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epoch = 20
	max_max_epoch = 70
	keep_prob = 1.0
	lr_decay = 0.9
	batch_size = 20
	vocab_size = 10000


class LargeConfig(object):
	"""Large config."""
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	lr_decay = 1 / 1.15
	batch_size = 20
	vocab_size = 10000


class TestConfig(object):
	"""Tiny config, for testing."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1
	num_layers = 1
	num_steps = 2
	hidden_size = 2
	max_epoch = 1
	max_max_epoch = 1
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)

	fetches = {
		"cost": model.cost,
		"final_state": model.final_state,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
			      (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
			       iters * model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)


def run():
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to PTB data directory")

	raw_data = reader.ptb_raw_data(FLAGS.data_path)
	train_data, valid_data, test_data, _ = raw_data

	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,
		                                            config.init_scale)

		with tf.name_scope("Train"):
			train_input = PTBInput(config=config, data=train_data, name="TrainInput")
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				m = PTBModel(is_training=True, config=config, input_=train_input)
			tf.summary.scalar("Training Loss", m.cost)
			tf.summary.scalar("Learning Rate", m.lr)

		with tf.name_scope("Valid"):
			valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
			tf.summary.scalar("Validation Loss", mvalid.cost)

		with tf.name_scope("Test"):
			test_input = PTBInput(
				config=eval_config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = PTBModel(is_training=False, config=eval_config,
				                 input_=test_input)

		models = {"Train": m, "Valid": mvalid, "Test": mtest}
		for name, model in models.items():
			model.export_ops(name)
		metagraph = tf.train.export_meta_graph()
		soft_placement = False

	with tf.Graph().as_default():
		tf.train.import_meta_graph(metagraph)
		for model in models.values():
			model.import_ops()
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
		with sv.managed_session(config=config_proto) as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
				m.assign_lr(session, config.learning_rate * lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, eval_op=m.train_op,
				                             verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

			test_perplexity = run_epoch(session, mtest)
			print("Test Perplexity: %.3f" % test_perplexity)

			if FLAGS.save_path:
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == '__main__':
	run()
