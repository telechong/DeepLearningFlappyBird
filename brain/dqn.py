import logging
import random

from collections import deque

import numpy as np
import tensorflow as tf

from .constants import GAMMA

from .constants import OBSERVE
from .constants import EXPLORE

from .constants import INITIAL_EPSILON
from .constants import FINAL_EPSILON

from .constants import REPLAY_MEMORY
from .constants import BATCH_SIZE
from .constants import UPDATE_TIME
from .constants import FRAME_PER_ACTION

LOGGER = logging.getLogger(__name__)


class DQN(object):

    """This class implements the DQN algorithm."""
    ACTION_SIZE = 2

    def __init__(self, init_observation, actions, checkpoint_mgmt, summary):
        self._replay_memory = deque()

        # Counters
        self._timestep = 0
        self._loss_function = tf.Variable(initial_value=0.0, dtype=tf.float32, name='loss')

        self._epsilon = tf.Variable(initial_value=INITIAL_EPSILON,
                                    dtype=tf.float32, name='epsilon')

        # init Q network
        (self._stateinput,
         self._qvalue,
         self._w_conv1,
         self._b_conv_1,
         self._w_conv_2,
         self._b_conv2,
         self._w_conv_3,
         self._b_conv3,
         self._w_fc_1,
         self._b_fc1,
         self._w_fc_2,
         self._b_fc2) = self._create_q_network()

        # init Target Q Network
        (self._stateinput_t,
         self._qvalue_t,
         self._w_conv_1_t,
         self._b_conv_1_t,
         self._w_conv_2_t,
         self._b_conv_2_t,
         self._w_conv_3_t,
         self._b_conv_3_t,
         self._w_fc_1_t,
         self._b_fc_1_t,
         self._w_fc_2_t,
         self._b_fc_2_t) = self._create_q_network()

        with tf.name_scope('copy_target_qnetwork'):
            self._copy_target_qnetwork_operation = [self._w_conv_1_t.assign(self._w_conv1),
                                                    self._b_conv_1_t.assign(self._b_conv_1),
                                                    self._w_conv_2_t.assign(self._w_conv_2),
                                                    self._b_conv_2_t.assign(self._b_conv2),
                                                    self._w_conv_3_t.assign(self._w_conv_3),
                                                    self._b_conv_3_t.assign(self._b_conv3),
                                                    self._w_fc_1_t.assign(self._w_fc_1),
                                                    self._b_fc_1_t.assign(self._b_fc1),
                                                    self._w_fc_2_t.assign(self._w_fc_2),
                                                    self._b_fc_2_t.assign(self._b_fc2)]
        with tf.name_scope('train_qnetwork'):
            self._action_input = tf.placeholder('float', [None, self.ACTION_SIZE])
            self._y_input = tf.placeholder('float', [None])
            q_action = tf.reduce_sum(tf.mul(self._qvalue, self._action_input),
                                     reduction_indices=1, name='q_action')
            self._cost = tf.reduce_mean(tf.square(self._y_input - q_action), name='loss_function')
            self._train_step = tf.train.AdamOptimizer(1e-6).minimize(self._cost)

        self._session = tf.InteractiveSession()

        self._summary = summary
        self._summary.setup(self._session.graph, self._loss_function, self._epsilon)

        # saving and loading networks
        # set max_to_keep=0 so that all the recent check points are saved
        self._session.run(tf.initialize_all_variables())

        self._cp_mgmt = checkpoint_mgmt
        self._cp_mgmt.load(self._session)

        self._current_state = np.stack((init_observation, init_observation,
                                        init_observation, init_observation), axis=2)

    def _create_q_network(self):
        with tf.name_scope('input_layer'):
            stateinput = tf.placeholder('float', [None, 80, 80, 4], name='stateinput')

        # hidden layers
        with tf.name_scope('conv1'):
            w_conv1 = self._weight_variable([8, 8, 4, 32], name='conv1_weights')
            b_conv1 = self._bias_variable([32], name='conv1_bias')
            h_conv1 = tf.nn.relu(self._conv2d(stateinput, w_conv1, 4) + b_conv1, name='conv1_relu')
            h_pool1 = self._max_pool_2x2(h_conv1)

        with tf.name_scope('conv2'):
            w_conv2 = self._weight_variable([4, 4, 32, 64], name='conv2_weights')
            b_conv2 = self._bias_variable([64], name='conv2_bias')
            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, w_conv2, 2) + b_conv2, name='conv2_relu')

        with tf.name_scope('conv3'):
            w_conv3 = self._weight_variable([3, 3, 64, 64], name='conv3_weights')
            b_conv3 = self._bias_variable([64], name='conv3_bias')
            h_conv3 = tf.nn.relu(self._conv2d(h_conv2, w_conv3, 1) + b_conv3, name='conv3_relu')
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600], name='conv3_flatten')

        with tf.name_scope('fc1'):
            w_fc1 = self._weight_variable([1600, 512], name='fc1_weights')
            b_fc1 = self._bias_variable([512], name='fc1_bias')
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1, name='fc1_relu')

        with tf.name_scope('fc2'):
            w_fc2 = self._weight_variable([512, self.ACTION_SIZE], name='fc2_weights')
            b_fc2 = self._bias_variable([self.ACTION_SIZE], name='fc2_bias')

        with tf.name_scope('q-value'):
            qvalue = tf.matmul(h_fc1, w_fc2) + b_fc2

        return (stateinput, qvalue,
                w_conv1, b_conv1,
                w_conv2, b_conv2,
                w_conv3, b_conv3,
                w_fc1, b_fc1,
                w_fc2, b_fc2)

    def _bias_variable(self, shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name)

    def _weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_action(self):
        qvalue = self._qvalue.eval(feed_dict={self._stateinput: [self._current_state]})[0]
        action = np.zeros(self.ACTION_SIZE)
        action_index = 0
        epsilon = self._epsilon.eval()
        if self._timestep % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(self.ACTION_SIZE)
                action[action_index] = 1
            else:
                action_index = np.argmax(qvalue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # Change episilon
        if epsilon > FINAL_EPSILON and self._timestep > OBSERVE:
            self._epsilon.assign_sub((INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE).op.run()

        return action

    def set_perception(self, next_observation, action, reward, terminal):
        # new_state = np.append(next_observation, self._current_state[:, :, 1:], axis=2)
        new_state = np.append(next_observation, self._current_state[:, :, :3], axis=2)
        self._replay_memory.append((self._current_state, action, reward, new_state, terminal))
        if len(self._replay_memory) > REPLAY_MEMORY:
            self._replay_memory.popleft()
        if self._timestep > OBSERVE:
            self._train_qnetwork()

        self._summary.update(self._session, self._timestep, reward, terminal)
        self._current_state = new_state
        self._timestep += 1

    def _train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self._replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        qvalue_batch = self._qvalue_t.eval(feed_dict={self._stateinput_t: next_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(qvalue_batch[i]))

        if self._timestep == OBSERVE + 1:
            run_metadata = tf.RunMetadata()
            self._session.run(self._train_step,
                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                              run_metadata=run_metadata,
                              feed_dict={self._y_input: y_batch,
                                         self._action_input: action_batch,
                                         self._stateinput: state_batch})
            self._summary.add_run_metadata(run_metadata, self._timestep)
        else:
            self._train_step.run(feed_dict={self._y_input: y_batch,
                                            self._action_input: action_batch,
                                            self._stateinput: state_batch})

        # calculate loss function (we do it via proxy variable since summaries are run separately)
        value = self._cost.eval(feed_dict={self._y_input: y_batch,
                                           self._action_input: action_batch,
                                           self._stateinput: state_batch})
        self._loss_function.assign(value).op.run()

        # save network every 10000 iteration
        if self._timestep % 10000 == 0:
            self._cp_mgmt.store(self._session, self._timestep)

        if self._timestep % UPDATE_TIME == 0:
            self._copy_target_qnetwork()

    def _copy_target_qnetwork(self):
        self._session.run(self._copy_target_qnetwork_operation)
