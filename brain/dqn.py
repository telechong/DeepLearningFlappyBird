from __future__ import print_function
import datetime
import logging
import os
import random
import time

from collections import deque

import numpy as np
import tensorflow as tf

ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 10000

LOGGER = logging.getLogger(__name__)


def _debub_array(array, name):
    shape = array.get_shape().as_list()[1:]
    LOGGER.debug('%s shape: %s, dimension: %s', name, shape, np.prod(shape))


class DQN:
    def __init__(self, init_observation, actions, checkpointpath=None, summarypath=None):
        self._replay_memory = deque()

        # Counters
        self._timestep = 0
        self._prev_timestep = 0
        self._prev_timestamp = time.clock()

        self._fps = tf.Variable(initial_value=0.0, dtype=tf.float32, name='frames_per_second')
        self._meta_state = tf.Variable(initial_value=0,
                                       dtype=tf.uint8,
                                       name='meta_state')  # (0-observe, 1-explore, 2-train)

        self._game_score = tf.Variable(initial_value=0, dtype=tf.float32, name='game_score')
        self._game_score_tmp = 0
        self._loss_function = tf.Variable(initial_value=0.0, dtype=tf.float32, name='loss')

        self._epsilon = tf.Variable(initial_value=INITIAL_EPSILON,
                                    dtype=tf.float32, name='epsilon')
        self._actions = actions

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
            self._action_input = tf.placeholder('float', [None, self._actions])
            self._y_input = tf.placeholder('float', [None])
            q_action = tf.reduce_sum(tf.mul(self._qvalue, self._action_input),
                                     reduction_indices=1, name='q_action')
            self._cost = tf.reduce_mean(tf.square(self._y_input - q_action), name='loss_function')
            self._train_step = tf.train.AdamOptimizer(1e-6).minimize(self._cost)
        self._session = tf.InteractiveSession()

        if summarypath is not None:
            tf.scalar_summary(r'overview/meta_state (0-observe, 1-explore, 2-train)',
                              self._meta_state)
            tf.scalar_summary(r'overview/cost (loss function)', self._loss_function)
            tf.scalar_summary(r'overview/epsilon (exploration probability)', self._epsilon)
            tf.scalar_summary(r'overview/game_score', self._game_score)
            tf.scalar_summary(r'performance/frames_per_second', self._fps)
            self._summaries = tf.merge_all_summaries()
            self._summarywriter = tf.train.SummaryWriter(os.path.join(summarypath,
                                                                      str(datetime.datetime.now())),
                                                         self._session.graph)

        # saving and loading networks
        # set max_to_keep=0 so that all the recent check points are saved
        self._saver = tf.train.Saver(max_to_keep=0)
        self._session.run(tf.initialize_all_variables())

        if checkpointpath is not None:
            try:
                os.makedirs(checkpointpath)
            except OSError:
                pass

            self._checkpoint = tf.train.get_checkpoint_state(checkpointpath)
            if self._checkpoint and self._checkpoint.model_checkpoint_path:
                self._saver.restore(self._session, self._checkpoint.model_checkpoint_path)
                LOGGER.info('Successfully loaded: %s', self._checkpoint.model_checkpoint_path)
            else:
                LOGGER.info('Could not find old network weights')

        self._current_state = np.stack((init_observation, init_observation,
                                        init_observation, init_observation), axis=2)

    def set_perception(self, next_observation, action, reward, terminal):
        # new_state = np.append(next_observation, self._current_state[:, :, 1:], axis=2)
        new_state = np.append(next_observation, self._current_state[:, :, :3], axis=2)
        self._replay_memory.append((self._current_state, action, reward, new_state, terminal))
        if len(self._replay_memory) > REPLAY_MEMORY:
            self._replay_memory.popleft()
        if self._timestep > OBSERVE:
            self._train_qnetwork()

        if self._timestep % 100 == 0:
            self._write_counters()

        if reward == 1:
            self._game_score_tmp += reward
        if terminal:
            self._game_score.assign(self._game_score_tmp).op.run()
            self._game_score_tmp = 0

        self._current_state = new_state
        self._timestep += 1

    def get_action(self):
        qvalue = self._qvalue.eval(feed_dict={self._stateinput: [self._current_state]})[0]
        action = np.zeros(self._actions)
        action_index = 0
        epsilon = self._epsilon.eval()
        if self._timestep % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(self._actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(qvalue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if epsilon > FINAL_EPSILON and self._timestep > OBSERVE:
            self._epsilon.assign_sub((INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE).op.run()

        return action

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
            w_fc2 = self._weight_variable([512, self._actions], name='fc2_weights')
            b_fc2 = self._bias_variable([self._actions], name='fc2_bias')

        with tf.name_scope('q-value'):
            qvalue = tf.matmul(h_fc1, w_fc2) + b_fc2

        return (stateinput, qvalue,
                w_conv1, b_conv1,
                w_conv2, b_conv2,
                w_conv3, b_conv3,
                w_fc1, b_fc1,
                w_fc2, b_fc2)

    def _copy_target_qnetwork(self):
        self._session.run(self._copy_target_qnetwork_operation)

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
            self._summarywriter.add_run_metadata(run_metadata, 'step_%d' % self._timestep)
        else:
            self._train_step.run(feed_dict={self._y_input: y_batch,
                                            self._action_input: action_batch,
                                            self._stateinput: state_batch})

        # calculate loss function (we do it via proxy variable since summaries are run separatly)
        value = self._cost.eval(feed_dict={self._y_input: y_batch,
                                           self._action_input: action_batch,
                                           self._stateinput: state_batch})
        self._loss_function.assign(value).op.run()

        # save network every 10000 iteration
        if self._timestep % 10000 == 0:
            self._saver.save(self._session, 'data/checkpoints/dqn', global_step=self._timestep)

        if self._timestep % UPDATE_TIME == 0:
            self._copy_target_qnetwork()

    def _write_counters(self):
        # FPS
        new_timestamp = time.clock()
        self._fps.assign((self._timestep - self._prev_timestep) / (new_timestamp - self._prev_timestamp)).op.run()
        self._prev_timestamp = new_timestamp
        self._prev_timestep = self._timestep

        # meta state (0-observe, 1-explore, 2-train)
        if self._timestep <= OBSERVE:
            state = 0
        elif self._timestep > OBSERVE and self._timestep <= OBSERVE + EXPLORE:
            state = 1
        else:
            state = 2
        self._meta_state.assign(state).op.run()

        summary = self._session.run(self._summaries)
        self._summarywriter.add_summary(summary, self._timestep)

    def _weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name)

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
