"""This module handles the data generated or needed in algorithm."""
import datetime
import logging
import os
import time

import tensorflow as tf

from .constants import OBSERVE
from .constants import EXPLORE

LOGGER = logging.getLogger(__name__)


class Summary(object):

    def __init__(self, summary_path):
        self._summary_path = summary_path

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

        self._summaries = None
        self._summarywriter = None

    def setup(self, session_graph, loss_func, epsilon):
        if self._summary_path is not None:
            tf.scalar_summary(r'overview/meta_state (0-observe, 1-explore, 2-train)',
                              self._meta_state)
            tf.scalar_summary(r'overview/cost (loss function)', loss_func)
            tf.scalar_summary(r'overview/epsilon (exploration probability)', epsilon)
            tf.scalar_summary(r'overview/game_score', self._game_score)
            tf.scalar_summary(r'performance/frames_per_second', self._fps)
            self._summaries = tf.merge_all_summaries()
            self._summarywriter = tf.train.SummaryWriter(
                os.path.join(self._summary_path, str(datetime.datetime.now())),
                session_graph)

    def update(self, session, time_step, reward, terminal):
        self._timestep = time_step
        if time_step % 100 == 0:
            self._write_counters(session)

        if reward == 1:
            self._game_score_tmp += reward
        if terminal:
            self._game_score.assign(self._game_score_tmp).op.run()
            self._game_score_tmp = 0

    def add_run_metadata(self, run_metadata, time_step):
        self._summarywriter.add_run_metadata(run_metadata, 'step_{}'.format(time_step))

    def _write_counters(self, session):
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

        summary = session.run(self._summaries)
        self._summarywriter.add_summary(summary, self._timestep)


class CheckPointMgmt(object):

    """This class is used to load and save checkpoints."""

    def __init__(self, restore_path, store_path):
        self._restore_path = restore_path
        self._store_path = store_path

        # set max_to_keep=0 so that all the recent check points are saved
        self._saver = tf.train.Saver(max_to_keep=0)

    def setup(self):
        """Create store path if not exist."""
        if self._store_path and not os.path.exists(self._store_path):
            os.makedirs(self._store_path)

    def load(self, session):
        """Load stored checkpoint to given session."""
        if self._restore_path is not None:
            checkpoint_state = tf.train.get_checkpoint_state(self._restore_path)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                model_path = checkpoint_state.model_checkpoint_path
                self._saver.restore(session, model_path)
                LOGGER.info('Successfully loaded: %s', model_path)
            else:
                LOGGER.info('Could not find old network weights from path %s',
                            self._restore_path)

    def store(self, session, time_step, name="dqn"):
        """Save the checkpoint from given session."""
        self._saver.save(session, "{}/{}".format(self._store_path, name),
                         global_step=time_step)
