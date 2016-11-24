#!/usr/bin/env python
from __future__ import print_function
import logging
import os
import random
import sys

from argparse import ArgumentParser

import cv2
import numpy as np

from game.wrapped_flappy_bird import GameState
from brain.dqn import DQN

ACTIONS = 2

LOGGER = logging.getLogger(__name__)


class Score(object):

    def __init__(self, debug):
        self._debug = debug
        self._current = 0
        self._highest = 0
        self._attempts = 1

    def update(self, terminal, reward):
        if reward == 1:
            self._update_current(reward)
        if terminal:
            self._reset_current()

    def _update_current(self, reward):
        """Update the the score of current game."""
        self._current = self._current + reward
        if self._highest < self._current:
            self._highest = self._current

    def _reset_current(self):
        """Reset the current score."""
        self._current = 0
        self._attempts += 1

    def show(self):
        if not self._debug:
            # Clear screen
            print(chr(27) + "[2J")
            print("statistics".upper())
            print("-------------")
            print(self.__str__())
        else:
            LOGGER.debug('ATTEMPT: %s, CURRENT SCORE: %s, HIGH SCORE: %s',
                         self._attempts, self._current, self._highest)

    def __str__(self):
        return ('Attempt\t\t: {} \nCurrent score\t: {} \nHighscore\t: {}'.
                format(self._attempts, self._current, self._highest))


def _preprocess(observation):
    """Preprocess raw image to 80*80 gray image."""
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))


def _parse_args():
    parser = ArgumentParser("Flappy bird DQN")
    data_basepath = 'data'

    parser.add_argument('-d', '--debug', action="store_true",
                        help='verbose information')
    parser.add_argument('-c', '--checkpoints', dest='checkpoints',
                        default=os.path.join(data_basepath, 'checkpoints'),
                        help='Path where to store checkpoints (i.e partial training)')
    parser.add_argument('-s', '--summary', dest='summary',
                        default=os.path.join(data_basepath, 'summary'),
                        help='Path where to store summary data (for tensorboard)')
    args = parser.parse_args()
    return args


def _setup_logging(debug=False):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(lvl)
    logging.basicConfig(format="%(message)s")


def _play_game(args):
    game_state = GameState()
    score = Score(args.debug)

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    observation, _, _ = game_state.frame_step(do_nothing)
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, observation0 = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)

    dqn = DQN(observation0, ACTIONS, args.checkpoints, args.summary)

    while True:
        action = dqn.get_action()
        next_observation, reward, terminal = game_state.frame_step(action)
        score.update(terminal, reward)
        score.show()
        next_observation = _preprocess(next_observation)
        dqn.set_perception(next_observation, action, reward, terminal)


def main():
    args = _parse_args()
    _setup_logging(args.debug)
    _play_game(args)


if __name__ == "__main__":
    main()
