from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from src import cluster
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class ClusterEnv(py_environment.PyEnvironment):

    def __init__(self):
        cluster.init_cluster()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(cluster.features,), dtype=np.int32, minimum=np.copy(cluster.cluster_state_min),
            maximum=np.copy(cluster.cluster_state_max),
            name='observation')
        self._state = np.copy(cluster.cluster_state_init)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        cluster.init_cluster()
        self._state = np.copy(cluster.cluster_state_init)
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action == 0:
            print('action 0')
            # moveJobs() <- finish one running job, update cluster states
            # if no running jobs but jobs waiting to be scheduled -> huge Neg Reward and episode ends
        elif action == 1:
            print('action 1')
            # if invalid placement -> Huge Neg Reward and episode ends
            # else place 1 ex in VM 1, update cluster states; check for episode ends
        elif action == 2:
            print('action 2')
        elif action == 3:
            print('action 3')
        else:
            raise ValueError('`action` should in 0 to 3.')

            # self._episode_ended = True -> when last job's last executor is placed or bad action

            # self._state = generate new state after executing the current action

        if self._episode_ended:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=0.9)


environment = ClusterEnv()
utils.validate_py_environment(environment, episodes=5)

get_new_card_action = 0
end_round_action = 1

environment = ClusterEnv()
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(3):
    time_step = environment.step(get_new_card_action)
    print(time_step)
    cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)
