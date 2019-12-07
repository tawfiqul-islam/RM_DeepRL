from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from src import cluster
from queue import PriorityQueue
from src import definitions as defs
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
            shape=(cluster.features,), dtype=np.int32, minimum=cluster.cluster_state_min,
            maximum=cluster.cluster_state_max,
            name='observation')
        self._state = np.copy(cluster.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = np.copy(cluster.VMS)
        self.jobs = np.copy(cluster.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        cluster.init_cluster()
        self._state = np.copy(cluster.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = np.copy(cluster.VMS)
        self.jobs = np.copy(cluster.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        print('current cluster state: ', self._state)
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action > 3 or action < 0:
            raise ValueError('`action` should in 0 to 3.')

        elif action == 0:
            print('action 0, clock', self.clock)
            # penalty for partial placement
            if self.jobs[self.job_idx].ex_placed > 0:
                self.reward += (-1000)
                self._episode_ended = True
            # if no running jobs but jobs waiting to be scheduled -> huge Neg Reward and episode ends
            elif self.job_queue.empty():
                self.reward += (-1000)
                self._episode_ended = True
            # finishOneJob() <- finish one running job, update cluster states-> "self._state"
            else:
                _, y = self.job_queue.get()
                self.finish_one_job(y)
            # TODO add check for large job which does not fit in the cluster
        else:
            print('action ', action, ' clock ', self.clock)
            # if valid placement, place 1 ex in the VM chosen, update cluster states -> "self._state";
            # check for episode end  -> update self._episode_ended
            if self.execute_placement(action):
                #print('placement successful, clock: ', self.clock)
                self.check_episode_end()
            # if invalid placement -> Huge Neg Reward and episode ends
            else:
                self.reward += (-1000)
                self._episode_ended = True
                print('bad action, episode ended')
            # self._episode_ended = True -> when last job's last executor is placed or bad action

            # self._state = generate new state after executing the current action

        if self._episode_ended:
            print('episode ended ')
            self.calculate_reward()
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)

        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0, discount=0.9)

    def finish_one_job(self, finished_job):
        finished_job.finished = True
        finished_job.running = False
        self.clock = finished_job.finish_time
        for i in range(len(finished_job.ex_placement_list)):
            vm = finished_job.ex_placement_list[i]
            vm.cpu_now += finished_job.cpu
            vm.mem_now += finished_job.mem
            self.vms[vm.id] = vm
        self._state = cluster.gen_cluster_state(self.job_idx, self.jobs,
                                                self.vms)
        print('finished execution of job ', finished_job.id)
        print('current cluster state: ', self._state, ' clock ', self.clock)

    def execute_placement(self, action):
        current_job = self.jobs[self.job_idx]
        vm = self.vms[action - 1]
        if current_job.cpu > vm.cpu_now or current_job.mem > vm.mem_now:
            return False

        if not current_job.running:
            current_job.running = True
            current_job.start_time = self.clock
            current_job.finish_time = self.clock + current_job.duration
            self.job_queue.put((current_job.finish_time, current_job))

        current_job.ex_placed += 1
        current_job.ex_placement_list.append(vm)
        vm.cpu_now -= current_job.cpu
        vm.mem_now -= current_job.mem

        if current_job.finish_time > vm.stop_use_clock:
            vm.stop_use_clock = current_job.finish_time
            vm.used_time += (vm.stop_use_clock - current_job.finish_time)

        self.vms[vm.id] = vm
        self.jobs[self.job_idx] = current_job

        if current_job.ex_placed == current_job.ex:
            print('finished placement of job ', current_job.id)
            self.job_idx += 1
            self.clock = self.jobs[self.job_idx].arrival_time

            while True:
                if self.job_queue.empty():
                    break
                _, next_finished_job = self.job_queue.get()
                if next_finished_job.finish_time <= self.clock:
                    self.finish_one_job(next_finished_job)
                else:
                    self.job_queue.put((next_finished_job.finish_time, next_finished_job))
                    break

        self._state = cluster.gen_cluster_state(self.job_idx, self.jobs,
                                                self.vms)
        # print('current cluster state: ', self._state)
        return True

    def check_episode_end(self):
        current_job = self.jobs[self.job_idx]
        if self.job_idx + 1 == len(self.jobs) and current_job.ex == current_job.ex_placed:
            self._episode_ended = True

    def calculate_reward(self):
        for i in range(len(self.vms)):
            self.reward -= (self.vms[i].price * self.vms[i].used_time)
        print('reward: ', self.reward, ' \n\n')


environment = ClusterEnv()
utils.validate_py_environment(environment, episodes=15)
#
# get_new_card_action = 0
# end_round_action = 1
#
# environment = ClusterEnv()
# time_step = environment.reset()
# print(time_step)
# cumulative_reward = time_step.reward
#
# for _ in range(3):
#     time_step = environment.step(get_new_card_action)
#     print(time_step)
#     cumulative_reward += time_step.reward
#
# time_step = environment.step(end_round_action)
# print(time_step)
# cumulative_reward += time_step.reward
# print('Final Reward = ', cumulative_reward)
