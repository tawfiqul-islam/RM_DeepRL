from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import abc
import tensorflow as tf
import numpy as np
import cluster
import constants
from queue import PriorityQueue
import definitions as defs
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

logging.basicConfig(level=logging.DEBUG, filename='../output/app.log', filemode='w')


# logging.debug('This will get logged to a file')

class ClusterEnv(py_environment.PyEnvironment):

    def __init__(self):
        # cluster.init_cluster()
        # logging.debug('length cluster_state_min ', len(cluster.cluster_state_min))
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(cluster.features,), dtype=np.int32, minimum=cluster.cluster_state_min,
            maximum=cluster.cluster_state_max,
            name='observation')
        self._state = copy.deepcopy(cluster.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = copy.deepcopy(cluster.VMS)
        self.jobs = copy.deepcopy(cluster.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        self.episode_success = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # cluster.init_cluster()
        self._state = copy.deepcopy(cluster.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = copy.deepcopy(cluster.VMS)
        self.jobs = copy.deepcopy(cluster.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        self.episode_success = False

        # print(self.jobs[self.job_idx].ex_placed)
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        # logging.debug("Current Cluster State: {}".format(self._state))
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            # print(' i was here to reset and episode!!!!!!!!!!!!!!1')
            return self.reset()

        if action > 9 or action < 0:
            raise ValueError('`action` should be in 0 to 9.')

        elif action == 0:
            logging.debug("CLOCK: {}: Action: {}".format(self.clock, action))
            # penalty for partial placement
            if self.jobs[self.job_idx].ex_placed > 0:
                self.reward = (-200)
                self._episode_ended = True
                logging.debug("CLOCK: {}: Partial Executor Placement for a Job. Episode Ended\n\n".format(self.clock))
            # if no running jobs but jobs waiting to be scheduled -> huge Neg Reward and episode ends
            elif self.job_queue.empty():
                self.reward = (-200)
                self._episode_ended = True
                logging.debug(
                    "CLOCK: {}: No Executor Placement When No Job was Running. Episode Ended\n\n".format(self.clock))
            # finishOneJob() <- finish one running job, update cluster states-> "self._state"
            else:
                self.reward = -10
                _, y = self.job_queue.get()
                self.clock = y.finish_time
                self.finish_one_job(y)
            # TODO add check for large job which does not fit in the cluster
        else:
            logging.debug("CLOCK: {}: Action: {}".format(self.clock, action))
            # if valid placement, place 1 ex in the VM chosen, update cluster states -> "self._state";
            # check for episode end  -> update self._episode_ended
            if self.execute_placement(action):
                # print('placement successful, clock: ', self.clock)
                self.reward = 1
                self.check_episode_end()
            # if invalid placement -> Huge Neg Reward and episode ends
            else:
                self.reward = (-200)
                self._episode_ended = True
                logging.debug("CLOCK: {}: Invalid Executor Placement, Episode Ended\n\n".format(self.clock))

            # self._episode_ended = True -> when last job's last executor is placed or bad action

            # self._state = generate new state after executing the current action
        if self._episode_ended:

            if self.episode_success:
                # while True:
                #     if self.job_queue.empty():
                #         break
                #     cur_clock, next_finished_job = self.job_queue.get()
                #     self.clock = cur_clock
                #     self.finish_one_job(next_finished_job)
                # Multi-Objective Reward Calculation
                epi_cost = self.calculate_vm_cost()
                cost_reward = constants.beta * (epi_cost / cluster.total_episode_cost)
                epi_avg_job_duration = self.calculate_avg_time()
                time_reward = (1 - constants.beta) * (1 / (epi_avg_job_duration - cluster.min_avg_job_duration + 1))
                self.reward = 1 + 100 / (cost_reward + time_reward)
                logging.debug("CLOCK: {}: ****** Episode ended Successfully!!!!!!!! \n\n".format(self.clock))

            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)

        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=self.reward, discount=.9)

    def finish_one_job(self, finished_job):
        finished_job.finished = True
        finished_job.running = False
        for i in range(len(finished_job.ex_placement_list)):
            vm = finished_job.ex_placement_list[i]
            vm.cpu_now += finished_job.cpu
            vm.mem_now += finished_job.mem
            self.vms[vm.id] = vm
        self._state = cluster.gen_cluster_state(self.job_idx, self.jobs, self.vms)
        logging.debug("CLOCK: {}: Finished execution of job: {}".format(self.clock, finished_job.id))
        logging.debug("CLOCK: {}: Current Cluster State: {}".format(self.clock, self._state))

    def execute_placement(self, action):
        current_job = self.jobs[self.job_idx]
        vm = self.vms[action - 1]
        if current_job.cpu > vm.cpu_now or current_job.mem > vm.mem_now:
            return False

        if not current_job.running:
            current_job.running = True
            current_job.start_time = self.clock
            current_job.finish_time = self.clock + current_job.duration
            # TODO comment out this line to avoid invalid queue problem
            self.job_queue.put((current_job.finish_time, current_job))

        if current_job.start_time > vm.stop_use_clock:
            vm.used_time += current_job.duration
            vm.stop_use_clock = current_job.finish_time
        else:
            if current_job.finish_time > vm.stop_use_clock:
                vm.used_time += (current_job.finish_time - vm.stop_use_clock)
                vm.stop_use_clock = current_job.finish_time

        vm.cpu_now -= current_job.cpu
        vm.mem_now -= current_job.mem
        current_job.ex_placed += 1
        current_job.ex_placement_list.append(vm)

        self.vms[vm.id] = copy.deepcopy(vm)
        self.jobs[self.job_idx] = copy.deepcopy(current_job)

        if current_job.ex_placed == current_job.ex:
            # self.reward = 10
            logging.debug("CLOCK: {}: Finished placement of job: {}".format(self.clock, current_job.id))
            # Apply Job Duration Variance depending on placement type
            # For CPU and Memory bound applications -> Consolidated placement is better
            # For IO / Network bound applications -> Distributed placement is better
            # If condition does not satisfy -> Apply a 20% job duration increase

            if constants.pp_apply == 'true':
                # IO / Network bound jobs
                if current_job.type == 3:
                    if len(set(current_job.ex_placement_list)) > 1:
                        duration_increase = current_job.duration * float(constants.placement_penalty) / 100
                        current_job.duration += duration_increase
                        current_job.finish_time += duration_increase
                # Compute or Memory bound jobs
                else:
                    if len(set(current_job.ex_placement_list)) != 1:
                        duration_increase = current_job.duration * float(constants.placement_penalty) / 100
                        current_job.duration += duration_increase
                        current_job.finish_time += duration_increase
            # TODO uncomment the next line and comment out the previous job_queue_put line
            # self.job_queue.put((current_job.finish_time, current_job))
            if self.job_idx + 1 == len(self.jobs):
                self._episode_ended = True
                self.episode_success = True
                return True
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
        logging.debug("CLOCK: {}: Current Cluster State: {}".format(self.clock, self._state))
        return True

    def check_episode_end(self):
        current_job = self.jobs[self.job_idx]
        if self.job_idx + 1 == len(self.jobs) and current_job.ex == current_job.ex_placed:
            self._episode_ended = True

    def calculate_vm_cost(self):
        cost = 0
        for i in range(len(self.vms)):
            cost += (self.vms[i].price * self.vms[i].used_time)
            logging.debug("VM: {}, Price: {}, Time: {}".format(i, self.vms[i].price, self.vms[i].used_time))
            # print('vm: ', i, ' price: ', self.vms[i].price, ' time: ', self.vms[i].used_time)
        logging.debug("***Episode VM Cost: {}".format(cost))
        return cost

    def calculate_avg_time(self):
        time = 0
        for i in range(len(self.jobs)):
            time += self.jobs[i].duration
            logging.debug("Job: {}, Duration: {}".format(i, self.jobs[i].id, self.jobs[i].duration))
        avg_time = time / len(self.jobs)
        logging.debug("***Episode AVG Job Duration: {}".format(avg_time))
        return avg_time

# environment = ClusterEnv()
# environment2 = ClusterEnv()

# environment = ClusterEnv()
# utils.validate_py_environment(environment, episodes=1000)
