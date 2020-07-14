from __future__ import absolute_import, division, print_function

import csv

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment

import utilities
from rm_environment import ClusterEnv
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import main
tf.compat.v1.enable_v2_behavior()


# Data Collection
def collect_episode(environment, policy, num_episodes, replay_buffer):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# ***Metrics and Evaluation ***
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        # print('\n\n evaluation started \n')
        while not time_step.is_last():
            action_step = policy.action(time_step)
            # print('action: ', action_step.action)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        # print('episode return: ', episode_return)

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train_reinforce(
        # ***Hyperparameters***

        num_iterations=20000,  # @param {type:"integer"}
        collect_episodes_per_iteration=2,  # @param {type:"integer"}
        replay_buffer_max_length=10000,  # @param {type:"integer"}
        fc_layer_params=(100,),
        learning_rate=1e-3,  # @param {type:"number"}
        log_interval=200,  # @param {type:"integer"}
        num_eval_episodes=10,  # @param {type:"integer"}
        eval_interval=1000  # @param {type:"integer"}
):
    file = open('../output/AVG_returns.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return"])
    # *** Environment***
    # 2 environments, 1 for training and 1 for evaluation
    train_py_env = ClusterEnv()
    eval_py_env = ClusterEnv()

    # converting pyenv to tfenv
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # ***Agent***

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)

    agent.initialize()

    # *** Policies ***

    # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    # *** Replay Buffer ***
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # *** Agent Training ***
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_env, agent.collect_policy, collect_episodes_per_iteration, replay_buffer)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        # print('experience\n')
        # print(experience)
        train_loss = agent.train(experience)
        replay_buffer.clear()

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            avg_return_writer.writerow([step, avg_return])
            returns.append(avg_return)

    # *** Visualizations ***

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    # plt.ylim(top=250)
    plt.show()
