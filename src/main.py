import utilities
import constants
import cluster
import REINFORCE_tfagent
import DQN_tfagent
import workload


def main():
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()

    if constants.algo == 'reinforce':
        print("Running Reinforce Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(constants.iteration, constants.workload, constants.beta))
        REINFORCE_tfagent.train_reinforce(num_iterations=constants.iteration)
    elif constants.algo == 'dqn':
        DQN_tfagent.train_dqn(num_iterations=constants.iteration)
    else:
        print('Please specify valid algo option in config.ini file\n')


if __name__ == '__main__':
    main()
