import configparser


def load_config():
    import constants
    config = configparser.ConfigParser()
    # Load the configuration file
    config.read(constants.root+'/settings/config.ini')
    # Load configs
    for section in config.sections():
        for options in config.options(section):
            if options == 'root':
                constants.root = config.get(section, options)
            elif options == 'algo':
                constants.algo = config.get(section, options)
            elif options == 'workload':
                constants.workload = config.get(section, options)
            elif options == 'beta':
                constants.beta = float(config.get(section, options))
            elif options == 'iteration':
                constants.iteration = int(config.get(section, options))
            elif options == 'fixed_episodic_reward':
                constants.fixed_episodic_reward = int(config.get(section, options))
            elif options == 'epsilon':
                constants.epsilon = float(config.get(section, options))
            elif options == 'learning_rate':
                constants.learning_rate = float(config.get(section, options))
            elif options == 'gamma':
                constants.gamma = float(config.get(section, options))
            elif options == 'placement_penalty':
                constants.placement_penalty = int(config.get(section, options))
            elif options == 'pp_apply':
                constants.pp_apply = config.get(section, options)
            else:
                print('Invalid Option found {}'.format(options))
