import yaml


def alpha_sync(net, tgt_net, alpha):
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = net.state_dict()
    tgt_state = tgt_net.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    tgt_net.load_state_dict(tgt_state)


def load_hyperparams(args):
    # Hyperparameters for the requried environment
    hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

    if args.env not in hypers:
        raise Exception(
            f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/dqn.yaml)')
    return hypers[args.env]
