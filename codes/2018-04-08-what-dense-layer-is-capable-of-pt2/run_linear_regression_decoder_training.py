import os.path

import tensorflow as tf

from train_linear_regression_decoder import main


def _main():
    arg_set = _get_args_for_activation_comparison()
    arg_set += _get_args_for_intermediate_size_comparison()
    for args in arg_set:
        print(args)
        tf.reset_default_graph()
        main(args)


def _get_args_for_activation_comparison(learning_rate=0.001, n_epochs=2500):
    base_prefix = 'results/linear_regression_decoder/exp1'
    activations = [
        # 'gaussian',
        'sigmoid', 'tanh', 'elu',
        'softplus', 'leaky_relu', 'relu', 'relu6',
        None,
    ]

    params = {
        'feature_shape': [16],
        'bias': [0],
        'optimizer': 'AdamOptimizer',
        'scale_input': True,
        'loss_type': 'l2',
        'learning_rate': learning_rate,
        'intermediate_dim': None,
    }
    ret = []
    for activation in activations:
        prefix = _get_output_prefix(
            base_prefix, activation=activation, **params)
        if os.path.exists(prefix):
            print('Skipping', prefix)
            continue
        args = _gen_args(
            prefix, activation=activation, n_epochs=n_epochs, **params)
        ret.append(args)
    return ret


def _get_args_for_intermediate_size_comparison(
        learning_rate=0.003, n_epochs=2500):
    base_prefix = 'results/linear_regression_decoder/exp2'
    intermediate_dims = [14, 15, 16, 17]

    params = {
        'feature_shape': [16],
        'bias': [0],
        'activation': 'sigmoid',
        'optimizer': 'AdamOptimizer',
        'scale_input': True,
        'loss_type': 'l2',
        'learning_rate': learning_rate,
    }
    ret = []
    for n_dim in intermediate_dims:
        prefix = _get_output_prefix(
            base_prefix, intermediate_dim=n_dim, **params)
        if os.path.exists(prefix):
            print('Skipping', prefix)
            continue
        args = _gen_args(
            prefix, intermediate_dim=n_dim, n_epochs=n_epochs, **params)
        ret.append(args)
    return ret


def _get_output_prefix(
        base_prefix, *, feature_shape, intermediate_dim, bias, activation,
        loss_type, optimizer, learning_rate, scale_input):
    pattern = (
        'feature_%s_intermediate_dim_%s_bias_%s_'
        'activation_%s_loss_%s_optimizer_%s_lr_%s'
    ) % (
        '_'.join([str(val) for val in feature_shape]),
        'none' if intermediate_dim is None else intermediate_dim,
        'none' if bias is None else '_'.join([str(val) for val in bias]),
        'none' if activation is None else activation,
        loss_type, optimizer, learning_rate
    )
    if scale_input:
        pattern += '_scaled'
    return os.path.join(base_prefix, pattern)


def _gen_args(
        prefix, *, feature_shape, n_epochs, loss_type, bias, activation,
        intermediate_dim, scale_input, optimizer, learning_rate):
    args = [prefix]
    args += ['--feature-shape'] + feature_shape
    args += [
        '--n-epochs', n_epochs,
        '--learning-rate', learning_rate,
        '--loss-type', loss_type,
        '--seed', 123,
        '--optimizer', optimizer,
    ]
    if scale_input:
        args.append('--scale-input')
    if bias is not None:
        args.extend(['--bias'] + bias)
    if intermediate_dim is not None:
        args.extend(['--intermediate-dim', intermediate_dim])
    if activation is not None:
        args.extend(['--activation', activation])
    return [str(val) for val in args]


if __name__ == '__main__':
    _main()
