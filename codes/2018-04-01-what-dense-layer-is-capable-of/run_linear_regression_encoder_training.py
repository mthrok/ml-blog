import os.path

import tensorflow as tf

from train_linear_regression_encoder import main


def _main():
    _run_training(input_shape=[3, 5])
    _run_training(input_shape=[13])


def _run_training(input_shape):
    arg_set = _gen_arg_set(input_shape)
    for args in arg_set:
        print(args)
        tf.reset_default_graph()
        main(args)


def _gen_arg_set(feature_shape, n_epochs=25):
    non_zero_bias = [(val - 1) // 2 for val in feature_shape]
    ret = []
    for loss_type in ['l2', 'l1']:
        for bias in [non_zero_bias, 0, None]:
            args = _gen_args(feature_shape, n_epochs, loss_type, bias)
            if args is not None:
                ret.append(args)
        args = _gen_args(
            feature_shape, n_epochs, loss_type, non_zero_bias, True)
        if args is not None:
            ret.append(args)
    return ret


def _gen_args(
        feature_shape, n_epochs, loss_type, bias,
        shuffle=False, optimizer='GradientDescentOptimizer',
        learning_rate=0.1):
    if bias == 0:
        bias = [bias]
    prefix = _get_output_prefix(feature_shape, bias, loss_type, shuffle)
    if os.path.exists(prefix):
        return None
    args = [prefix]
    args += ['--feature-shape'] + list(feature_shape)
    args += [
        '--n-epochs', n_epochs,
        '--learning-rate', learning_rate,
        '--loss-type', loss_type,
        '--seed', 123,
        '--optimizer', optimizer
    ]
    if bias is None:
        pass
    else:
        args.extend(['--bias'] + bias)
    if shuffle:
        args.append('--shuffle')
    return [str(val) for val in args]


def _get_output_prefix(feature_shape, bias, loss_type, shuffle):
    n_dim = len(feature_shape)
    pattern = 'feature_%s_bias_%s_loss_%s' % (
        '_'.join([str(val) for val in feature_shape]),
        'none' if bias is None else '_'.join([str(val) for val in bias]),
        loss_type,
    )
    if shuffle:
        pattern += '_shuffle'
    return 'results/linear_regression_encoder_%sd/%s' % (n_dim, pattern)


if __name__ == '__main__':
    _main()
