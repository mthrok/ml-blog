import numpy as np
import tensorflow as tf

import tf_util
from generator import MatGenerator


def _parse_args(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Train a dense layer that maps feature to parameter.'
    )
    parser.add_argument('output_prefix')
    parser.add_argument('--feature-shape', type=int, default=[32], nargs='+')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--optimizer', default='GradientDescentOptimizer')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--loss-type', choices=['l1', 'l2'], default='l2')
    parser.add_argument('--bias', type=float, default=None, nargs='+')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int)
    return parser.parse_args(args=args)


def main(args=None):
    args = _parse_args(args=args)

    tf.set_random_seed(args.seed)
    funcs = _get_funcs(
        args.feature_shape, args.bias, args.shuffle,
        args.loss_type, args.optimizer, args.learning_rate, args.output_prefix)
    generator = MatGenerator(
        args.feature_shape, shuffle_axes=args.shuffle, seed=args.seed)
    _run_training(generator, funcs, n_epochs=args.n_epochs)


def _run_training(generator, funcs, n_epochs=10):
    test_batch = generator.get_ref_batch()
    test_input, test_label = test_batch['data'], test_batch['label']
    n_train = 0
    for epoch in range(n_epochs):
        for _ in range(100):
            batch = generator.get_batch()
            if (n_train & (~n_train + 1)) == n_train:
                test_output = funcs['infer'](test_input)
                test_loss = funcs['loss'](test_input, test_label)
                funcs['save_output'](
                    test_input, test_label, n_train, test_output, test_loss)
            if n_train % 100 == 0:
                funcs['summarize'](test_input, test_label, n_train)
            funcs['train'](batch['data'], batch['label'])
            n_train += 1
        test_output = funcs['infer'](test_input)
        test_loss = funcs['loss'](test_input, test_label)
        funcs['save_output'](
            test_input, test_label, n_train, test_output, test_loss)
        if epoch % 10 == 0:
            funcs['save_model'](n_train)
        print(epoch, test_loss)
    funcs['save_model'](n_train)
    funcs['summarize'](test_input, test_label, n_train)


def _get_funcs(
        feature_shape, bias, shuffle, loss_type,
        optimizer, learning_rate, save_prefix):
    tensors = _get_model(feature_shape, bias, loss_type)
    opt = getattr(tf.train, optimizer)(learning_rate=learning_rate)
    train_op = opt.minimize(tensors['loss'], var_list=tensors['params'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model_prefix = '%s/models/model' % save_prefix
    output_file = '%s/outputs.npz' % save_prefix
    summary_prefix = '%s/summary' % save_prefix

    saver = tf.train.Saver(max_to_keep=None)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summary_prefix, sess.graph)

    def _train_func(input_data, input_label):
        feed_dict = {
            tensors['input']: input_data,
            tensors['label']: input_label,
        }
        return sess.run([tensors['loss'], train_op], feed_dict=feed_dict)[0]

    def _loss_func(input_data, input_label):
        feed_dict = {
            tensors['input']: input_data,
            tensors['label']: input_label,
        }
        return sess.run(tensors['loss'], feed_dict=feed_dict)

    def _inference_func(input_data):
        feed_dict = {tensors['input']: input_data}
        return sess.run(tensors['output'], feed_dict=feed_dict)

    def _get_param_func():
        params = {
            tensor.name.split(':')[0]: tensor
            for tensor in tensors['params']
        }
        return sess.run(params)

    def _save_model_func(n_train):
        saver.save(sess, model_prefix, global_step=n_train)

    n_trains, output_data, losses = [], [], []
    def _save_output_func(
            input_data, input_label, n_train, output_datum, loss):
        n_trains.append(n_train)
        output_data.append(output_datum)
        losses.append(loss)
        with open(output_file, 'bw') as fileobj:
            np.savez(fileobj, **{
                'feature_shape': feature_shape,
                'bias': bias,
                'loss_type': loss_type,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'input_data': input_data,
                'input_label': input_label,
                'n_train': n_trains,
                'output_data': output_data,
                'shuffle': shuffle,
                'loss': losses,
            })

    def _summarize_func(input_data, input_label, n_train):
        feed_dict = {
            tensors['input']: input_data,
            tensors['label']: input_label,
        }
        summary = sess.run(merged, feed_dict=feed_dict)
        writer.add_summary(summary, n_train)

    return {
        'train': _train_func,
        'loss': _loss_func,
        'infer': _inference_func,
        'save_model': _save_model_func,
        'save_output': _save_output_func,
        'summarize': _summarize_func,
        'get_params': _get_param_func,
    }


def _get_model(feature_shape, bias, loss_type):
    dtype = tf_util.get_float_dtype()
    n_dim = len(feature_shape)
    feature_size = np.prod(feature_shape)

    input_shape = [None, feature_size]
    output_shape = [None, n_dim]

    input_x = tf.placeholder(dtype=dtype, shape=input_shape, name='input')
    label_x = tf.placeholder(dtype=dtype, shape=output_shape, name='label')

    if bias is None:
        use_bias = False
        bias_initializer = None
    else:
        use_bias = True
        bias_initializer = tf.constant_initializer(bias, dtype=dtype)

    dense = tf.layers.Dense(
        units=n_dim, activation=None, dtype=dtype,
        use_bias=use_bias, bias_initializer=bias_initializer,
    )
    output_x = dense(input_x)

    diff = label_x - output_x
    with tf.variable_scope('loss'):
        # Note if there is no noise in data, optimization with L1 loss
        # will fluctuate around optima
        diff = tf.abs(diff) if loss_type == 'l1' else tf.square(diff)
        loss = tf.reduce_mean(diff)

    params = list(dense.variables)
    for var in params:
        with tf.name_scope(var.name.split(':')[0]):
            tf.summary.histogram('histogram', var)
    with tf.name_scope('loss'):
        tf.summary.scalar('loss', loss)
    with tf.name_scope('output'):
        tf.summary.histogram('output_value', output_x)

    return {
        'input': input_x,
        'label': label_x,
        'output': output_x,
        'params': params,
        'loss': loss
    }


if __name__ == '__main__':
    main()
