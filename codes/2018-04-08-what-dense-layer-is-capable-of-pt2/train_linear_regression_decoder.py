import numpy as np
import tensorflow as tf

import tf_util
from generator import MatGenerator


def _parse_args(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Train a dense layer that maps parameter to feature.'
    )
    parser.add_argument('output_prefix')
    parser.add_argument('--feature-shape', type=int, default=[16], nargs='+')
    parser.add_argument('--intermediate-dim', type=int)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--optimizer', default='AdamOptimizer')
    parser.add_argument('--activation')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--loss-type', choices=['l1', 'l2'], default='l2')
    parser.add_argument('--scale-input', action='store_true')
    parser.add_argument('--bias', type=float, nargs='+')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int)
    return parser.parse_args(args=args)


def main(args=None):
    args = _parse_args(args=args)
    tf.set_random_seed(args.seed)
    funcs = _get_funcs(
        feature_shape=args.feature_shape,
        scale_input=args.scale_input,
        intermediate_dim=args.intermediate_dim,
        bias=args.bias, activation=args.activation, loss_type=args.loss_type,
        optimizer=args.optimizer, learning_rate=args.learning_rate,
        save_prefix=args.output_prefix)
    generator = MatGenerator(
        feature_shape=args.feature_shape,
        shuffle_axes=args.shuffle, normalize_label=True, seed=args.seed)
    _run_training(generator, funcs, n_epochs=args.n_epochs)


def _run_training(generator, funcs, n_epochs=10):
    test_batch = generator.get_ref_batch()
    test_input, test_label = test_batch['data'], test_batch['label']
    n_train = 0
    for epoch in range(n_epochs):
        for _ in range(1000):
            batch = generator.get_batch()
            if (n_train & (~n_train + 1)) == n_train:
                funcs['save_output'](test_label, test_input, n_train)
            if n_train % 100 == 0:
                funcs['summarize'](test_label, test_input, n_train)
            funcs['train'](batch['label'], batch['data'])
            n_train += 1
        funcs['save_output'](test_label, test_input, n_train)
        if epoch % 10 == 0:
            funcs['save_model'](n_train)
        test_loss = funcs['loss'](test_label, test_input)
        print(epoch, test_loss)
        if epoch >= 500 and test_loss < 1e-10:
            break
    funcs['save_model'](n_train)
    funcs['summarize'](test_label, test_input, n_train)


def _get_funcs(
        feature_shape, scale_input, intermediate_dim,
        bias, activation, loss_type, optimizer, learning_rate, save_prefix):
    tensors = _get_model(
        feature_shape=feature_shape, scale_input=scale_input,
        intermediate_dim=intermediate_dim, bias=bias,
        activation=activation, loss_type=loss_type)
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

    def _intermediate_func(input_data):
        feed_dict = {tensors['input']: input_data}
        return sess.run(tensors['intermediate'], feed_dict=feed_dict)

    def _get_param_func():
        params = {
            tensor.name.split(':')[0]: tensor
            for tensor in tensors['params']
        }
        return sess.run(params)

    def _save_model_func(n_train):
        saver.save(sess, model_prefix, global_step=n_train)

    n_trains, output_data, losses, intermediates = [], [], [], []
    def _save_output_func(input_data, input_label, n_train):
        n_trains.append(n_train)
        output_data.append(_inference_func(input_data))
        losses.append(_loss_func(input_data, input_label))
        intermediates.append(_intermediate_func(input_data))
        with open(output_file, 'bw') as fileobj:
            np.savez(fileobj, **{
                'feature_shape': feature_shape,
                'scale_input': scale_input,
                'intermediate_dim': intermediate_dim,
                'bias': bias,
                'loss_type': loss_type,
                'activation': activation,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'input_data': input_data,
                'input_label': input_label,
                'n_train': n_trains,
                'output_data': output_data,
                'intermediates': intermediates,
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


def gaussian(x):
    return tf.exp(-tf.square(x))


def _scale(x, name='scale'):
    dtype = tf_util.get_float_dtype()
    shape = x.get_shape().as_list()[1:]
    with tf.variable_scope(name):
        scale = tf.get_variable(name='scale', shape=shape, dtype=dtype)
        center = tf.get_variable(name='center', shape=shape, dtype=dtype)
        ret = tf.multiply(scale, (x - center))
    return ret, [scale, center]


def _get_model(
        feature_shape, scale_input, intermediate_dim,
        bias, activation, loss_type):
    dtype = tf_util.get_float_dtype()
    n_dim = len(feature_shape)
    feature_size = np.prod(feature_shape)

    input_shape = [None, n_dim]
    output_shape = [None, feature_size]

    input_x = tf.placeholder(dtype=dtype, shape=input_shape, name='input')
    label_x = tf.placeholder(dtype=dtype, shape=output_shape, name='label')

    if intermediate_dim is None:
        intermediate_dim = feature_size

    if bias is None:
        use_bias = False
        bias_initializer = None
    else:
        use_bias = True
        bias_initializer = tf.constant_initializer(bias, dtype=dtype)

    if activation is not None:
        activation = getattr(tf.nn, activation)

    variable = input_x
    params = []

    if scale_input:
        variable, scale_params = _scale(variable)
        params.extend(scale_params)

        with tf.name_scope('scaled'):
            tf.summary.histogram('value', variable)

    dense1 = tf.layers.Dense(
        units=intermediate_dim, dtype=dtype, activation=activation,
        use_bias=use_bias, bias_initializer=bias_initializer,
        name='decoder_dense1',
    )
    dense2 = tf.layers.Dense(
        units=feature_size, dtype=dtype,
        use_bias=use_bias, bias_initializer=bias_initializer,
        name='decoder_dense2',
    )
    intermediate = dense1(variable)
    output_x = dense2(intermediate)
    params.extend(dense1.variables)
    params.extend(dense2.variables)

    diff = label_x - output_x
    with tf.variable_scope('loss'):
        diff = tf.abs(diff) if loss_type == 'l1' else tf.square(diff)
        loss = tf.reduce_mean(diff)

    for var in params:
        with tf.name_scope(var.name.split(':')[0]):
            tf.summary.histogram('histogram', var)
    with tf.name_scope('intermediate'):
        tf.summary.histogram('intermediate_value', intermediate)
    with tf.name_scope('output'):
        tf.summary.histogram('output_value', output_x)
    with tf.name_scope('loss'):
        tf.summary.scalar('loss', loss)

    return {
        'input': input_x,
        'label': label_x,
        'output': output_x,
        'intermediate': intermediate,
        'params': params,
        'loss': loss
    }


if __name__ == '__main__':
    main()
