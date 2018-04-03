import os


def get_int_dtype():
    return os.environ.get('TF_INT_DTYPE', 'int32')


def get_float_dtype():
    return os.environ.get('TF_DTYPE', 'float32')


def get_conv_format():
    return os.environ.get('TF_CONV_FORMAT', 'NHWC')
