from collections import defaultdict

import numpy as np
import bokeh.layouts
from bokeh.palettes import d3
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', required=True, nargs='+')
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def _main():
    args = _parse_args()
    results = []
    for input_file in args.input_files:
        print('Loading:', input_file)
        results.append(_parse_file(input_file))
    legend_keys = _get_different_meta_keys([r[1] for r in results])
    bokeh.io.output_file(args.output_file)
    plot = _plot_loss(results, legend_keys)
    if args.quiet:
        bokeh.plotting.save(plot)
    else:
        bokeh.plotting.show(plot)


def _parse_file(input_file):
    data = np.load(input_file)
    result = {key: val for key, val in data.items()}
    return _preprocess_result(result)


def _to_tuple(val):
    try:
        return tuple(_to_tuple(v) for v in val)
    except Exception:  # pylint: disable=broad-except
        pass
    return val


def _to_hashable(val):
    if not isinstance(val, np.ndarray):
        return val
    if str(val.dtype).startswith('<U'):
        return str(val)
    return _to_tuple(val.tolist())


def _preprocess_result(result):
    data = {
        'n_train': result['n_train'],
        'loss': result['loss'],
    }
    meta = {}
    ignore_keys = {'input_data', 'n_train', 'input_label'}
    for key, val in result.items():
        if key in ignore_keys:
            continue
        try:
            if len(val) != len(result['n_train']):
                meta[key] = _to_hashable(val)
        except Exception:  # pylint: disable=broad-except
            meta[key] = _to_hashable(val)
    return data, meta


def _get_different_meta_keys(metas):
    key_patterns = defaultdict(set)
    ignore_keys = {'input_data', 'n_train', 'input_label', 'v_max', 'v_min'}
    for meta in metas:
        for key, val in meta.items():
            if key not in ignore_keys:
                key_patterns[key].add(val)
    ret = []
    for key, value in key_patterns.items():
        if len(value) > 1:
            ret.append(key)
    return ret


def _format(string):
    return ' '.join([s.capitalize() for s in str(string).split('_')])


def _plot_loss(results, legend_keys):
    plot = figure(
        x_axis_label='#Training',
        y_axis_label='Loss',
        # y_axis_type='log',
        x_axis_type='log',
        y_range=(0, 0.1),
    )

    palettes = d3['Category20'][2*len(results)]
    for i, (data, meta) in enumerate(results):
        components = [
            '%s: %s' % (_format(key), _format(meta[key]))
            for key in legend_keys]
        legend = ' '.join(components)

        src = ColumnDataSource(data=data)
        plot.line(
            x='n_train', y='loss', source=src,
            line_width=1.5,
            line_color=palettes[2*i+1],
            legend=legend,
        )
    plot.legend.location = 'top_right'
    plot.legend.click_policy = 'hide'
    return plot


if __name__ == '__main__':
    _main()
