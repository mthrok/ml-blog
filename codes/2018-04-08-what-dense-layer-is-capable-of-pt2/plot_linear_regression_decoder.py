from collections import defaultdict

import numpy as np
import bokeh.io
import bokeh.models
import bokeh.layouts
import bokeh.plotting


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', required=True, nargs='+')
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--plot-attr', default='output_data')
    parser.add_argument('--no-ref', dest='plot_ref', action='store_false')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def _main():
    args = _parse_args()
    results = []
    for input_file in args.input_files:
        print('Loading:', input_file)
        results.append(_parse_file(input_file, args.plot_attr))
    legend_keys = _get_different_meta_keys([r[1] for r in results])
    results.sort(key=_sort)
    bokeh.io.output_file(args.output_file)
    ref_plot, plots, sources, n_train = None, [], [], []
    for data, meta in results:
        if args.plot_ref and not ref_plot:
            ref_plot, _ = _plot_1d_result(data, meta, legend_keys, ref=True)
        plot, cds = _plot_1d_result(data, meta, legend_keys)
        plots.append(plot)
        sources.append(cds)
        if len(meta['n_train']) > len(n_train):
            n_train = meta['n_train']
    layout = _construct_plot(ref_plot, plots, sources, n_train)
    if args.quiet:
        bokeh.plotting.save(layout)
    else:
        bokeh.plotting.show(layout)


def _parse_file(input_file, src_key):
    data = np.load(input_file)
    result = {key: val for key, val in data.items()}
    return _preprocess_result_1d(result, src_key=src_key)


def _sort(data):
    data = data[1]
    activation = str(data['activation'])
    act = {
        'None': 0,
        'relu': 20, 'softplus': 23, 'leaky_relu': 26,
        'elu': 30, 'tanh': 40, 'sigmoid': 50, 'gaussian': 60,
    }
    return act.get(activation, 10)


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


def _preprocess_result_1d(result, src_key='output_data'):
    n_trains, data = [], {}
    v_max, v_min = -np.inf, np.inf
    for i, n in enumerate(result['n_train']):
        if n < 50000 and (n and not n & (n - 1)) or n % 50000 == 0:
            n_trains.append(n)
            key = 'pred_%s' % n
            data[key] = [result[src_key][i, ...]]
            v_max = max(v_max, np.amax(result[src_key][i, ...]))
            v_min = max(v_min, np.amax(result[src_key][i, ...]))
            data['pred_last'] = data[key]
    data['ref'] = [result['input_label']]
    data['pred'] = data['pred_0']

    meta = {'n_train': n_trains, 'v_max': v_max, 'v_min': v_min}
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


def _plot_1d_result(data, meta, legend_keys, ref=False):
    plot = bokeh.plotting.figure(
        x_range=(0, 1),
        y_range=(0, 1),
        plot_width=200 if ref else 100,
        plot_height=100,
        tools='',
        toolbar_location=None,
        match_aspect=True)
    plot.outline_line_alpha = 0.0
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None

    color_mapper = bokeh.models.LinearColorMapper(
        palette='Viridis256', low=-1.05, high=1.05)
    cds = bokeh.models.ColumnDataSource(data=data)

    if ref:
        dw = 0.47
    else:
        dw = data['pred'][0].shape[1] / data['pred'][0].shape[0]
    plot.image(
        image='ref' if ref else 'pred', source=cds,
        x=0, y=0, dw=dw, dh=1, color_mapper=color_mapper,
    )
    plot.legend.location = 'top_left'
    plot.legend.click_policy = 'hide'

    if ref:
        color_bar = bokeh.models.ColorBar(
            color_mapper=color_mapper,
            width=9, height=85, margin=0, padding=0,
            border_line_color=None, location=(100, 3))
        plot.add_layout(color_bar)

    if ref:
        text = 'Expected Output'
    else:
        components = [
            '%s: %s' % (_format(key), _format(meta[key]))
            for key in legend_keys]
        text = '<br>'.join(components)
    div = bokeh.models.Div(text=text, width=100, style={'font-size': '70%'})

    return bokeh.layouts.column([plot, div]), cds


def _tile(plots, n_col=None):
    n_plots = len(plots)
    n_col = n_col or int(np.ceil(np.sqrt(n_plots)))

    rows = []
    for i in range(0, n_plots, n_col):
        rows.append(bokeh.layouts.row(plots[i:i+n_col]))
    return bokeh.layouts.column(rows)


def _construct_plot(ref_plot, plots, sources, n_train):
    cb_args = {}
    for i, src in enumerate(sources):
        cb_args['_animation_%s' % i] = src

    meta_values = bokeh.models.ColumnDataSource(
        data={'n_trains': n_train})

    cb_args['meta_values'] = meta_values
    callback = bokeh.models.CustomJS(args=cb_args, code="""
    var args = this.callback.args;
    function _normalize(src) {
       var dst = new Float32Array(src.length);
       var max_val = Math.max.apply(Math, src);
       var min_val = Math.min.apply(Math, src);
       var range = max_val - min_val;
       src.forEach(function(val, index) {
         dst[index] = (val - min_val) / range * 2 - 1;
       });
       return dst
    }
    function _update_plot() {
      var n_train = meta_values.data['n_trains'][slider.value];
      var index_str = n_train.toString();
      var changes = [];
      // Update display 1
      slider.title = "#Trainings: " + n_train;
      changes.push(slider.change);
      // Iterate data source
      for (var key in args) {
        try {
          if (key.startsWith('_animation_')) {
            cdn = args[key];
            var data = cdn.data['pred_' + index_str];
            if (data === undefined) {
                data = cdn.data['pred_last']
            }
            if (normalized.active) {
              data = [_normalize(data[0])];
            }
            cdn.data['pred'] = data
            changes.push(cdn.change);
          }
        } catch (err) {
          console.log('Failed to update' + key);
          throw err;
        }
      }
      changes.map(function(change){change.emit()});
    }
    function _update_plot_periodically() {
      slider.value = (slider.value + slider.step) % (slider.end + 1)
      _update_plot()
    }

    switch (this.name) {
    case 'controll_button':
      if (play_button.active) {
        timer = setInterval(_update_plot_periodically, 300);
        play_button.label = '\u25A0';
      } else {
        clearInterval(timer);
        play_button.label = '\u25B6';
      }
      play_button.change.emit();
      break;
    default:
      _update_plot()
    }
    """)

    slider = bokeh.models.Slider(
        start=0, end=len(n_train)-1, value=0, step=1,
        title="#Trainings: 0", callback=callback,
        show_value=False, tooltips=False, name='slider')
    callback.args['slider'] = slider

    btn = bokeh.models.Toggle(
        label=u'\u25B6', button_type='success', width=50,
        active=False, callback=callback, name='controll_button')
    callback.args['play_button'] = btn

    checkbox = bokeh.models.Toggle(
        label='Normalize', button_type='primary', width=50,
        active=False, callback=callback, name='normalize_bubtton')
    callback.args['normalized'] = checkbox

    rows = [
        _tile(plots, n_col=4),
        slider,
        bokeh.layouts.row([btn, bokeh.layouts.Spacer(), checkbox]),
    ]

    if ref_plot is not None:
        rows = [ref_plot] + rows

    layout = bokeh.layouts.column(rows)
    return layout


if __name__ == '__main__':
    _main()
