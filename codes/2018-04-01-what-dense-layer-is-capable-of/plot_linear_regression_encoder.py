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
    return parser.parse_args()


def _main():
    args = _parse_args()
    results = [dict(np.load(input_file)) for input_file in args.input_files]
    print([res['bias'].dtype for res in results])
    bokeh.io.output_file(args.output_file)
    n_dim = len(results[0]['feature_shape'])
    if n_dim == 1:
        results = [_preprocess_result_1d(res) for res in results]
        plot, sources = _plot_results_1d(results)
        layout = _construct_layout_1d(plot, sources, results[0][1])
    elif n_dim == 2:
        results = [_preprocess_result_2d(res) for res in results]
        plot, sources = _plot_results_2d(results)
        layout = _construct_layout_2d(plot, sources, results[0][1])
    bokeh.plotting.show(layout)


def _preprocess_result_1d(result):
    n_trains, data = [], {}
    for i, n in enumerate(result['n_train']):
        if n > 2500:
            break
        if (
                n == 0 or
                (n < 1000 and (n and not n & (n - 1))) or
                (n >= 1000 and n % 100 == 0)
        ):
            n_trains.append(n)
            data['pred_%s' % n] = result['output_data'][i, :, 0]
    data['ref'] = result['input_label'][:, 0]
    data['pred'] = data['pred_0']
    meta = {
        'n_train': n_trains,
        'bias': None if result['bias'] == 'none' else result['bias'],
        'loss_type': result['loss_type'],
        'optimizer': str(result['optimizer']).replace('Optimizer', ''),
        'shuffle': result.get('shuffle', False),
    }
    return data, meta


def _plot_results_1d(results):
    x_range, y_range = _get_range_1d([res[0] for res in results])
    plot = figure(
        x_axis_label='Expected value',
        y_axis_label='Predicted value',
        x_range=x_range, y_range=y_range)

    palettes = d3['Category20'][2*len(results)]
    sources = []
    for i, (data, meta) in enumerate(results):
        legend = 'Loss: %s, Bias: %s' % (meta['loss_type'], meta['bias'])
        if meta['shuffle']:
            legend += ', Shuffle'
        src = ColumnDataSource(data=data)
        sources.append(src)
        plot.scatter(
            x='ref', y='pred', source=src,
            fill_color=palettes[2*i],
            line_color=palettes[2*i+1],
            marker='circle', size=12, alpha=0.3,
            legend=legend,
        )
        if i == len(results) - 1:
            plot.scatter(
                x='ref', y='ref', source=src,
                fill_color=None,
                line_color='black',
                marker='circle', size=12, alpha=1.0,
            )
    plot.legend.location = 'top_left'
    plot.legend.click_policy = 'hide'
    return plot, sources


def _construct_layout_1d(plot, sources, meta):
    from bokeh.models import CustomJS, Toggle, Slider, CheckboxGroup, WidgetBox

    cb_args = {}
    for i, src in enumerate(sources):
        cb_args['_animation_%s' % i] = src

    meta_values = bokeh.models.ColumnDataSource(
        data={'n_trains': meta['n_train']})

    cb_args['meta_values'] = meta_values
    callback = CustomJS(args=cb_args, code="""
    var args = this.callback.args;
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
            cdn.data['pred'] = cdn.data['pred_' + index_str];
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
        timer = setInterval(_update_plot_periodically, 400);
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

    slider = Slider(
        start=0, end=len(meta['n_train'])-1, value=0, step=1,
        title="#Trainings: 0", callback=callback,
        show_value=False, tooltips=False, name='slider')
    callback.args['slider'] = slider

    btn = Toggle(
        label=u'\u25B6', button_type='success', width=50,
        active=False, callback=callback, name='controll_button')
    callback.args['play_button'] = btn

    from bokeh.layouts import column, row, Spacer
    layout = column(
        plot,
        row(slider, btn),
    )
    return layout


def _extend(v_min, v_max, ratio=0.8):
    real_range = v_max - v_min
    extend = real_range * (1.0 - ratio) / 2
    v_min -= extend
    v_max += extend
    return v_min, v_max


def _get_range_1d(results):
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for res in results:
        x_min = min(x_min, res['ref'].min())
        x_max = max(x_max, res['ref'].max())

        for key, value in res.items():
            if key.startswith('pred_'):
                y_min = min(y_min, value.min())
                y_max = max(y_max, value.max())

    x_min, x_max = _extend(x_min, x_max, ratio=0.8)
    y_min, y_max = _extend(y_min, y_max, ratio=0.8)
    return (x_min, x_max), (y_min, y_max)


def _preprocess_result_2d(result):
    n_trains, data = [], {}
    for i, n in enumerate(result['n_train']):
        if n > 2500:
            break
        if (
                n == 0 or
                (n < 1000 and (n and not n & (n - 1))) or
                (n >= 1000 and n % 100 == 0)
        ):
            n_trains.append(n)
            data['pred_x_%s' % n] = result['output_data'][i, :, 0]
            data['pred_y_%s' % n] = result['output_data'][i, :, 1]
    data['ref_x'] = result['input_label'][:, 0]
    data['ref_y'] = result['input_label'][:, 1]
    data['pred_x'] = data['pred_x_0']
    data['pred_y'] = data['pred_y_0']
    meta = {
        'n_train': n_trains,
        'bias': result['bias'],
        'loss_type': result['loss_type'],
        'optimizer': str(result['optimizer']).replace('Optimizer', ''),
        'shuffle': result.get('shuffle', False),
    }
    return data, meta


def _get_range_2d(results):
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for res in results:
        for key, value in res.items():
            if key.startswith('ref_x'):
                x_min = min(x_min, value.min())
                x_max = max(x_max, value.max())
            elif key.startswith('ref_y'):
                y_min = min(y_min, value.min())
                y_max = max(y_max, value.max())
    x_min, x_max = _extend(x_min, x_max, ratio=0.35)
    y_min, y_max = _extend(y_min, y_max, ratio=0.35)
    return (x_min, x_max), (y_min, y_max)


def _plot_results_2d(results):
    x_range, y_range = _get_range_2d([res[0] for res in results])
    plot = figure(
        x_axis_label='Feature dimension 1',
        y_axis_label='Feature dimension 2',
        x_range=x_range, y_range=y_range)
    palettes = d3['Category20'][2*len(results)]

    sources = []
    for i, (data, meta) in enumerate(results):
        legend = 'Loss: %s, Bias: %s' % (meta['loss_type'], meta['bias'])
        if meta['shuffle']:
            legend += ', Shuffle'
        src = ColumnDataSource(data=data)
        sources.append(src)
        plot.scatter(
            x='pred_x', y='pred_y', source=src,
            fill_color=palettes[2*i],
            line_color=palettes[2*i+1],
            marker='circle', size=8, alpha=0.3,
            legend=legend,
        )
        if i == len(results) - 1:
            plot.scatter(
                x='ref_x', y='ref_y', source=src,
                fill_color=None,
                line_color='black',
                marker='circle', size=8, alpha=1.0,
            )
    plot.legend.location = 'top_left'
    plot.legend.click_policy = 'hide'
    return plot, sources


def _construct_layout_2d(plot, sources, meta):
    from bokeh.models import CustomJS, Toggle, Slider, CheckboxGroup, WidgetBox

    cb_args = {}
    for i, src in enumerate(sources):
        cb_args['_animation_%s' % i] = src

    meta_values = ColumnDataSource(
        data={'n_trains': meta['n_train']})

    cb_args['meta_values'] = meta_values
    callback = CustomJS(args=cb_args, code="""
    var args = this.callback.args;
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
            cdn.data['pred_x'] = cdn.data['pred_x_' + index_str];
            cdn.data['pred_y'] = cdn.data['pred_y_' + index_str];
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
        timer = setInterval(_update_plot_periodically, 400);
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

    slider = Slider(
        start=0, end=len(meta['n_train'])-1, value=0, step=1,
        title="#Trainings: 0", callback=callback,
        show_value=False, tooltips=False, name='slider')
    callback.args['slider'] = slider

    btn = Toggle(
        label=u'\u25B6', button_type='success', width=50,
        active=False, callback=callback, name='controll_button')
    callback.args['play_button'] = btn

    from bokeh.layouts import column, row, Spacer
    layout = column(
        plot,
        row(slider, btn),
    )
    return layout


if __name__ == '__main__':
    _main()
