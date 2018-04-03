## Code for [this blog entry](https://mthrok.github.io/ml-blog/deep_learning/2018/04/01/what-dense-layer-is-capable-of/)

Run training with

```
python run_linear_regression_encoder_training.py
```

Then plot results with

```
python plot_linear_regression_encoder.py --input results/linear_regression_encoder_1d/*/outputs.npz --output results/linear_regression_encoder_1d.html
python plot_linear_regression_encoder.py --input results/linear_regression_encoder_2d/*/outputs.npz --output results/linear_regression_encoder_2d.html
```
