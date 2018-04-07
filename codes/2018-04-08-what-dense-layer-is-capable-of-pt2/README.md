## Code for [this blog entry](https://mthrok.github.io/ml-blog/deep_learning/dense/2018/04/08/what-dense-layer-is-capable-of-pt2/)

Run training with

```
python run_linear_regression_decoder_training.py
```

Then plot results with

```
set -x
for exp in 'exp1' 'exp2'; do
    python parameter_learning/plot_linear_regression_loss.py \
	   --input results/linear_regression_decoder/"${exp}"/*/outputs.npz \
	   --output results/linear_regression_decoder/"${exp}"_loss.html \
	   --quiet
    python parameter_learning/plot_linear_regression_decoder.py \
	   --input results/linear_regression_decoder/"${exp}"/*/outputs.npz \
	   --output results/linear_regression_decoder/"${exp}"_output.html \
	   --quiet
    python parameter_learning/plot_linear_regression_decoder.py \
	   --input results/linear_regression_decoder/"${exp}"/*/outputs.npz \
	   --output results/linear_regression_decoder/"${exp}"_intermediate.html \
	   --no-ref --plot-attr intermediates \
	   --quiet
done
```
