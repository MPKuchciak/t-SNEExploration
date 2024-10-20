# t-SNEExploration
 t-SNE exploration method on MNIST dataset

## Notes on TensorFlow Performance Warning
When running TensorFlow operations, you may encounter the following message: ... oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.

This message is informational and refers to TensorFlow using oneDNN (a performance library) for certain operations. It might lead to slight numerical differences due to floating-point rounding. However, this usually doesn’t affect most results significantly.

- If you are fine with these slight differences, you can ignore this message.
- If you prefer consistency, you can disable oneDNN optimizations by setting the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

### How to Disable oneDNN Optimizations
To disable oneDNN optimizations, you can set the environment variable as follows:

- In R, use this command before running your code:
  Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")