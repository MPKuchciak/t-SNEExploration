# t-SNE Exploration
**Author**: Maciej Kuchciak  
**Date**: January 2024

[t-SNE Exploration RPubs link](https://rpubs.com/TusVasMit/T-SNEExploration)

This project explores the t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction technique, applied to the MNIST dataset. t-SNE is a powerful method for visualizing high-dimensional data in a low-dimensional space, preserving local data relationships and revealing clusters.

## Project Overview
The t-SNE algorithm is applied to the MNIST dataset, which contains 60,000 training and 10,000 test grayscale images of handwritten digits. By flattening and normalizing the data, we visualize complex patterns in a 2D space.

This project highlights the use of t-SNE for understanding high-dimensional data structures, while also discussing important parameters like perplexity and the number of iterations used in the optimization process.

## Notes on TensorFlow Performance Warning
When running TensorFlow operations, you may encounter the following message: 

> ... oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

This message indicates that TensorFlow is using oneDNN for certain operations. If you’re okay with slight numerical differences, you can ignore the message. Otherwise, you can disable these optimizations by setting the environment variable.

### How to Disable oneDNN Optimizations
To disable oneDNN optimizations, set the environment variable as follows:

In R, before running your code, use this command:
  Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")
