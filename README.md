# t-SNE Exploration

**Author**: Maciej Kuchciak  
**Date**: January 2024  

[t-SNE Exploration RPubs link](https://rpubs.com/TusVasMit/T-SNEExploration)

This project explores the *t-SNE* (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction technique, applied to the MNIST dataset. t-SNE is a powerful method for visualizing high-dimensional data in a low-dimensional space, preserving local data relationships and revealing clusters.

## Project Structure

```bash
t-SNEExploration/
├── README.md                     # Project overview
├── LICENSE                       # License file
├── .gitignore                    # Files to ignore in version control
├── .gitattributes                # Git attributes
├── t-SNEExploration.Rproj        # R project file
├── src/                          # RMarkdown file and other scripts
│   ├── Dimension_reduction.Rmd   # RMarkdown analysis file
├── results/                      # Outputs generated from analysis
│   ├── Dimension_reduction.md    # Markdown output
│   ├── Dimension_reduction_files/ # Auto-generated figures from RMarkdown
│   └── HTML_version/             # (Optional) HTML outputs
├── docs/                         # GitHub Pages website content
│   └── default.html              # Default layout for site
├── images/                       # All images (plots, figures, etc.)
│   └── mnist_animation_enlarged.gif
├── data/                         # Data files (optional, if applicable)
│   └── mnist_data.csv
```

## Project Overview

The t-SNE algorithm is applied to the MNIST dataset, which contains 60,000 training and 10,000 test grayscale images of handwritten digits. By flattening and normalizing the data, we visualize complex patterns in a 2D space.

This project highlights the use of t-SNE for understanding high-dimensional data structures while discussing important parameters like **perplexity** and the number of iterations used in the optimization process.

## Live Links

- [t-SNE Exploration RPubs](https://rpubs.com/TusVasMit/T-SNEExploration)
- [GitHub Pages link to project](https://your-github-page-link-here)

## Visualization

Below is a animation of the MNIST dataset elements:

![t-SNE Animation](images/mnist_animation_enlarged.gif)

## Dependencies

To replicate this analysis, the following packages and libraries are required:

```r
# Install necessary packages
if (!requireNamespace("keras", quietly = TRUE)) install.packages("keras")
if (!requireNamespace("magick", quietly = TRUE)) install.packages("magick")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("Rtsne", quietly = TRUE)) install.packages("Rtsne")
