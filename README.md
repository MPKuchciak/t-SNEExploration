# t-SNE Exploration

**Author**: Maciej Kuchciak  
**Date**: January 2024  

[t-SNE Exploration RPubs link](https://rpubs.com/TusVasMit/T-SNEExploration)

[GitHub Pages link to project](https://mpkuchciak.github.io/tsne-dimensionality-reduction/)

This project explores the *t-SNE* (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction technique, applied to the MNIST dataset. t-SNE is a powerful method for visualizing high-dimensional data in a low-dimensional space, preserving local data relationships and revealing clusters.

## Project Overview

The t-SNE algorithm is applied to the MNIST dataset. This dataset was originally created by [Yann LeCun, Corinna Cortes, and Chris Burges](http://yann.lecun.com/exdb/mnist/), which contains 60,000 training and 10,000 test grayscale images of handwritten digits. These images are 28x28 grayscale, representing the digits 0 to 9. By flattening and normalizing the data, we visualize complex patterns in a 2D space.

This project highlights the use of t-SNE for understanding high-dimensional data structures while discussing important parameters like **perplexity** and the number of iterations used in the optimization process.

## Visualization

Below is an animation of the MNIST dataset elements:

![t-SNE Animation](images/mnist_animation_enlarged.gif)

## Project Structure

```bash
t-SNEExploration/
├── README.md                     # Project overview and instructions
├── LICENSE                       # License file for the project
├── .gitignore                    # List of files/folders ignored in version control
├── .gitattributes                # Git settings for line endings and other attributes
├── t-SNEExploration.Rproj        # R project file for easy setup in RStudio
├── src/                          # Source files for analysis
│   ├── Dimension_reduction.Rmd   # Main RMarkdown file for t-SNE analysis
│   ├── Dimension_reduction.md    # Generated Markdown output (without adjustments for GitHub Pages)
│   ├── Dimension_reduction_files/ # Auto-generated figures and images from RMarkdown
│   ├── HTML_version/             # (Optional) HTML outputs from RMarkdown
│   └── R.File/                   # Folder for R script outputs from purling RMarkdown
│       ├── main.R                # Script to purl Dimension_reduction.Rmd into R code
│       └── Dimension_reduction.R # Auto-generated R code extracted from Dimension_reduction.Rmd
├── docs/                         # GitHub Pages website files
│   ├── index.md                  # Main page for GitHub Pages site (based on Dimension_reduction.md)
│   ├── _config.yml               # Jekyll configuration for GitHub Pages
│   ├── Dimension_reduction_files/ # Auto-generated figures and images for GitHub Pages
│   └── images/                   # Additional images for the website
├── images/                       # Extra project images (plots, figures, etc.)
│   └── mnist_animation_enlarged.gif # Animation illustrating the dataset

```

## Getting Started

To replicate this analysis, you need to install the required R packages and set up your environment:

**Prerequisites:**
- R (>= 4.0)
- RStudio (Optional but recommended)
- TensorFlow backend for Keras

```r
# Install necessary packages
if (!requireNamespace("keras", quietly = TRUE)) install.packages("keras")
if (!requireNamespace("magick", quietly = TRUE)) install.packages("magick")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("Rtsne", quietly = TRUE)) install.packages("Rtsne")
```

```r
# Install TensorFlow
keras::install_keras()
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). You are free to modify, share, and use this project under the terms of the GPL-3.0 license. If you distribute modified versions of this work, you must release them under the same license.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request. 
