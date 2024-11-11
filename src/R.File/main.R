# 1. PURLING Rmd files into R files ============================================
# In this section we can transform the Rmd files into R files. 
# An important part of the transformation is the `documentation` parameter.
# Value `2` will include text in the resulting `R` files as  `roxygen2` comments.
knitr::purl(input    = "src/Dimension_reduction.Rmd",
            output   = "src/R.File/Dimension_reduction.R",
            documentation = 2)