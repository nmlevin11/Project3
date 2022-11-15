# Overview:
The purpose of this repo is to automate reports looking at a dataset of features about articles published by Mashable over a two year period. Each report is specific to one of six data channels and contains some summary statistics and plots, model-fitting for a linear regression model and a boosted tree, and a comparison of the predictive abilities of the two models. A list of required packages to run the reports in RStudio, links to the individual reports, and the code for rendering the reports are below.

y<sub>i</sub> = β<sup>0</sup> + β<sup>1</sup>x<sup>i1</sup> + . . . β<sup>p</sup>x<sup>ip</sup>

# Required packages:  
  
1. `tidyverse`
2. `caret`
3. `leaps`
4. `rmarkdown`
5. `knitr`

# Links to Reports:
[Lifestyle report is available here](lifestyle.html).  
[Entertainment report is available here](entertainment.html).  
[Business report is available here](bus.html).    
[Social media report is available here](socmed.html).  
[Technology report is available here](tech.html).   
[World report is available here](world.html).  

# Code for rendering:
channel_list <- c("lifestyle", "entertainment", "bus", "socmed", "tech", "world")   
output_file <- paste0(channel_list, ".md")  
params <- lapply(channel_list, FUN = function(x){list(channel = x)})  
reports <- tibble(output_file, params)  
apply(reports, MARGIN = 1, FUN = function(x){render(input = "Project3.Rmd", output_file = x[[1]], params = x[[2]])})

