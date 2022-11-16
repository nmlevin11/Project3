Project 3
================
Nicole Levin
11/16/22

# Analysis of world channel

## Introduction

This report analyzes one data channel of a dataset of features about
articles published by Mashable over a two year period. This report
contains some summary statistics and plots, model-fitting for a linear
regression model and a boosted tree, and a comparison of the predictive
abilities of the two models. There are six data channels in the complete
dataset: lifestyle, entertainment, business, social media, technology,
and world. Results for the other channels can be seen in their
respective reports. The full dataset contains 61 attributes for each
article, but we will focus our attention on shares as the response
variable and the following six predictor variables for summarizing and
modeling.

1.  num_hrefs: Number of links
2.  n_tokens_title: Number of words in the title
3.  kw_avg_avg: Average keyword
4.  average_token_length: Average length of the words in the content
5.  num_imgs: Number of images
6.  n_non_stop_unique_tokens: Rate of unique non-stop words in the
    content

The packages required for creating this report are the following:

1.  `tidyverse`
2.  `caret`
3.  `leaps`
4.  `rmarkdown`
5.  `knitr`

We will start with loading the required packages and reading in the
data.

``` r
#Load packages
library(tidyverse)
library(caret)
library(leaps)
library(rmarkdown)
library(knitr)

#Use a relative path to import data. 
news_data <- read_csv("OnlineNewsPopularity.csv")
```

    ## Rows: 39644 Columns: 61
    ## ── Column specification ───────────────────────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (1): url
    ## dbl (60): timedelta, n_tokens_title, n_tokens_content, n_unique_tokens, n_non_stop_words, n_non...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
#Filter data for just the desired channel.
channel_filter <- paste0("data_channel_is_", params[[1]])
selected_data <- filter(news_data, get(channel_filter) == 1)
selected_data <- selected_data %>% select(num_hrefs, n_tokens_title, kw_avg_avg, average_token_length, num_imgs, n_non_stop_unique_tokens, shares)
```

## Summary Statistics

Before modeling, we’ll look at some basic summary statistics and graphs,
starting with a summary table of means and standard deviations of all of
our variables of interest. These will give us an idea of the center and
spread of the distributions of each of our variables.

``` r
#Calculate means and standard deviations
col_means <- colMeans(selected_data)
col_sds <- apply(selected_data,2,sd)

#Put into a table
data_table <- rbind(t(col_means), t(col_sds))
row.names(data_table) <- c("Mean", "Std. Dev.")
kable(data_table)
```

|           | num_hrefs | n_tokens_title | kw_avg_avg | average_token_length | num_imgs | n_non_stop_unique_tokens |   shares |
|:----------|----------:|---------------:|-----------:|---------------------:|---------:|-------------------------:|---------:|
| Mean      | 10.195206 |      10.599027 |  2524.7406 |            4.6781214 | 2.841225 |                0.6652934 | 2287.734 |
| Std. Dev. |  9.226685 |       2.083929 |   853.3302 |            0.8650829 | 5.217095 |                0.1453257 | 6089.669 |

Next, we will look at a scatterplot of number of links vs. shares. An
upward trend in this graph would indicate that articles with additional
links tend to be shared more often. A downward trend would indicate that
articles with additional links tend to be shared less often.

``` r
#Create a scatterplot for num_hrefs vs shares
g <- ggplot(data=selected_data, aes(x=num_hrefs, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of links")
```

![](world_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Next, we will look at a scatterplot of number of images vs. shares. An
upward trend in this graph would indicate that articles with more images
tend to be shared more often. A downward trend would indicate that
articles with additional images tend to be shared less often.

``` r
#Plot num_imgs vs shares
g <- ggplot(data=selected_data, aes(x=num_imgs, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of Images")
```

![](world_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Next, we will look at a scatterplot of number of words in the title
vs. shares. An upward trend in this graph would indicate that articles
with additional words in the title tend to be shared more often. A
downward trend would indicate that articles with additional words in the
title tend to be shared less often.

``` r
#Plot words in title vs. shares
g <- ggplot(data=selected_data, aes(x=n_tokens_title, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of Words in Title")
```

![](world_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Next, we will look at a scatterplot of average word length vs. shares.
An upward trend in this graph would indicate that articles with a larger
average word length tend to be shared more often. A downward trend would
indicate that articles with a larger average word length tend to be
shared less often.

``` r
#Plot average word length vs. shares
g <- ggplot(data=selected_data, aes(x=average_token_length, y=shares))
g + geom_point() + labs(title = "Shares vs. Average Token Length")
```

![](world_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## Model Preparation

Next, we will prepare for modeling by splitting the data into a training
and test set. We will use the training set to fit two models, a linear
regression and a boosted tree. The test set will be then used to
evaluate the abilities of the models to predict out of sample results
for number of shares.

``` r
#Split data for modeling into train and test sets.
set.seed(371)
train_index <- createDataPartition(selected_data$shares, p=0.7, list=FALSE)
data_train <- selected_data[train_index, ]
data_test <- selected_data[-train_index, ]
```

## Linear Regression Model

The first model we will look at is a linear regression model. The goal
with linear regression is to model the linear relationship between the
predictor variables and the response variable with an equation like the
one below.

y<sub>i</sub> = β<sub>0</sub> + β<sub>1</sub>x<sub>i1</sub> + . . . +
β<sub>p</sub>x<sub>ip</sub>

The best-fit linear model is found by solving for the parameter
estimates (the betas above) that minimize the sum of the squares of the
residuals. The regression equation is then used for prediction of future
values, finding confidence intervals for mean values, etc. Linear
regression is often the simplest modeling option and can be more
interpretable than some of the ensemble methods, but it often loses out
when prediction is the most important goal.

``` r
#Create a linear regression. 
linear_reg <- lm(shares ~ num_hrefs + n_tokens_title + num_imgs + average_token_length + kw_avg_avg + n_non_stop_unique_tokens, data = data_train)
summary(linear_reg)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ num_hrefs + n_tokens_title + num_imgs + 
    ##     average_token_length + kw_avg_avg + n_non_stop_unique_tokens, 
    ##     data = data_train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -8147  -1408   -875   -169 282006 
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)               823.87082  676.90986   1.217  0.22361    
    ## num_hrefs                  27.53937    9.69008   2.842  0.00450 ** 
    ## n_tokens_title            105.39732   37.77980   2.790  0.00529 ** 
    ## num_imgs                   93.76231   17.21786   5.446 5.37e-08 ***
    ## average_token_length     -928.34016  159.96785  -5.803 6.84e-09 ***
    ## kw_avg_avg                  0.49247    0.09325   5.281 1.33e-07 ***
    ## n_non_stop_unique_tokens 4347.24057  979.92054   4.436 9.32e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6043 on 5893 degrees of freedom
    ## Multiple R-squared:  0.01882,    Adjusted R-squared:  0.01783 
    ## F-statistic: 18.84 on 6 and 5893 DF,  p-value: < 2.2e-16

## Boosted Tree Model

Tree-based methods are another modeling option available. The
methodology for trees is to split the predictor space into regions with
different predictions for each region. For a continuous response, the
prediction for each region is the mean response for the observed values
that fall in that predictor region.

Boosting trees is a way to improve the predictive ability over a single
tree fit. Boosting is slow fitting of trees where trees are grown
sequentially. Each tree is grown on a modified version of the original
data and the predictions update as the trees are grown. Boosting
typically improves the predictive performance over a single tree fit.

``` r
#Create a boosted tree fit. 
tuneGrid = expand.grid(n.trees = c(25, 50, 100, 150, 200), interaction.depth = 1:4, shrinkage = c(0.05, 0.1, 0.2), n.minobsinnode = 10)
boosted_tree <- train(shares ~ ., data = data_train, method = "gbm", 
                      preProcess = c("center", "scale"),
                      trControl = trainControl(method = "cv", number = 10), 
                      tuneGrid = tuneGrid, verbose = FALSE)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 5900 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (6), scaled (6) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 5310, 5309, 5310, 5311, 5310, 5311, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   shrinkage  interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   0.05       1                   25      5487.794  0.020263660  1887.398
    ##   0.05       1                   50      5474.269  0.024633673  1874.554
    ##   0.05       1                  100      5469.875  0.027305213  1864.711
    ##   0.05       1                  150      5474.658  0.027776451  1863.891
    ##   0.05       1                  200      5474.525  0.028333912  1865.802
    ##   0.05       2                   25      5497.455  0.019165747  1885.187
    ##   0.05       2                   50      5519.115  0.019024812  1883.403
    ##   0.05       2                  100      5577.465  0.017816219  1898.430
    ##   0.05       2                  150      5597.475  0.018927747  1894.818
    ##   0.05       2                  200      5626.015  0.019357963  1913.670
    ##   0.05       3                   25      5500.172  0.020493512  1889.119
    ##   0.05       3                   50      5531.386  0.019772646  1887.338
    ##   0.05       3                  100      5600.423  0.017868440  1907.707
    ##   0.05       3                  150      5629.630  0.018489537  1914.605
    ##   0.05       3                  200      5641.929  0.018538577  1922.215
    ##   0.05       4                   25      5505.028  0.017538806  1888.237
    ##   0.05       4                   50      5549.101  0.018716491  1902.481
    ##   0.05       4                  100      5591.647  0.020111037  1904.401
    ##   0.05       4                  150      5636.507  0.017007177  1927.005
    ##   0.05       4                  200      5669.311  0.016255725  1939.831
    ##   0.10       1                   25      5478.148  0.023198051  1872.040
    ##   0.10       1                   50      5476.890  0.025213468  1869.290
    ##   0.10       1                  100      5472.756  0.027808775  1867.804
    ##   0.10       1                  150      5476.443  0.028333069  1861.953
    ##   0.10       1                  200      5480.024  0.028408127  1864.062
    ##   0.10       2                   25      5541.012  0.012881824  1886.147
    ##   0.10       2                   50      5593.392  0.014671016  1902.645
    ##   0.10       2                  100      5629.767  0.016676591  1914.734
    ##   0.10       2                  150      5682.012  0.015318284  1935.891
    ##   0.10       2                  200      5726.555  0.012858100  1952.208
    ##   0.10       3                   25      5553.449  0.013840789  1896.971
    ##   0.10       3                   50      5603.407  0.013879885  1905.604
    ##   0.10       3                  100      5673.184  0.012609249  1936.739
    ##   0.10       3                  150      5720.297  0.012533123  1967.848
    ##   0.10       3                  200      5781.387  0.008885488  1994.377
    ##   0.10       4                   25      5594.198  0.010656226  1914.708
    ##   0.10       4                   50      5645.190  0.011390841  1924.075
    ##   0.10       4                  100      5684.162  0.015546160  1966.907
    ##   0.10       4                  150      5727.379  0.015109448  1992.525
    ##   0.10       4                  200      5786.919  0.015104195  2031.936
    ##   0.20       1                   25      5479.804  0.024796933  1870.807
    ##   0.20       1                   50      5478.625  0.026714492  1862.511
    ##   0.20       1                  100      5484.377  0.027238549  1867.737
    ##   0.20       1                  150      5495.786  0.026286595  1881.252
    ##   0.20       1                  200      5493.365  0.027431544  1878.624
    ##   0.20       2                   25      5644.661  0.011889264  1916.645
    ##   0.20       2                   50      5704.028  0.011221278  1946.613
    ##   0.20       2                  100      5756.189  0.020596188  1988.858
    ##   0.20       2                  150      5844.609  0.017135428  2027.690
    ##   0.20       2                  200      5888.850  0.013228941  2055.272
    ##   0.20       3                   25      5686.527  0.009962779  1937.780
    ##   0.20       3                   50      5728.430  0.011467852  1947.176
    ##   0.20       3                  100      5826.499  0.007290822  2008.522
    ##   0.20       3                  150      5953.097  0.006694593  2080.193
    ##   0.20       3                  200      5965.583  0.008038759  2117.421
    ##   0.20       4                   25      5647.525  0.026553723  1936.721
    ##   0.20       4                   50      5703.264  0.020515655  1966.889
    ##   0.20       4                  100      5835.134  0.017690362  2052.943
    ##   0.20       4                  150      5919.192  0.012781112  2121.073
    ##   0.20       4                  200      5954.950  0.015068505  2159.986
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 1, shrinkage =
    ##  0.05 and n.minobsinnode = 10.

## Model Comparison

Now the two models will be compared based on their ability to predict
out of sample results for number of shares. The model with the lower
RMSE will be selected as the better model.

``` r
#Make predictions using the test data
pred_reg <- predict(linear_reg, newdata = data_test)
pred_boost <- predict(boosted_tree, newdata = data_test)
results_reg <- postResample(pred_reg, obs = data_test$shares)
results_boost <- postResample(pred_boost, obs = data_test$shares)

#Create table of results
results_table <- rbind(t(results_reg), t(results_boost))
row.names(results_table) <- c("Linear Regression", "Boosted Tree")
kable(results_table)
```

|                   |     RMSE |  Rsquared |      MAE |
|:------------------|---------:|----------:|---------:|
| Linear Regression | 6002.183 | 0.0225297 | 1945.706 |
| Boosted Tree      | 6007.033 | 0.0209576 | 1919.868 |

``` r
#Select the better model
if(results_reg[1] < results_boost[1]){winner <- "linear regression"
  } else{winner <- "boosted tree"}
```

Based on resulting RMSE, the better performing model for prediction is
the linear regression model.

## Citation

Data used to prepare this report is from:

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
Support System for Predicting the Popularity of Online News. Proceedings
of the 17th EPIA 2015 - Portuguese Conference on Artificial
Intelligence, September, Coimbra, Portugal.
