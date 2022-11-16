Project 3
================
Nicole Levin
11/16/22

# Analysis of tech channel

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
| Mean      |  9.416825 |      10.191669 |  2746.2662 |            4.5821243 | 4.434522 |                0.6828719 | 3072.283 |
| Std. Dev. |  8.526926 |       2.111337 |   737.3789 |            0.3503738 | 7.024018 |                0.1106509 | 9024.344 |

Next, we will look at a scatterplot of number of links vs. shares. An
upward trend in this graph would indicate that articles with additional
links tend to be shared more often. A downward trend would indicate that
articles with additional links tend to be shared less often.

``` r
#Create a scatterplot for num_hrefs vs shares
g <- ggplot(data=selected_data, aes(x=num_hrefs, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of links")
```

![](tech_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Next, we will look at a scatterplot of number of images vs. shares. An
upward trend in this graph would indicate that articles with more images
tend to be shared more often. A downward trend would indicate that
articles with additional images tend to be shared less often.

``` r
#Plot num_imgs vs shares
g <- ggplot(data=selected_data, aes(x=num_imgs, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of Images")
```

![](tech_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

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

![](tech_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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

![](tech_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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
    ## -10724  -1893  -1219     28 657506 
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)               3727.8091  2245.7347   1.660 0.096985 .  
    ## num_hrefs                   67.7104    19.3136   3.506 0.000459 ***
    ## n_tokens_title              56.7649    68.5193   0.828 0.407453    
    ## num_imgs                   -64.1690    25.3070  -2.536 0.011254 *  
    ## average_token_length         6.7818   458.7622   0.015 0.988206    
    ## kw_avg_avg                   0.5118     0.1950   2.624 0.008707 ** 
    ## n_non_stop_unique_tokens -4394.4419  1702.4531  -2.581 0.009872 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10290 on 5138 degrees of freedom
    ## Multiple R-squared:  0.006854,   Adjusted R-squared:  0.005694 
    ## F-statistic:  5.91 on 6 and 5138 DF,  p-value: 3.686e-06

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
    ## 5145 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (6), scaled (6) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4631, 4631, 4631, 4631, 4630, 4630, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   shrinkage  interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   0.05       1                   25      7278.888  0.002638338  2418.192
    ##   0.05       1                   50      7303.264  0.002881315  2414.277
    ##   0.05       1                  100      7333.027  0.002818805  2424.543
    ##   0.05       1                  150      7326.492  0.002792610  2419.816
    ##   0.05       1                  200      7379.439  0.002522139  2427.970
    ##   0.05       2                   25      7286.622  0.004218859  2407.812
    ##   0.05       2                   50      7360.875  0.004961333  2411.840
    ##   0.05       2                  100      7370.306  0.005825108  2399.858
    ##   0.05       2                  150      7436.647  0.006079699  2411.465
    ##   0.05       2                  200      7579.677  0.006258246  2421.808
    ##   0.05       3                   25      7238.480  0.005285553  2391.505
    ##   0.05       3                   50      7317.179  0.005049116  2400.793
    ##   0.05       3                  100      7428.219  0.004950402  2407.715
    ##   0.05       3                  150      7537.137  0.006515841  2414.670
    ##   0.05       3                  200      7649.677  0.007558835  2426.637
    ##   0.05       4                   25      7247.972  0.005081490  2392.892
    ##   0.05       4                   50      7312.393  0.005681880  2399.612
    ##   0.05       4                  100      7481.006  0.005416787  2423.925
    ##   0.05       4                  150      7502.744  0.007665747  2416.452
    ##   0.05       4                  200      7672.430  0.006440891  2428.663
    ##   0.10       1                   25      7257.708  0.002340532  2400.347
    ##   0.10       1                   50      7368.665  0.002573808  2436.128
    ##   0.10       1                  100      7398.713  0.002915865  2427.889
    ##   0.10       1                  150      7519.226  0.003076410  2430.232
    ##   0.10       1                  200      7585.106  0.003061874  2424.939
    ##   0.10       2                   25      7361.411  0.004460534  2416.376
    ##   0.10       2                   50      7420.301  0.006688945  2416.376
    ##   0.10       2                  100      7717.111  0.006353504  2433.459
    ##   0.10       2                  150      7897.151  0.007017404  2415.401
    ##   0.10       2                  200      8135.717  0.007764000  2433.238
    ##   0.10       3                   25      7376.423  0.006187411  2402.418
    ##   0.10       3                   50      7471.897  0.006726074  2406.584
    ##   0.10       3                  100      7756.464  0.005683732  2434.596
    ##   0.10       3                  150      7924.386  0.005288605  2440.155
    ##   0.10       3                  200      8150.922  0.006126078  2446.190
    ##   0.10       4                   25      7335.568  0.005722094  2408.329
    ##   0.10       4                   50      7443.723  0.006457230  2420.169
    ##   0.10       4                  100      7492.238  0.006029834  2412.683
    ##   0.10       4                  150      7738.916  0.006744944  2427.416
    ##   0.10       4                  200      8003.021  0.007342625  2459.110
    ##   0.20       1                   25      7399.059  0.003130968  2427.501
    ##   0.20       1                   50      7463.883  0.003709702  2440.465
    ##   0.20       1                  100      7753.798  0.004618445  2456.404
    ##   0.20       1                  150      7990.417  0.006348702  2425.632
    ##   0.20       1                  200      8279.461  0.006733833  2441.812
    ##   0.20       2                   25      7406.115  0.005132147  2412.932
    ##   0.20       2                   50      7521.586  0.006801144  2399.090
    ##   0.20       2                  100      8066.475  0.004925522  2439.385
    ##   0.20       2                  150      8554.335  0.004537801  2465.844
    ##   0.20       2                  200      8843.085  0.004672764  2474.628
    ##   0.20       3                   25      7480.388  0.004799173  2429.965
    ##   0.20       3                   50      7855.618  0.007210680  2434.796
    ##   0.20       3                  100      8303.031  0.005319712  2456.496
    ##   0.20       3                  150      8824.358  0.005252014  2503.921
    ##   0.20       3                  200      9115.640  0.005716632  2510.366
    ##   0.20       4                   25      7700.941  0.005831119  2448.531
    ##   0.20       4                   50      7895.889  0.005913871  2456.899
    ##   0.20       4                  100      8533.844  0.008741351  2510.206
    ##   0.20       4                  150      8971.154  0.006065101  2533.166
    ##   0.20       4                  200      9417.441  0.005264897  2578.287
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 3, shrinkage = 0.05
    ##  and n.minobsinnode = 10.

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
| Linear Regression | 4695.133 | 0.0332256 | 2304.114 |
| Boosted Tree      | 4771.837 | 0.0166797 | 2307.554 |

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
