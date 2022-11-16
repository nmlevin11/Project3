Project 3
================
Nicole Levin
11/15/22

# Analysis of entertainment channel

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

|           | num_hrefs | n_tokens_title | kw_avg_avg | average_token_length |  num_imgs | n_non_stop_unique_tokens |   shares |
|:----------|----------:|---------------:|-----------:|---------------------:|----------:|-------------------------:|---------:|
| Mean      |  10.68967 |      11.001984 |   3155.900 |            4.4768152 |  6.317699 |                0.7632158 | 2970.487 |
| Std. Dev. |  12.92069 |       2.087105 |   1099.092 |            0.8093148 | 11.627069 |                7.7311595 | 7858.134 |

Next, we will look at a scatterplot of number of links vs. shares. An
upward trend in this graph would indicate that articles with additional
links tend to be shared more often. A downward trend would indicate that
articles with additional links tend to be shared less often.

``` r
#Create a scatterplot for num_hrefs vs shares
g <- ggplot(data=selected_data, aes(x=num_hrefs, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of links")
```

![](entertainment_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Next, we will look at a scatterplot of number of images vs. shares. An
upward trend in this graph would indicate that articles with more images
tend to be shared more often. A downward trend would indicate that
articles with additional images tend to be shared less often.

``` r
#Plot num_imgs vs shares
g <- ggplot(data=selected_data, aes(x=num_imgs, y=shares))
g + geom_point() + labs(title = "Shares vs. Number of Images")
```

![](entertainment_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

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

![](entertainment_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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

![](entertainment_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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
    ## -22173  -2227  -1302   -252 208499 
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)              -2583.1916   963.3773  -2.681  0.00736 ** 
    ## num_hrefs                   25.0314     9.0575   2.764  0.00574 ** 
    ## n_tokens_title              -7.4592    54.0075  -0.138  0.89016    
    ## num_imgs                    12.8586     9.8651   1.303  0.19249    
    ## average_token_length       148.8146   141.3198   1.053  0.29238    
    ## kw_avg_avg                   1.4745     0.1092  13.508  < 2e-16 ***
    ## n_non_stop_unique_tokens     4.1077    12.1907   0.337  0.73617    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7905 on 4934 degrees of freedom
    ## Multiple R-squared:  0.039,  Adjusted R-squared:  0.03783 
    ## F-statistic: 33.37 on 6 and 4934 DF,  p-value: < 2.2e-16

## Boosted Tree Model

Tree-based methods are another modeling option available. The
methodology for trees is to split the predictor space into regions with
different predictions for each region. For a continuous response, the
prediction for each region is the mean response for the observed values
that fall in that predictor region.

Boosting trees is a way to improve the predictive ability over a single
tree fit. Boosting is slow fitting of trees where trees are grown
sequentially. Each tree is grown on a modified version of the original
data and the predictions update as the trees grow. Boosting typically
improves the predictive performance over a single tree fit.

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
    ## 4941 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (6), scaled (6) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4446, 4446, 4447, 4448, 4447, 4447, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   shrinkage  interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   0.05       1                   25      7832.237  0.015481051  2982.617
    ##   0.05       1                   50      7833.223  0.015312631  2977.283
    ##   0.05       1                  100      7838.655  0.013758981  2976.579
    ##   0.05       1                  150      7842.203  0.014271702  2968.208
    ##   0.05       1                  200      7843.334  0.013958194  2975.536
    ##   0.05       2                   25      7829.442  0.013357688  2975.626
    ##   0.05       2                   50      7848.054  0.011304507  2976.493
    ##   0.05       2                  100      7871.203  0.010282017  2982.927
    ##   0.05       2                  150      7886.822  0.009278116  2985.172
    ##   0.05       2                  200      7897.189  0.009374558  2991.303
    ##   0.05       3                   25      7821.340  0.017106537  2976.237
    ##   0.05       3                   50      7843.252  0.014393170  2976.961
    ##   0.05       3                  100      7878.463  0.013037944  2990.499
    ##   0.05       3                  150      7902.729  0.012414699  2992.601
    ##   0.05       3                  200      7915.353  0.013145288  3000.083
    ##   0.05       4                   25      7827.844  0.014461759  2982.187
    ##   0.05       4                   50      7851.600  0.014119198  2987.740
    ##   0.05       4                  100      7889.280  0.014758648  3003.711
    ##   0.05       4                  150      7907.940  0.015856182  3008.087
    ##   0.05       4                  200      7939.839  0.015265564  3028.890
    ##   0.10       1                   25      7842.100  0.013138015  2981.544
    ##   0.10       1                   50      7840.552  0.013007569  2969.114
    ##   0.10       1                  100      7841.675  0.013414177  2960.037
    ##   0.10       1                  150      7844.365  0.013616106  2966.234
    ##   0.10       1                  200      7849.469  0.013267424  2969.172
    ##   0.10       2                   25      7846.905  0.010964185  2980.324
    ##   0.10       2                   50      7864.451  0.011142759  2975.166
    ##   0.10       2                  100      7883.382  0.011416829  3002.538
    ##   0.10       2                  150      7900.060  0.011109039  3009.651
    ##   0.10       2                  200      7926.898  0.009885285  3018.636
    ##   0.10       3                   25      7868.442  0.009178011  2989.889
    ##   0.10       3                   50      7901.652  0.008271615  2998.318
    ##   0.10       3                  100      7944.078  0.007662474  3017.485
    ##   0.10       3                  150      7981.169  0.007052809  3039.775
    ##   0.10       3                  200      8013.739  0.009347353  3063.621
    ##   0.10       4                   25      7884.320  0.013721199  2986.625
    ##   0.10       4                   50      7932.907  0.011731448  3016.343
    ##   0.10       4                  100      7992.381  0.007821312  3056.329
    ##   0.10       4                  150      8004.630  0.012804180  3082.559
    ##   0.10       4                  200      8048.178  0.012945055  3100.292
    ##   0.20       1                   25      7842.990  0.013632361  2976.809
    ##   0.20       1                   50      7857.875  0.012844911  2980.798
    ##   0.20       1                  100      7864.556  0.013886207  2973.669
    ##   0.20       1                  150      7874.215  0.013159438  2981.813
    ##   0.20       1                  200      7872.318  0.011799057  2989.143
    ##   0.20       2                   25      7873.832  0.008734004  2969.328
    ##   0.20       2                   50      7916.559  0.008011304  2998.719
    ##   0.20       2                  100      8006.111  0.005551160  3040.539
    ##   0.20       2                  150      8025.144  0.009508361  3076.744
    ##   0.20       2                  200      8042.571  0.007427790  3055.872
    ##   0.20       3                   25      7914.476  0.008856671  2998.414
    ##   0.20       3                   50      7971.304  0.008969405  3038.212
    ##   0.20       3                  100      8126.776  0.005454861  3120.113
    ##   0.20       3                  150      8189.287  0.005829074  3186.570
    ##   0.20       3                  200      8253.223  0.006906827  3262.588
    ##   0.20       4                   25      7976.118  0.011716401  3016.842
    ##   0.20       4                   50      8000.514  0.012129342  3059.532
    ##   0.20       4                  100      8147.845  0.009844490  3156.815
    ##   0.20       4                  150      8223.663  0.016733312  3215.365
    ##   0.20       4                  200      8293.165  0.017575744  3263.779
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
| Linear Regression | 7323.071 | 0.0227219 | 2794.578 |
| Boosted Tree      | 7189.175 | 0.0594528 | 2754.328 |

``` r
#Select the better model
if(results_reg[1] < results_boost[1]){winner <- "linear regression"
  } else{winner <- "boosted tree"}
```

Based on resulting RMSE, the better performing model for prediction is
the boosted tree model.

## Citation

Data used to prepare this report is from:

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
Support System for Predicting the Popularity of Online News. Proceedings
of the 17th EPIA 2015 - Portuguese Conference on Artificial
Intelligence, September, Coimbra, Portugal.
