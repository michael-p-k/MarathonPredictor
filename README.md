# US Marathon Race Factors and Adjusting Your Pacing Accordingly

This is a data science project I created to scrape marathon times, weather data, and elevation data in order to predict marathon times. Data for around 400 marathon instances represented the aggregation of around 20 races from year 2000 to 2022. Linear and tree-based regressors are used to model the average run times, female win times, and male win times. Ridge model coefficients for average run times are extracted to help the typical runner adjust their pacing plan to account for their race day conditions and run a smart race.

If you are interested in my code for scraping data and compiling my data set, please take a look at the MarathonDataCollection.ipynb jupyter notebook.

The data visualization and modeling is contained in the MarathonPredictor.ipynb jupyter notebook. You will also need my most recent data table csv from the full_df_csvs folder. The units in this data set are listed in the appendix.

__Objective 1:__ Are variations in average marathon times connected to weather conditions and course elevation changes?

__Objective 2:__ Are marathon win times similarly connected to weather conditions and elevation changes?

__Objective 3:__ How can an average runner adjust their pacing plan to account for race day conditions and run a smart marathon?

### Exploratory Data Analysis

![Image](figures/Ave_by_event.png)
![Image](figures/General_plots.png)
![Image](figures/Weather_plots.png)
![Image](figures/Elevation_plots.png)

### Summary of Objective 1 Findings

The best model tested here was the random forest regressor with an R2 of 0.94 for the test data split, indicating the vast majority of the variation in average marathon run times could be predicted based on the input variables, including weather, elevation, and location. 5-fold cross-validation on the training data split showed an R2 st. dev. <0.02, which suggests this high model performance holds steady on previously unseen data. The gradient boosting tree ensemble showed very similar performance.

Linear models (including ridge regression and lasso regression) each had a test split R2 of 0.76, so these methods were less capable of predicting the variation in average run times. Linear models likely have lower predictive performance for this data set because some of the important input variables are not expected to have linear effects on race times. For example, each race location (described by latitude and longitude) tends to attract a different population of marathon runners due to factors like race reputation or proximity to home. Tree-based methods are better able to learn and account for these non-monotonic effects.

#### Random Forest Predictions
![Image](figures/Ave_Tree_Models.png)

### Summary of Objective 2 Findings

In general, female win times and male win times show less variance than average run times for the marathon data analyzed here. Weather and course conditions were able to predict a smaller proportion of variance for win times than for average run times.

Ridge regressors (chosen here as a representative linear model) for male and female win times had R2 scores about 0.2 lower than the ridge model for average run times. One possible interpretation is that elite marathoners are better than average runners at adapting to weather and to other factors with fairly linear effects. Win times are probably heavily influenced by factors such as the prestige of the marathon, which is difficult to quantify and account for in linear models. These non-linear factors can be learned and predicted by tree-based models, so random forest models had R2 scores about 0.27 higher than the associated ridge models for female and male win times.

Random forest regressors (the best model type tested here) for male and female win times had R2 scores about 0.15 lower than the random forest for average run times. This indicates variables not present in my data set account for a larger portion of variance in win times. These might include scheduling conflicts with competing races (Olympic Games, Olympic Trials, prestigeous marathons) and prize money.

#### Random Forest Predictions
![Image](figures/Random_Forests_plots.png)

#### Model Performance

|   Model | Best Params  |   Train R2 |   Test R2 |   CV R2 mean |   CV R2 stdev |   Train RMSE |   Test RMSE |   Train Split stdev |   Test Split stdev |
|----------:|:-----------|:-----------:|:----------:|:--------:|:-----------:|:----------:|:------------:|:-------------:|:------------:|
| Ridge Ave                | [{'alpha': 0.1}]                                              |   0.770101 |  0.760004 |     0.720597 |     0.0383809 |     14.1206  |    16.0569  |            29.4499  |           32.7762  |
| Lasso Ave                | [{'alpha': 1}]                                                |   0.769705 |  0.760624 |     0.723293 |     0.0386154 |     14.1327  |    16.0361  |            29.4499  |           32.7762  |
| Linear Ave               | N/A                                                           |   0.772337 |  0.761092 |     0.696343 |     0.0487966 |     14.0517  |    16.0204  |            29.4499  |           32.7762  |
| Gradient Boosting Ave    | [{'learning_rate': 0.2, 'max_depth': 1, 'n_estimators': 250}] |   0.976153 |  0.93377  |     0.936253 |     0.018465  |      4.54782 |     8.43501 |            29.4499  |           32.7762  |
| Random Forest Ave        | [{'max_features': 0.6, 'max_samples': 1.0}]                   |   0.993026 |  __0.949048__ |     0.941806 |     0.016268  |      2.45943 |     7.3984  |            29.4499  |           32.7762  |
| Ridge Female Win         | [{'alpha': 0.01}]                                             |   0.48608  |  0.545343 |     0.432089 |     0.074542  |      8.77589 |     8.43108 |            12.2417  |           12.5038  |
| Random Forest Female Win | [{'max_features': 0.3, 'max_samples': 1.0}]                   |   0.972522 |  __0.822653__ |     0.795746 |     0.0351898 |      2.02924 |     5.26566 |            12.2417  |           12.5038  |
| Ridge Male Win           | [{'alpha': 0.01}]                                             |   0.508736 |  0.605073 |     0.440895 |     0.0518985 |      5.74884 |     5.20146 |             8.20205 |            8.27689 |
| Random Forest Male Win   | [{'max_features': 0.2, 'max_samples': 1.0}]                   |   0.974109 |  __0.86465__  |     0.790649 |     0.0434219 |      1.31976 |     3.04506 |             8.20205 |            8.27689 |


### Summary of Objective 3 Findings

Many marathon runners have faced challenging race conditions, including hot temperatures, precipitation, and hilly courses. Pacing goals should be adjusted based on these factors to ensure runner start their races at a sustainable speed and are less likely to resign to walking the last several miles. The modeling of this data set can offer some rough guidance in adjusting pacing plans for an average marathoner.

While linear models did not perform as well as tree-based methods for this data set, they are still able to account for the majority of run time variance and offer interpretable results via the model coefficients. Prior to fitting the linear models, input variables were each normalized using a standard scaler. After fitting, each coefficient was transformed back into original units based on the respective variable's st. dev. Unscaled coefficients for the ridge model are listed for some variables that runners might want to account for.

#### Highlights on how to adjust goal race time:
(These are are in relation to what you consider "normal" training conditions.)

| Variable | Units | Ave Effect | Female Win Effect | Male Win Effect |
|------:|:------:|:------:|:------:|:------:|
| Temperature | sec/deg F | +52 | +26 | +30 |
| Relative Humidity | sec/% | -3 | +17 | +13 |
| Precipitation | sec/inch | +263 | -26 | +78 |
| Wind | sec/mph | -49 | -6 | 0 |
| Elevation | sec/foot | +0.5 | 0.0 | 0.0 |
| Cumulative Elev Loss | sec/foot | -3.6 | -2.4 | -1.9 |
| Cumulative Elev Gain | sec/foot | +3.1 | +2.6 | +1.9 |

These numbers are only to give a rough idea on how to quantify the added difficulty of race conditions while controlling for other variables. Experienced runners know flat courses and easier/faster than hilly courses with zero net elevation change, but the coefficients for the model examined here suggest the hilly course would be about the same. Data from a larger variety of race courses than the 20 examined here would help to improve these coefficient estimates. Runners likely will still find these quantified adjustments valuable if they choose to travel to unfamiliar territory for a marathon with a notably different climate and elevation profile than they are used to.

## Appendix

### Data Units

__Marathon Data:__
(scraped from http://www.marathonguide.com/index.cfm)
- Date - YYYY-MM-DD
- Finishers - count of total finishers
- Males - count of male finishers
- Females - count of female finishers
- Male Win - time in HH:MM:SS
- Female Win - time in HH:MM:SS
- Average Time - HH:MM:SS
- Time STD - HH:MM:SS
- Percent Female - percent of finishers

__Elevation Data:__
(scraped from https://findmymarathon.com/)
- Elev Gain - cumulative elevation gain in feet
- Elev Loss - cumulative elevation loss in feet
- Max Elev - in feet
- Min Elev - in feet

__Weather Data Types:__
(queried from https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt)
- STATION - Station number (WMO/DATSAV3 possibly combined w/WBAN number) 
- LATITUDE - Given in decimated degrees (Southern Hemisphere values are negative)
- LONGITUDE - Given in decimated degrees (Western Hemisphere values are negative)
- ELEVATION - Given in meters
- TEMP - Mean temperature (Fahrenheit)
- DEWP - Mean dew point (Fahrenheit)
- SLP - Mean sea level pressure (mb)
- STP - Mean station pressure (mb)
- VISIB - Mean visibility (miles)
- WDSP – Mean wind speed (knots)
- MXSPD - Maximum sustained wind speed (knots)
- GUST - Maximum wind gust (knots)
- MAX - Maximum temperature (Fahrenheit)
- MIN - Minimum temperature (Fahrenheit)
- PRCP - Precipitation amount (inches)

All 9's in a field (e.g., 99.99 for PRCP) indicates no report or insufficient data.

### Additional Exploratory Data Analysis

![Image](figures/Female_by_event.png)
![Image](figures/Male_by_event.png)

### Win Time Models Plotted Individually

![Image](figures/Wins_Random_Forest.png)

