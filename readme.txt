This is a data science project I created to scrape marathon times, weather data, and elevation data in order to predict marathon times. Data for around 400 marathon instances represented the aggregation of around 20 races from year 2000 to 2022. Linear and tree-based regressors are used to model the average run times, female win times, and male win times. Ridge model coefficients for average run times are extracted to help the typical runner adjust their pacing plan to account for their race day conditions and run a smart race.

If you are interested in my code for scraping data and compiling my data set, please take a look at the MarathonDataCollection.ipynb jupyter notebook.

The data visualization and predictive modeling is contained in the MarathonPredictor.ipynb jupyter notebook. You will also need my most recent data table csv from the full_df_csvs folder. The units in this data set are as follows.

Marathon Data:
Date - YYYY-MM-DD
Finishers - count of total finishers
Males - count of male finishers
Females - count of female finishers
Male Win - time in HH:MM:SS
Female Win - time in HH:MM:SS
Average Time - HH:MM:SS
Time STD - HH:MM:SS
Percent Female - percent of finishers

Elevation Data:
(https://findmymarathon.com/)
Elev Gain - cumulative elevation gain in feet
Elev Loss - cumulative elevation loss in feet
Max Elev - in feet
Min Elev - in feet

Weather Data Types:
(https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt)
STATION - Station number (WMO/DATSAV3 possibly combined w/WBAN number) 
LATITUDE - Given in decimated degrees (Southern Hemisphere values are negative)
LONGITUDE - Given in decimated degrees (Western Hemisphere values are negative)
ELEVATION - Given in meters
TEMP - Mean temperature (Fahrenheit)
DEWP - Mean dew point (Fahrenheit)
SLP - Mean sea level pressure (mb)
STP - Mean station pressure (mb)
VISIB - Mean visibility (miles)
WDSP – Mean wind speed (knots)
MXSPD - Maximum sustained wind speed (knots)
GUST - Maximum wind gust (knots)
MAX - Maximum temperature (Fahrenheit)
MIN - Minimum temperature (Fahrenheit)
PRCP - Precipitation amount (inches)
All 9's in a field (e.g., 99.99 for PRCP) indicates no report or insufficient data.