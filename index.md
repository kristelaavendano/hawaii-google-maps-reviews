# Hawaii Google Maps Restaurant Rating Analysis

## Introduction

I am working on the `Hawaii Google Maps Reviews` dataset from Jiacheng Li, Jingbo Shang, and Julian McAuley. My project is centered around extracting insight from business metadata and guest reviews to estimate the average star review of a restaurant.

This project uses two datasets -- the `meta` dataset, which provides business metadata, and `reviews` dataset, which collates an extensive list of guest reviews.

In this project, I will be developing the following features. Below is a list of all features in three datasets.

#### `meta`
| Column Name | Description |
|:-------------|:-------------|
|name|name of the business
|address|	address of the business
|gmap_id|ID of the business
|description |description of the business
|latitude |latitude of the business
|longitude|longitude of the business
|category|category of the business
|avg_rating|average rating of the business
|num_of_reviews|number of reviews
|price|price of the business
|hours|open hours
|MISC|MISC information
|state|the current status of the business (e.g., permanently closed)
|relative_results|	relative businesses recommended by Google
|url|URL of the business
|zipcode|zipcode of the business
|county|county of the business (Hawaii)
|center_lat|geographic center of the county (latitude)
|center_long|geographic center of the county (longitude)
|center_distance|distance of business from geographic center
|coastal_threshold|threshold for farthest 30% of businesses from geographic center
|is_coastal|coastal or inland status

#### `review`
|Column Name|Description|
|-----------|-----------|
|user_id|	ID of the reviewer
|name|	name of the reviewer
|time|	time of the review (unix time)
|rating|	rating of the business
|text|	text of the review
|pics|	pictures of the review
|resp|	business response to the review including unix time and text of the response
|gmap_id|	ID of the business
|review_length|character length of review
|sentiment|positive-negative sentiment score. -1 to 1.
|sentiment_labeled|positive-negative sentiment score. binned


#### `restaurants`
| Column Name | Description |
|:-------------|:-------------|
|name_x         |name of the business
|address      |	address of the business
|gmap_id      |ID of the business
|description  |description of the business
|latitude     |latitude of the business
|longitude    |longitude of the business
|category     |category of the business
|avg_rating   |average rating of the business
|num_of_reviews|number of reviews
|price         |price of the business
|hours         |open hours
|MISC          |MISC information
|state          |the current status of the business (e.g., permanently closed)
|relative_results|	relative businesses recommended by Google
|url             |URL of the business
|zipcode         |zipcode of the business
|county          |county of the business (Hawaii)
|center_lat      |geographic center of the county (latitude)
|center_long     |geographic center of the county (longitude)
|center_distance |distance of business from geographic center
|coastal_threshold|threshold for farthest 30% of businesses from geographic center
|is_coastal       |coastal or inland status
|user_id|	ID of the reviewer
|name_y|	name of the reviewer
|time|	time of the review (unix time)
|rating|	rating of the business
|text|	text of the review
|pics|	pictures of the review. **EDITED: 1 if photos included, 0 if not**
|resp|	business response to the review including unix time and text of the response
|review_length|character length of review
|sentiment|positive-negative sentiment score. -1 to 1.
|sentiment_labeled|positive-negative sentiment score. binned
|**NEW: zipcode_restaurant_density**|count of all restaurants in zipcode.
|**NEW: sentiment_mean**|average sentiment score per business
|**NEW: review_length_mean**|average review character length per business
|**NEW: pics_prop***|proportion of reviews with pictures per business


## Data Cleaning and EDA
**`meta` Dataset**  
✅ Zipcode column  
✅ County column  
✅ Center Distance column  
✅ Coastal vs. Inland column  
✅ Restaurant density/competition column

To create our `county` column, we first needed to create a `zipcode` column. We would then 


**`reviews` Dataset**  
✅ Review length column  
✅ Sentiment column

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis
