# Hawaii Google Maps Restaurant Rating Analysis

## Introduction

I am working on the `Hawaii Google Maps Reviews` dataset from Jiacheng Li, Jingbo Shang, and Julian McAuley. My project is centered around extracting insight from business metadata and guest reviews to estimate the average star review of a restaurant.

This project uses two datasets -- the `meta` dataset, which provides business metadata, and `reviews` dataset, which collates an extensive list of guest reviews.

In this project, I will be developing several new features to use in a Linear Regression and Random Forest model. Below is a list of all features in the three datasets I will be working on.

#### `meta`, 21,507 rows

| Column Name | Description |
|:------------|:------------|
| name | name of the business |
| address | address of the business |
| gmap_id | ID of the business |
| description | description of the business |
| latitude | latitude of the business |
| longitude | longitude of the business |
| category | category of the business |
| avg_rating | average rating of the business |
| num_of_reviews | number of reviews |
| price | price of the business |
| hours | open hours |
| MISC | MISC information |
| state | the current status of the business (e.g., permanently closed) |
| relative_results | relative businesses recommended by Google |
| url | URL of the business |
| zipcode | zipcode of the business |
| county | county of the business (Hawaii) |
| center_lat | geographic center of the county (latitude) |
| center_long | geographic center of the county (longitude) |
| center_distance | distance of business from geographic center |
| coastal_threshold | threshold for farthest 30% of businesses from geographic center |
| is_coastal | coastal or inland status |

#### `review`, 1,504,347 rows

| Column Name | Description |
|:------------|:------------|
| user_id | ID of the reviewer |
| name | name of the reviewer |
| time | time of the review (unix time) |
| rating | rating of the business |
| text | text of the review |
| pics | pictures of the review |
| resp | business response to the review including unix time and text of the response |
| gmap_id | ID of the business |
| review_length | character length of review |
| sentiment | positive-negative sentiment score. -1 to 1. |
| sentiment_labeled | positive-negative sentiment score. binned |

#### `restaurants`, 567,012 rows

| Column Name | Description |
|:------------|:------------|
| name_x | name of the business |
| address | address of the business |
| gmap_id | ID of the business |
| description | description of the business |
| latitude | latitude of the business |
| longitude | longitude of the business |
| category | category of the business |
| avg_rating | average rating of the business |
| num_of_reviews | number of reviews |
| price | price of the business |
| hours | open hours |
| MISC | MISC information |
| state | the current status of the business (e.g., permanently closed) |
| relative_results | relative businesses recommended by Google |
| url | URL of the business |
| zipcode | zipcode of the business |
| county | county of the business (Hawaii) |
| center_lat | geographic center of the county (latitude) |
| center_long | geographic center of the county (longitude) |
| center_distance | distance of business from geographic center |
| coastal_threshold | threshold for farthest 30% of businesses from geographic center |
| is_coastal | coastal or inland status |
| user_id | ID of the reviewer |
| name_y | name of the reviewer |
| time | time of the review (unix time) |
| rating | rating of the business |
| text | text of the review |
| pics | pictures of the review. **EDITED: 1 if photos included, 0 if not** |
| resp | business response to the review including unix time and text of the response |
| review_length | character length of review |
| sentiment | positive-negative sentiment score. -1 to 1. |
| sentiment_labeled | positive-negative sentiment score. binned |
| **NEW: zipcode_restaurant_density** | count of all restaurants in zipcode |
| **NEW: sentiment_mean** | average sentiment score per business |
| **NEW: review_length_mean** | average review character length per business |
| **NEW: pics_prop** | proportion of reviews with pictures per business |



## Data Cleaning and EDA
**`meta` Dataset**  
✅ Zipcode column  
✅ County column  
✅ Center Distance column  
✅ Coastal vs. Inland column  
✅ Restaurant density/competition column

### Zipcode Column
To create our `county` column, we first needed to create a `zipcode` column. I created a `get_zipcode` function that extracted the zipcode of each business. I applied this function to `meta['address']` to create the new column. 

### County Column
Then, I created a custom dictionary that mapped all zipcodes in Hawaii to their corresponding counties. This was a manual process, and initially, when I compared the number of missing `county` values to `zipcode` values, I got this tuple: `(835, 417)`.

If the missingness values of `county` were dependent on `zipcode`, the values in the tuple should be equal to each other. A closer look into the corresponding addresses of missing `county` values revealed an array of zipcodes I overlooked in the creation of my dictionary.

```
array(['96857', None, '96733', '96848', '96796', '96726', '96795',
       '96853', '96860', '96127', '96745', '96123', '96751', '96863',
       '96861', '96718', '96830', '96767', '96858', '96867', '96824',
       '96859'], dtype=object)
```

I updated the dictionary with the list of zipcodes. Interestingly, the zipcodes `96127` and `96123` are both located in Lassen County, California. I pulled up the specific address and business to see what was going on.

```python
meta[meta['zipcode'] == '96127']
print(meta.iat[1219, 1])
```
```
Titanium Rim Repair, 961272 Waihona St, Pearl City, HI 96782
```

```python
meta[meta['zipcode'] == '96123']
print(meta.iat[1663, 1])
```
```
Elite Discount Furniture Warehouse, 961237 Waihona St, Pearl City, HI 96782
```

The problem with regex's `re.search` is that it returns the first match. These addresses happened to have a string pattern that matched the zipcode pattern, BEFORE the zipcode. Note how they are both located on the same street and have the same zipcode; it's likely they're both in the same plaza and follow a similar naming pattern. I manually updated the zipcode value in these rows.

Finally, it also appeared that zipcode `96867` didn't exist in unitedstateszipcodes.org nor in a regular search.

```python
meta[meta['zipcode'] == '96867']
print(f"Business Name: {meta.iat[6655, 0]}")
print(f"Business Address: {meta.iat[6655, 1]}")
```
```
Business Name: Gas Lanes
Business Address: Gas Lanes, Building 3071, 3rd St, MCBH, HI 96867
```

This address, called `Gas Lanes` and referred to as `Marine Mart` is located on Marine Corps Base, HI. While the zipcode itself does not seem to exist, the address is correct through search verification. I have a feeling that this zipcode scenario has something to do with the fact that this gas station/convenience store is only open to MCBH access only. We'll impute NaN values for this business' `county` and `zipcode` values to omit this from analysis.


**`reviews` Dataset**  
✅ Review length column  
✅ Sentiment column

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis
