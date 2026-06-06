<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


# Hawaii Google Maps Restaurant Rating Analysis

## Introduction

I am working on the `Hawaii Google Maps Reviews` dataset from Jiacheng Li, Jingbo Shang, and Julian McAuley. My project is centered around extracting insight from business metadata and guest reviews to estimate the average star review of a restaurant.

This project uses three datasets -- the `meta` dataset, which provides business metadata, the `reviews` dataset, which collates an extensive list of guest reviews, and `restaurants`, a subset of the merged `meta` and `reviews` dataframes.

In this project, I will be developing several new features to use in a Linear Regression and Random Forest model. The bolded columns are features that I added to the existing dataset.

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
| **NEW: zipcode** | zipcode of the business |
| **NEW: county** | county of the business (Hawaii) |
| **NEW: center_lat** | geographic center of the county (latitude) |
| **NEW: center_long** | geographic center of the county (longitude) |
| **NEW: center_distance** | distance of business from geographic center |
| **NEW: coastal_threshold** | threshold for farthest 30% of businesses from geographic center |
| **NEW: is_coastal** | coastal or inland status |

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
| **NEW: review_length** | character length of review |
| **NEW: sentiment** | positive-negative sentiment score. -1 to 1. |
| **NEW: sentiment_labeled** | positive-negative sentiment score. binned |

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
| **zipcode** | zipcode of the business |
| **county** | county of the business (Hawaii) |
| **center_lat** | geographic center of the county (latitude) |
| **center_long** | geographic center of the county (longitude) |
| **center_distance** | distance of business from geographic center |
| **coastal_threshold** | threshold for farthest 30% of businesses from geographic center |
| **is_coastal** | coastal or inland status |
| user_id | ID of the reviewer |
| name_y | name of the reviewer |
| time | time of the review (unix time) |
| rating | rating of the business |
| text | text of the review |
| pics | pictures of the review. **EDITED: 1 if photos included, 0 if not** |
| resp | business response to the review including unix time and text of the response |
| **review_length** | character length of review |
| **sentiment** | positive-negative sentiment score. -1 to 1. |
| **sentiment_labeled** | positive-negative sentiment score. binned |
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
⚠️ Restaurant density/competition column

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

This address, called `Gas Lanes` and referred to as `Marine Mart` is located on Marine Corps Base, HI. While the zipcode itself does not seem to exist, the address is correct through search verification. I have a feeling that this zipcode scenario has something to do with the fact that this gas station/convenience store is only open to MCBH access only. I imputed NaN values for this business' `county` and `zipcode` values to omit this from analysis.

With my dictionary now updated, I checked my missingness distribution again.

```python
print("County distribution:")
print(meta['county'].value_counts())
print(f"\nMissing: {meta['county'].isna().sum()}")
```
```
County distribution:
county
Honolulu    11944
Hawaii       3793
Maui         3680
Kauai        1672
Name: count, dtype: int64

Missing: 418
```
Those fixes reduced the number of our `county` missing values from **835** to **418**. Note that our number of missing zipcodes went up from 417 to 418 because of our changes to Gas Mart.

```python
(meta[meta['county'].isna()].shape[0], meta[meta['zipcode'].isna()].shape[0])
```
```
(418, 418)
```

I then moved on to the discrepancy between missing `address` and `zipcode` values.

```python
(meta[meta['address'].isna()].shape[0], meta[meta['zipcode'].isna()].shape[0])
```
```
(318, 418)
```

We can assume that 318/418 of the missing zipcodes is due to missing zipcodes in the address. Let's try to figure out what's up with the last 100 missing zipcode values.

```python
missing_zipcodes = meta[meta['county'].isna()]['zipcode'].unique()
### UNCOMMENT ONLY TO CREATE DF IF YOU HAVEN'T DONE SO ALREADY
# meta[meta['zipcode'].isna()][['name', 'address', 'category']].to_csv('missing_zipcodes.txt', index = False)

missing_zipcodes_df = pd.read_csv('missing_zipcodes.txt')
missing_zipcodes_df['address'].unique().shape[0]
```
```
99
```

The shape of 99 seems to match up almost exactly with our 100 missing zipcode values.

```python
meta[meta['zipcode'].isna()]['address'].unique()[:10]
```
```
array([None, 'Black Pot Beach, Hanalei, HI',
       'Mauna Kea Observatory, Island of, HI', 'Kahului Bay, Hawaii',
       'Old Mānā Plantation Camp, Hawaii', 'Kawākiu Nui, Hawaii',
       'Kuia Stream, Hawaii', 'Kawainui Stream, Hawaii',
       'Kaho‘olawe, Hawaii', 'Lōʻihi Seamount'], dtype=object)
```
```python
missing_zipcode_categories = meta[meta['zipcode'].isna()]['address']

mzc_samples = np.random.choice(missing_zipcode_categories, size = 10, replace = False)
mzc_samples
```
```
array([None, None, 'Midway Island, United States Minor Outlying Islands',
       None, None, None, None, None, None, 'Lahaina Roads'], dtype=object)
```

The total number of our remaining missing zipcodes is either because there is  
* NO provided address
* NO provided zipcode in the address.

For our analysis, we'll skip all rows with no zipcode/county.

### Coastal vs. Inland Column
I define "coastal" as top 30% of businesses furthest from center of a county. We'll approximate center of a county by taking the average latitude and longitude values for each county. I saved these into dictionaries with corresponding counties as the keys.

I then mapped the dictionaries to two new columns, `center_lat` and `center_long`.

|    | county   |   center_lat |
|---:|:---------|-------------:|
|  0 | Honolulu |      21.3513 |
|  1 | Honolulu |      21.3513 |
|  2 | Maui     |      20.8521 |
|  3 | Maui     |      20.8521 |
|  4 | Honolulu |      21.3513 |

|    | county   |   center_long |
|---:|:---------|--------------:|
|  0 | Honolulu |      -157.896 |
|  1 | Honolulu |      -157.896 |
|  2 | Maui     |      -156.509 |
|  3 | Maui     |      -156.509 |
|  4 | Honolulu |      -157.896 |

Next, I faced the task of actually calculating distances. Since the Earth is mostly spherical and not flat, typical Euclidean distance calculations can't be used. Instead, I had to use the Haversine formula for calculating great-circle distances. Though I wish I could say I was a flat earth believer in order to use Euclidean distances instead, alas, I must face reality.

I'll use the haversine formula with atan2 for more stability, as that's what Stackoverflow recommends.

The Haversine distance $d$ between two points is given by:

$$
d = 2R \cdot \operatorname{arctan2}\left(\sqrt{a}, \sqrt{1-a}\right)
$$

where the intermediate value $a$ is:

$$
a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\Delta\lambda}{2}\right)
$$

**Variable Explanations**  
$d$: The distance between the two points along the sphere's surface.  
$R$: The radius of the Earth (mean radius $\approx$ 6,371 km or 3,959 mi).  
$\phi_1, \phi_2$: The latitude of point 1 and point 2, expressed in radians.  
$\lambda_1, \lambda_2$: The longitude of point 1 and point 2, expressed in radians.  
$\Delta\phi$: The difference in latitude ($\phi_2 - \phi_1$).  
$\Delta\lambda$: The difference in longitude ($\lambda_2 - \lambda_1$).  
$a$: The square of half the chord length between the points (a dimensionless value between 0 and 1).  
$\operatorname{atan2}(y, x)$: The two-argument arctangent function, which computes the angle whose tangent is $y/x$, using the signs of both arguments to determine the correct quadrant. Here, it effectively calculates the central angle $c = 2 \times \operatorname{atan2}(\sqrt{a}, \sqrt{1-a})$.

I first created a `haversine_distance` function, which calculated the haversine distance given two latitude and longitude points. Then, I calculated `center_distance`, or a business' distance from its county's geographic center. When I first ran this, I got the following maximum values:

|county |
|:------|
|Hawaii|       184.590626|
|Honolulu |   1506.077762|
|Kauai     |   113.637627|
|Maui      |   105.639856|

If I wasn't mistaken, my original distances were larger than the state of Hawaii, much less the county. The NaN values for center_lat and center_long line up with not having `county` values, so missingness of the latitude longitude data is not the problem.
```python
print(meta['center_lat'].isna().sum())
print(meta['center_long'].isna().sum())
```
```
418
418
```

I took a look at the rows with the aforementioned maximum values.

| | county | latitude | longitude | center_lat | center_long | center_distance |
|:--|:--------|:---------|:----------|:-----------|:------------|:----------------|
| 9766 | Hawaii | 21.393852 | -157.742967 | 19.692992 | -155.542907 | 184.590626 |
| 10396 | Honolulu | -0.352130 | -159.955034 | 21.351255 | -157.895734 | 1506.077762 |
| 12399 | Honolulu | 28.289839 | -177.372887 | 21.351255 | -157.895734 | 1310.107968 |
| 15413 | Kauai | 21.456171 | -157.768809 | 22.026183 | -159.429620 | 113.637627 |
| 15448 | Honolulu | 19.758484 | -155.455961 | 21.351255 | -157.895734 | 192.422146 |


Google search shows that the geographic boundaries of Hawaii are

**min_latitude:** 18.910361  
**max_latitude:** 28.402123  
**min_longitude:** -178.334698  
**max_longitude:** -154.806773  

The outliers in our dataset have impossible latitude/longitude values. I got rid of all rows with invalid latitude and longitude values and ran it again.

| | county | latitude | longitude | center_lat | center_long | center_distance |
|:--|:--------|:---------|:----------|:-----------|:------------|:----------------|
| 9766 | Hawaii | 21.393852 | -157.742967 | 19.692992 | -155.542907 | 184.590626 |
| 15413 | Kauai | 21.456171 | -157.768809 | 22.026183 | -159.429620 | 113.637627 |
| 15448 | Honolulu | 19.758484 | -155.455961 | 21.351255 | -157.895734 | 192.422146 |
| 15780 | Honolulu | 19.644531 | -155.993291 | 21.351255 | -157.895734 | 170.490843 |
| 18630 | Maui | 21.321267 | -158.068642 | 20.852130 | -156.509107 | 105.639856 |

> We did it, Joe!  
> -VP Kamala Harris

`center_distance` columns acquired. Finally, I could combine everything into the `is_coastal` column.

```python
# Check the distribution
print(meta['is_coastal'].value_counts())
```

| is_coastal | count |
|:-----------|------:|
| False | 14890 |
| True | 6244 |

Just checking by county.

| county | is_coastal | count |
|:-------|:-----------|------:|
| Hawaii | False | 2655 |
| Hawaii | True | 1138 |
| Honolulu | False | 8359 |
| Honolulu | True | 3583 |
| Kauai | False | 978 |
| Kauai | True | 419 |
| Maui | False | 2576 |
| Maui | True | 1104 |

### Restaurant Density Column
The `meta` dataset includes all businesses, not just restaurants. I'm going to approximate restaurant density by doing a simple count of how many restaurants are in each county, but I'll create this column once I subset meta to a smaller `restaurants` dataframe. So, keep this column in mind!

I ended the `meta` data cleaning section by replacing all None values with `np.nan`.

**`reviews` Dataset**  
✅ Review length column  
✅ Sentiment column

### Review Length Column
This was fairly simple. I took the length of the total review.

|        | text                                                                                |   review_length |
|-------:|:--------------------------------------------------------------------------------------------------------------------------------------|----------------:|
| 220427 | Very very expensive for fast food restaurant                                                                                          |              44 |
| 969030 | Free coffee samples, great walking, been drinking coffee from this location all year.  Spent $240 on coffee for our fiends back home. |             133 |

|         | text                                                                               |   review_length |
|--------:|:-----------------------------------------------------------------------------------|----------------:|
|  898479 | The service and food was good                                                      |              29 |
| 1458303 | The actual Memorial is closed due to ramp repairs. Not sure when it will be fixed. |              82 |
|  251276 | Its Jersey Mike's. You're a fool if you dont like the subs here.                   |              64 |

### Sentiment Column
In a previous class, when we conducted sentiment analysis on a corpus, we did everything in R with a package. I assumed that there would also be a Python library out there that could calculate sentiment very simply, rather than me using a rudimentary custom words dictionary.

I chose to use VADER and its SentimentIntensityAnalyzer from the mltk module. VADER scores sentiment on a scale of -1 to 1 (focus on the 'compound' key in the score dictionary the polarity_scores method generates).

Here's a sample review I extracted.

```
'Last night the lobster bisque soup was too thin, too little.\nDictionary says bisque should be creamy, rich, and flavorful.\nAnd the prime rib was too fat untrimmed, bloody Rare instead of "medium-rare".\n\nNext night the Onion Soup was B+ good and the 4 half inch thin lamb chops,cooked medium, were Delicious!\nGood chocolate cake, but could use ice cream, a bit dry.\nDeuce table tipsy and it spilled my beer oops. Good Service from Taylor!'
```

This was the output.
```
{'neg': 0.024, 'neu': 0.82, 'pos': 0.157, 'compound': 0.876}
```

I was surprised by this score, and it revealed some limitations with VADER. This is a mixed review, clearly, but it's scored quite positively in the compound value at 0.876. I think it weighed words like "Delicious", "good", "good", "good service" more compared to "too thin", "too little", "too fat", etc. But this is one instance, I think it should generalize ok.

Here's another sample of some reviews accompanied by their sentiment score.

|         | text                                                                                            |   sentiment |
|--------:|:------------------------------------------------------------------------------------------------|------------:|
| 1393541 | It's a beautiful place, it's in Hawaii and the view is beautiful, and the water is so beautiful |      0.9136 |
|  347974 | Was okay                                                                                        |      0.2263 |
|  106319 |                                                                                                 |    nan      |

**`restaurants` DataFrame**  
This is a subset of the `meta` dataframe. From here on out, we will refer to this dataset for most of our research questions, as I wanted to focus on restaurants. Here is a sample of the `restaurants` dataframe, which has all columns from the `meta` and `review` dataframes.

|        | name_x         | address                                                    | description                                                                                        | text                                                                                      |   sentiment | sentiment_labeled   |
|-------:|:---------------|:-----------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|------------:|:--------------------|
| 538808 | Dunkin'        | Dunkin', 3270 Ualena St, Honolulu, HI 96819                | Long-running chain serving signature donuts, breakfast sandwiches & a variety of coffee drinks.    | They have  great variety of different donuts I love the glazed ones there very delicious! |      0.9272 | very positive       |
| 191348 | Starbucks      | Starbucks, 91-1105 Keaunui Dr #500, Ewa Beach, HI 96706    | Seattle-based coffeehouse chain known for its signature roasts, light bites and WiFi availability. |                                                                                           |    nan      | nan                 |
| 486453 | Duke's Waikiki | Duke's Waikiki, 2335 Kalakaua Ave #116, Honolulu, HI 96815 | Popular option known for its beachfront location, surf 'n' turf, tiki vibe & umbrella drinks.      | Good food. Great staff. Stunning view!                                                    |      0.8718 | very positive       |

## Want Some Visuals?! 😎🏖️🐳🏄‍♀️
### Univariate Data Visualizations
<iframe src="assets/num-businesses-county.html" width="800" height="600" frameborder="0"></iframe>
<iframe src="assets/distance-county.html" width="800" height="600" frameborder="0"></iframe>

###

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis
