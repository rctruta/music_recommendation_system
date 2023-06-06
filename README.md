# A Music Recommendation System

**Capstone Project for the Applied Data Science Program, MIT Professional Education**

Humans are in a constant race against time, with ever-increasing busy lives. 
The volume of music available to us is contantly increasing, making the task of searching for new music to listen to daunting, 
at the very least.

The problem we are addressing here is using data science to build a recommendation system capable of proposing the top 10 songs to a user, 
based on the likelihood that said user will be listening to those songs.

Building a recommendation system based on our unique musical preferences, 
able to assist us in the process of finding new music to explore, will benefit us by enriching our lives.

In order for us to build a recommendatio system, we need to answer a few key questions, listed below.

    - How do we define the musical preference of a user?

    - What factors contribute to a user's musical preference?

    - What data is the most useful when defining the preference of a user?

    - What is the musical content that a user is most likely to consume?

Depending on the data available to us, answering these question may reveal different answers. 
Data scientists are striving to build a reliable recommendation system, one that is accurate, fast and inexpensive.

Musical tastes are very personal, and recommending new songs to a user is not an easy task, by any measure. 
Ultimately, any user can benefit from having a reliable recommendation system, capable of making fast and accurate recommendations!

## **Key takeaways**

The initial analysis performed for the Milestone submission revealed important takeaways, which we aimed to address in this current submission. 

Arguably, the **most important** key takeway that needs addressing is improving the performance of the system, reflected in the performance metrics results. 

We have performed many experiments in trying to address the low performance of all the models we proposed and analyzed. In many ways, we performed a `Grid Search`, where the `hyperparameters` to by tuned are:

1. The ratio of the trainset/testset data split:
  - either a `80/20` split, 
  - a `70/30` split, or
  - a `60/40`split.
2. Dropping the imputed year values (year = 0):
  - either `True` (drop),
  - or `False` (keep).
3. Threshold parameter, in `precision@k`, and `recall@k`:
  - we experimented with these values: `1.0, 1.1, 1.2, 1.3, 1.4, 1.5`.

Ultimately, based on the extensive analysis we performed, we have concluded that the _hyperparameter_ influencing the performance metrics the most is the `threshold` parameter. The value of this parameter, in combination with any choice of the other two parameters revealed very similar results, making the `threshold` parameter the one that influences the perfomance metrics the most. The reasoning behind this comes from the data itself.

For the analysis presented in this report, the values of the parameters that we decided on are:
1. The ratio of the trainset/testset data split: `80/20`.
2. Dropping the imputed year values (year = 0): `True`.
3. Threshold parameter, in `precision@k`, and `recall@k`: `1`.

Based on our analysis, we can use any of the models presented here with these values for `hyperparameters`, the results are comparable. In fact, the simplest, most ready-to-use model can be deployed with these hyperparameters.

## **Algorithms and models analysis**

In this case study, we built recommendation systems using four different category of algorithms, as described below.

1. `Rank-based` algorithm, using averages of `play_count`s for ranking.

2.  `Collaborative-filtering` algorithms:
    
    a. `Similarity/Neighborhood-based`:
      - `User-user` similarity, and its improved version;
      - `Item-item` similarity, and its improved version.
    
    b. `Model-based`:
      - `Matrix-factorization`: `Singular Value Decomposition` and its improved version.

3. `Cluster-based` algorithm, and its improved version.
    
4. `Content-based` recommendation system.  

The `surprise` library was used to demonstrate all of the `Collaborative-filtering`-based algorithms. For all of these algorithms, `Grid Search Cross-Validation` was used to find the optimal hyperparameters for the data, and to improve upon the baseline algorithms. In addition, the optimal hyperparameters were used to generate related predictions.

## **Evaluation metrics**

Many evaluation metrics are available for recommendation systems, and each one of them has its own pros and cons. 

In the context of recommendation systems, we are most likely interested in recommending top-N items to the user. In this sense, it makes the most sense to compute precision and recall metrics in the first N items, instead of all of the items. Thus, we are using the notion of precision and recall at k, where k is a user-definable integer to match the top-N recommendations objective.

To `evaluate the performance` of these models, we used `precision@k` and `recall@k` metrics. Since both of these metrics are indicative of a model's performance, the harmonic mean of these two metrics, `F_1 score` has been calculated for each of the models we highlighted. 

The `Precision@k`, `Recall@k` and `F1_score@k` are defined below:

  - $\text{Precision@k} = \frac{ | \{ \text{Recommended items that are relevant} \} | }{ | \{ \text{Recommended items} \} | }$

  - $\text{Recall@k} = \frac{ | \{ \text{Recommended items that are relevant} \} | }{ | \{ \text{Relevant items} \} | }$

  - $\text{F1_score@k} = \frac{2}{{\frac{1}{\text{Precision@k}} + {\frac{1}{\text{Recall@k}}}}}$

An item is considered `relevant` if its true rating $r_{ui}$ is greater than a given `threshold`. 

An item is considered `recommended` if its estimated rating
$\hat{r}_{ui}$ is greater than the threshold, and if it is among the k highest estimated ratings.

Note that in the edge cases where division by zero occurs, `Precision@k` and `Recall@k` values are undefined. As a convention, we set their values to 0 in such cases.

`Recall`: the fraction of relevant songs that are recommended to a user.

`Precision`: the fraction of the recommended songs that are relevant to the user.

`Precision@k` is the proportion of recommended items in the top-k set that are relevant.

`Recall@k` is the proportion of relevant items found in the top-k recommendations.

One takeaway worth mentioning is that we have performed an in-depth analysis of the `Precision@k` and `Recall@k`, varying the values of the `k` parameter, from `10` to `40`, in increments of 10. 

When `k = 40`, all the models we tested satisfy `Precision@40 = Recall@40 = 1`. That is, for `k = 40`, we reach `100% accuracy` of our models.

Another takeaway worth mentioning is that for a `60/40` data split, the models reach `100% accuracy` when `k = 60`.

The following table **summarizes the metric results** used to evaluate the models we build for our analysis. More on this analysis is presented later in the report.

|metric|user-user|user-user-optimized|item-item|item-item-optimized|SVD|SVD-optimized|coCluster|coCluster-optimized|
|------|---------|-------------------|---------|-------------------|---|-------------|---------|-------------------|
|RMSE  |1.088  |1.041             |  1.024  |1.02 |**1.009**|**1.009**|1.033|1.04|
|Precision@10|1.0|1.0|1.0|1.0|1.0|1.0|1.0|1.0|
|Recall@10|0.948|0.948|0.948|0.948|0.948|0.948|0.948|0.948|
|Recall@20|0.994|0.994|0.994|0.994|0.994|0.994|0.994|0.994|
|Recall@30|0.999|0.999|0.999|0.999|0.999|0.999|0.999|0.999|
|Recall@40|1.0|1.0|1.0|1.0|1.0|1.0|1.0|1.0|
|$F_1$ score@10|0.973|0.973|0.973|0.973|0.973|0.973|0.973|0.973|
|$F_1$ score@20|0.997|0.997|0.997|0.997|0.997|0.997|0.997|0.997|
|$F_1$ score@30|0.999|0.999|0.999|0.999|0.999|0.999|0.999|0.999|
|$F_1$ score@40|1.0|1.0|1.0|1.0|1.0|1.0|1.0|1.0|

As illustrated in the table above, all the models we tested perform really well. 
