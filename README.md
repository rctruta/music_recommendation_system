# <img src="http://millionsongdataset.com/sites/default/files/millionsong2-128.jpg" width="60" align="left"> Music Recommendation System

## Capstone Project for the Applied Data Science Program, MIT Professional Education

### Project Overview

In an era of ever-increasing digital music catalogs, the task of discovering new music tailored to individual preferences can be daunting. This capstone project addresses this challenge by developing a **music recommendation system** capable of proposing the top 10 songs to a user based on their predicted listening likelihood. Our goal is to enrich the user's music exploration experience by providing fast, accurate, and personalized recommendations.

## Problem & Solution Summary

The core problem tackled is leveraging data science to build a recommendation system that can effectively suggest the top 10 songs to a user, aligning with their probable listening habits.

To achieve this, we addressed several key questions:
* How do we define the musical preference of a user?
* What factors contribute to a user's musical preference?
* What data is most useful when defining a user's preference?
* What musical content is a user most likely to consume?

Our solution involved extensive data preprocessing, exploration of various recommendation algorithms, rigorous hyperparameter tuning, and detailed performance evaluation.

## Data

The project utilizes the **Taste Profile Subset** from the **Million Song Dataset**.

**Data Source:** [http://millionsongdataset.com/](http://millionsongdataset.com/)

### Data Dictionary

* **`song_data`**: Contains details about songs.
    * `song_id`: A unique ID for every song.
    * `title`: Title of the song.
    * `release`: Name of the released album.
    * `artist_name`: Name of the artist.
    * `year`: Year of release.
* **`count_data`**: Records user listening activity.
    * `user_id`: A unique ID for the user.
    * `song_id`: A unique ID for the song.
    * `play_count`: Number of times the song was played by the user.

### Initial Data Exploratory Analysis (EDA)

* **`count_df` Observations:**
    * Contains `2,000,000` entries.
    * `10,000` unique song IDs, implying approximately 200 entries (user listens) per song on average.
    * No missing values.
    * Average `play_count` is 3, with a minimum of 1 and a maximum of 2213 (potential outlier).
* **`song_df` Observations:**
    * Contains `1,000,000` entries.
    * `999,056` unique song IDs and `702,428` unique song titles, indicating multiple songs share the same title.
    * Minor missing values (15 for `title`, 5 for `release`).
* **Year-based Observations:**
    * A significant drop in song counts for the period `1969-1986`, with several years completely missing, suggesting potential data imputation needs.
    * ![Yearly Song Distribution](path/to/your/yearly_song_distribution.png) *(Placeholder for your plot output)*

### Data Preprocessing Steps

The raw datasets were preprocessed through the following stages to prepare them for modeling:

1.  **Merge DataFrames:** `count_df` and `song_df` were merged (left-merge on `song_id`), dropping duplicate `song_id` entries from `song_df` and the `Unnamed: 0` column.
2.  **Encode IDs:** `user_id` and `song_id` attributes were label-encoded from strings to numeric values.
3.  **Filter Dataset:**
    * Users who listened to less than 90 songs were removed.
    * Songs listened to by less than 120 users were removed.
4.  **Generate Rating Scale:** `play_count` was re-scaled as a rating system from `1` to `5` by restricting data to users who listened to a song at most 5 times.
5.  **Drop Imputed Years:** Entries with `year = 0` were removed.

These steps significantly reduced the dataset from `2,000,000` rows to `97,227` rows (4.86% of the original size), addressing computational intensity.

| Step | Shape before step | Shape after step |
| :--- | :---------------- | :--------------- |
| 3a   | (2000000, 7)      | (438390, 7)      |
| 3b   | (438390, 7)       | (130398, 7)      |
| 4    | (130398, 7)       | (117876, 7)      |
| 5    | (117876, 7)       | (97227, 7)       |

### Final Dataset Overview

After preprocessing, the final dataset contains:
* `3,154` unique users.
* `478` unique song IDs.
* `476` unique song titles (confirming duplicate titles for different song IDs).
* `185` unique artists.
* No missing values.
* ![Play Count Distribution](path/to/your/play_count_distribution.png) *(Placeholder for your plot output)*

## Recommendation System Workflow

Our recommendation system workflow follows standard machine learning practices:
1.  A training set is fed to a recommendation algorithm, which produces a recommendation model.
2.  A held-out test set is used to generate predictions from the learned model for each user-item pair.
3.  Predictions with known true values are then input into an evaluation algorithm to produce performance results.

## Methodology and Algorithms

This case study built and analyzed recommendation systems using four distinct categories of algorithms:

1.  **Rank-Based Algorithm:**
    * Recommendations are based on the overall popularity of a song, defined by the frequency and average of its play counts.
2.  **Collaborative-Filtering Algorithms:**
    * **Similarity/Neighborhood-Based:**
        * User-user similarity (and an improved version).
        * Item-item similarity (and an improved version).
    * **Model-Based:**
        * Matrix Factorization: Singular Value Decomposition (SVD) and its optimized version.
3.  **Cluster-Based Algorithm:**
    * Groups similar users based on listening patterns and recommends songs popular within the same cluster.
4.  **Content-Based Recommendation System:**
    * Recommends items based on the features/attributes of the items themselves.

The **[Surprise](https://surprise.readthedocs.io/en/stable/index.html)** library was extensively used to demonstrate all Collaborative Filtering algorithms. **Grid Search Cross-Validation** was applied to find optimal hyperparameters and improve model performance.

## Experimental Design & Hyperparameter Tuning

A key focus was on improving system performance. We conducted extensive experiments via a Grid Search across critical hyperparameters:

* **Train/Test Data Split Ratio:** `80/20`, `70/30`, or `60/40`.
* **Handling Imputed Year Values (`year = 0`):** `True` (drop) or `False` (keep).
* **Relevance Threshold Parameter** (for `precision@k` and `recall@k`): `1.0, 1.1, 1.2, 1.3, 1.4, 1.5`.

**Key Takeaway from Tuning:** The **relevance threshold parameter** was found to be the most influential hyperparameter affecting performance metrics, with other parameters yielding similar results regardless of their choice. This insight is attributed to the inherent characteristics of the data.

**Final Hyperparameter Values for Analysis:**
* **Train/Test Data Split Ratio:** `80/20`
* **Dropping Imputed Year Values (`year = 0`):** `True`
* **Relevance Threshold Parameter:** `1.0`

These values enable comparable results across models, simplifying the deployment of even the simplest models.

## Evaluation Metrics

For recommendation systems, evaluating top-N recommendations is crucial. We used **Precision@k**, **Recall@k**, and their harmonic mean, the **F1-score@k**.

**Definitions:**

* **Precision@k:** The proportion of recommended items within the top-k set that are actually relevant to the user.
    $$ \text{Precision@k} = \frac{ | \{ \text{Recommended items that are relevant} \} | }{ | \{ \text{Recommended items} \} | } $$
* **Recall@k:** The proportion of all relevant items that were successfully found within the top-k recommendations.
    $$ \text{Recall@k} = \frac{ | \{ \text{Recommended items that are relevant} \} | }{ | \{ \text{Relevant items} \} | } $$
* **F1-score@k:** The harmonic mean of Precision@k and Recall@k, balancing both metrics.
    $$ \text{F1_score@k} = \frac{2}{{\frac{1}{\text{Precision@k}} + {\frac{1}{\text{Recall@k}}}}} $$
    * An item is considered **relevant** if its true rating `r_ui` is greater than a given threshold.
    * An item is considered **recommended** if its estimated rating `r_hat_ui` is greater than the threshold AND it is among the k highest estimated ratings.
    * *Note:* In edge cases where division by zero occurs, Precision@k and Recall@k values are conventionally set to 0.

Our in-depth analysis varied `k` from 10 to 40. Notably, at `k = 40`, all models achieved `Precision@40 = Recall@40 = 1`, indicating 100% accuracy within the top 40 recommendations. For a `60/40` data split, models reached 100% accuracy when `k = 60`.

### Performance Metrics Summary

The table below summarizes the evaluation metric results for the models. Detailed analysis and visualizations are available in the project's Jupyter Notebook.

| Metric              | User-User | User-User-Optimized | Item-Item | Item-Item-Optimized | SVD   | SVD-Optimized | CoCluster | CoCluster-Optimized |
| :------------------ | :-------- | :------------------ | :-------- | :------------------ | :---- | :------------ | :-------- | :------------------ |
| RMSE                | 1.088     | 1.041               | 1.024     | 1.02                | 1.009 | 1.009         | 1.033     | 1.04                |
| Precision@10        | 1.0       | 1.0                 | 1.0       | 1.0                 | 1.0   | 1.0           | 1.0       | 1.0                 |
| Recall@10           | 0.948     | 0.948               | 0.948     | 0.948               | 0.948 | 0.948         | 0.948     | 0.948               |
| F1-score@10         | 0.973     | 0.973               | 0.973     | 0.973               | 0.973 | 0.973         | 0.973     | 0.973               |
| Recall@20           | 0.994     | 0.994               | 0.994     | 0.994               | 0.994 | 0.994         | 0.994     | 0.994               |
| F1-score@20         | 0.997     | 0.997               | 0.997     | 0.997               | 0.997 | 0.997         | 0.997     | 0.997               |
| Recall@30           | 0.999     | 0.999               | 0.999     | 0.999               | 0.999 | 0.999         | 0.999     | 0.999               |
| F1-score@30         | 0.999     | 0.999               | 0.999     | 0.999               | 0.999 | 0.999         | 0.999     | 0.999               |
| Recall@40           | 1.0       | 1.0                 | 1.0       | 1.0                 | 1.0   | 1.0           | 1.0       | 1.0                 |
| F1-score@40         | 1.0       | 1.0                 | 1.0       | 1.0                 | 1.0   | 1.0           | 1.0       | 1.0                 |

All models performed exceptionally well, particularly at higher `k` values.

### Sample Prediction Summary

| Algorithm           | min_predicted_play_count | max_predicted_play_count | (6958, 1671, 2) | (6958, 3232, None) |
| :------------------ | :----------------------- | :----------------------- | :-------------- | :----------------- |
| user-user           |                          |                          | 1.67            | 1.67               |
| user-user-optimized | 2.69                     | 3.05                     | **1.91** | 1.29               |
| item-item           |                          |                          | 1.31            | 1.3                |
| item-item-optimized | 2.04                     | 2.72                     | **1.91** | 1.29               |
| SVD                 |                          |                          | 1.31            | 1.23               |
| SVD-optimized       | 2.04                     | 2.72                     | 1.36            | 1.44               |
| coCluster           |                          |                          | 1.2             | 1.02               |
| coCluster-optimized | 1.98                     | 3.14                     | 1.0             | 1.0                |

Based on predictions for user `6958`, the `user-user-optimized` model provided the most accurate prediction for a known song (`1.91` vs. `2.0` actual).

### Final Model Proposal

We propose the **Singular Value Decomposition (SVD)** model as a robust and ready-to-use solution. SVD is a personalized, model-based collaborative filtering algorithm that relies on latent features extracted from past user behavior, making recommendations independent of additional external information.

## Recommendations for Implementation

### Implementation Strategy

An example implementation strategy would involve:
1.  **Establishing a Goal:** E.g., implementing the model to recommend songs when a `genre` feature is added.
2.  **Scheduling Milestones:** E.g., adding a new feature and testing its usefulness every month.
3.  **Defining Success:** E.g., successful implementation and training in 4 weeks or less; model still in use 3 months or more after implementation.
4.  **Re-assessing Metrics:** Continuously evaluating time, cost, and human resources needed for implementation.

### Cost Analysis

The current analysis was performed on Google Colab (free tier). For a large-scale recommendation system, a **cloud solution (e.g., IaaS, PaaS, or SaaS)** is recommended over an on-premise setup due to data volume, efficiency, and cost offsets in maintenance. A detailed cost analysis would require an economist or business analyst.

### Risks and Challenges

* **Threshold Parameter Variability:** The optimal relevance threshold (currently `1.0`) may need to change if more data is added, requiring re-evaluation of all models.
* **Content-Based Accuracy Dilemma:** Our content-based model, trained for top 10 recommendations, sometimes provided only half as many genuinely similar songs (e.g., 5 for "Learn To Fly" instead of 10). This raises an ethical dilemma regarding whether to "recommend" or merely "suggest." Simple NLP techniques could improve this.

### Further Analysis & Future Work

We believe the system can be significantly enhanced by:
* **Incorporating a Larger NLP Component:** Utilizing ready-to-use NLP models could bring substantial benefits.
* **Exploring Neural Networks:** While computationally expensive, Neural Networks could improve recommendations if cost is not a limiting factor.
* **Expanding Data Sources:**
    * Including more original features like `danceability`, `energy`, `loudness`, `mode`, `similar artists`, `song hotttnesss`, `tempo` for a more robust content-based model.
    * Integrating Spotify-related data, such as `genre` information (which is missing from the current dataset) via vast Spotify datasets or the `Spotipy` library.
    * Utilizing data from radio stations (new songs, oldies) or music discovery apps like `Shazam` to capture user preferences.
* **Genre-Based Clustering:** A cluster-based system using genre information would be beneficial for users with limited musical tastes.
* **Sentiment Analysis:** For users with eclectic tastes, sentiment analysis could provide more accurate predictions.
* **Playlist-Based Recommendations:** Instead of individual songs, recommending entire playlists, potentially integrated with IoT wearable device data (e.g., BPM to suggest "Relaxing/Meditative" or "High Energy/Dancing" playlists).

## Technologies Used

* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Preprocessing:** scikit-learn (`LabelEncoder`)
* **Recommendation Algorithms:** [Surprise](https://surprise.readthedocs.io/en/stable/index.html) (`KNNBasic`, `SVD`, `CoClustering`, `Reader`, `Dataset`, `GridSearchCV`, `RandomizedSearchCV`, `accuracy`, `train_test_split`)
* **Natural Language Processing:** NLTK (`punkt`, `stopwords`, `wordnet`, `word_tokenize`, `WordNetLemmatizer`), `re` (regular expressions), `TfidfVectorizer` (from scikit-learn)
* **Similarity Calculation:** scikit-learn (`cosine_similarity`)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab

## Repository Structure
```
├── music_recommendation_system_final_submission.ipynb # The main Jupyter Notebook for the project.
├── data/                                              # Directory for raw and processed datasets (e.g., song_data.csv, count_data.csv).
├── README.md                                          # This README file.
└── requirements.txt                                   # Lists Python package dependencies for reproducibility.
```
## Installation & How to Run

To set up and run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```
    *(Replace `yourusername` and `your-repo-name` with your actual GitHub details.)*

2.  **Create `data/` directory and place datasets:**
    * Create a folder named `data` in the root of the repository.
    * Place `song_data.csv` and `count_data.csv` (obtained from [Million Song Dataset](http://millionsongdataset.com/)) inside the `data/` folder.

3.  **Install dependencies:**
    * Ensure you have Python (3.6+) and `pip` installed.
    * Install all required libraries using the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Jupyter Notebook:**
    * Launch Jupyter Notebook or JupyterLab from your repository's root directory:
        ```bash
        jupyter notebook
        ```
        or
        ```bash
        jupyter lab
        ```
    * Open `music_recommendation_system_final_submission.ipynb` and run all cells to execute the analysis, modeling, and generate results.

**Viewing on GitHub:**
* You can directly view the `music_recommendation_system_final_submission.ipynb` file on GitHub, as it supports rendering Jupyter Notebooks directly in the browser.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

### Appendix: Acknowledgements

Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011.

---
