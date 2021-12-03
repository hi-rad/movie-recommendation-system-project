# Movie Recommendation System Code

> This repository contains the code for a Graduate Coursework Project (CP8305 - Knowledge Discovery) for Fall 2021.
## 1. Introduction snd Setup
The project is a Movie Recommendation System using multiple models.

To run the project, a python 3.9 environment must be created and the following packages must be installed (either by using conda or pip):
- PyTorch
- Pandas
- Matplotlib
- Numpy
- SKLearn
- joblib
- beautifultable

Also, Jupyter Notebook support must be added.

The order in which the files must be executed are:
1. `data_preprocessing.ipynb`: This file contains some preprocessing steps needed for all models. It might take hours to run the file as the dataset is huge
2. `train_autoencoder.py`: This file contains training of the AutoEncoder. The file can be edited to change the AutoEncoder network structure. Training the network may take hours since the dataset is huge and the network has lots of neurons. By the end of running this file, the final AutoEncoder network will be trained and saved for future use.
3. `kmean_preprocessing.ipynb`: This file contains some preprocessing steps needed for creating the KMeans Clustering approach and finding a good candidate for K. Running this file takes hours since the dataset is huge and the KMeans Training is done multiple times. By the end of running this file, the final KMeans model will be trained and saved for future use.
4. `kd_tree_preprocessing.ipynb` : This file contains some preprocessing steps needed for creating the KD-Tree approach. By the end of running this file, the final KD-Tree model will be trained and saved for future use. Also, all movie distances will be calculated and stored for faster use in the recommendation engine.
5. `validation_approaches.ipynb`: This file contains the evaluations for all approaches.

## 2. Dataset
The Movie Dataset was used for this project. There are 26,000,000 ratings provided by 270,000 users for 45,000 movies in the dataset. The dataset also provides genre and cast information for each movie which was used in this project. The contents of the downloaded dataset must be placed inside the data directory.

The dataset can be downloaded from [this link](https://www.kaggle.com/rounakbanik/the-movies-dataset).

[https://www.kaggle.com/rounakbanik/the-movies-dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)

## 3. Results
First, the timestamp for a 0.90 quantile of ratings was found to keep ratings before this timestamp as training and the rest for evaluations.

Then, I trained the autoencoder with two different architectures. The first model, as shown in figure 1, take a users ratings to all available movies (the unrated movies are set to zero) and forwards them to a hidden layer with 400 neurons. This is the encoder part. The second part tries to recreate the inputs by turning the encoded ratings (values produced by the hidden layer) to the users original ratings, which is the decoder part. Figure 2 shows the autoencoder network with three hidden layer which works in the same way as the previous network.

![Figure 1. First Autoencoder Network](/autoencoder_400_network.png)
Figure 1. Autoencoder with One Hidden Layer

![Figure 2. Second Autoencoder Network](/autoencoder_400_200_400_network.png)
Figure 2. Autoencoder with Three Hidden Layers

As can be seen from the results in figure 3, the second model does a better job, and has been chosen for the recommender system. Both models have been trained for 60 epochs (more epochs is better, but due to hardware constraints I could not train for more epochs or deeper networks). 70% of the users were chosen for training and the remaining 30% were used for testing.

![Figure 3. AutoEncoder Models Comparison](/ae_loss_comparison.png)
Figure 3. AutoEncoder Models Comparison

I first found the top 200 casts based on the number of a cast was mentioned in movies. Figures 4 to 6 shows genre and cast occurrences. Then I multi-hot encoded genres and casts. Because a movie can have multiple genres or casts, a multi-hot encoding approach was chosen instead of one-hot encoding.

![Figure 4. Genre Occurrence in Ratings](/genre_occurrence.png)
Figure5. Genre Occurrence in Ratings

![Figure 5. Genre Occurrence in Movies](/movies_only_based_genre_occurrence.png)
Figure 5. Genre Occurrence in Movies

![Figure 6. Cast Occurrence in Movies](/cast_occurrence.png)
Figure 6. Cast Occurrence in Movies

Then, using different values for K, I checked the sum of squared distances of samples to their closest cluster center to decide which value of K is better. Figure 7 shows that at around 50 the value of K is optimal and values greater than that, will not have a significantly lower value of sum of squared distances of samples to their closest cluster center.

![Figure 7. Different K Values for KMeans Clustering](/candidates_for_k_means_sum_of_squared_distances.png)
Figure 7. Different K Values for KMeans Clustering

Finally, I created three methods for recommendations, KMeans Clustering (Collaborative Filtering and Content Based Filtering), KD-Tree (Content Based Filtering), Autoencoder (Collaborative Filtering), and a Hybrid approach using Autoencoder and KD-Tree (Collaborative Filtering and Content Based Filtering). Figure 8 and table 1 show the results for these methods.

![Figure 8. Methods Comparison](/methods_comparison.png)
Figure 8. Methods Comparison

Table 1. Methods Comparison

|               | MSE     | Accuracy   | Precision   | Recall   | F1 Measure   |
|---------------|---------|------------|-------------|----------|--------------|
| Clustering    | 2.618   | 0.515      | 0.672       | 0.377    | 0.483        |
| ------------- | ------- | ---------- | ----------- | -------- | ------------ |
| KD Tree       | 0.921   | 0.662      | 0.742       | 0.672    | 0.706        |
| ------------- | ------- | ---------- | ----------- | -------- | ------------ |
| Autoencoder   | 0.912   | 0.699      | 0.782       | 0.694    | 0.735        |
| ------------- | ------- | ---------- | ----------- | -------- | ------------ |
| Hybrid        | 0.824   | 0.704      | 0.783       | 0.702    | 0.74         |

## 4. Conclusion

As can be seen by the above table and chart, the Hybrid approach has the lowest Mean Squared Error and the highest Accuracy, Precision, and Recall. Therefore, the hybrid approach would be a better option.

The KD-Tree approach is only based on the genres and casts. It is a content-based only approach. The Clustering approach takes the ratings and genere into account. As a result, it is using both the collaborative filtering and content-based filtering approaches. However, it fails to produce more accurate results compared to the other methods. The autoencoder is a collaborative filtering only approach. The input is the user ratings. The movies use has not yet rated will be set to zero. Then the model tries to recreate the input ratings in the output. In the process of learning, the weights will be updated based on the training users ratings. As a result, the optimization of the network's weights is done using the collaboration of the previous users' ratings. When the model generates the outputs, it also predicts ratings for the unseen movies and these ratings will be used as predictions.

Finally, The hybrid approach uses both the collaborative filtering approach (the autoencoder) and the content-based filtering approach (KD-Tree). We have assigned a coefficient of 0.7 for the autoencoder results and a coefficient of 0.3 for the KD-Tree results, and added the final values together. The experiments show that the final Hybrid results are better than the other three approaches.