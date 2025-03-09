# MachineLearningGuidedLiteratureReview
Using ML to speed the discovery of relevent articles during a literature review

'An hour in the library saves a day in the lab' ~ Unknown

Background: At the beginning of a project researches spend hours reviewing the literature to understand the state of the art. Standard databases offer search tools based on boolean logic find those relevant documents. However because these search terms are single token based they often return far to few or far to many documents.

In the case of too many documents machine learning can speed this review by learning from the Researcher judgement of inscope and out of scope and sorting documents with probable relevancy to the top of the list to read.

Method: A boolean search has returned almost 4000 patents and we have downloaded the title and abstract using code found in my ArticleClassifier Repo. 

These are randomly shuffled and the researcher reviews 5-10% manually marking relevant documents with a '5' possibly relevant with a '3' and not relevant with a '1'

This random walk produced 33 relevant documents. The code in 1.1TextProcessingModelingPrediction.ipynb loads the data base, creates 10 balanced data sets by randomly sampling 33 not relevant documents. We then train 50 neural network models per dataset by utilizing a random search in keras tuner to sample the hyperparameters. The 10 best models for each data set are saved to serve as an ensemble of models for predicting the unreviewed documents.

We look for consensus in the model predictions by summing the predictions for 100 models for each document. We can also calculate an uncertanty score that increases if the ensemble is unsure if a document is relevant or not. This ensemble method is vital because with only 33 relevant documents it is unlikely that any one model will have acceptable performance.

Documents can be sorted based based on the prediciton score to help the researcher find more relevant articles faster. At early stages it is also advisable to reveiw some documents with high uncertanty to improve subsequent retraining.

In the sample data, the researcher reviewed 87 new documents with the highest prediction score of which 86 were relevant. The researcher also reviewed 59 with the highest uncertanty of which 18 were relevant. Finally the researcher reviewed the 44 with the lowest prediciton score and found 1 relevant. These new documents were added to the original reivew and the ensemble was retrained.

Using the provided sample data the review of the sorted predictions
