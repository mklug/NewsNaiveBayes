# News naive Bayes

We use the [naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to approach a text classification problem, namely, given the title and a short summary of a news article, we would like to determine the category that the article belongs to (e.g., politics, sports, etc.).  The dataset we will explore is a set of about 210k news articles from the Huffington Post published between 2012 and 2022.  This dataset was prepared by [Rishabh Misra](https://arxiv.org/abs/2209.11429) and is available also on [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset). 
The dataset is in the form of a json file and is about 90mb.

The entire notebook, which creates the figures found in the `figs` directory and saves the model in the `models` directory, takes about 4 minutes to run on my desktop.  A more complete description of the dataset together with an exploration of the model parameters is found in the notebook.  
