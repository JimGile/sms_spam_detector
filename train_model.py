# Import the required dependencies
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Define the sms_classification function
def sms_classification(sms_text_df) -> Pipeline:
    """
    Perform SMS classification using a pipeline with TF-IDF vectorization and Linear Support Vector Classification.

    Parameters:
    - sms_text_df (pd.DataFrame): DataFrame containing 'text_message' and 'label' columns for SMS classification.

    Returns:
    - text_clf (Pipeline): Fitted pipeline model for SMS classification.

    This function takes a DataFrame with 'text_message' and 'label' columns, splits the data into
    training and testing sets, builds a pipeline with TF-IDF vectorization and Linear Support Vector
    Classification, and fits the model to the training data. 
    The fitted pipeline is returned to make future predictions.
    """
    # Set the features variable to the text message.
    X = sms_text_df['text_message']

    # Set the target variable to the "label" column.
    y = sms_text_df['label']

    # Split data into training and testing and set the test_size = 33%
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Build a pipeline to transform the test set to compare to the training set.
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=42)),
    ])

    # Fit the model to the transformed training data and return model.
    text_clf.fit(X_train, y_train)
    return text_clf


def train_and_save_model():
    # Load the dataset into a DataFrame
    sms_text_df = pd.read_csv('Resources/SMSSpamCollection.csv')
    text_clf = sms_classification(sms_text_df)
    with open('sms_text_clf_model.pkl', 'wb') as file:
        # Use pickle.dump to store the pipeline
        pickle.dump(text_clf, file)


# Run the application
if __name__ == "__main__":
    train_and_save_model()
