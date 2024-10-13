# Import the required dependencies
from sklearn.pipeline import Pipeline
import gradio as gr
import pickle


# Load the text classification model pipeline
with open('./sms_text_clf_model.pkl', 'rb') as file:
    clf: Pipeline = pickle.load(file)


# Define the sms_prediction function
def sms_prediction(text) -> tuple[str, bool]:
    """
    Predict the spam/ham classification of a given text message using a pre-trained model.

    Parameters:
    - text (str): The text message to be classified.

    Returns: a tuple of the following:
    - str: A message indicating whether the text message is classified as spam or not.
    - bool: A boolean value indicating whether the text message is classified as spam or not.

    This function takes a text message and a pre-trained pipeline model, then predicts the
    spam/ham classification of the text. The result is a message stating whether the text is
    classified as spam or not.
    """
    # Create a variable that will hold the prediction of a new text.
    prediction = clf.predict([text])

    # Using a conditional if the prediction is "ham" return the message:
    # f'The text message: "{text}", is not spam.' Else, return f'The text message: "{text}", is spam.'
    if prediction == 'ham':
        return (f'The text message: "{text}"\n, is not spam.', False)
    else:
        return (f'The text message: "{text}"\n, is spam.', True)


# Define the gradio interface
def create_gradio_app():
    sms_app = gr.Interface(
        sms_prediction,
        gr.Textbox(
            label="What is the SMS Text message you want to test?",
            placeholder="Enter text here..."
        ),
        [
            gr.Textbox(
                label="The app has determined that:",
                placeholder="Prediction will show here..."
            ),
            gr.Checkbox(
                label="Spam?"
            )
        ],
        title="SMS Text Spam Detector",
        description="Enter an SMS text message and our app will determine if it is spam or not",
        examples=[
            "You are a lucky winner of $5000!",
            "You won 2 free tickets to the Super Bowl.",
            "You won 2 free tickets to the Super Bowl. Text us to claim your prize.",
            "Thanks for registering. Text 4343 to receive free updates on medicare.",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2000. Text FA to 87121 to receive entry.",
        ],
        cache_examples=False
    )
    return sms_app


# Run the application
if __name__ == "__main__":
    sms_app = create_gradio_app()
    sms_app.launch()
