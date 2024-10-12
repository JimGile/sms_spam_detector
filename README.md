DU-VIRT-AI-PT-05-2024-U-LOLC-MWTH - Module 21 Challenge (Oct 12, 2024)

# Module 21 Challenge - SMS Classification

[notebooks/gradio_sms_text_classification.ipynb](https://github.com/JimGile/sms_spam_detector/blob/main/notebooks/gradio_sms_text_classification.ipynb)

## Focus

Module 21 focuses on the evolution of Natural Language Processing (NLP), including the increasing sophistication of language models and their integration with user-friendly interfaces like Gradio.

## History of Natural Language Processing

- **1950 - 1970**

  Early NLP models relied heavily on rule-based systems, like ELIZA, which, while groundbreaking, proved unscalable.

- **1990**

  The 1990s ushered in the era of statistical NLP, leveraging machine learning techniques like support vector machines and decision trees. This shift facilitated data-driven approaches to language processing, enabling predictions on unseen data for tasks such as part-of-speech tagging and named entity recognition.

- **2000**

  The arrival of deep learning and neural networks in the 2000s revolutionized NLP. Models like RNNs and CNNs made significant strides in areas like machine translation and sentiment analysis. However, these models came with challenges like high computational costs, the need for vast training data, and the “exploding or vanishing gradient problem."

- **2010**

  The introduction of LSTM networks in 2007 addressed the gradient problem inherent in RNNs, while the concept of "attention" in 2014 enabled models to develop internal representations of word relationships. This paved the way for the transformer model in 2017, which utilized attention mechanisms as a core component and became the foundation for many state-of-the-art NLP models.

- **2020**

  The late 2010s and 2020s witnessed the rise of modern Large Language Models (LLMs) and Generative Pre-trained Transformer (GPT) models. These models, trained on massive text datasets, demonstrated remarkable abilities in generating human-like text and performing various language tasks, showcasing the power of scaling up model size for training.

## Key Concepts

- **Tokenization:** the process of breaking down text into smaller units (tokens) for processing by language models.
- **Embeddings:** a mathematical representation of text that captures its meaning.
- **Similarity Measures:** used to assess the semantic closeness of different text pieces based on their embeddings.
- **Hugging Face:** a platform hosting various pre-trained transformer models, which are highlighted for its ease of use in NLP tasks.
- **Gradio:** a tool for building user-friendly interfaces for AI applications. It allows for creating interactive web-based demos, making complex models accessible to a wider audience.

## Challenge

The challenge is to refactor code from an SMS text classification solution into a function that constructs a linear Support Vector Classification (SVC) model. Once the model is created and trained, create a Gradio app to host the application and enable users to test text messages. The application will provide feedback to users, indicating whether the text is classified as spam or not, based on the model's performance.

## Solution

The solution is in the Jupyter Notebook file [notebooks/gradio_sms_text_classification.ipynb](https://github.com/JimGile/sms_spam_detector/blob/main/notebooks/gradio_sms_text_classification.ipynb).
