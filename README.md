# Fine-tuned Text Summarization using PyTorch and Hugging Face Transformers

This application uses PyTorch and the Hugging Face transformers library to extract important text from a given paragraph and summarize it in minimal text. The model can be fine-tuned on any desired dataset.

## Steps Used

1. Loading data from S3
2. Loading the tokenizer and model with pre-trained weights
3. Creating a custom dataset and dataloader using PyTorch Lightning
4. Creating a custom finetuner using PyTorch Lightning
5. Fine-tuning the model using GPU
6. Building a Flask-based web application


## 🧑‍💻 How to setup
create fresh conda environment 
```python
conda create -p venv python=3.10 -y
```
activate conda environment
```python
conda activate venv/
```
Install requirements
```python
pip install -r requirements.txt
```
Run the web app
```python
python app.py
```
To launch swagger ui
```python
http://localhost:8080/
```

## 🧑‍💻 Tech Used
1. Deep Learning
2. Hugging Face Models
3. Pytorch Lightning
4. Flask
5. Docker
6. GCP(Compute engine)

## 🏭 Industrial Use-cases 
1. Newsletters
2. Social media marketing

## 👋 Conclusion
This text summarization application, which uses PyTorch, Hugging Face transformers, and fine-tuning on custom datasets, has the potential to be used in a variety of real-world scenarios. Some examples include:

* News summarization: automatically summarizing news articles to quickly get the main points
* Business or legal document summarization: to quickly extract the most important information from long reports or contracts
* Social media summarization: to quickly summarize long social media posts or conversations
* Customer feedback summarization: to extract key themes or issues from customer feedback and reviews.

Overall, this application can be useful in any context where time is limited and quickly understanding the main points of large amounts of text is important.
