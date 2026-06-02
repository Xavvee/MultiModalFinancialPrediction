import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

class NLPProcessor:
    def __init__(self, input_csv, output_pkl, sample_size=100):
        self.input_csv = input_csv
        self.output_pkl = output_pkl
        self.sample_size = sample_size
        
        self.device = torch.device('cpu')
        print("FinBERT Loading FinBERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert").to(self.device)
        self.model.eval()

    def chunk_and_vectorize(self, text):
        """Cutting long text into chunks and generating FinBERT vectors"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.zeros(768)

        # POC Optimization: We're taking only the FIRST 15,000 words of the text for vectorization. This is a practical limit to avoid memory issues and long processing times, especially since some days might have an extremely large number of tweets.
        # In a real-world scenario, we could consider more sophisticated approaches like selecting the most relevant tweets or using a sliding window approach to capture different parts of the text.
        words = text.split()
        if len(words) > 15000:
            words = words[:15000]
        shortened_text = " ".join(words)

        tokens = self.tokenizer.encode(shortened_text, add_special_tokens=False)
        
        # Splitting tokens into chunks of 500 (the max input size for FinBERT)
        chunk_size = 500
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        
        chunk_embeddings = []
        
        with torch.no_grad():
            for chunk in chunks:
                # Changing tokens back to text for the tokenizer, because FinBERT's tokenizer can handle raw text and will add special tokens as needed. This also ensures that we don't lose any important tokenization nuances.
                chunk_text = self.tokenizer.decode(chunk)
                
                inputs = self.tokenizer(
                    chunk_text, 
                    return_tensors='pt',
                    padding='max_length',
                    max_length=512,
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Passing through FinBERT
                outputs = self.model(**inputs)
                
                # Extracting the [CLS] vector from the [batch, sequence, hidden_size] dimension
                cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
                chunk_embeddings.append(cls_embedding)
        
        # Mean pooling over all chunk embeddings to get a single vector for the entire day's text
        if len(chunk_embeddings) > 0:
            return np.mean(chunk_embeddings, axis=0)
        else:
            return np.zeros(768)

    def process(self):
        print(f"Loading data from {self.input_csv}...")
        df = pd.read_csv(self.input_csv)
        
        # Taking only the last 'sample_size' rows for processing to speed up testing. In a real scenario, we would process the entire dataset.
        df = df.tail(self.sample_size).copy()
        df.reset_index(drop=True, inplace=True)
        
        print(f"Generating vectors for {len(df)} days...")
        embeddings = []
        
        for text in tqdm(df['cleaned_text'], desc="Processing Tweets"):
            vec = self.chunk_and_vectorize(text)
            embeddings.append(vec)
            
        df['finbert_vector'] = embeddings
        
        df.to_pickle(self.output_pkl)
        print(f"Saved processed data with vectors to: {self.output_pkl}")

if __name__ == "__main__":
    processor = NLPProcessor(
        input_csv='final_multimodal_dataset.csv', 
        output_pkl='dataset_with_vectors.pkl',
        sample_size=100
    )
    processor.process()