import pandas as pd
import re
import fasttext
import os
from huggingface_hub import hf_hub_download

class DataLoader:
    def __init__(self, tweets_file_path, output_file='clean_tweets.csv'):
        """
        Initializes the DataLoader for pure ETL (Extract, Transform, Load) tasks.
        """
        self.tweets_path = tweets_file_path
        self.output_file = output_file
        self.chunk_size = 100_000

        print("--- INITIALIZING FASTTEXT LANGUAGE MODEL ---")
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.lang_model = fasttext.load_model(model_path)

    def is_english(self, text):
        """
        Checks if the provided text is classified as English.
        """
        try:
            clean_text = str(text).replace('\n', ' ').replace('\r', ' ')
            predictions = self.lang_model.predict(clean_text)
            return '__label__en' in predictions[0][0]
        except Exception:
            return False

    def clean_text(self, text):
        """
        Removes URLs, user tags, and extra spaces from tweets.
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_and_save(self):
        """
        Reads a large CSV in chunks, cleans text, filters English,
        and appends the raw, unmerged data directly to a new CSV file.
        """
        print(f"Starting ETL process: {self.tweets_path} -> {self.output_file}")
        
        # Remove the old file if it exists to start fresh
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            print("Removed old output file to start fresh.")

        # LOAD ONLY THE REQUIRED COLUMNS
        chunk_iterator = pd.read_csv(
            self.tweets_path, 
            sep=';', 
            usecols=['user', 'timestamp', 'replies', 'likes', 'retweets', 'text'], 
            chunksize=self.chunk_size, 
            on_bad_lines='skip',
            engine='python'
        )

        total_processed = 0
        total_kept = 0

        for i, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {i+1}...")

            # 1. Drop empty values
            chunk = chunk.dropna(subset=['timestamp', 'text']).copy()
            initial_len = len(chunk)
            total_processed += initial_len
            
            # 2. Text cleaning
            chunk['cleaned_text'] = chunk['text'].apply(self.clean_text)
            
            # 3. Filter English
            mask = chunk['cleaned_text'].apply(self.is_english)
            chunk = chunk[mask].copy()
            
            kept_len = len(chunk)
            dropped = initial_len - kept_len
            total_kept += kept_len
            
            print(f"   -> Dropped {dropped} tweets. Kept: {kept_len} English tweets.")
            
            # 4. Date conversion (Simplifying the timestamp to just Date for easier merging later)
            chunk['date'] = pd.to_datetime(chunk['timestamp'], errors='coerce').dt.tz_localize(None).dt.floor('D')
            chunk = chunk.dropna(subset=['date'])
            
            # 5. Select final columns to save
            final_cols = ['date', 'user', 'cleaned_text', 'likes', 'retweets', 'replies']
            clean_chunk = chunk[final_cols]
            
            # 6. Save incrementally to avoid RAM issues
            clean_chunk.to_csv(self.output_file, mode='a', index=False, header=(i == 0))

        print("\n" + "="*50)
        print("--- ETL COMPLETE ---")
        print(f"Total rows processed: {total_processed}")
        print(f"Total pure English tweets saved: {total_kept}")
        print(f"Data saved to: {self.output_file}")
        print("="*50 + "\n")

if __name__ == "__main__":
    input_file = 'tweets.csv' 
    output_file = 'clean_tweets.csv' 
    
    loader = DataLoader(input_file, output_file)
    loader.process_and_save()