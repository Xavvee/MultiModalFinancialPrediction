import pandas as pd
import re
import fasttext
import os
from huggingface_hub import hf_hub_download
import csv

class DataLoader:
    def __init__(self, tweets_file_path, output_file='clean_tweets.csv'):
        """
        Initializes the DataLoader for pure ETL (Extract, Transform, Load) tasks.
        Adapted for the 2025-2026 Kaggle dataset focusing on User Authority.
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

    import csv

    def process_and_save(self):
        print(f"Starting Robust ETL (Manual Parsing): {self.tweets_path}")
        
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        # Otwieramy plik w sposób absolutnie bezpieczny
        with open(self.tweets_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(self.output_file, 'w', encoding='utf-8') as f_out:
            
            # Zapisujemy nagłówek
            f_out.write("date,user_name,cleaned_text,user_followers,user_verified\n")
            
            reader = csv.reader(f_in)
            next(reader)  # Pomiń oryginalny nagłówek
            
            count = 0
            for row in reader:
                try:
                    # Wiemy, że date jest w kolumnie 8, user w 0, text w 9, followers 4, verified 7
                    # Jeśli wiersz jest za krótki, pomijamy
                    if len(row) < 10: continue
                    
                    date_val = row[8]
                    user_val = row[0]
                    text_val = row[9] 
                    followers_val = row[4]
                    verified_val = row[7]
                    
                    # Czyszczenie
                    clean_txt = self.clean_text(text_val)
                    if not self.is_english(clean_txt): continue
                    
                    # Zapisujemy po przecinku, dbając o cudzysłowy w tekście
                    f_out.write(f"{date_val},{user_val},\"{clean_txt}\",{followers_val},{verified_val}\n")
                    count += 1
                    if count % 10000 == 0: print(f"Processed {count} rows...")
                    
                except Exception:
                    continue
        print(f"ETL Done! Saved {count} rows.")

if __name__ == "__main__":
    # UPDATED FILENAME FOR THE NEW DATASET
    input_file = 'bitcoin_tweets_2025_26.csv' 
    output_file = 'clean_tweets_2025_26.csv' 
    
    loader = DataLoader(input_file, output_file)
    loader.process_and_save()