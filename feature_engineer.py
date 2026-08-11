import pandas as pd
import numpy as np
import os
import csv

class FeatureEngineer:
    def __init__(self, input_file='data/new_dataset/interim/clean_tweets_2021_23.csv', output_file='data/new_dataset/interim/weighted_tweets_2021_23.csv'):
        self.input_file = input_file
        self.output_file = output_file

    def calculate_authority_weights(self):
        print(f"--- STARTING ROBUST AUTHORITY ENGINEERING ---")

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.input_file, 'r', encoding='utf-8') as f_in, \
             open(self.output_file, 'w', encoding='utf-8', newline='') as f_out:
            
            reader = csv.reader(f_in)
            header = next(reader)
            # Dodajemy nową kolumnę do nagłówka
            f_out.write(",".join(header) + ",engagement_weight\n")
            
            writer = csv.writer(f_out)
            
            count = 0
            for row in reader:
                # row[0]: date, row[1]: user_name, row[2]: cleaned_text, row[3]: followers, row[4]: verified
                if len(row) < 5: continue
                
                try:
                    followers = float(row[3])
                    verified = (row[4].strip().lower() == 'true')
                    
                    # Logika wielorybów
                    weight = 5.0 if (verified and followers > 100000) else 1.0
                    
                    writer.writerow(row + [weight])
                    count += 1
                except ValueError:
                    continue
                    
        print(f"--- ENGINEERING COMPLETE: Processed {count} rows. ---")

if __name__ == "__main__":
    engineer = FeatureEngineer(input_file='data/new_dataset/interim/clean_tweets_2021_23.csv', output_file='data/new_dataset/interim/weighted_tweets_2021_23.csv')
    engineer.calculate_authority_weights()