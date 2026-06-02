import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self, input_file='clean_tweets.csv', output_file='weighted_tweets.csv'):
        """
        Initializes the Feature Engineer to calculate engagement weights.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = 500_000

    def calculate_weights(self):
        """
        Reads clean tweets, calculates the logarithmic engagement weight 
        based on likes, retweets, and replies, and saves to a new CSV.
        """
        print(f"--- STARTING FEATURE ENGINEERING ---")
        print(f"Input: {self.input_file} | Output: {self.output_file}")
        
        # Remove the old output file to prevent appending to old data
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            print("Removed old output file to start fresh.")

        chunk_iterator = pd.read_csv(self.input_file, chunksize=self.chunk_size)
        total_processed = 0

        for i, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {i+1}...")
            
            # Ensure engagement columns are numeric and fill NaNs with 0
            for col in ['likes', 'retweets', 'replies']:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
            
            # THE ALGORITHM: Base formula simulating Twitter's internal heavy ranker
            engagement_score = chunk['likes'] + (20 * chunk['retweets']) + (15 * chunk['replies'])
            
            # Apply natural logarithm: ln(1 + x) to flatten massive viral outliers
            chunk['engagement_weight'] = np.log1p(engagement_score)
            
            # Save the augmented chunk to the new file
            chunk.to_csv(self.output_file, mode='a', index=False, header=(i == 0))
            total_processed += len(chunk)

        print("\n" + "="*50)
        print("--- FEATURE ENGINEERING COMPLETE ---")
        print(f"Total weighted tweets saved: {total_processed}")
        print("="*50 + "\n")

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.calculate_weights()