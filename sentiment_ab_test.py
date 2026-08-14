import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_sentiment_ab_test(weighted_pkl='data/old_dataset/processed/full_dataset_weighted.pkl', unweighted_pkl='data/old_dataset/archive/dataset_unweighted.pkl', output_file='results/sentiment_ab_test.png'):
    print("--- GENERATING SENTIMENT PLOT ---")
    
    df_w = pd.read_pickle(weighted_pkl)
    df_u = pd.read_pickle(unweighted_pkl)
    
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # We now use 'combined_sentiment' instead of 'final_sentiment' due to the Dual-Stream NLP.
    # Normalize to -1 to 1 scale for easy visual comparison.
    df_w['norm_sentiment'] = df_w['combined_sentiment'] / df_w['combined_sentiment'].abs().max()
    df_u['norm_sentiment'] = df_u['combined_sentiment'] / df_u['combined_sentiment'].abs().max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # TOP PANEL: BTC Close Price
    ax1.plot(df_w['date'], df_w['Close'], color='black', linewidth=1.5, label='BTC Close Price (USD)')
    ax1.set_title('A/B Experiment: BTC Price vs. Dual-Stream Sentiment', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # BOTTOM PANEL: Sentiment Comparison
    ax2.plot(df_u['date'], df_u['norm_sentiment'], color='orange', linewidth=1.5, alpha=0.6, label='Variant B (Naive - Unweighted)')
    ax2.plot(df_w['date'], df_w['norm_sentiment'], color='blue', linewidth=2, alpha=0.9, label='Variant A (Intelligent - Weighted)')
    
    # Neutral line (Zero)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.8)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Normalized Sentiment', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Success! Plot saved as: {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_sentiment_ab_test()