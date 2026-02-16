#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD DATA CHUNKING PIPELINE (Local PC Version)
=================================================
Xử lý XAUUSD.parquet thành chunks nhỏ - MEMORY EFFICIENT VERSION

Chỉ làm:
1. Load parquet từng phần (không load toàn bộ vào RAM)
2. Tính features
3. Tạo sequences
4. Label
5. Lưu chunk xuống thư mục

Không train - để train sau với chunks này!
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from collections import Counter
import json
import gc

print("="*80)
print("  XAUUSD CHUNKING PIPELINE - MEMORY EFFICIENT")
print("="*80)
print()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'parquet_path': os.path.join(SCRIPT_DIR, 'XAUUSD.parquet'),
    'output_dir': os.path.join(SCRIPT_DIR, 'XAUUSD_Chunks'),
    'chunk_rows': 20000,  # Giảm xuống 20K - lưu nhanh hơn, tránh timeout
    'sequence_length': 60,
    'score_threshold': 1.5,  # Balanced threshold
    'max_row_groups': None,  # None = process toàn bộ dataset
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"Parquet: {CONFIG['parquet_path']}")
print(f"Output: {CONFIG['output_dir']}")
print(f"Chunk size: {CONFIG['chunk_rows']:,} rows")
print()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_features(df):
    """Calculate 12 features from bid/ask data"""
    
    # Use 'mid' as price for calculations
    price = df['mid']
    
    # Returns
    df['returns_1s'] = price.pct_change().fillna(0)
    
    # Spread-based range (thay thế hl_range)
    df['spread_norm'] = df['spread'] / price
    df['spread_norm'] = df['spread_norm'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Bid-Ask imbalance (thay body_ratio)
    df['ba_imbalance'] = (df['ask'] - df['bid']) / (df['spread'] + 1e-10)
    df['ba_imbalance'] = df['ba_imbalance'].replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 1)
    
    # Price position relative to range (thay close_position)
    df['price_position'] = 0.5  # Đơn giản hóa vì không có high/low
    
    # Tick count (không có volume, dùng spread activity)
    df['tick_activity'] = np.log1p(1.0 / (df['spread'] + 1e-10))
    df['tick_activity'] = df['tick_activity'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Price imbalance
    mid_dev = price - price.rolling(20, min_periods=1).mean()
    mid_std = price.rolling(20, min_periods=1).std() + 1e-10
    df['price_deviation'] = (mid_dev / mid_std).clip(-3, 3).fillna(0)
    
    # Spread (đã có)
    # df['spread_norm'] đã tính ở trên
    
    # Momentum
    df['m1_momentum'] = price.pct_change(5).fillna(0)
    df['m5_momentum'] = price.pct_change(20).fillna(0)
    
    # Volatility
    df['m1_volatility'] = df['returns_1s'].rolling(5, min_periods=1).std().fillna(0)
    df['m5_volatility'] = df['returns_1s'].rolling(20, min_periods=1).std().fillna(0)
    
    # Relative spread (thay relative_volume)
    avg_spread = df['spread'].rolling(20, min_periods=1).mean()
    df['relative_spread'] = np.where(avg_spread > 0, df['spread'] / avg_spread, 1.0)
    
    # Extract 12 features
    features = df[[
        'returns_1s', 'spread_norm', 'ba_imbalance', 'price_position',
        'tick_activity', 'price_deviation', 'spread_norm',
        'm1_momentum', 'm1_volatility', 'm5_momentum',
        'm5_volatility', 'relative_spread'
    ]].values.astype('float32')
    
    # Clean
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(features, seq_len):
    """Create overlapping sequences"""
    n = len(features) - seq_len + 1
    if n <= 0:
        return np.array([])
    
    sequences = np.zeros((n, seq_len, 12), dtype='float32')
    for i in range(n):
        sequences[i] = features[i:i+seq_len]
    
    return sequences

# ══════════════════════════════════════════════════════════════════════════════
# LABELING
# ══════════════════════════════════════════════════════════════════════════════

def calculate_labels(sequences, threshold=None):
    """Balanced percentile-based labeling"""
    if len(sequences) == 0:
        return np.array([])
    
    n = len(sequences)
    labels = np.ones(n, dtype='int8')  # Default = HOLD (1)
    
    last_step = sequences[:, -1, :]
    
    # Feature indices:
    # 0: returns_1s, 1: spread_norm, 2: ba_imbalance, 3: price_position,
    # 4: tick_activity, 5: price_deviation, 6: spread_norm (dup),
    # 7: m1_momentum, 8: m1_volatility, 9: m5_momentum, 10: m5_volatility, 11: relative_spread
    
    # Bullish score: combination of momentum + position
    bull = (
        last_step[:, 7] * 100 +      # m1_momentum (scaled)
        last_step[:, 9] * 50 +       # m5_momentum (scaled)
        last_step[:, 2] * 10 +       # ba_imbalance
        last_step[:, 5] * 5 +        # price_deviation
        (last_step[:, 3] - 0.5) * 20 # price_position centered
    )
    
    # Bearish score: inverse of bull signals
    bear = -bull
    
    # Volatility adjustment - high vol = toward HOLD
    m1_vol = last_step[:, 8]
    m5_vol = last_step[:, 10]
    avg_vol = (m1_vol + m5_vol / 2)
    vol_threshold = np.percentile(avg_vol, 75)  # Use 75th percentile as "high vol"
    high_vol = avg_vol > vol_threshold
    
    # Use percentile-based labeling for balance (33% each class)
    # Calculate percentiles
    p33_bull = np.percentile(bull, 33)
    p67_bull = np.percentile(bull, 67)
    
    # Assign labels based on ranges
    labels[(bull > p67_bull) & ~high_vol] = 2  # Top 33% = BUY
    labels[(bull < p33_bull) & ~high_vol] = 0  # Bottom 33% = SELL  
    labels[high_vol] = 1                        # High vol = HOLD
    labels[(bull >= p33_bull) & (bull <= p67_bull)] = 1  # Middle 34% = HOLD
    
    return labels

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA - MEMORY EFFICIENT
# ══════════════════════════════════════════════════════════════════════════════

print("Analyzing parquet file...")
try:
    parquet_file = pq.ParquetFile(CONFIG['parquet_path'])
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.num_row_groups
    
    print(f"✓ File info:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Row groups: {num_row_groups}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# PROCESS CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

chunk_files = []
chunk_stats = []
chunk_idx = 0

# Đọc từng row group (memory efficient)
row_groups_to_process = num_row_groups
if CONFIG['max_row_groups'] and CONFIG['max_row_groups'] > 0:
    row_groups_to_process = min(CONFIG['max_row_groups'], num_row_groups)
    print(f"⚠️  Test mode: processing {row_groups_to_process} row group(s) only")
    print()

for row_group_idx in range(row_groups_to_process):
    print(f"Processing Row Group {row_group_idx+1}/{num_row_groups}")
    print("-" * 60)
    
    try:
        # Đọc 1 row group tại 1 thời điểm
        table = parquet_file.read_row_group(row_group_idx)
        df_chunk = table.to_pandas()
        
        print(f"  Loaded: {len(df_chunk):,} rows")
        
        # Features
        features = calculate_features(df_chunk)
        print(f"  Features: {features.shape}")
        
        # Sequences
        sequences = create_sequences(features, CONFIG['sequence_length'])
        if len(sequences) == 0:
            print(f"  ⚠️  No sequences - skip")
            continue
        print(f"  Sequences: {len(sequences):,}")
        
        # Labels
        labels = calculate_labels(sequences, CONFIG['score_threshold'])
        
        # Distribution
        dist = Counter(labels)
        sell_pct = dist.get(0,0)/len(labels)*100
        hold_pct = dist.get(1,0)/len(labels)*100
        buy_pct = dist.get(2,0)/len(labels)*100
        
        print(f"  Labels: SELL={sell_pct:.1f}% HOLD={hold_pct:.1f}% BUY={buy_pct:.1f}%")
        
        # Save - NO compression để tránh timeout
        chunk_name = f'chunk_{chunk_idx:04d}.npz'
        chunk_path = os.path.join(CONFIG['output_dir'], chunk_name)
        
        # Convert float32 -> float16 (giảm 50% size)
        X_f16 = sequences.astype('float16')
        
        # Lưu KHÔNG NÉN (nhanh hơn)
        np.savez(chunk_path, X=X_f16, y=labels)
        
        size_mb = os.path.getsize(chunk_path) / (1024**2)
        print(f"  ✓ Saved: {chunk_name} ({size_mb:.1f} MB)")
        print()
        
        chunk_files.append(chunk_name)
        chunk_stats.append({
            'filename': chunk_name,
            'n_sequences': int(len(sequences)),
            'sell_pct': float(sell_pct),
            'hold_pct': float(hold_pct),
            'buy_pct': float(buy_pct),
            'size_mb': float(size_mb),
        })
        
        chunk_idx += 1
        
        # Free memory
        del df_chunk, features, sequences, labels, table
        gc.collect()
        
    except Exception as e:
        import traceback
        print(f"  ❌ Error processing row group {row_group_idx}: {e}")
        print(traceback.format_exc())
        continue

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("  ✅ CHUNKING COMPLETE")
print("="*80)
print()

if len(chunk_stats) == 0:
    print("❌ No chunks created!")
    exit(1)

total_sequences = sum(s['n_sequences'] for s in chunk_stats)
avg_sell = np.mean([s['sell_pct'] for s in chunk_stats])
avg_hold = np.mean([s['hold_pct'] for s in chunk_stats])
avg_buy = np.mean([s['buy_pct'] for s in chunk_stats])

print(f"Chunks created: {len(chunk_files)}")
print(f"Total sequences: {total_sequences:,}")
print(f"Average distribution: SELL={avg_sell:.1f}% HOLD={avg_hold:.1f}% BUY={avg_buy:.1f}%")
print()

# Save manifest
manifest = {
    'created': datetime.now().isoformat(),
    'source': CONFIG['parquet_path'],
    'total_rows': int(total_rows),
    'chunk_rows': CONFIG['chunk_rows'],
    'sequence_length': CONFIG['sequence_length'],
    'score_threshold': CONFIG['score_threshold'],
    'n_chunks': len(chunk_files),
    'total_sequences': int(total_sequences),
    'chunks': chunk_stats,
}

manifest_path = os.path.join(CONFIG['output_dir'], 'manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest: {manifest_path}")
print()
print(f"All chunks saved to: {CONFIG['output_dir']}")
print()
print("Next step: Train model using these chunks!")
print("="*80)
