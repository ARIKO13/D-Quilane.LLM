"""
Convert dataset to Parquet + ZSTD format
"""
import pyarrow as pa
import pyarrow.parquet as pq
import json, os, zipfile, shutil
from pathlib import Path
from collections import defaultdict

# Load all JSONL from dataset_OVERPOWER
DATA_DIR = Path("/home/z/my-project/download/llm/dataset_OVERPOWER")
all_items = []

for f in DATA_DIR.glob("*_train.jsonl"):
    cat = f.stem.replace("_train", '')
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            item = json.loads(line.strip())
            item['bidang'] = cat
            all_items.append(item)
for f in DATA_DIR.glob("*_validation.jsonl"):
    cat = f.stem.replace("_validation", '')
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            item = json.loads(line.strip())
            item['bidang'] = cat
            all_items.append(item)

# Deduplicate
seen = set()
unique = []
for item in all_items:
    key = item['input']
    if key not in seen:
        seen.add(key)
        unique.append(item)

print(f"Total unique items: {len(unique)}")

# Group by category
by_cat = defaultdict(list)
for item in unique:
    by_cat[item['bidang']].append(item)

# Output directory
OUT = Path("/home/z/my-project/download/llm/dataset_parquet")
OUT.mkdir(parents=True, exist_ok=True)

# Write per-category Parquet files
for cat, items in sorted(by_cat.items()):
    safe = cat.replace(' ', '_')
    
    inputs = [item['input'] for item in items]
    outputs = [item['output'] for item in items]
    bidangs = [item.get('bidang', cat) for item in items]
    
    table = pa.Table.from_arrays(
        [pa.array(inputs, type=pa.string()), 
         pa.array(outputs, type=pa.string()),
         pa.array(bidangs, type=pa.string())],
        names=['input', 'output', 'bidang']
    )
    
    # Write with ZSTD compression
    pq.write_table(table, OUT / f"{safe}.parquet", compression='zstd', compression_level=19)
    print(f"  {cat}: {len(items)} items -> {safe}.parquet ({os.path.getsize(OUT / f'{safe}.parquet')/1024:.1f} KB)")

# Combined train/validation split (80/20)
import random
random.seed(42)
random.shuffle(unique)
split = int(len(unique) * 0.8)
train = unique[:split]
val = unique[split:]

for name, data in [("dataset_train", train), ("dataset_validation", val)]:
    inputs = [item['input'] for item in data]
    outputs = [item['output'] for item in data]
    bidangs = [item.get('bidang', '') for item in data]
    
    table = pa.Table.from_arrays(
        [pa.array(inputs, type=pa.string()),
         pa.array(outputs, type=pa.string()),
         pa.array(bidangs, type=pa.string())],
        names=['input', 'output', 'bidang']
    )
    pq.write_table(table, OUT / f"{name}.parquet", compression='zstd', compression_level=19)
    print(f"  {name}: {len(data)} items -> {os.path.getsize(OUT / f'{name}.parquet')/1024:.1f} KB")

# Stats
stats = {
    'total': len(unique),
    'train': len(train),
    'validation': len(val),
    'categories': {c: len(i) for c, i in sorted(by_cat.items())},
    'total_categories': len(by_cat),
    'format': 'parquet',
    'compression': 'zstd',
    'compression_level': 19,
    'columns': ['input', 'output', 'bidang'],
}
with open(OUT / 'dataset_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

# ZIP
zip_path = Path("/home/z/my-project/download/llm/dataset.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.iterdir()):
        zf.write(f, f.name)

print(f"\n{'='*60}")
print(f"  PARQUET + ZSTD DATASET COMPLETE")
print(f"{'='*60}")
print(f"Total: {len(unique)} Q&A")
print(f"Train: {len(train)} | Val: {len(val)}")
print(f"Categories: {len(by_cat)}")
print(f"Files: {len(list(OUT.iterdir()))}")
print(f"ZIP: {zip_path.stat().st_size/1024:.1f} KB")

# Compare sizes
jsonl_size = sum(f.stat().st_size for f in DATA_DIR.glob("*.jsonl")) / 1024
parquet_size = sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 1024
print(f"\nSize comparison:")
print(f"  JSONL total:     {jsonl_size:.1f} KB")
print(f"  Parquet+ZSTD:    {parquet_size:.1f} KB")
print(f"  Compression ratio: {parquet_size/jsonl_size*100:.1f}% of original")
print(f"  Space saved:      {100 - parquet_size/jsonl_size*100:.1f}%")
