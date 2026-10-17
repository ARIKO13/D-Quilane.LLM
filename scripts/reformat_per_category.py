"""
Reformat dataset ke per-kategori: kategori_train.jsonl + kategori_validation.jsonl + kategori_stats.json
"""
import json, random, zipfile
from pathlib import Path
from collections import defaultdict

random.seed(42)

# Load all existing data from dataset_v3 + final
all_items = []

# Load from v3
V3_DIR = Path("/home/z/my-project/download/llm/dataset_v3")
for jsonl_file in V3_DIR.rglob("*.jsonl"):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            all_items.append({
                'input': item.get('prompt', item.get('input', '')),
                'output': item.get('response', item.get('output', '')),
                'bidang': item.get('bidang', 'unknown'),
            })

# Load from final (syllogisms, analogies, etc.)
FINAL_DIR = Path("/home/z/my-project/download/llm/dataset_final")
if (FINAL_DIR / 'dataset_train.jsonl').exists():
    for fname in ['dataset_train.jsonl', 'dataset_validation.jsonl']:
        with open(FINAL_DIR / fname, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
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
by_category = defaultdict(list)
for item in unique:
    by_category[item.get('bidang', 'unknown')].append(item)

# Create output
OUTPUT_DIR = Path("/home/z/my-project/download/llm/dataset_per_category")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also create combined
all_train = []
all_val = []

for cat, items in sorted(by_category.items()):
    random.shuffle(items)
    split = int(len(items) * 0.8)
    train = items[:split]
    val = items[split:]
    
    # Sanitize category name for filename
    safe_cat = cat.replace(' ', '_').replace('/', '_')
    
    # Write train
    train_path = OUTPUT_DIR / f"{safe_cat}_train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train:
            f.write(json.dumps({'input': item['input'], 'output': item['output']}, ensure_ascii=False) + '\n')
    
    # Write validation
    val_path = OUTPUT_DIR / f"{safe_cat}_validation.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val:
            f.write(json.dumps({'input': item['input'], 'output': item['output']}, ensure_ascii=False) + '\n')
    
    # Write stats
    stats = {
        'category': cat,
        'total_items': len(items),
        'train_items': len(train),
        'validation_items': len(val),
        'split_ratio': '80/20',
    }
    stats_path = OUTPUT_DIR / f"{safe_cat}_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    all_train.extend(train)
    all_val.extend(val)
    print(f"  {cat}: {len(items)} → train {len(train)} + val {len(val)}")

# Combined files
random.shuffle(all_train)
random.shuffle(all_val)

with open(OUTPUT_DIR / 'dataset_train.jsonl', 'w', encoding='utf-8') as f:
    for item in all_train:
        f.write(json.dumps({'input': item['input'], 'output': item['output']}, ensure_ascii=False) + '\n')

with open(OUTPUT_DIR / 'dataset_validation.jsonl', 'w', encoding='utf-8') as f:
    for item in all_val:
        f.write(json.dumps({'input': item['input'], 'output': item['output']}, ensure_ascii=False) + '\n')

# Combined stats
combined_stats = {
    'total_items': len(unique),
    'train_items': len(all_train),
    'validation_items': len(all_val),
    'split_ratio': '80/20',
    'format': 'jsonl',
    'keys': ['input', 'output'],
    'categories': {cat: len(items) for cat, items in sorted(by_category.items())},
    'total_categories': len(by_category),
    'files_per_category': 3,  # train, validation, stats
}
with open(OUTPUT_DIR / 'dataset_stats.json', 'w', encoding='utf-8') as f:
    json.dump(combined_stats, f, indent=2, ensure_ascii=False)

# ZIP
zip_path = Path("/home/z/my-project/download/llm/dataset.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUTPUT_DIR.iterdir()):
        zf.write(f, f.name)

print(f"\n{'='*60}")
print(f"  DATASET PER-CATEGORY COMPLETE")
print(f"{'='*60}")
print(f"Total: {len(unique)} Q&A pairs")
print(f"Train: {len(all_train)} (80%)")
print(f"Validation: {len(all_val)} (20%)")
print(f"Categories: {len(by_category)}")
print(f"Files: {len(list(OUTPUT_DIR.iterdir()))} ({len(by_category)} cats x 3 + 3 combined)")
print(f"ZIP: {zip_path} ({zip_path.stat().st_size/1024:.1f} KB)")
print(f"\nFiles in ZIP:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")
