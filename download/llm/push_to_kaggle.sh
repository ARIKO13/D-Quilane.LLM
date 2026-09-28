#!/bin/bash
# ============================================================
#  D'QUILANE — Push to Kaggle
# ============================================================
# Cara pakai:
#   1. Dapatkan API key dari https://www.kaggle.com/settings
#      -> Create New API Token -> download kaggle.json
#   2. Copy kaggle.json ke ~/.kaggle/kaggle.json
#   3. Run script ini
# ============================================================

set -e

LLM_DIR="/home/z/my-project/download/llm"
KAGGLE_DIR="$HOME/.kaggle"

echo "=== D'QUILANE Push to Kaggle ==="
echo ""

# 1. cek kredensial
if [ ! -f "$KAGGLE_DIR/kaggle.json" ]; then
    echo "[ERROR] kaggle.json belum ada di $KAGGLE_DIR/"
    echo ""
    echo "Cara dapatkan:"
    echo "  1. Buka https://www.kaggle.com/settings"
    echo "  2. Scroll ke 'API' section -> klik 'Create New Token'"
    echo "  3. Download file kaggle.json"
    echo "  4. Jalankan perintah ini (ganti path kalau beda):"
    echo "     mkdir -p $KAGGLE_DIR"
    echo "     cp /path/ke/kaggle.json $KAGGLE_DIR/kaggle.json"
    echo "     chmod 600 $KAGGLE_DIR/kaggle.json"
    exit 1
fi

# 2. set permission
chmod 600 "$KAGGLE_DIR/kaggle.json"
echo "[OK] kaggle.json terdeteksi"

# 3. cek isi username
USERNAME=$(python3 -c "import json; print(json.load(open('$KAGGLE_DIR/kaggle.json'))['username')")
echo "[OK] Username Kaggle: $USERNAME"

# 4. update kernel-metadata.json dengan username yg benar
METADATA="$LLM_DIR/kernel-metadata.json"
python3 -c "
import json
m = json.load(open('$METADATA'))
m['id'] = '$USERNAME/d-quilane-llm-v0-1'
m['title'] = 'd-quilane-llm-v0-1'
json.dump(m, open('$METADATA', 'w'), indent=2)
print('[OK] kernel-metadata.json updated with username:', '$USERNAME')
"

# 5. push!
echo ""
echo "=== Pushing notebook to Kaggle... ==="
cd "$LLM_DIR"
kaggle kernels push -p .

echo ""
echo "=== DONE! ==="
echo "Cek notebook kamu di:"
echo "  https://www.kaggle.com/code/$USERNAME/d-quilane-llm-v0-1"
echo ""
echo "Setelah run, output (mini_llm.pt + tokenizer.json) ada di tab 'Output'."
