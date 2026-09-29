# D'QUILANE Local Trainer

Run D'QUILANE training di GPU kamu (RTX 3050 atau CUDA GPU lainnya).

## Cara Pakai (Windows)

1. **Download semua file** ke satu folder, mis `C:\dquilane`
2. **Klik 2x `setup.bat`** — one-time install (Python deps + PyTorch CUDA)
3. **Klik 2x `train.bat`** — train model di GPU
4. **Klik 2x `serve.bat`** — start local web UI di http://localhost:8000

## Yang Terjadi Saat Train

- GPU kamu kepake buat training (VRAM terisi ~1-2 GB)
- Setelah training selesai, GPU otomatis freed (`torch.cuda.empty_cache()`)
- RAM & GPU VRAM balik ke kondisi normal

## Spesifikasi

- Architecture: d=192, n_layers=16, n_heads=8 (parameter sharing)
- Params: ~700K unique, ~11M effective
- Training time: ~5-15 menit (GPU T4/3050)
- File output: web/dquilane_model.json (~10 MB)

## Troubleshooting

- **OOM (out of memory)**: edit dquilane_local.py, turunkan batch_size dari 32 ke 16
- **CUDA gak available**: pastikan NVIDIA driver + CUDA Toolkit terinstall
- **Python gak ada**: download dari python.org, centang "Add to PATH"
