# Dungeon PCGRL Environment - Fixed Version

Versi ini memperbaiki masalah utama pada environment RL dungeon generation.

## Perubahan utama

1. `DEFAULT_TARGET_RANGES` sekarang mengikuti statistik `dungeon_metadata.csv`.
2. Reward tidak lagi memberi bonus floor tanpa batas; floor yang melewati target atas diberi penalti.
3. `target_delta` sekarang memberi reward saat membaik dan penalti saat memburuk.
4. `STOP` menjadi ketat secara default lewat `strict_stop=True`.
5. Metrik `valid` sekarang berarti map playable dan semua target inti berada dalam range.
6. Ditambahkan `basic_valid` untuk map playable yang hanya memenuhi floor dan jarak Start-Goal.
7. Auto-place marker tidak aktif pada training script dan tidak berjalan saat episode ter-truncate, kecuali `auto_place_on_truncation=True`.
8. Ditambahkan `action_masks()` agar kompatibel dengan `sb3-contrib` MaskablePPO.
9. BFS memakai `collections.deque`.
10. Project dilengkapi `requirements.txt`, `.gitignore`, `train.py`, `evaluate.py`, dan smoke test.

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training contoh

```bash
python train.py --difficulty easy --size 16 --timesteps 200000
```

Untuk memakai metadata CSV eksternal:

```bash
python train.py --difficulty easy --size 16 --metadata-csv ../dungeon_dataset_100/dungeon_metadata.csv
```

## Evaluasi contoh

```bash
python evaluate.py ppo_dungeon_easy_16x16_fixed.zip --difficulty easy --size 16 --episodes 20 --render-maps-dir generated_maps
```

## Catatan penting

Model lama yang sudah dilatih dengan reward/target lama sebaiknya dilatih ulang, karena distribusi target dan reward sekarang berubah cukup besar.
