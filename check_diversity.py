from pathlib import Path
import sys

folder = Path(sys.argv[1])
maps = sorted(folder.glob("map_*.txt"))

contents = [p.read_text(encoding="utf-8") for p in maps]
unique_maps = set(contents)

print(f"Total maps     : {len(maps)}")
print(f"Unique maps    : {len(unique_maps)}")

if len(maps) > 0:
    print(f"Diversity rate : {len(unique_maps) / len(maps) * 100:.2f}%")
else:
    print("No map files found.")