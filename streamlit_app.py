from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import tempfile
import time

import numpy as np
import streamlit as st

try:
    from sb3_contrib import MaskablePPO
except Exception:  # pragma: no cover - handled in UI
    MaskablePPO = None

from dungeon_pcgrl_env import DungeonPCGRLEnv, Tile


TILE_CHARS = {
    int(Tile.WALL): "#",
    int(Tile.FLOOR): ".",
    int(Tile.START): "S",
    int(Tile.GOAL): "G",
}

TILE_LABELS = {
    int(Tile.WALL): "Wall",
    int(Tile.FLOOR): "Floor",
    int(Tile.START): "Start",
    int(Tile.GOAL): "Goal",
}

TILE_COLORS = {
    int(Tile.WALL): "#1f2937",
    int(Tile.FLOOR): "#f3e8c8",
    int(Tile.START): "#16a34a",
    int(Tile.GOAL): "#dc2626",
}


def export_map_text(grid: np.ndarray) -> str:
    lines = []
    for row in grid:
        lines.append("".join(TILE_CHARS.get(int(tile), "#") for tile in row))
    return "\n".join(lines)


def render_map_html(grid: np.ndarray, cell_size: int = 24) -> str:
    rows: List[str] = []
    for row in grid:
        cells: List[str] = []
        for tile in row:
            tile_int = int(tile)
            color = TILE_COLORS.get(tile_int, "#111827")
            char = TILE_CHARS.get(tile_int, "#")
            text = char if char in {"S", "G"} else ""
            cells.append(
                f"<td title='{TILE_LABELS.get(tile_int, 'Tile')}' "
                f"style='width:{cell_size}px;height:{cell_size}px;"
                f"background:{color};text-align:center;vertical-align:middle;"
                f"font-weight:700;color:white;border:1px solid rgba(0,0,0,.25);"
                f"font-family:Arial, sans-serif;font-size:{max(10, cell_size-8)}px;'>"
                f"{text}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table style='border-collapse:collapse;margin:0 auto;'>" + "".join(rows) + "</table>"


def is_quality_map(info: Dict[str, Any], save_only: str) -> bool:
    if save_only == "all":
        return True
    if save_only == "playable":
        return int(info.get("playable", 0)) == 1
    if save_only == "strict":
        return int(info.get("strict_valid", 0)) == 1
    return int(info.get("target_match", info.get("basic_valid", 0))) == 1


@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    if MaskablePPO is None:
        raise RuntimeError("Package sb3-contrib belum terpasang. Jalankan: pip install -r requirements.txt")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    return MaskablePPO.load(model_path)


def load_model_from_upload(uploaded_file):
    if MaskablePPO is None:
        raise RuntimeError("Package sb3-contrib belum terpasang. Jalankan: pip install -r requirements.txt")
    suffix = ".zip"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    return MaskablePPO.load(tmp_path)


def rollout_model(
    model,
    difficulty: str,
    size: int,
    seed: int,
    stochastic: bool,
    strict_stop: bool,
    auto_place_markers: bool,
    auto_place_on_truncation: bool,
    max_steps: int | None,
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    env = DungeonPCGRLEnv(
        width=size,
        height=size,
        target_difficulty=difficulty,
        observation_mode="flat",
        max_steps=max_steps,
        auto_place_markers=auto_place_markers,
        auto_place_on_truncation=auto_place_on_truncation,
        strict_stop=strict_stop,
        seed=seed,
        randomize_initial_position=True,
        randomize_auto_place=True,
    )
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=not stochastic,
            action_masks=env.action_masks(),
        )
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        done = bool(terminated or truncated)

    grid = env.grid.copy()
    env.close()
    return grid, info, total_reward


def rollout_random_baseline(
    difficulty: str,
    size: int,
    seed: int,
    strict_stop: bool,
    auto_place_markers: bool,
    auto_place_on_truncation: bool,
    max_steps: int | None,
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    env = DungeonPCGRLEnv(
        width=size,
        height=size,
        target_difficulty=difficulty,
        observation_mode="flat",
        max_steps=max_steps,
        auto_place_markers=auto_place_markers,
        auto_place_on_truncation=auto_place_on_truncation,
        strict_stop=strict_stop,
        seed=seed,
        randomize_initial_position=True,
        randomize_auto_place=True,
    )
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        mask = env.action_masks()
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            break
        action = int(rng.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    grid = env.grid.copy()
    env.close()
    return grid, info, total_reward


def metric_row(info: Dict[str, Any], total_reward: float) -> Dict[str, Any]:
    return {
        "reward": round(float(total_reward), 2),
        "valid": int(info.get("valid", 0)),
        "playable": int(info.get("playable", 0)),
        "target_match": int(info.get("target_match", 0)),
        "strict_valid": int(info.get("strict_valid", 0)),
        "floor_tiles": int(info.get("floor_tiles", 0)),
        "dead_ends": int(info.get("num_dead_ends", 0)),
        "loops": int(info.get("num_loops", 0)),
        "distance": int(info.get("start_goal_distance", -1)),
        "termination": str(info.get("termination_reason", "unknown")),
        "auto_placed": int(info.get("auto_placed_markers", 0)),
    }


def main() -> None:
    st.set_page_config(page_title="Dungeon Map Generator", layout="wide")
    st.title("Dungeon Map Generator")
    st.caption("Hanya menampilkan hasil dari PCGRL / MaskablePPO dungeon generation yang memenuhi target difficulty yang dipilih")

    with st.sidebar:
        st.header("Input User")
        difficulty_label = st.radio(
            "Tingkat kesulitan",
            ["Easy", "Medium", "Hard"],
            index=0,
            horizontal=True,
        )

        difficulty = difficulty_label.lower()
        n_maps = st.slider(
            "Jumlah map yang ingin dibuat",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )

        # Seed disembunyikan dari user. Nilainya dibuat otomatis agar setiap klik generate
        # dapat menghasilkan variasi map tanpa membingungkan user umum/dosen.
        base_seed = int(time.time() * 1000) % 1_000_000

        # Konfigurasi teknis disembunyikan dari user agar web lebih sederhana.
        # Nilai ini mengikuti konfigurasi evaluasi yang sudah terbukti berhasil.
        size = 16
        stochastic = True
        mode = "Trained RL model"
        model_path = f"models/ppo_{difficulty}.zip"
        save_only = "target"
        strict_stop = False
        auto_place_markers = True
        auto_place_on_truncation = True
        max_steps = None


    generate = st.button("Generate Dungeon", type="primary")

    if not generate:
        st.info("Pilih input di sidebar, lalu klik Generate Dungeon.")
        return

    try:
        model = load_model_from_path(str(model_path))
    except Exception as exc:
        st.error(f"Gagal load model: {exc}")
        st.info("Pastikan file model tersedia di folder models/ dengan nama ppo_easy.zip, ppo_medium.zip, dan ppo_hard.zip.")
        return

    results = []
    for idx in range(n_maps):
        current_seed = int(base_seed) + idx
        try:
            grid, info, total_reward = rollout_model(
                model=model,
                difficulty=difficulty,
                size=int(size),
                seed=current_seed,
                stochastic=stochastic,
                strict_stop=strict_stop,
                auto_place_markers=auto_place_markers,
                auto_place_on_truncation=auto_place_on_truncation,
                max_steps=max_steps,
            )
        except Exception as exc:
            st.error(f"Generate map ke-{idx + 1} gagal: {exc}")
            continue

        if is_quality_map(info, save_only):
            results.append((current_seed, grid, info, total_reward))

    if not results:
        st.warning("Belum ada map yang lolos kriteria target. Coba klik Generate lagi atau pastikan model yang digunakan sudah sesuai dengan tingkat kesulitan.")
        return

    rows = [metric_row(info, reward) for _, _, info, reward in results]
    st.subheader("Ringkasan Metrik")
    st.dataframe(rows, use_container_width=True)

    st.subheader("Generated Maps")
    for i, (seed_value, grid, info, total_reward) in enumerate(results, start=1):
        with st.expander(f"Map {i} | target_match={int(info.get('target_match', 0))}", expanded=(i == 1)):
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown(render_map_html(grid), unsafe_allow_html=True)
            with col2:
                map_text = export_map_text(grid)
                st.download_button(
                    label="Download map .txt",
                    data=map_text,
                    file_name=f"dungeon_{difficulty}_{size}x{size}_map{i}.txt",
                    mime="text/plain",
                    key=f"download-{i}",
                )
                st.code(map_text, language="text")


if __name__ == "__main__":
    main()
