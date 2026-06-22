from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sb3_contrib import MaskablePPO

from dungeon_pcgrl_env import DungeonPCGRLEnv


def export_clean_map(env):
    tile_chars = {
        0: "#",  # WALL
        1: ".",  # FLOOR
        2: "S",  # START
        3: "G",  # GOAL
    }

    lines = []
    for row in env.grid:
        line = "".join(tile_chars.get(int(tile), "#") for tile in row)
        lines.append(line)

    return "\n".join(lines)


def is_quality_map(info, save_only: str = "target") -> bool:
    """Filter for exported maps.

    all       : save every generated map
    playable  : save maps with connected START-GOAL path
    target    : save maps matching paper-level targets: floor_tiles + distance
    strict    : save maps also matching dead-end and loop ranges
    """
    if save_only == "all":
        return True
    if save_only == "playable":
        return int(info.get("playable", 0)) == 1
    if save_only == "strict":
        return int(info.get("strict_valid", 0)) == 1
    return int(info.get("target_match", info.get("basic_valid", 0))) == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DungeonPCGRL MaskablePPO model.")
    parser.add_argument("model_path")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--size", type=int, choices=[16, 24], default=16)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--render-maps-dir", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy for ablation only. Default evaluation is deterministic.")
    parser.add_argument("--save-only", choices=["strict", "target", "playable", "all"], default="target")
    parser.add_argument("--auto-place-markers", action="store_true", help="Allow automatic START/GOAL placement at terminal time. Use this only for demo-assisted evaluation.")
    parser.add_argument("--auto-place-on-truncation", action="store_true", help="Allow automatic START/GOAL placement when max_steps/stagnation truncates the episode.")
    parser.add_argument("--strict-stop", action="store_true", help="Require dead-end and loop ranges before STOP.")
    parser.add_argument("--fixed-reset", action="store_true", help="Use fixed center initial room. Default uses randomized initial room per episode.")
    parser.add_argument("--fixed-auto-place", action="store_true", help="Use deterministic START/GOAL auto-placement. Default randomizes tie-breaking.")
    parser.add_argument("--auto-place-top-k", type=int, default=8, help="Top candidate START/GOAL pairs used for randomized auto-placement.")
    args = parser.parse_args()

    env = DungeonPCGRLEnv(
        width=args.size,
        height=args.size,
        target_difficulty=args.difficulty,
        metadata_csv=args.metadata_csv,
        observation_mode="flat",
        auto_place_markers=args.auto_place_markers,
        auto_place_on_truncation=args.auto_place_on_truncation,
        strict_stop=args.strict_stop,
        seed=args.seed,
        randomize_initial_position=not args.fixed_reset,
        randomize_auto_place=not args.fixed_auto_place,
        auto_place_top_k=args.auto_place_top_k,
    )
    model = MaskablePPO.load(args.model_path)

    output_dir = Path(args.render_maps_dir) if args.render_maps_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    reason_counts: dict[str, int] = {}
    total_reward = 0.0
    saved_count = 0

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=not args.stochastic,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            done = terminated or truncated

        total_reward += ep_reward
        reason = str(info.get("termination_reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        row = {
            "episode": episode + 1,
            "reward": ep_reward,
            "difficulty": args.difficulty,
            "valid": int(info.get("valid", 0)),
            "structural_valid": int(info.get("structural_valid", info.get("valid", 0))),
            "playable": int(info.get("playable", 0)),
            "target_match": int(info.get("target_match", info.get("basic_valid", 0))),
            "strict_valid": int(info.get("strict_valid", 0)),
            "floor_tiles": int(info.get("floor_tiles", 0)),
            "dead_ends": int(info.get("num_dead_ends", 0)),
            "loops": int(info.get("num_loops", 0)),
            "distance": int(info.get("start_goal_distance", -1)),
            "auto_placed": int(info.get("auto_placed_markers", 0)),
            "termination_reason": reason,
        }
        metric_rows.append(row)

        print(
            f"Episode {episode + 1:03d}: reward={ep_reward:.2f}, "
            f"valid={row['valid']}, playable={row['playable']}, "
            f"target_match={row['target_match']}, strict_valid={row['strict_valid']}, "
            f"reason={reason}, metrics="
            f"floor={row['floor_tiles']}, dead_ends={row['dead_ends']}, "
            f"loops={row['loops']}, distance={row['distance']}, "
            f"auto_place={row['auto_placed']}"
        )

        if output_dir and is_quality_map(info, args.save_only):
            saved_count += 1
            path = output_dir / f"map_{saved_count:03d}.txt"
            path.write_text(export_clean_map(env), encoding="utf-8")

    n = max(args.episodes, 1)
    valid_count = sum(r["valid"] for r in metric_rows)
    playable_count = sum(r["playable"] for r in metric_rows)
    target_count = sum(r["target_match"] for r in metric_rows)
    strict_count = sum(r["strict_valid"] for r in metric_rows)
    floor_total = sum(r["floor_tiles"] for r in metric_rows)
    positive_distances = [r["distance"] for r in metric_rows if r["distance"] > 0]
    mean_distance = sum(positive_distances) / max(len(positive_distances), 1)

    print("---")
    print(f"Average reward: {total_reward / n:.2f}")
    print(f"Valid rate: {valid_count}/{args.episodes} = {100 * valid_count / n:.2f}%")
    print(f"Playable rate: {playable_count}/{args.episodes} = {100 * playable_count / n:.2f}%")
    print(f"Target match rate: {target_count}/{args.episodes} = {100 * target_count / n:.2f}%")
    print(f"Strict-valid rate: {strict_count}/{args.episodes} = {100 * strict_count / n:.2f}%")
    print(f"Mean floor tiles: {floor_total / n:.2f}")
    print(f"Mean start-goal distance: {mean_distance:.2f}")
    print(f"Termination reasons: {reason_counts}")
    print(f"Saved clean maps: {saved_count}")

    if args.metrics_csv:
        csv_path = Path(args.metrics_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Saved metrics CSV to {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
