from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from dungeon_pcgrl_env import DungeonPCGRLEnv


def recommended_max_steps(size: int, difficulty: str) -> int:
    if size == 16:
        return {"easy": 240, "medium": 300, "hard": 360}[difficulty]
    return {"easy": 480, "medium": 600, "hard": 720}[difficulty]


def make_env(args: argparse.Namespace) -> Monitor:
    max_steps = args.max_steps or recommended_max_steps(args.size, args.difficulty)
    env = DungeonPCGRLEnv(
        width=args.size,
        height=args.size,
        target_difficulty=args.difficulty,
        observation_mode=args.observation_mode,
        metadata_csv=args.metadata_csv,
        max_steps=max_steps,
        auto_place_markers=args.auto_place_markers,
        auto_place_on_truncation=args.auto_place_on_truncation,
        strict_stop=args.strict_stop,
        initial_room_size=args.initial_room_size,
        seed=args.seed,
        randomize_initial_position=not args.fixed_reset,
        randomize_auto_place=not args.fixed_auto_place,
        auto_place_top_k=args.auto_place_top_k,
    )
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO for DungeonPCGRLEnv.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--size", type=int, choices=[16, 24], default=16)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--initial-room-size", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--observation-mode", choices=["flat", "dict"], default="flat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--auto-place-markers", action="store_true", help="Allow automatic START/GOAL placement at terminal time. Use for demos, not for pure academic training.")
    parser.add_argument("--auto-place-on-truncation", action="store_true", help="Allow automatic START/GOAL placement when episode is truncated.")
    parser.add_argument("--strict-stop", action="store_true", help="Require dead-end and loop ranges before STOP. Default requires only playable + floor/distance target match.")
    parser.add_argument("--fixed-reset", action="store_true", help="Use fixed center initial room. Default uses randomized initial room per episode.")
    parser.add_argument("--fixed-auto-place", action="store_true", help="Use deterministic START/GOAL auto-placement. Default randomizes tie-breaking.")
    parser.add_argument("--auto-place-top-k", type=int, default=8, help="Top candidate START/GOAL pairs used for randomized auto-placement.")
    args = parser.parse_args()

    # Medium/hard butuh horizon dan eksplorasi lebih besar. Default lama 200k
    # terlalu kecil dan sering berakhir pada 0% valid rate.
    if args.timesteps is None:
        args.timesteps = {"easy": 300_000, "medium": 700_000, "hard": 1_000_000}[args.difficulty]

    env = make_env(args)
    policy = "MlpPolicy" if args.observation_mode == "flat" else "MultiInputPolicy"
    model = MaskablePPO(
        policy,
        env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        clip_range=0.20,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    save_path = args.save_path or f"ppo_dungeon_{args.difficulty}_{args.size}x{args.size}_revisi.zip"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    env.close()
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
