from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from dungeon_pcgrl_env import DungeonPCGRLEnv
from train import recommended_max_steps


def make_stage_env(args: argparse.Namespace, difficulty: str) -> Monitor:
    env = DungeonPCGRLEnv(
        width=args.size,
        height=args.size,
        target_difficulty=difficulty,
        observation_mode=args.observation_mode,
        metadata_csv=args.metadata_csv,
        max_steps=args.max_steps or recommended_max_steps(args.size, difficulty),
        auto_place_markers=args.auto_place_markers,
        auto_place_on_truncation=args.auto_place_on_truncation,
        strict_stop=args.strict_stop,
        initial_room_size=args.initial_room_size,
        seed=args.seed,
    )
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum training for medium/hard DungeonPCGRL.")
    parser.add_argument("--target", choices=["medium", "hard"], default="hard")
    parser.add_argument("--size", type=int, choices=[16, 24], default=16)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--initial-room-size", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--observation-mode", choices=["flat", "dict"], default="flat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models_curriculum")
    parser.add_argument("--easy-steps", type=int, default=250_000)
    parser.add_argument("--medium-steps", type=int, default=600_000)
    parser.add_argument("--hard-steps", type=int, default=900_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--auto-place-markers", action="store_true", help="Allow automatic START/GOAL placement at terminal time.")
    parser.add_argument("--auto-place-on-truncation", action="store_true", help="Allow automatic START/GOAL placement when episode is truncated.")
    parser.add_argument("--strict-stop", action="store_true", help="Require dead-end and loop ranges before STOP.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stages = ["easy", "medium"] if args.target == "medium" else ["easy", "medium", "hard"]
    stage_steps = {
        "easy": args.easy_steps,
        "medium": args.medium_steps,
        "hard": args.hard_steps,
    }

    model = None
    policy = "MlpPolicy" if args.observation_mode == "flat" else "MultiInputPolicy"

    for stage in stages:
        env = make_stage_env(args, stage)
        if model is None:
            model = MaskablePPO(
                policy,
                env,
                verbose=1,
                seed=args.seed,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                learning_rate=args.learning_rate,
                clip_range=0.20,
            )
        else:
            model.set_env(env)

        print(f"=== Training stage: {stage} for {stage_steps[stage]} timesteps ===")
        model.learn(
            total_timesteps=stage_steps[stage],
            reset_num_timesteps=False,
            progress_bar=True,
        )
        stage_path = save_dir / f"ppo_dungeon_{stage}_{args.size}x{args.size}_curriculum.zip"
        model.save(stage_path)
        env.close()
        print(f"Saved stage model to {stage_path}")

    final_path = save_dir / f"ppo_dungeon_{args.target}_{args.size}x{args.size}_final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
