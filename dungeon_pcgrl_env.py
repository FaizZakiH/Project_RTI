
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import csv

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:  # pragma: no cover
        class _SimpleEnv:
            metadata = {}

            def reset(self, *, seed=None, options=None):
                return None

        class _Discrete:
            def __init__(self, n: int):
                self.n = int(n)

            def sample(self) -> int:
                return int(np.random.randint(self.n))

        class _Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = low
                self.high = high
                self.shape = tuple(shape)
                self.dtype = dtype

        class _Dict:
            def __init__(self, spaces_dict):
                self.spaces = dict(spaces_dict)

        class _SpacesNamespace:
            Discrete = _Discrete
            Box = _Box
            Dict = _Dict

        class _GymNamespace:
            Env = _SimpleEnv

        gym = _GymNamespace()  # type: ignore
        spaces = _SpacesNamespace()  # type: ignore


class DungeonAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    CARVE_FLOOR = 4
    PLACE_WALL = 5
    PLACE_START = 6
    PLACE_GOAL = 7
    STOP = 8


class Tile(IntEnum):
    WALL = 0
    FLOOR = 1
    START = 2
    GOAL = 3


DEFAULT_TARGET_RANGES: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]] = {
    # Target ranges aligned with the paper table for 16x16 maps:
    # Easy ~= 21 walkable tiles and distance ~= 12
    # Medium ~= 42 walkable tiles and distance ~= 18
    # Hard ~= 40 walkable tiles and distance ~= 24
    # Dead ends and loops are retained as optional strict-quality indicators,
    # not as the main validity criteria.
    16: {
        "easy": {
            "floor_tiles": (18, 26),
            "num_dead_ends": (1, 5),
            "num_loops": (0, 6),
            "start_goal_distance": (10, 14),
        },
        "medium": {
            "floor_tiles": (35, 50),
            "num_dead_ends": (2, 7),
            "num_loops": (1, 10),
            "start_goal_distance": (15, 21),
        },
        "hard": {
            "floor_tiles": (35, 48),
            "num_dead_ends": (3, 9),
            "num_loops": (0, 8),
            "start_goal_distance": (22, 28),
        },
    },
    # Scaled targets for 24x24 are experimental. Use 16x16 for reproducing
    # the reported paper table.
    24: {
        "easy": {
            "floor_tiles": (45, 70),
            "num_dead_ends": (2, 8),
            "num_loops": (0, 12),
            "start_goal_distance": (18, 28),
        },
        "medium": {
            "floor_tiles": (80, 120),
            "num_dead_ends": (4, 12),
            "num_loops": (2, 18),
            "start_goal_distance": (30, 45),
        },
        "hard": {
            "floor_tiles": (80, 130),
            "num_dead_ends": (5, 16),
            "num_loops": (0, 16),
            "start_goal_distance": (46, 70),
        },
    },
}

@dataclass
class RewardConfig:
    step_penalty: float = -0.02
    invalid_move_penalty: float = -0.10
    carve_new_floor_reward: float = 0.45
    carve_existing_floor_penalty: float = -0.05
    place_wall_from_floor_penalty: float = -0.08
    remove_marker_penalty: float = -0.80
    first_marker_reward: float = 0.10
    reposition_marker_reward: float = -0.20
    explore_new_cursor_reward: float = 0.02
    frontier_cursor_reward: float = 0.03
    connectivity_improve_reward: float = 0.05
    connectivity_worse_penalty: float = -0.20
    first_playable_bonus: float = 0.00
    path_improve_bonus: float = 0.15
    target_delta_scale: float = 0.60
    target_regression_scale: float = 0.40
    floor_progress_scale: float = 0.25
    floor_overflow_penalty_scale: float = -0.12
    valid_terminal_reward: float = 10.0
    target_match_terminal_reward: float = 45.0
    auto_place_terminal_reward_scale: float = 0.35
    invalid_terminal_penalty: float = -4.0
    premature_stop_penalty: float = -0.50
    stagnation_penalty: float = -1.50


def make_reward_config_for_difficulty(difficulty: str) -> RewardConfig:
    """Difficulty-aware reward configuration.

    The values below are tuned for the compact 16x16 targets used in the paper.
    They reduce the old over-carving tendency and make distance/path quality more
    important, especially for the hard difficulty.
    """
    difficulty = str(difficulty).lower()

    if difficulty == "easy":
        return RewardConfig(
            step_penalty=-0.02,
            invalid_move_penalty=-0.10,
            carve_new_floor_reward=0.45,
            carve_existing_floor_penalty=-0.05,
            place_wall_from_floor_penalty=-0.08,
            remove_marker_penalty=-0.80,
            first_marker_reward=0.20,
            reposition_marker_reward=-0.10,
            explore_new_cursor_reward=0.01,
            frontier_cursor_reward=0.02,
            connectivity_improve_reward=0.10,
            connectivity_worse_penalty=-0.20,
            first_playable_bonus=1.00,
            path_improve_bonus=0.20,
            target_delta_scale=0.80,
            target_regression_scale=0.45,
            floor_progress_scale=0.25,
            floor_overflow_penalty_scale=-0.25,
            valid_terminal_reward=10.0,
            target_match_terminal_reward=45.0,
            auto_place_terminal_reward_scale=0.35,
            invalid_terminal_penalty=-4.0,
            premature_stop_penalty=-1.00,
            stagnation_penalty=-3.00,
        )

    if difficulty == "medium":
        return RewardConfig(
            step_penalty=-0.025,
            invalid_move_penalty=-0.12,
            carve_new_floor_reward=0.50,
            carve_existing_floor_penalty=-0.05,
            place_wall_from_floor_penalty=-0.08,
            remove_marker_penalty=-0.90,
            first_marker_reward=0.25,
            reposition_marker_reward=-0.12,
            explore_new_cursor_reward=0.01,
            frontier_cursor_reward=0.02,
            connectivity_improve_reward=0.10,
            connectivity_worse_penalty=-0.22,
            first_playable_bonus=1.25,
            path_improve_bonus=0.30,
            target_delta_scale=1.00,
            target_regression_scale=0.50,
            floor_progress_scale=0.25,
            floor_overflow_penalty_scale=-0.30,
            valid_terminal_reward=10.0,
            target_match_terminal_reward=55.0,
            auto_place_terminal_reward_scale=0.35,
            invalid_terminal_penalty=-5.0,
            premature_stop_penalty=-1.20,
            stagnation_penalty=-3.00,
        )

    if difficulty == "hard":
        return RewardConfig(
            step_penalty=-0.03,
            invalid_move_penalty=-0.14,
            carve_new_floor_reward=0.45,
            carve_existing_floor_penalty=-0.06,
            place_wall_from_floor_penalty=-0.08,
            remove_marker_penalty=-1.00,
            first_marker_reward=0.30,
            reposition_marker_reward=-0.15,
            explore_new_cursor_reward=0.02,
            frontier_cursor_reward=0.03,
            connectivity_improve_reward=0.10,
            connectivity_worse_penalty=-0.25,
            first_playable_bonus=1.50,
            path_improve_bonus=0.45,
            target_delta_scale=1.20,
            target_regression_scale=0.55,
            floor_progress_scale=0.20,
            floor_overflow_penalty_scale=-0.35,
            valid_terminal_reward=10.0,
            target_match_terminal_reward=65.0,
            auto_place_terminal_reward_scale=0.35,
            invalid_terminal_penalty=-6.0,
            premature_stop_penalty=-1.50,
            stagnation_penalty=-3.00,
        )

    raise ValueError("difficulty must be one of easy, medium, hard")

class DungeonPCGRLEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        target_difficulty: str = "easy",
        observation_mode: str = "flat",
        max_steps: Optional[int] = None,
        metadata_csv: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        auto_place_markers: bool = True,
        auto_place_on_truncation: bool = False,
        strict_stop: bool = True,
        min_stop_steps: Optional[int] = None,
        initial_room_size: int = 3,
        stagnation_patience: Optional[int] = None,
        randomize_initial_position: bool = True,
        randomize_auto_place: bool = True,
        auto_place_top_k: int = 8,
    ) -> None:
        super().__init__()

        if width != height:
            raise ValueError("This environment currently expects square maps.")
        if target_difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("target_difficulty must be one of: easy, medium, hard")
        if observation_mode not in {"flat", "dict"}:
            raise ValueError("observation_mode must be 'flat' or 'dict'")
        if initial_room_size not in {1, 3, 5}:
            raise ValueError("initial_room_size must be one of: 1, 3, 5")

        self.width = int(width)
        self.height = int(height)
        self.size = self.width
        self.target_difficulty = target_difficulty
        self.observation_mode = observation_mode
        self.max_steps = int(max_steps or (220 if self.size == 16 else 420))
        self.min_stop_steps = int(min_stop_steps or max(30, self.size * 2))
        self.stagnation_patience = int(stagnation_patience or (120 if self.size == 16 else 180))
        self.render_mode = render_mode
        self._uses_custom_reward_config = reward_config is not None
        self.reward_cfg = reward_config or make_reward_config_for_difficulty(target_difficulty)
        self.auto_place_markers = bool(auto_place_markers)
        self.auto_place_on_truncation = bool(auto_place_on_truncation)
        self.strict_stop = bool(strict_stop)
        self.initial_room_size = int(initial_room_size)
        self.randomize_initial_position = bool(randomize_initial_position)
        self.randomize_auto_place = bool(randomize_auto_place)
        self.auto_place_top_k = max(1, int(auto_place_top_k))

        if metadata_csv is not None:
            self.target_ranges_all = self._load_target_ranges_from_csv(metadata_csv)
        else:
            self.target_ranges_all = DEFAULT_TARGET_RANGES

        if self.size not in self.target_ranges_all:
            raise ValueError(
                f"No target ranges found for map size {self.size}. "
                f"Provide metadata_csv or extend DEFAULT_TARGET_RANGES."
            )
        self.target_ranges = self.target_ranges_all[self.size][self.target_difficulty]

        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.cursor = np.array([self.height // 2, self.width // 2], dtype=np.int32)
        self.cursor_visited = np.zeros((self.height, self.width), dtype=bool)
        self.start_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.current_step = 0
        self.first_playable_given = False
        self.best_path_gap = float("inf")
        self.best_target_score = 0.0
        self.best_floor_tiles = 0
        self.steps_since_progress = 0
        self.last_info: Dict[str, Any] = {}
        self.last_auto_placed = False
        self.np_random = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(len(DungeonAction))
        self.feature_dim = 18

        if self.observation_mode == "dict":
            self.observation_space = spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.height, self.width, 6),
                        dtype=np.float32,
                    ),
                    "features": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(self.feature_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            flat_dim = (self.height * self.width * 6) + self.feature_dim
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(flat_dim,),
                dtype=np.float32,
            )

        self.reset(seed=seed)

    @staticmethod
    def _load_target_ranges_from_csv(metadata_csv: str) -> Dict[int, Dict[str, Dict[str, Tuple[int, int]]]]:
        """Build target ranges from dataset metadata.

        The function deliberately uses the standard-library csv module so the
        environment can be imported without pandas. Only training/data-analysis
        scripts need heavier dependencies.
        """
        csv_path = Path(metadata_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")

        required = {
            "difficulty",
            "width",
            "floor_tiles",
            "num_dead_ends",
            "num_loops",
            "start_goal_distance",
        }
        grouped: Dict[Tuple[int, str], Dict[str, list[int]]] = {}
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"metadata_csv is missing required columns: {missing_text}")

            for row in reader:
                difficulty = str(row["difficulty"]).lower()
                if difficulty == "invalid":
                    continue
                width = int(row["width"])
                key = (width, difficulty)
                bucket = grouped.setdefault(
                    key,
                    {
                        "floor_tiles": [],
                        "num_dead_ends": [],
                        "num_loops": [],
                        "start_goal_distance": [],
                    },
                )
                for metric_name in bucket:
                    bucket[metric_name].append(int(row[metric_name]))

        result: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]] = {}
        for (width, difficulty), values in grouped.items():
            result.setdefault(width, {})[difficulty] = {
                metric_name: (min(metric_values), max(metric_values))
                for metric_name, metric_values in values.items()
                if metric_values
            }

        if not result:
            raise ValueError("metadata_csv did not contain any non-invalid dungeon rows.")
        return result


    def set_target_difficulty(self, difficulty: str) -> None:
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be one of: easy, medium, hard")

        self.target_difficulty = difficulty
        self.target_ranges = self.target_ranges_all[self.size][difficulty]
        if not self._uses_custom_reward_config:
            self.reward_cfg = make_reward_config_for_difficulty(difficulty)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            pass

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.grid.fill(Tile.WALL)
        self.cursor_visited.fill(False)
        self.last_auto_placed = False

        room_radius = self.initial_room_size // 2
        if self.randomize_initial_position:
            # Randomized initial seed room prevents deterministic policy from
            # producing the exact same dungeon in every evaluation episode.
            min_r = 1 + room_radius
            max_r = self.height - 2 - room_radius
            min_c = 1 + room_radius
            max_c = self.width - 2 - room_radius
            if min_r <= max_r and min_c <= max_c:
                center = (
                    int(self.np_random.integers(min_r, max_r + 1)),
                    int(self.np_random.integers(min_c, max_c + 1)),
                )
            else:
                center = (self.height // 2, self.width // 2)
        else:
            center = (self.height // 2, self.width // 2)
            if center[0] <= 0 or center[0] >= self.height - 1 or center[1] <= 0 or center[1] >= self.width - 1:
                center = (1, 1)

        for dr in range(-room_radius, room_radius + 1):
            for dc in range(-room_radius, room_radius + 1):
                rr, cc = center[0] + dr, center[1] + dc
                if not self._is_border(rr, cc):
                    self.grid[rr, cc] = Tile.FLOOR

        self.cursor = np.array(center, dtype=np.int32)
        self.cursor_visited[center] = True
        self.start_pos = None
        self.goal_pos = None
        self.current_step = 0
        self.first_playable_given = False
        self.best_path_gap = float("inf")

        metrics = self._compute_metrics()
        self.best_floor_tiles = int(metrics["floor_tiles"])
        self.steps_since_progress = 0
        self.best_target_score = self._target_score(metrics)

        info = self._build_info(metrics, reward_breakdown=None, termination_reason=None)
        self.last_info = info
        return self._get_obs(metrics), info

    def step(self, action: int):
        action_enum = DungeonAction(int(action))
        self.current_step += 1
        prev_metrics = self._compute_metrics()
        prev_target_score = self._target_score(prev_metrics)

        reward_breakdown = {
            "step": self.reward_cfg.step_penalty,
            "edit": 0.0,
            "connectivity": 0.0,
            "path": 0.0,
            "target_delta": 0.0,
            "floor_progress": 0.0,
            "floor_overflow": 0.0,
            "terminal": 0.0,
        }
        reward = self.reward_cfg.step_penalty

        terminated = False
        truncated = False
        termination_reason: Optional[str] = None
        made_progress = False

        if action_enum == DungeonAction.STOP and not self._can_stop(prev_metrics):
            reward_breakdown["edit"] = self.reward_cfg.premature_stop_penalty
            reward += reward_breakdown["edit"]
            current_metrics = prev_metrics
            self.steps_since_progress += 1
        else:
            reward_breakdown["edit"] = self._apply_action(action_enum)
            reward += reward_breakdown["edit"]
            current_metrics = self._compute_metrics()

            reward_breakdown["connectivity"] = self._connectivity_reward(prev_metrics, current_metrics)
            reward += reward_breakdown["connectivity"]

            reward_breakdown["path"] = self._path_reward(prev_metrics, current_metrics)
            reward += reward_breakdown["path"]

            current_target_score = self._target_score(current_metrics)
            target_delta = current_target_score - prev_target_score
            if target_delta >= 0:
                reward_breakdown["target_delta"] = self.reward_cfg.target_delta_scale * target_delta
            else:
                reward_breakdown["target_delta"] = self.reward_cfg.target_regression_scale * target_delta
            reward += reward_breakdown["target_delta"]

            if current_target_score > self.best_target_score + 1e-6:
                made_progress = True
                self.best_target_score = current_target_score

            floor_low, floor_high = self.target_ranges["floor_tiles"]
            current_floor = int(current_metrics["floor_tiles"])
            previous_best_floor = int(self.best_floor_tiles)

            # Reward carving only while it moves the map toward the target range.
            # Once the upper bound is reached, extra floors are penalized instead
            # of being treated as progress.
            bounded_before = min(previous_best_floor, floor_high)
            bounded_after = min(current_floor, floor_high)
            gained_toward_target = max(0, bounded_after - bounded_before)
            if gained_toward_target > 0:
                reward_breakdown["floor_progress"] = (
                    self.reward_cfg.floor_progress_scale * float(gained_toward_target)
                )
                reward += reward_breakdown["floor_progress"]
                self.best_floor_tiles = max(previous_best_floor, current_floor)
                made_progress = True

            if current_floor > floor_high:
                overflow = current_floor - floor_high
                reward_breakdown["floor_overflow"] = (
                    self.reward_cfg.floor_overflow_penalty_scale * float(overflow)
                )
                reward += reward_breakdown["floor_overflow"]

            if reward_breakdown["connectivity"] > 0 or reward_breakdown["path"] > 0:
                made_progress = True

            if made_progress:
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1

        if self.steps_since_progress >= self.stagnation_patience:
            truncated = True
            termination_reason = "stagnation"
        elif self.current_step >= self.max_steps:
            truncated = True
            termination_reason = "max_steps"
        elif action_enum == DungeonAction.STOP and self._can_stop(current_metrics):
            terminated = True
            termination_reason = "agent_stop"

        if terminated or truncated:
            allow_auto_place = terminated or (truncated and self.auto_place_on_truncation)
            current_metrics = self._prepare_terminal_metrics(
                current_metrics,
                allow_auto_place=allow_auto_place,
            )
            terminal_reward = self._terminal_reward(current_metrics)
            if self.last_auto_placed:
                terminal_reward *= self.reward_cfg.auto_place_terminal_reward_scale
            if termination_reason == "stagnation":
                terminal_reward += self.reward_cfg.stagnation_penalty
            reward_breakdown["terminal"] = terminal_reward
            reward += reward_breakdown["terminal"]

        info = self._build_info(current_metrics, reward_breakdown, termination_reason)
        self.last_info = info
        return self._get_obs(current_metrics), float(reward), terminated, truncated, info


    def _metric_in_range(self, metrics: Dict[str, Any], metric_name: str) -> bool:
        low, high = self.target_ranges[metric_name]
        value = int(metrics[metric_name])
        return low <= value <= high

    def _all_core_targets_in_range(self, metrics: Dict[str, Any]) -> bool:
        return all(
            self._metric_in_range(metrics, metric_name)
            for metric_name in (
                "floor_tiles",
                "start_goal_distance",
                "num_dead_ends",
                "num_loops",
            )
        )

    def _basic_targets_in_range(self, metrics: Dict[str, Any]) -> bool:
        return self._metric_in_range(metrics, "floor_tiles") and self._metric_in_range(
            metrics,
            "start_goal_distance",
        )

    def _can_stop(self, metrics: Optional[Dict[str, Any]] = None) -> bool:
        if metrics is None:
            metrics = self._compute_metrics()

        if self.current_step < self.min_stop_steps:
            return False
        if not metrics["has_start"] or not metrics["has_goal"]:
            return False
        if not metrics["playable"]:
            return False
        if not self._basic_targets_in_range(metrics):
            return False
        if self.strict_stop and not self._all_core_targets_in_range(metrics):
            return False

        return True


    def _apply_action(self, action: DungeonAction) -> float:
        if action in {
            DungeonAction.MOVE_UP,
            DungeonAction.MOVE_DOWN,
            DungeonAction.MOVE_LEFT,
            DungeonAction.MOVE_RIGHT,
        }:
            return self._move_cursor(action)
        if action == DungeonAction.CARVE_FLOOR:
            return self._carve_floor()
        if action == DungeonAction.PLACE_WALL:
            return self._place_wall()
        if action == DungeonAction.PLACE_START:
            return self._place_marker("start")
        if action == DungeonAction.PLACE_GOAL:
            return self._place_marker("goal")
        if action == DungeonAction.STOP:
            return 0.0
        return 0.0

    def _cursor_can_enter(self, r: int, c: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        if self._is_border(r, c):
            return False

        tile = Tile(int(self.grid[r, c]))
        if tile != Tile.WALL:
            return True

        return self._count_walkable_neighbors(r, c) > 0

    def _move_cursor(self, action: DungeonAction) -> float:
        dr, dc = 0, 0
        if action == DungeonAction.MOVE_UP:
            dr = -1
        elif action == DungeonAction.MOVE_DOWN:
            dr = 1
        elif action == DungeonAction.MOVE_LEFT:
            dc = -1
        elif action == DungeonAction.MOVE_RIGHT:
            dc = 1

        nr = int(self.cursor[0] + dr)
        nc = int(self.cursor[1] + dc)

        if not self._cursor_can_enter(nr, nc):
            return self.reward_cfg.invalid_move_penalty

        self.cursor[:] = (nr, nc)

        reward = 0.0
        if not self.cursor_visited[nr, nc]:
            self.cursor_visited[nr, nc] = True
            reward += self.reward_cfg.explore_new_cursor_reward

        if Tile(int(self.grid[nr, nc])) == Tile.WALL and self._count_walkable_neighbors(nr, nc) > 0:
            reward += self.reward_cfg.frontier_cursor_reward

        return reward

    def _iter_cardinal_neighbors(self, r: int, c: int) -> Iterable[Tuple[int, int]]:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                yield nr, nc

    def _count_walkable_neighbors(self, r: int, c: int) -> int:
        count = 0
        for nr, nc in self._iter_cardinal_neighbors(r, c):
            if not self._is_border(nr, nc) and self.grid[nr, nc] != Tile.WALL:
                count += 1
        return count

    def _is_frontier(self, r: int, c: int) -> bool:
        if self._is_border(r, c):
            return False
        tile = Tile(int(self.grid[r, c]))
        if tile == Tile.WALL:
            return self._count_walkable_neighbors(r, c) > 0
        for nr, nc in self._iter_cardinal_neighbors(r, c):
            if not self._is_border(nr, nc) and self.grid[nr, nc] == Tile.WALL:
                return True
        return False

    def _component_count_for_mask(self, walkable_mask: np.ndarray) -> int:
        walkable_positions = np.argwhere(walkable_mask)
        if walkable_positions.size == 0:
            return 0

        visited = np.zeros_like(walkable_mask, dtype=bool)
        num_components = 0

        for r, c in walkable_positions:
            r, c = int(r), int(c)
            if visited[r, c]:
                continue
            num_components += 1
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in self._iter_cardinal_neighbors(cr, cc):
                    if walkable_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

        return num_components

    def _carve_floor(self) -> float:
        r, c = int(self.cursor[0]), int(self.cursor[1])
        tile = Tile(int(self.grid[r, c]))

        if self._is_border(r, c):
            return self.reward_cfg.invalid_move_penalty

        if tile == Tile.WALL:
            if self._count_walkable_neighbors(r, c) == 0:
                return self.reward_cfg.invalid_move_penalty
            self.grid[r, c] = Tile.FLOOR
            return self.reward_cfg.carve_new_floor_reward

        if tile == Tile.FLOOR:
            return self.reward_cfg.carve_existing_floor_penalty

        if tile == Tile.START:
            self.grid[r, c] = Tile.FLOOR
            self.start_pos = None
            return self.reward_cfg.remove_marker_penalty

        if tile == Tile.GOAL:
            self.grid[r, c] = Tile.FLOOR
            self.goal_pos = None
            return self.reward_cfg.remove_marker_penalty

        return 0.0

    def _place_wall(self) -> float:
        r, c = int(self.cursor[0]), int(self.cursor[1])
        tile = Tile(int(self.grid[r, c]))

        if self._is_border(r, c):
            return self.reward_cfg.invalid_move_penalty
        if tile == Tile.WALL:
            return self.reward_cfg.carve_existing_floor_penalty
        if tile in {Tile.START, Tile.GOAL}:
            return self.reward_cfg.invalid_move_penalty

        walkable_mask = self.grid != Tile.WALL
        if int(walkable_mask.sum()) <= 4:
            return self.reward_cfg.invalid_move_penalty

        original = self.grid[r, c]
        self.grid[r, c] = Tile.WALL
        new_walkable_mask = self.grid != Tile.WALL
        components_after = self._component_count_for_mask(new_walkable_mask)

        if components_after > 1:
            self.grid[r, c] = original
            return self.reward_cfg.invalid_move_penalty

        return self.reward_cfg.place_wall_from_floor_penalty

    def _place_marker(self, marker_type: str) -> float:
        r, c = int(self.cursor[0]), int(self.cursor[1])

        if self._is_border(r, c):
            return self.reward_cfg.invalid_move_penalty

        current_tile = Tile(int(self.grid[r, c]))
        if current_tile == Tile.WALL:
            return self.reward_cfg.invalid_move_penalty
        if marker_type == "start" and current_tile == Tile.GOAL:
            return self.reward_cfg.invalid_move_penalty
        if marker_type == "goal" and current_tile == Tile.START:
            return self.reward_cfg.invalid_move_penalty

        metrics = self._compute_metrics()
        min_floor_target = self.target_ranges["floor_tiles"][0]
        min_floor_for_marker = max(10, int(0.5 * min_floor_target))

        if metrics["floor_tiles"] < min_floor_for_marker:
            return self.reward_cfg.invalid_move_penalty
        if metrics["num_components"] != 1:
            return self.reward_cfg.invalid_move_penalty

        min_dist_low = self.target_ranges["start_goal_distance"][0]
        walkable_mask = self.grid != Tile.WALL

        if marker_type == "start" and self.goal_pos is not None:
            dist = self._shortest_path_distance((r, c), self.goal_pos, walkable_mask)
            if dist < min_dist_low:
                return self.reward_cfg.invalid_move_penalty

        if marker_type == "goal" and self.start_pos is not None:
            dist = self._shortest_path_distance(self.start_pos, (r, c), walkable_mask)
            if dist < min_dist_low:
                return self.reward_cfg.invalid_move_penalty

        if marker_type == "start":
            if self.start_pos == (r, c):
                return 0.0

            had_marker = self.start_pos is not None
            if self.start_pos is not None:
                sr, sc = self.start_pos
                self.grid[sr, sc] = Tile.FLOOR

            self.grid[r, c] = Tile.START
            self.start_pos = (r, c)
            return self.reward_cfg.first_marker_reward if not had_marker else self.reward_cfg.reposition_marker_reward

        if self.goal_pos == (r, c):
            return 0.0

        had_marker = self.goal_pos is not None
        if self.goal_pos is not None:
            gr, gc = self.goal_pos
            self.grid[gr, gc] = Tile.FLOOR

        self.grid[r, c] = Tile.GOAL
        self.goal_pos = (r, c)
        return self.reward_cfg.first_marker_reward if not had_marker else self.reward_cfg.reposition_marker_reward

    def _connectivity_reward(self, prev_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> float:
        prev_components = int(prev_metrics["num_components"])
        current_components = int(current_metrics["num_components"])

        if current_components < prev_components:
            return self.reward_cfg.connectivity_improve_reward
        if current_components > prev_components:
            return self.reward_cfg.connectivity_worse_penalty
        return 0.0

    def _path_reward(self, prev_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> float:
        reward = 0.0

        min_floor_target = self.target_ranges["floor_tiles"][0]
        min_floor_for_path_reward = max(12, int(0.65 * min_floor_target))

        if current_metrics["floor_tiles"] < min_floor_for_path_reward:
            return 0.0

        current_dist = current_metrics["start_goal_distance"]

        if (not prev_metrics["playable"]) and current_metrics["playable"] and (not self.first_playable_given):
            reward += self.reward_cfg.first_playable_bonus
            self.first_playable_given = True

        if current_metrics["playable"] and current_dist > 0:
            target_low, target_high = self.target_ranges["start_goal_distance"]
            current_gap = self._distance_to_range(current_dist, (target_low, target_high))
            if current_gap < self.best_path_gap:
                reward += self.reward_cfg.path_improve_bonus
                self.best_path_gap = current_gap

        return reward

    def _target_score(self, metrics: Dict[str, Any]) -> float:
        min_floor_target = self.target_ranges["floor_tiles"][0]
        floor_score = self._range_score(
            metrics["floor_tiles"],
            self.target_ranges["floor_tiles"],
            tau=max(10.0, self.size * self.size * 0.10),
        )

        components = max(1, int(metrics["num_components"]))
        if metrics["floor_tiles"] < 4:
            connectivity_score = 0.0
        elif metrics["connected"]:
            connectivity_score = 1.0
        else:
            connectivity_score = 1.0 / float(components)

        marker_ready = metrics["floor_tiles"] >= max(10, int(0.5 * min_floor_target))
        path_ready = metrics["floor_tiles"] >= max(12, int(0.65 * min_floor_target))
        structure_ready = metrics["floor_tiles"] >= max(16, int(0.7 * min_floor_target))

        marker_score = 0.5 * (float(metrics["has_start"]) + float(metrics["has_goal"])) if marker_ready else 0.0

        path_score = 0.0
        if path_ready and metrics["start_goal_distance"] > 0:
            path_score = self._range_score(
                metrics["start_goal_distance"],
                self.target_ranges["start_goal_distance"],
                tau=max(6.0, self.size / 2),
            )

        dead_end_score = 0.0
        loop_score = 0.0
        if structure_ready and metrics["connected"]:
            dead_end_score = self._range_score(metrics["num_dead_ends"], self.target_ranges["num_dead_ends"], tau=4.0)
            loop_score = self._range_score(metrics["num_loops"], self.target_ranges["num_loops"], tau=max(5.0, self.size / 2))

        playable_score = float(metrics["playable"]) if path_ready else 0.0

        score = (
            0.34 * floor_score
            + 0.12 * connectivity_score
            + 0.08 * marker_score
            + 0.18 * path_score
            + 0.10 * dead_end_score
            + 0.10 * loop_score
            + 0.08 * playable_score
        )
        return float(min(1.0, max(0.0, score)))

    @staticmethod
    def _distance_to_range(value: float, bounds: Tuple[int, int]) -> float:
        low, high = bounds
        if low <= value <= high:
            return 0.0
        if value < low:
            return float(low - value)
        return float(value - high)

    def _range_score(self, value: float, bounds: Tuple[int, int], tau: float) -> float:
        gap = self._distance_to_range(value, bounds)
        return max(0.0, 1.0 - (gap / max(tau, 1e-6)))

    def _prepare_terminal_metrics(
        self,
        metrics: Dict[str, Any],
        allow_auto_place: bool = True,
    ) -> Dict[str, Any]:
        self.last_auto_placed = False
        if not allow_auto_place or not self.auto_place_markers:
            return metrics

        min_floor_target = self.target_ranges["floor_tiles"][0]
        min_floor_for_marker = max(10, int(0.5 * min_floor_target))
        if metrics["floor_tiles"] < min_floor_for_marker or metrics["num_components"] != 1:
            return metrics
        if self.start_pos is not None and self.goal_pos is not None:
            return metrics

        pair = self._farthest_walkable_pair()
        if pair is None:
            return metrics

        start_pos, goal_pos, dist = pair
        dist_low, dist_high = self.target_ranges["start_goal_distance"]
        if dist < dist_low:
            return metrics

        # allow auto-place to overshoot slightly, but prefer staying close to target
        self._clear_existing_markers()
        sr, sc = start_pos
        gr, gc = goal_pos
        self.grid[sr, sc] = Tile.START
        self.grid[gr, gc] = Tile.GOAL
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.last_auto_placed = True
        return self._compute_metrics()

    def _clear_existing_markers(self) -> None:
        if self.start_pos is not None:
            sr, sc = self.start_pos
            if self.grid[sr, sc] == Tile.START:
                self.grid[sr, sc] = Tile.FLOOR
        if self.goal_pos is not None:
            gr, gc = self.goal_pos
            if self.grid[gr, gc] == Tile.GOAL:
                self.grid[gr, gc] = Tile.FLOOR
        self.start_pos = None
        self.goal_pos = None

    def _farthest_walkable_pair(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        walkable_positions = [tuple(map(int, pos)) for pos in np.argwhere(self.grid != Tile.WALL)]
        if len(walkable_positions) < 2:
            return None

        dist_low, dist_high = self.target_ranges["start_goal_distance"]
        candidates: list[Tuple[float, int, Tuple[int, int], Tuple[int, int]]] = []

        for start in walkable_positions:
            dist_map = self._bfs_distances(start)
            for goal, d in dist_map.items():
                if goal == start:
                    continue
                gap = self._distance_to_range(d, (dist_low, dist_high))
                candidates.append((float(gap), int(d), start, goal))

        if not candidates:
            return None

        # Prefer pairs inside or closest to the target distance range. When many
        # pairs have the same quality, random tie-breaking prevents identical
        # S/G placement across episodes.
        candidates.sort(key=lambda item: (item[0], -item[1]))
        if self.randomize_auto_place:
            best_gap = candidates[0][0]
            near_best = [item for item in candidates if abs(item[0] - best_gap) < 1e-9]
            if len(near_best) > self.auto_place_top_k:
                # Keep the best-distance subset but randomize within it.
                near_best = near_best[: self.auto_place_top_k]
            chosen = near_best[int(self.np_random.integers(0, len(near_best)))]
        else:
            chosen = candidates[0]

        _gap, dist, start, goal = chosen
        return (start, goal, int(dist))

    def _bfs_distances(self, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        queue = deque([start])
        dist = {start: 0}

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._iter_cardinal_neighbors(r, c):
                pos = (nr, nc)
                if self.grid[nr, nc] != Tile.WALL and pos not in dist:
                    dist[pos] = dist[(r, c)] + 1
                    queue.append(pos)

        return dist

    def _terminal_reward(self, metrics: Dict[str, Any]) -> float:
        if metrics.get("target_match", metrics.get("basic_valid", 0)):
            return self.reward_cfg.target_match_terminal_reward
        if metrics.get("playable", 0):
            return self.reward_cfg.valid_terminal_reward
        return self.reward_cfg.invalid_terminal_penalty


    def _compute_metrics(self) -> Dict[str, Any]:
        walkable_mask = self.grid != Tile.WALL
        walkable_positions = np.argwhere(walkable_mask)
        floor_tiles = int(walkable_mask.sum())
        wall_tiles = int((~walkable_mask).sum())

        has_start = self.start_pos is not None
        has_goal = self.goal_pos is not None

        if floor_tiles == 0:
            return {
                "floor_tiles": 0,
                "wall_tiles": wall_tiles,
                "connected": 0,
                "num_components": 0,
                "num_dead_ends": 0,
                "num_loops": 0,
                "start_goal_distance": -1,
                "has_start": int(has_start),
                "has_goal": int(has_goal),
                "structural_valid": 0,
                "playable": 0,
                "target_match": 0,
                "basic_valid": 0,
                "strict_valid": 0,
                "valid": 0,
            }

        visited = np.zeros_like(walkable_mask, dtype=bool)
        num_components = 0

        def neighbors(r: int, c: int):
            for nr, nc in self._iter_cardinal_neighbors(r, c):
                if walkable_mask[nr, nc]:
                    yield nr, nc

        for r, c in walkable_positions:
            r, c = int(r), int(c)
            if visited[r, c]:
                continue
            num_components += 1
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in neighbors(cr, cc):
                    if not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

        connected = int(num_components == 1)

        edge_count = 0
        dead_ends = 0
        for r, c in walkable_positions:
            r, c = int(r), int(c)
            degree = 0
            for _nr, _nc in neighbors(r, c):
                degree += 1
                edge_count += 1
            tile = Tile(int(self.grid[r, c]))
            if degree == 1 and tile not in {Tile.START, Tile.GOAL}:
                dead_ends += 1

        edge_count //= 2
        num_loops = max(edge_count - floor_tiles + num_components, 0)

        start_goal_distance = -1
        if has_start and has_goal:
            start_goal_distance = self._shortest_path_distance(self.start_pos, self.goal_pos, walkable_mask)

        structural_valid = int(has_start and has_goal and connected == 1)
        playable = int(structural_valid and start_goal_distance > 0)
        metric_snapshot = {
            "floor_tiles": floor_tiles,
            "num_dead_ends": int(dead_ends),
            "num_loops": int(num_loops),
            "start_goal_distance": int(start_goal_distance),
        }

        # Main paper-level target match: floor count + start-goal distance.
        # Dead ends and loops are stricter diagnostic metrics, not mandatory
        # for the main reported valid/playable/target-match table.
        target_match = int(playable and self._basic_targets_in_range(metric_snapshot))
        strict_valid = int(playable and self._all_core_targets_in_range(metric_snapshot))
        valid = structural_valid

        return {
            "floor_tiles": floor_tiles,
            "wall_tiles": wall_tiles,
            "connected": connected,
            "num_components": num_components,
            "num_dead_ends": int(dead_ends),
            "num_loops": int(num_loops),
            "start_goal_distance": int(start_goal_distance),
            "has_start": int(has_start),
            "has_goal": int(has_goal),
            "structural_valid": structural_valid,
            "playable": playable,
            "target_match": target_match,
            "basic_valid": target_match,
            "strict_valid": strict_valid,
            "valid": valid,
        }

    def _shortest_path_distance(
        self,
        start_pos: Optional[Tuple[int, int]],
        goal_pos: Optional[Tuple[int, int]],
        walkable_mask: np.ndarray,
    ) -> int:
        if start_pos is None or goal_pos is None:
            return -1

        sr, sc = start_pos
        gr, gc = goal_pos

        queue = deque([(sr, sc)])
        dist = {start_pos: 0}

        while queue:
            r, c = queue.popleft()

            if (r, c) == (gr, gc):
                return int(dist[(r, c)])

            for nr, nc in self._iter_cardinal_neighbors(r, c):
                pos = (nr, nc)
                if walkable_mask[nr, nc] and pos not in dist:
                    dist[pos] = dist[(r, c)] + 1
                    queue.append(pos)

        return -1

    def _build_grid_tensor(self) -> np.ndarray:
        grid_tensor = np.zeros((self.height, self.width, 6), dtype=np.float32)
        grid_tensor[:, :, 0] = (self.grid == Tile.WALL).astype(np.float32)
        grid_tensor[:, :, 1] = (self.grid == Tile.FLOOR).astype(np.float32)
        grid_tensor[:, :, 2] = (self.grid == Tile.START).astype(np.float32)
        grid_tensor[:, :, 3] = (self.grid == Tile.GOAL).astype(np.float32)
        grid_tensor[:, :, 4] = self.cursor_visited.astype(np.float32)
        grid_tensor[self.cursor[0], self.cursor[1], 5] = 1.0
        return grid_tensor

    def _build_features(self, metrics: Dict[str, Any]) -> np.ndarray:
        difficulty_one_hot = {
            "easy": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "medium": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "hard": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }[self.target_difficulty]

        max_walkable = float((self.height - 2) * (self.width - 2))
        max_reasonable_distance = float(max(1, (self.height - 2) * (self.width - 2)))

        start_goal_distance = metrics["start_goal_distance"]
        norm_distance = -1.0 if start_goal_distance < 0 else min(1.0, start_goal_distance / max_reasonable_distance)
        norm_dead_ends = min(1.0, metrics["num_dead_ends"] / max(1.0, max_walkable))
        norm_loops = min(1.0, metrics["num_loops"] / max(1.0, max_walkable))
        floor_gap = self._distance_to_range(metrics["floor_tiles"], self.target_ranges["floor_tiles"])
        path_gap = 0.0 if start_goal_distance < 0 else self._distance_to_range(
            start_goal_distance,
            self.target_ranges["start_goal_distance"],
        )
        dead_end_gap = self._distance_to_range(metrics["num_dead_ends"], self.target_ranges["num_dead_ends"])
        loop_gap = self._distance_to_range(metrics["num_loops"], self.target_ranges["num_loops"])
        stagnation_ratio = min(1.0, self.steps_since_progress / max(1.0, float(self.stagnation_patience)))

        features = np.concatenate(
            [
                difficulty_one_hot,
                np.array(
                    [
                        self.current_step / float(self.max_steps),
                        metrics["floor_tiles"] / float(self.height * self.width),
                        float(metrics["has_start"]),
                        float(metrics["has_goal"]),
                        float(metrics["connected"]),
                        float(metrics["playable"]),
                        norm_distance,
                        norm_dead_ends,
                        norm_loops,
                        min(1.0, floor_gap / max(1.0, self.height * self.width)),
                        min(1.0, path_gap / max(1.0, self.height * self.width)),
                        min(1.0, dead_end_gap / 10.0),
                        min(1.0, loop_gap / 10.0),
                        float(self._is_frontier(int(self.cursor[0]), int(self.cursor[1]))),
                        stagnation_ratio,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return features.astype(np.float32)

    def _get_obs(self, metrics: Optional[Dict[str, Any]] = None):
        metrics = metrics or self._compute_metrics()
        grid_tensor = self._build_grid_tensor()
        features = self._build_features(metrics)

        if self.observation_mode == "dict":
            return {"grid": grid_tensor, "features": features}

        flat = np.concatenate([grid_tensor.reshape(-1), features]).astype(np.float32)
        return flat.astype(np.float32)

    def get_action_mask(self, metrics: Optional[Dict[str, Any]] = None) -> np.ndarray:
        mask = np.zeros(len(DungeonAction), dtype=bool)

        r, c = int(self.cursor[0]), int(self.cursor[1])
        tile = Tile(int(self.grid[r, c]))

        if metrics is None:
            metrics = self._compute_metrics()

        floor_low, floor_high = self.target_ranges["floor_tiles"]
        current_floor = int(metrics["floor_tiles"])

        # 1. Jika cursor berada di WALL frontier, izinkan CARVE_FLOOR secara selektif.
        #    Perubahan ini tidak mengubah fungsi utama _carve_floor(); hanya membatasi
        #    aksi yang terlihat oleh MaskablePPO agar medium/hard tidak berubah menjadi
        #    area blob penuh loop.
        max_carve_neighbors = self._max_carve_neighbors_for_mask()
        if tile == Tile.WALL:
            walkable_neighbors = self._count_walkable_neighbors(r, c)
            if 0 < walkable_neighbors <= max_carve_neighbors and current_floor < floor_high:
                mask[DungeonAction.CARVE_FLOOR] = True
            else:
                self._enable_safe_moves_for_mask(mask, current_floor, floor_high)
            return mask

        # 2. Jika floor belum cukup, prioritaskan frontier yang aman.
        #    Pada kode lama, agen dipaksa masuk ke semua frontier sehingga medium/hard
        #    sering menghasilkan loop berlebihan dan valid rate jatuh ke 0%.
        if current_floor < floor_low:
            frontier_moves = []

            candidates = [
                (DungeonAction.MOVE_UP, r - 1, c),
                (DungeonAction.MOVE_DOWN, r + 1, c),
                (DungeonAction.MOVE_LEFT, r, c - 1),
                (DungeonAction.MOVE_RIGHT, r, c + 1),
            ]

            for action, nr, nc in candidates:
                if self._cursor_can_enter(nr, nc):
                    if (
                        Tile(int(self.grid[nr, nc])) == Tile.WALL
                        and 0 < self._count_walkable_neighbors(nr, nc) <= max_carve_neighbors
                    ):
                        frontier_moves.append(action)

            if frontier_moves:
                for action in frontier_moves:
                    mask[action] = True
                return mask

        # 3. Gerakan normal. Ketika floor sudah menyentuh batas atas, hindari masuk
        #    ke wall frontier lagi agar agen fokus menempatkan marker/merapikan struktur.
        self._enable_safe_moves_for_mask(mask, current_floor, floor_high)

        # 3b. Aksi PLACE_WALL sebelumnya tersedia di environment tetapi tidak pernah
        #     diaktifkan di mask. Ini membuat agen mustahil mengurangi loop setelah
        #     salah carving. Sekarang aksi ini dibuka secara aman tanpa mengubah
        #     logika utama _place_wall().
        if current_floor > floor_low and tile == Tile.FLOOR:
            if self._can_place_wall_without_disconnect(r, c):
                mask[DungeonAction.PLACE_WALL] = True

        # 4. Setelah floor cukup, izinkan marker START dan GOAL.
        min_floor_for_marker = max(10, int(0.5 * floor_low))

        if current_floor >= min_floor_for_marker and metrics["num_components"] == 1:
            if tile == Tile.FLOOR:
                walkable_mask = self.grid != Tile.WALL
                min_dist_low = self.target_ranges["start_goal_distance"][0]

                if self.start_pos != (r, c):
                    if self.goal_pos is None:
                        mask[DungeonAction.PLACE_START] = True
                    else:
                        dist = self._shortest_path_distance((r, c), self.goal_pos, walkable_mask)
                        if dist >= min_dist_low:
                            mask[DungeonAction.PLACE_START] = True

                if self.goal_pos != (r, c):
                    if self.start_pos is None:
                        mask[DungeonAction.PLACE_GOAL] = True
                    else:
                        dist = self._shortest_path_distance(self.start_pos, (r, c), walkable_mask)
                        if dist >= min_dist_low:
                            mask[DungeonAction.PLACE_GOAL] = True

        # 5. STOP hanya boleh jika map memang boleh berhenti.
        if self._can_stop(metrics):
            mask[DungeonAction.STOP] = True

        # 6. Fallback supaya mask tidak kosong.
        if not mask.any():
            if self._cursor_can_enter(r - 1, c):
                mask[DungeonAction.MOVE_UP] = True
            if self._cursor_can_enter(r + 1, c):
                mask[DungeonAction.MOVE_DOWN] = True
            if self._cursor_can_enter(r, c - 1):
                mask[DungeonAction.MOVE_LEFT] = True
            if self._cursor_can_enter(r, c + 1):
                mask[DungeonAction.MOVE_RIGHT] = True

        return mask


    def _max_carve_neighbors_for_mask(self) -> int:
        """Difficulty-aware carving guard used only by action masking.

        Easy maps may tolerate dense carving. Medium and hard maps need longer
        paths and controlled loops, so the mask discourages carving a wall cell
        that already touches too many walkable cells. The underlying CARVE_FLOOR
        action is intentionally left unchanged.
        """
        if self.target_difficulty == "hard":
            return 1
        if self.target_difficulty == "medium":
            return 2
        return 3

    def _enable_safe_moves_for_mask(
        self,
        mask: np.ndarray,
        current_floor: int,
        floor_high: int,
    ) -> None:
        """Enable movement actions while avoiding unnecessary over-carving states."""
        r, c = int(self.cursor[0]), int(self.cursor[1])
        candidates = [
            (DungeonAction.MOVE_UP, r - 1, c),
            (DungeonAction.MOVE_DOWN, r + 1, c),
            (DungeonAction.MOVE_LEFT, r, c - 1),
            (DungeonAction.MOVE_RIGHT, r, c + 1),
        ]
        for action, nr, nc in candidates:
            if not self._cursor_can_enter(nr, nc):
                continue
            if current_floor >= floor_high and Tile(int(self.grid[nr, nc])) == Tile.WALL:
                continue
            mask[action] = True

    def _can_place_wall_without_disconnect(self, r: int, c: int) -> bool:
        """Return True when PLACE_WALL would keep all walkable tiles connected."""
        if self._is_border(r, c):
            return False
        if Tile(int(self.grid[r, c])) != Tile.FLOOR:
            return False
        walkable_mask = self.grid != Tile.WALL
        if int(walkable_mask.sum()) <= 4:
            return False
        original = self.grid[r, c]
        self.grid[r, c] = Tile.WALL
        try:
            new_walkable_mask = self.grid != Tile.WALL
            return self._component_count_for_mask(new_walkable_mask) == 1
        finally:
            self.grid[r, c] = original

    def action_masks(self) -> np.ndarray:
        """Compatibility method for sb3-contrib MaskablePPO.

        MaskablePPO commonly looks for env.action_masks(). Keeping
        get_action_mask() preserves backward compatibility with existing code.
        """
        return self.get_action_mask()

    def _build_info(
        self,
        metrics: Dict[str, Any],
        reward_breakdown: Optional[Dict[str, float]],
        termination_reason: Optional[str],
    ) -> Dict[str, Any]:
        info = {
            "target_difficulty": self.target_difficulty,
            "step": self.current_step,
            "cursor_row": int(self.cursor[0]),
            "cursor_col": int(self.cursor[1]),
            "target_ranges": self.target_ranges,
            "target_score": self._target_score(metrics),
            "strict_stop": int(self.strict_stop),
            "auto_placed_markers": int(self.last_auto_placed),
            "action_mask": self.get_action_mask(metrics),
            **metrics,
        }
        if reward_breakdown is not None:
            info["reward_breakdown"] = reward_breakdown
        if termination_reason is not None:
            info["termination_reason"] = termination_reason
        return info

    def render(self):
        lines = []
        for r in range(self.height):
            chars = []
            for c in range(self.width):
                tile = Tile(int(self.grid[r, c]))
                if (r, c) == (int(self.cursor[0]), int(self.cursor[1])):
                    if tile == Tile.START:
                        chars.append("s")
                    elif tile == Tile.GOAL:
                        chars.append("g")
                    else:
                        chars.append("@")
                    continue

                if tile == Tile.WALL:
                    chars.append("#")
                elif tile == Tile.FLOOR:
                    chars.append(".")
                elif tile == Tile.START:
                    chars.append("S")
                elif tile == Tile.GOAL:
                    chars.append("G")

            lines.append("".join(chars))

        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            print(self.last_info)
            return None
        return text

    def close(self) -> None:
        return None

    def _is_border(self, r: int, c: int) -> bool:
        return r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1


def register_env(env_id: str = "DungeonPCGRL-v0") -> None:
    try:
        from gymnasium.envs.registration import register
    except ImportError:  # pragma: no cover
        from gym.envs.registration import register  # type: ignore

    register(id=env_id, entry_point=f"{__name__}:DungeonPCGRLEnv")
