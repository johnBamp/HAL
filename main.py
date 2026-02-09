from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import pygame

from hal import HalEnv, NamingGameTrainer, NoOpScheduler


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str


class HalPygameApp:
    def __init__(self, seed: int) -> None:
        pygame.init()
        pygame.display.set_caption("Heuristic Algorithmic Language (HAL)")

        self.seed = seed
        self.cell_size = 5
        self.margin = 14
        self.sidebar_width = 440
        self.flash_duration_s = 0.22
        self.row_height = 20

        self.env = HalEnv(seed=seed)
        self.trainer = NamingGameTrainer(
            env=self.env,
            seed=seed + 1,
            scheduler=NoOpScheduler(),
        )

        self.map_px = self.env.grid.width * self.cell_size
        self.window_w = self.margin * 3 + self.map_px + self.sidebar_width
        self.window_h = self.margin * 2 + self.map_px
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Courier New", 15)
        self.small_font = pygame.font.SysFont("Courier New", 13)

        self.map_origin = (self.margin, self.margin)
        panel_x = self.margin * 2 + self.map_px
        self.panel_rect = pygame.Rect(panel_x, self.margin, self.sidebar_width, self.map_px)

        self.buttons: list[Button] = []
        self.info_rect = pygame.Rect(0, 0, 0, 0)
        self.list_rect = pygame.Rect(0, 0, 0, 0)
        self._build_ui_layout()

        self.static_map = pygame.Surface((self.map_px, self.map_px))
        self._build_static_map()

        self.running = True
        self.paused = False
        self.speed_steps_per_s = 120.0
        self.training_accumulator = 0.0

        self.flash_tile: Optional[int] = None
        self.flash_timer = 0.0
        self.selected_event_idx: Optional[int] = None
        self.selected_tile: Optional[int] = None
        self.selected_confidence = 0.0
        self.selected_confidence_x = 0.0
        self.selected_confidence_y = 0.0
        self.selected_col_x: Optional[int] = None
        self.selected_row_y: Optional[int] = None
        self.scroll_offset = 0

    def _build_ui_layout(self) -> None:
        panel_left = self.panel_rect.left + 10
        btn_y = self.panel_rect.top + 10
        btn_w = (self.panel_rect.width - 40) // 3
        btn_h = 32
        gap = 10
        self.buttons = [
            Button("Pause", pygame.Rect(panel_left, btn_y, btn_w, btn_h), "toggle_pause"),
            Button(
                "Step",
                pygame.Rect(panel_left + btn_w + gap, btn_y, btn_w, btn_h),
                "single_step",
            ),
            Button(
                "Reset",
                pygame.Rect(panel_left + (btn_w + gap) * 2, btn_y, btn_w, btn_h),
                "reset",
            ),
        ]

        info_y = btn_y + btn_h + 10
        self.info_rect = pygame.Rect(
            panel_left,
            info_y,
            self.panel_rect.width - 20,
            138,
        )

        list_y = self.info_rect.bottom + 10
        self.list_rect = pygame.Rect(
            panel_left,
            list_y,
            self.panel_rect.width - 20,
            self.panel_rect.bottom - list_y - 10,
        )

    def _build_static_map(self) -> None:
        outside = (22, 26, 32)
        inside = (38, 63, 88)
        self.static_map.fill(outside)
        visible = set(self.env.visible_tiles())
        for tile in visible:
            x, y = self.env.tile_xy(tile)
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            self.static_map.fill(inside, rect)

    def _reset(self) -> None:
        self.env = HalEnv(seed=self.seed)
        self.trainer = NamingGameTrainer(
            env=self.env,
            seed=self.seed + 1,
            scheduler=NoOpScheduler(),
        )
        self._build_static_map()
        self.paused = False
        self.speed_steps_per_s = 120.0
        self.training_accumulator = 0.0
        self.flash_tile = None
        self.flash_timer = 0.0
        self.selected_event_idx = None
        self.selected_tile = None
        self.selected_confidence = 0.0
        self.selected_confidence_x = 0.0
        self.selected_confidence_y = 0.0
        self.selected_col_x = None
        self.selected_row_y = None
        self.scroll_offset = 0

    def _format_term(self, term: tuple[int, ...]) -> str:
        if not term:
            return "-"
        return "".join(str(token) for token in term)

    def _visible_rows(self) -> int:
        return max(1, self.list_rect.height // self.row_height)

    def _set_scroll(self, next_offset: int) -> None:
        max_scroll = max(0, len(self.trainer.events) - self._visible_rows())
        self.scroll_offset = max(0, min(max_scroll, next_offset))

    def _scroll_by(self, delta: int) -> None:
        self._set_scroll(self.scroll_offset + delta)

    def _tile_rect(self, tile: int) -> pygame.Rect:
        x, y = self.env.tile_xy(tile)
        ox, oy = self.map_origin
        return pygame.Rect(
            ox + x * self.cell_size,
            oy + y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _center_of_tile(self, tile: int) -> tuple[int, int]:
        rect = self._tile_rect(tile)
        return rect.centerx, rect.centery

    def _step_once(self) -> None:
        self.trainer.step()
        self.flash_tile = self.trainer.last_prediction_tile
        self.flash_timer = self.flash_duration_s
        if self.trainer.is_complete():
            self.paused = True

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _set_speed(self, multiplier: float) -> None:
        self.speed_steps_per_s = max(1.0, min(1200.0, self.speed_steps_per_s * multiplier))

    def _handle_button_action(self, action: str) -> None:
        if action == "toggle_pause":
            self._toggle_pause()
            return
        if action == "single_step":
            if self.paused:
                self._step_once()
            return
        if action == "reset":
            self._reset()
            return

    def _select_event_at(self, mouse_pos: tuple[int, int]) -> None:
        if not self.paused or not self.list_rect.collidepoint(mouse_pos):
            return
        row = (mouse_pos[1] - self.list_rect.top) // self.row_height
        idx = self.scroll_offset + row
        if idx < 0 or idx >= len(self.trainer.events):
            return
        self.selected_event_idx = idx
        event = self.trainer.events[idx]
        (
            pred_tile,
            confidence,
            confidence_x,
            confidence_y,
            _,
            _,
        ) = self.trainer.decode_terms(
            event.x_word,
            event.y_word,
            fallback_random=False,
        )
        self.selected_tile = pred_tile
        self.selected_confidence = confidence
        self.selected_confidence_x = confidence_x
        self.selected_confidence_y = confidence_y
        pred_x, pred_y = self.env.tile_xy(pred_tile)
        self.selected_col_x = pred_x
        self.selected_row_y = pred_y

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._toggle_pause()
                elif event.key == pygame.K_n:
                    if self.paused:
                        self._step_once()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._set_speed(1.25)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._set_speed(0.8)
                elif event.key == pygame.K_UP:
                    self._scroll_by(-3)
                elif event.key == pygame.K_DOWN:
                    self._scroll_by(3)
            elif event.type == pygame.MOUSEWHEEL:
                self._scroll_by(-event.y * 3)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for button in self.buttons:
                    if button.rect.collidepoint(pos):
                        self._handle_button_action(button.action)
                        break
                else:
                    self._select_event_at(pos)

    def _draw_button(self, button: Button) -> None:
        is_pause = button.action == "toggle_pause"
        label = "Resume" if (is_pause and self.paused) else button.label
        bg = (69, 86, 108)
        border = (154, 175, 202)
        pygame.draw.rect(self.screen, bg, button.rect, border_radius=5)
        pygame.draw.rect(self.screen, border, button.rect, width=1, border_radius=5)
        text = self.font.render(label, True, (234, 240, 248))
        self.screen.blit(text, text.get_rect(center=button.rect.center))

    def _draw_grid(self) -> None:
        ox, oy = self.map_origin
        self.screen.blit(self.static_map, (ox, oy))
        border = pygame.Rect(ox - 1, oy - 1, self.map_px + 2, self.map_px + 2)
        pygame.draw.rect(self.screen, (115, 127, 143), border, width=1)

        if self.selected_col_x is not None:
            col_rect = pygame.Rect(
                ox + self.selected_col_x * self.cell_size,
                oy,
                self.cell_size,
                self.map_px,
            )
            col_overlay = pygame.Surface((col_rect.width, col_rect.height), pygame.SRCALPHA)
            col_overlay.fill((68, 195, 214, 55))
            self.screen.blit(col_overlay, (col_rect.x, col_rect.y))
            pygame.draw.rect(self.screen, (68, 195, 214), col_rect, width=1)

        if self.selected_row_y is not None:
            row_rect = pygame.Rect(
                ox,
                oy + self.selected_row_y * self.cell_size,
                self.map_px,
                self.cell_size,
            )
            row_overlay = pygame.Surface((row_rect.width, row_rect.height), pygame.SRCALPHA)
            row_overlay.fill((68, 195, 214, 55))
            self.screen.blit(row_overlay, (row_rect.x, row_rect.y))
            pygame.draw.rect(self.screen, (68, 195, 214), row_rect, width=1)

        if self.flash_tile is not None and self.flash_timer > 0:
            rect = self._tile_rect(self.flash_tile)
            pygame.draw.rect(self.screen, (227, 183, 66), rect)

        if self.selected_tile is not None:
            rect = self._tile_rect(self.selected_tile)
            pygame.draw.rect(self.screen, (68, 195, 214), rect, width=2)

        # Agent 1 marker and heading/FOV rays.
        anchor_tile = self.env.tile_id(self.env.agent1.x, self.env.agent1.y)
        ax, ay = self._center_of_tile(anchor_tile)
        pygame.draw.circle(self.screen, (245, 102, 102), (ax, ay), max(2, self.cell_size // 2))

        fov = self.env.current_stage.fov
        heading = self.env.agent1.heading_deg
        view_len_px = int(fov.range_tiles * self.cell_size)
        heading_rad = math.radians(heading)
        hx = int(ax + math.cos(heading_rad) * view_len_px)
        hy = int(ay + math.sin(heading_rad) * view_len_px)
        pygame.draw.line(self.screen, (245, 102, 102), (ax, ay), (hx, hy), width=1)

        half = fov.angle_deg / 2.0
        for offset in (-half, half):
            ray_rad = math.radians(heading + offset)
            rx = int(ax + math.cos(ray_rad) * view_len_px)
            ry = int(ay + math.sin(ray_rad) * view_len_px)
            pygame.draw.line(self.screen, (161, 119, 119), (ax, ay), (rx, ry), width=1)

        # Agent 2 marker.
        agent2_tile = self.env.agent2_tile_id
        bx, by = self._center_of_tile(agent2_tile)
        pygame.draw.circle(self.screen, (104, 220, 132), (bx, by), max(2, self.cell_size // 2))

    def _draw_info(self) -> None:
        panel_bg = (27, 33, 41)
        panel_border = (92, 103, 119)
        pygame.draw.rect(self.screen, panel_bg, self.panel_rect, border_radius=6)
        pygame.draw.rect(self.screen, panel_border, self.panel_rect, width=1, border_radius=6)

        for button in self.buttons:
            self._draw_button(button)

        visible_count = len(self.env.visible_tiles())
        covered_count = len(self.trainer.successful_tiles.intersection(set(self.env.visible_tiles())))
        status = "PAUSED" if self.paused else "TRAINING"
        complete = "YES" if self.trainer.is_complete() else "NO"
        selected_word = (
            (
                f"{self._format_term(self.trainer.events[self.selected_event_idx].x_word)}|"
                f"{self._format_term(self.trainer.events[self.selected_event_idx].y_word)}"
            )
            if self.selected_event_idx is not None
            else "-"
        )

        lines = [
            f"Status: {status}",
            f"Complete: {complete}",
            f"Steps: {self.trainer.step_count}",
            f"Speed: {self.speed_steps_per_s:.1f} steps/s",
            f"Rolling acc (1k): {self.trainer.rolling_accuracy(1000):.3f}",
            f"Coverage: {covered_count}/{visible_count}",
            f"Selected terms: {selected_word}",
            f"Selected conf: {self.selected_confidence:.2f}",
            f"Selected cX/cY: {self.selected_confidence_x:.2f}/{self.selected_confidence_y:.2f}",
            "Controls: Space pause, N step, +/- speed",
        ]
        y = self.info_rect.top + 4
        for line in lines:
            text = self.small_font.render(line, True, (218, 226, 237))
            self.screen.blit(text, (self.info_rect.left, y))
            y += 13

    def _draw_event_list(self) -> None:
        bg = (17, 22, 27)
        border = (90, 102, 119)
        pygame.draw.rect(self.screen, bg, self.list_rect, border_radius=4)
        pygame.draw.rect(self.screen, border, self.list_rect, width=1, border_radius=4)

        rows = self._visible_rows()
        self._set_scroll(self.scroll_offset)
        start = self.scroll_offset
        end = min(len(self.trainer.events), start + rows)
        y = self.list_rect.top

        for idx in range(start, end):
            event = self.trainer.events[idx]
            row_rect = pygame.Rect(self.list_rect.left, y, self.list_rect.width, self.row_height)
            if idx == self.selected_event_idx:
                pygame.draw.rect(self.screen, (45, 86, 120), row_rect)
            elif idx % 2 == 0:
                pygame.draw.rect(self.screen, (21, 27, 34), row_rect)

            status = "S" if event.success else "F"
            payload = (
                f"{event.step:06d}  "
                f"{self._format_term(event.x_word):<4}|{self._format_term(event.y_word):<4}  "
                f"{status}  c={event.confidence:.2f}"
            )
            text = self.small_font.render(payload, True, (214, 224, 236))
            self.screen.blit(text, (row_rect.left + 6, row_rect.top + 3))
            y += self.row_height

    def _render(self) -> None:
        self.screen.fill((10, 13, 17))
        self._draw_grid()
        self._draw_info()
        self._draw_event_list()
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self._process_events()

            if not self.paused:
                self.training_accumulator += dt * self.speed_steps_per_s
                steps = min(200, int(self.training_accumulator))
                if steps > 0:
                    self.training_accumulator -= steps
                    for _ in range(steps):
                        self._step_once()
                        if self.paused:
                            break

            if self.flash_timer > 0.0:
                self.flash_timer = max(0.0, self.flash_timer - dt)
                if self.flash_timer <= 0.0:
                    self.flash_tile = None

            self._render()

        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="HAL Part 1 pygame prototype")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()
    HalPygameApp(seed=args.seed).run()


if __name__ == "__main__":
    main()
