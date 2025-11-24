"""
2D visualization for the EcoTrack environment using pygame.

This renderer:
- Draws a grid-based city map
- Shows the depot, normal bins, high-priority bins, and the truck
- Encodes bin fill levels with colours
- Displays HUD information (step, load, overflows, serviced bins, etc.)
"""

from __future__ import annotations

from typing import Tuple

import pygame


class EcoTrackRenderer:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        cell_size: int = 48,
        hud_width: int = 260,
        fps: int = 10,
    ):
        """
        Args:
            grid_width: number of grid cells horizontally
            grid_height: number of grid cells vertically
            cell_size: pixel size of each cell (square)
            hud_width: width of side panel for text
            fps: target frames per second for rendering
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.hud_width = hud_width
        self.fps = fps

        self.window_width = grid_width * cell_size + hud_width
        self.window_height = grid_height * cell_size

        pygame.init()
        pygame.display.set_caption("EcoTrack – Smart Waste Collection")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_large = pygame.font.SysFont("Arial", 20, bold=True)

        # Basic colours
        self.COLOR_BG = (25, 25, 30)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_DEPOT = (66, 135, 245)
        self.COLOR_TRUCK = (255, 165, 0)
        self.COLOR_BIN_NORMAL_LOW = (60, 160, 60)
        self.COLOR_BIN_NORMAL_MED = (200, 180, 60)
        self.COLOR_BIN_NORMAL_HIGH = (200, 60, 60)
        self.COLOR_BIN_HIGHPRIO_BORDER = (255, 255, 255)
        self.COLOR_HUD_BG = (15, 15, 20)
        self.COLOR_TEXT = (230, 230, 230)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(self, env) -> None:
        """
        Render the current state of the environment.

        Args:
            env: an instance of EcoTrackEnv (or compatible object) with:
                - grid_width, grid_height
                - depot_pos
                - bin_positions, bin_priority, bin_fill
                - truck_pos, truck_load, max_capacity
                - t_step, max_steps
                - overflow_count, serviced_bins_count, serviced_high_priority_count
        """
        # Process basic events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(self.COLOR_BG)

        # Draw grid + map
        self._draw_grid()
        self._draw_bins(env)
        self._draw_depot(env)
        self._draw_truck(env)

        # Draw HUD
        self._draw_hud(env)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _cell_rect(self, x: int, y: int) -> pygame.Rect:
        """Return the pixel rect for a grid cell (x, y)."""
        return pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _draw_grid(self) -> None:
        """Draw the base grid lines."""
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = self._cell_rect(x, y)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, width=1)

    def _draw_depot(self, env) -> None:
        """Draw the depot cell."""
        dep_x, dep_y = env.depot_pos
        rect = self._cell_rect(dep_x, dep_y)
        pygame.draw.rect(self.screen, self.COLOR_DEPOT, rect)

        # Label "D"
        text = self.font_small.render("D", True, (0, 0, 0))
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def _bin_colour_from_fill(self, fill_ratio: float) -> Tuple[int, int, int]:
        """
        Choose a colour for a bin based on its fill ratio [0, 1].

        - Low fill: green-ish
        - Medium: yellow-ish
        - High: red-ish
        """
        if fill_ratio < 0.33:
            return self.COLOR_BIN_NORMAL_LOW
        elif fill_ratio < 0.66:
            return self.COLOR_BIN_NORMAL_MED
        else:
            return self.COLOR_BIN_NORMAL_HIGH

    def _draw_bins(self, env) -> None:
        """Draw all bins (normal + high-priority)."""
        if env.bin_fill is None:
            return

        for idx, (bx, by) in enumerate(env.bin_positions):
            rect = self._cell_rect(bx, by)

            fill = float(env.bin_fill[idx])
            fill_ratio = min(max(fill / env.max_capacity, 0.0), 1.0)
            base_color = self._bin_colour_from_fill(fill_ratio)

            # Draw bin base
            pygame.draw.rect(self.screen, base_color, rect)

            # High-priority bins get a bright border
            if env.bin_priority[idx] == 1:
                pygame.draw.rect(self.screen, self.COLOR_BIN_HIGHPRIO_BORDER, rect, width=3)

            # Draw a small vertical fill bar inside the bin
            bar_margin = 4
            bar_height = int((self.cell_size - 2 * bar_margin) * fill_ratio)
            bar_rect = pygame.Rect(
                rect.left + rect.width - bar_margin * 2,
                rect.bottom - bar_margin - bar_height,
                bar_margin,
                bar_height,
            )
            pygame.draw.rect(self.screen, (20, 20, 20), bar_rect)

    def _draw_truck(self, env) -> None:
        """Draw the truck at its current position."""
        tx, ty = env.truck_pos
        rect = self._cell_rect(tx, ty)

        # Slightly inset rectangle so it's visually distinct from bins
        inset = 4
        truck_rect = pygame.Rect(
            rect.left + inset,
            rect.top + inset,
            rect.width - 2 * inset,
            rect.height - 2 * inset,
        )
        pygame.draw.rect(self.screen, self.COLOR_TRUCK, truck_rect)

        # Indicate load ratio as a small bar under the truck
        load_ratio = min(max(env.truck_load / env.max_capacity, 0.0), 1.0)
        bar_width = int(truck_rect.width * load_ratio)
        bar_height = 4
        bar_rect = pygame.Rect(
            truck_rect.left,
            truck_rect.bottom - bar_height,
            bar_width,
            bar_height,
        )
        pygame.draw.rect(self.screen, (0, 0, 0), bar_rect)

    def _draw_hud(self, env) -> None:
        """Draw side HUD with environment stats."""
        # HUD background
        hud_rect = pygame.Rect(
            self.grid_width * self.cell_size,
            0,
            self.hud_width,
            self.window_height,
        )
        pygame.draw.rect(self.screen, self.COLOR_HUD_BG, hud_rect)

        # Some padding
        x0 = hud_rect.left + 12
        y = hud_rect.top + 16
        line_spacing = 22

        def draw_line(text: str, bold: bool = False):
            nonlocal y
            font = self.font_large if bold else self.font_small
            surf = font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (x0, y))
            y += line_spacing

        draw_line("EcoTrack", bold=True)
        y += 4

        draw_line(f"Step: {env.t_step} / {env.max_steps}")
        draw_line(
            f"Truck: ({env.truck_pos[0]}, {env.truck_pos[1]})"
        )
        draw_line(
            f"Load: {env.truck_load:.1f} / {env.max_capacity:.1f}"
        )
        y += 8

        draw_line(f"Overflows: {env.overflow_count}")
        draw_line(
            f"Serviced bins: {env.serviced_bins_count}"
        )
        draw_line(
            f"Serviced high-prio: {env.serviced_high_priority_count}"
        )

        # Ratios (if counts available)
        if env.total_bins > 0:
            serviced_ratio = env.serviced_bins_count / env.total_bins
            draw_line(f"Serviced ratio: {serviced_ratio:.2f}")

        if env.total_high_priority_bins > 0:
            high_ratio = (
                env.serviced_high_priority_count / env.total_high_priority_bins
            )
            draw_line(f"High-prio ratio: {high_ratio:.2f}")

        # Legend
        y += 10
        draw_line("Legend:", bold=True)

        # Tiny squares for legend
        def draw_legend_square(color, label):
            nonlocal y
            size = 14
            square_rect = pygame.Rect(x0, y, size, size)
            pygame.draw.rect(self.screen, color, square_rect)
            text_surf = self.font_small.render(label, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (x0 + size + 6, y - 1))
            y += line_spacing

        draw_legend_square(self.COLOR_DEPOT, "Depot")
        draw_legend_square(self.COLOR_TRUCK, "Truck")
        draw_legend_square(self.COLOR_BIN_NORMAL_LOW, "Bin (low fill)")
        draw_legend_square(self.COLOR_BIN_NORMAL_MED, "Bin (medium fill)")
        draw_legend_square(self.COLOR_BIN_NORMAL_HIGH, "Bin (high fill)")
