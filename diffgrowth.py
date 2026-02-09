#!/usr/bin/env python3
"""
Differential Growth Pattern Generator (Optimized Pure Python)

Implements differential growth algorithm with spatial hashing and
optimized hot paths for maximum pure Python performance.

Author: Claude
License: MIT
"""

import argparse
import logging
import math
import random
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Set


class Node:
    """Optimized node using __slots__ for faster attribute access."""
    __slots__ = ('x', 'y', 'vx', 'vy', 'fx', 'fy')

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.fx = 0.0
        self.fy = 0.0


class SpatialHash:
    """Spatial hash grid for O(1) neighbor lookups."""
    __slots__ = ('cell_size', 'grid', 'edge_grid')

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        self.edge_grid: Dict[Tuple[int, int], List[int]] = {}

    def clear(self) -> None:
        self.grid.clear()
        self.edge_grid.clear()

    def insert(self, idx: int, x: float, y: float) -> None:
        key = (int(x // self.cell_size), int(y // self.cell_size))
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(idx)

    def insert_edge(self, edge_idx: int, x1: float, y1: float, x2: float, y2: float) -> None:
        cell_size = self.cell_size
        min_cx = int(min(x1, x2) // cell_size)
        max_cx = int(max(x1, x2) // cell_size)
        min_cy = int(min(y1, y2) // cell_size)
        max_cy = int(max(y1, y2) // cell_size)

        edge_grid = self.edge_grid
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                key = (cx, cy)
                if key not in edge_grid:
                    edge_grid[key] = []
                if edge_idx not in edge_grid[key]:
                    edge_grid[key].append(edge_idx)

    def query_radius(self, x: float, y: float, radius: float) -> List[int]:
        result = []
        cell_size = self.cell_size
        cells = int(radius // cell_size) + 1
        cx, cy = int(x // cell_size), int(y // cell_size)
        grid = self.grid

        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                key = (cx + dx, cy + dy)
                if key in grid:
                    result.extend(grid[key])
        return result

    def query_edges_near(self, x: float, y: float, radius: float) -> Set[int]:
        result: Set[int] = set()
        cell_size = self.cell_size
        cells = int(radius // cell_size) + 1
        cx, cy = int(x // cell_size), int(y // cell_size)
        edge_grid = self.edge_grid

        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                key = (cx + dx, cy + dy)
                if key in edge_grid:
                    result.update(edge_grid[key])
        return result


class DifferentialGrowth:
    """Optimized differential growth with spatial hashing."""

    logger = logging.getLogger(__name__)

    _ATTRACTION_SCALE = 0.2
    _REPULSION_SCALE = 0.3
    _ALIGNMENT_SCALE = 0.25
    _NOISE_SCALE = 0.1
    _GROWTH_SCALE = 0.08

    @staticmethod
    def is_safe_params(growth: float, repulsion: float, noise: float) -> bool:
        """Check if parameters are safe for running without intersection checking.

        Safe constraint: repulsion > 0.2 * growth + 3 * noise
        """
        min_repulsion = 0.2 * growth + 3 * noise
        return repulsion > min_repulsion

    @staticmethod
    def get_safe_repulsion(growth: float, noise: float) -> float:
        """Get minimum safe repulsion value for given growth and noise."""
        return 0.2 * growth + 3 * noise + 0.1  # Add margin

    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        initial_nodes: int = 20,
        min_edge_ratio: float = 0.25,
        max_edge_ratio: float = 0.75,
        repulsion_radius_ratio: float = 1.5,
        attraction: float = 0.5,
        repulsion: float = 0.5,
        alignment: float = 0.5,
        growth: float = 0.5,
        noise: float = 0.1,
        damping: float = 0.5,
        random_seed: Optional[int] = None,
        shape: str = 'circle',
        bounds: Optional[Tuple[float, float, float, float]] = None,
        bound_shape: str = 'rectangle',
        boundary_repulsion: float = 0.0
    ):
        self.width = width
        self.height = height
        self.base_length = min(width, height) / 40.0

        self.min_edge_length = max(1.0, min_edge_ratio * self.base_length)
        self.max_edge_length = max(self.min_edge_length + 1, max_edge_ratio * self.base_length)
        self.repulsion_radius = repulsion_radius_ratio * self.base_length

        # Pre-compute commonly used values
        self.min_edge_sq = self.min_edge_length * self.min_edge_length
        self.repulsion_radius_sq = self.repulsion_radius * self.repulsion_radius
        self.max_velocity = self.min_edge_length * 0.3
        self.min_separation = self.min_edge_length * 0.5

        # Scaled force factors
        self.attraction_factor = attraction * self._ATTRACTION_SCALE
        self.repulsion_factor = repulsion * self._REPULSION_SCALE
        self.alignment_factor = alignment * self._ALIGNMENT_SCALE
        self.growth_factor = growth * self._GROWTH_SCALE
        self.noise_factor = noise * self._NOISE_SCALE * self.base_length
        self.damping = max(0.0, min(1.0, damping))

        if random_seed is not None:
            random.seed(random_seed)

        # Bounding constraint (min_x, min_y, max_x, max_y defines the bounding region)
        if bounds is not None:
            self.bounds = bounds
            # Pre-compute bound shape parameters
            self.bound_cx = (bounds[0] + bounds[2]) / 2
            self.bound_cy = (bounds[1] + bounds[3]) / 2
            self.bound_w = bounds[2] - bounds[0]
            self.bound_h = bounds[3] - bounds[1]
            self.bound_radius = min(self.bound_w, self.bound_h) / 2
        else:
            self.bounds = None
        self.bound_shape = bound_shape
        self.boundary_repulsion = boundary_repulsion * self._REPULSION_SCALE

        # Statistics tracking
        self.intersections_blocked = 0
        self.intersection_checks = 0

        self.shape = shape
        self.nodes: List[Node] = []
        self._create_initial_nodes(initial_nodes, shape)
        self.spatial_hash = SpatialHash(self.repulsion_radius)

        self.logger.info(
            f"Initialized: base={self.base_length:.1f}, "
            f"edges=[{self.min_edge_length:.1f}, {self.max_edge_length:.1f}], "
            f"repulsion_r={self.repulsion_radius:.1f}"
        )

    def _create_initial_nodes(self, count: int, shape: str = 'circle') -> None:
        cx, cy = self.width / 2, self.height / 2
        size = min(self.width, self.height) / 8

        if shape == 'circle':
            two_pi = 2 * math.pi
            for i in range(count):
                angle = (two_pi * i) / count
                r = size * (1 + random.uniform(-0.1, 0.1))
                x = cx + r * math.cos(angle) + random.uniform(-2, 2)
                y = cy + r * math.sin(angle) + random.uniform(-2, 2)
                self.nodes.append(Node(x, y))

        elif shape == 'rectangle':
            # Distribute points along rectangle perimeter
            w, h = size * 2, size * 1.5
            perimeter = 2 * (w + h)
            for i in range(count):
                d = (perimeter * i) / count
                if d < w:  # Top edge
                    x, y = cx - w/2 + d, cy - h/2
                elif d < w + h:  # Right edge
                    x, y = cx + w/2, cy - h/2 + (d - w)
                elif d < 2*w + h:  # Bottom edge
                    x, y = cx + w/2 - (d - w - h), cy + h/2
                else:  # Left edge
                    x, y = cx - w/2, cy + h/2 - (d - 2*w - h)
                x += random.uniform(-2, 2)
                y += random.uniform(-2, 2)
                self.nodes.append(Node(x, y))

        elif shape == 'line':
            # Horizontal line - will grow outward
            length = size * 3
            for i in range(count):
                t = i / (count - 1) if count > 1 else 0.5
                x = cx - length/2 + t * length + random.uniform(-2, 2)
                y = cy + random.uniform(-2, 2)
                self.nodes.append(Node(x, y))

        elif shape == 'triangle':
            # Equilateral triangle
            points = [
                (cx, cy - size),  # Top
                (cx - size * 0.866, cy + size * 0.5),  # Bottom left
                (cx + size * 0.866, cy + size * 0.5),  # Bottom right
            ]
            edges = [(0, 1), (1, 2), (2, 0)]
            nodes_per_edge = count // 3
            for ei, (a, b) in enumerate(edges):
                p1, p2 = points[a], points[b]
                n = nodes_per_edge + (1 if ei < count % 3 else 0)
                for i in range(n):
                    t = i / n
                    x = p1[0] + t * (p2[0] - p1[0]) + random.uniform(-2, 2)
                    y = p1[1] + t * (p2[1] - p1[1]) + random.uniform(-2, 2)
                    self.nodes.append(Node(x, y))

        elif shape == 'star':
            # 5-pointed star
            two_pi = 2 * math.pi
            outer_r, inner_r = size, size * 0.4
            for i in range(count):
                angle = (two_pi * i) / count
                r = outer_r if (i % (count // 5)) < (count // 10) else inner_r
                r *= (1 + random.uniform(-0.1, 0.1))
                x = cx + r * math.cos(angle - math.pi/2) + random.uniform(-2, 2)
                y = cy + r * math.sin(angle - math.pi/2) + random.uniform(-2, 2)
                self.nodes.append(Node(x, y))

        else:
            raise ValueError(f"Unknown shape: {shape}. Use: circle, rectangle, line, triangle, star")

    def _rebuild_spatial_hash(self) -> None:
        sh = self.spatial_hash
        sh.clear()
        nodes = self.nodes
        n = len(nodes)

        for i in range(n):
            node = nodes[i]
            x, y = node.x, node.y
            sh.insert(i, x, y)

            next_i = (i + 1) % n
            nn = nodes[next_i]
            sh.insert_edge(i, x, y, nn.x, nn.y)

    def apply_forces(self, check_intersections: bool = True) -> None:
        nodes = self.nodes
        n = len(nodes)
        if n == 0:
            return

        self._rebuild_spatial_hash()

        # Reset forces
        for node in nodes:
            node.fx = 0.0
            node.fy = 0.0

        # Cache frequently used values
        min_edge = self.min_edge_length
        base_len = self.base_length
        rep_radius = self.repulsion_radius
        rep_radius_sq = self.repulsion_radius_sq
        attr_factor = self.attraction_factor
        rep_factor = self.repulsion_factor
        align_factor = self.alignment_factor
        growth_factor = self.growth_factor
        noise_factor = self.noise_factor
        sh = self.spatial_hash

        for i in range(n):
            node = nodes[i]
            nx, ny = node.x, node.y

            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            prev_node = nodes[prev_i]
            next_node = nodes[next_i]

            # --- Attraction to neighbors ---
            for neighbor in (prev_node, next_node):
                dx = neighbor.x - nx
                dy = neighbor.y - ny
                dist_sq = dx * dx + dy * dy
                if dist_sq > 1e-12:
                    dist = math.sqrt(dist_sq)
                    if dist > min_edge:
                        excess = (dist - min_edge) / base_len
                        force = excess * attr_factor / dist
                        node.fx += dx * force
                        node.fy += dy * force

            # --- Growth along normal ---
            tx = next_node.x - prev_node.x
            ty = next_node.y - prev_node.y
            tlen_sq = tx * tx + ty * ty
            if tlen_sq > 1e-12:
                tlen = math.sqrt(tlen_sq)
                # Normal is 90° CCW rotation of tangent
                node.fx += (-ty / tlen) * growth_factor
                node.fy += (tx / tlen) * growth_factor

            # --- Node repulsion (spatial hash) ---
            nearby = sh.query_radius(nx, ny, rep_radius)
            for j in nearby:
                if j == i:
                    continue
                other = nodes[j]
                dx = nx - other.x
                dy = ny - other.y
                dist_sq = dx * dx + dy * dy

                if dist_sq < rep_radius_sq and dist_sq > 1e-12:
                    dist = math.sqrt(dist_sq)
                    norm = 1.0 - dist / rep_radius
                    force = norm * rep_factor / dist
                    node.fx += dx * force
                    node.fy += dy * force

            # --- Edge repulsion (spatial hash) ---
            skip = {(i - 2) % n, (i - 1) % n, i, (i + 1) % n, (i + 2) % n}
            nearby_edges = sh.query_edges_near(nx, ny, rep_radius)

            for ei in nearby_edges:
                if ei in skip:
                    continue
                eni = (ei + 1) % n
                if eni in skip:
                    continue

                p1, p2 = nodes[ei], nodes[eni]
                # Inline point-to-segment distance
                abx, aby = p2.x - p1.x, p2.y - p1.y
                apx, apy = nx - p1.x, ny - p1.y
                ab_len_sq = abx * abx + aby * aby

                if ab_len_sq < 1e-10:
                    dist = math.sqrt(apx * apx + apy * apy)
                    cx, cy = p1.x, p1.y
                else:
                    t = (apx * abx + apy * aby) / ab_len_sq
                    t = max(0.0, min(1.0, t))
                    cx = p1.x + t * abx
                    cy = p1.y + t * aby
                    dx, dy = nx - cx, ny - cy
                    dist = math.sqrt(dx * dx + dy * dy)

                if dist < rep_radius and dist > 1e-6:
                    norm = 1.0 - dist / rep_radius
                    force = norm * norm * rep_factor * 1.5 / dist
                    node.fx += (nx - cx) * force
                    node.fy += (ny - cy) * force

            # --- Alignment ---
            mid_x = (prev_node.x + next_node.x) * 0.5
            mid_y = (prev_node.y + next_node.y) * 0.5
            dx, dy = mid_x - nx, mid_y - ny
            dist_sq = dx * dx + dy * dy
            if dist_sq > 1e-12:
                dist = math.sqrt(dist_sq)
                norm_dist = dist / base_len
                force = norm_dist * align_factor / dist
                node.fx += dx * force
                node.fy += dy * force

            # --- Noise ---
            node.fx += random.uniform(-1, 1) * noise_factor
            node.fy += random.uniform(-1, 1) * noise_factor

            # --- Boundary repulsion ---
            if self.bounds is not None and self.boundary_repulsion > 0:
                br = self.boundary_repulsion
                margin = rep_radius

                if self.bound_shape == 'rectangle':
                    bmin_x, bmin_y, bmax_x, bmax_y = self.bounds
                    # Repel from left edge
                    if nx - bmin_x < margin:
                        dist = max(1.0, nx - bmin_x)
                        node.fx += br * (margin / dist - 1)
                    # Repel from right edge
                    if bmax_x - nx < margin:
                        dist = max(1.0, bmax_x - nx)
                        node.fx -= br * (margin / dist - 1)
                    # Repel from top edge
                    if ny - bmin_y < margin:
                        dist = max(1.0, ny - bmin_y)
                        node.fy += br * (margin / dist - 1)
                    # Repel from bottom edge
                    if bmax_y - ny < margin:
                        dist = max(1.0, bmax_y - ny)
                        node.fy -= br * (margin / dist - 1)

                elif self.bound_shape == 'circle':
                    # Distance from center to node
                    dx = nx - self.bound_cx
                    dy = ny - self.bound_cy
                    dist_from_center = math.sqrt(dx * dx + dy * dy)
                    dist_to_edge = self.bound_radius - dist_from_center

                    if dist_to_edge < margin and dist_from_center > 1e-6:
                        # Push toward center
                        force = br * (margin / max(1.0, dist_to_edge) - 1)
                        node.fx -= (dx / dist_from_center) * force
                        node.fy -= (dy / dist_from_center) * force

                elif self.bound_shape == 'star':
                    # 5-pointed star boundary
                    dx = nx - self.bound_cx
                    dy = ny - self.bound_cy
                    dist_from_center = math.sqrt(dx * dx + dy * dy)
                    angle = math.atan2(dy, dx)

                    # Star radius varies with angle (5 points)
                    outer_r = self.bound_radius
                    inner_r = self.bound_radius * 0.4
                    # Modulate between inner and outer based on angle
                    t = (math.cos(5 * angle) + 1) / 2  # 0 to 1
                    star_radius = inner_r + t * (outer_r - inner_r)
                    dist_to_edge = star_radius - dist_from_center

                    if dist_to_edge < margin and dist_from_center > 1e-6:
                        force = br * (margin / max(1.0, dist_to_edge) - 1)
                        node.fx -= (dx / dist_from_center) * force
                        node.fy -= (dy / dist_from_center) * force

        # Apply forces and update positions
        damping = self.damping
        max_vel = self.max_velocity
        max_vel_sq = max_vel * max_vel

        for i in range(n):
            node = nodes[i]

            # Apply forces with damping
            vx = node.vx * damping + node.fx
            vy = node.vy * damping + node.fy

            # Cap velocity
            speed_sq = vx * vx + vy * vy
            if speed_sq > max_vel_sq:
                scale = max_vel / math.sqrt(speed_sq)
                vx *= scale
                vy *= scale

            node.vx = vx
            node.vy = vy

            new_x = node.x + vx
            new_y = node.y + vy

            if check_intersections:
                self.intersection_checks += 1
                if not self._would_cause_intersection(i, new_x, new_y):
                    node.x = new_x
                    node.y = new_y
                else:
                    self.intersections_blocked += 1
                    node.vx = 0
                    node.vy = 0
            else:
                node.x = new_x
                node.y = new_y

            # Clamp to bounding shape
            if self.bounds is not None:
                if self.bound_shape == 'rectangle':
                    bmin_x, bmin_y, bmax_x, bmax_y = self.bounds
                    node.x = max(bmin_x, min(bmax_x, node.x))
                    node.y = max(bmin_y, min(bmax_y, node.y))

                elif self.bound_shape == 'circle':
                    dx = node.x - self.bound_cx
                    dy = node.y - self.bound_cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > self.bound_radius:
                        # Project back onto circle edge
                        node.x = self.bound_cx + (dx / dist) * self.bound_radius
                        node.y = self.bound_cy + (dy / dist) * self.bound_radius

                elif self.bound_shape == 'star':
                    dx = node.x - self.bound_cx
                    dy = node.y - self.bound_cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > 1e-6:
                        angle = math.atan2(dy, dx)
                        outer_r = self.bound_radius
                        inner_r = self.bound_radius * 0.4
                        t = (math.cos(5 * angle) + 1) / 2
                        star_radius = inner_r + t * (outer_r - inner_r)
                        if dist > star_radius:
                            node.x = self.bound_cx + (dx / dist) * star_radius
                            node.y = self.bound_cy + (dy / dist) * star_radius

    def _would_cause_intersection(self, node_idx: int, new_x: float, new_y: float) -> bool:
        nodes = self.nodes
        n = len(nodes)
        if n < 4:
            return False

        pred_idx = (node_idx - 1) % n
        succ_idx = (node_idx + 1) % n
        pred, succ = nodes[pred_idx], nodes[succ_idx]

        min_sep = self.min_separation
        skip = {pred_idx, node_idx, succ_idx}

        # Query nearby edges
        sh = self.spatial_hash
        rep_r = self.repulsion_radius
        nearby: Set[int] = set()
        nearby.update(sh.query_edges_near(new_x, new_y, rep_r))
        nearby.update(sh.query_edges_near(pred.x, pred.y, rep_r))
        nearby.update(sh.query_edges_near(succ.x, succ.y, rep_r))

        for i in nearby:
            next_i = (i + 1) % n
            if i in skip or next_i in skip:
                continue

            p1, p2 = nodes[i], nodes[next_i]

            # Check intersection with line1 (pred -> new)
            if self._segments_intersect(pred.x, pred.y, new_x, new_y,
                                        p1.x, p1.y, p2.x, p2.y):
                return True

            # Check intersection with line2 (new -> succ)
            if self._segments_intersect(new_x, new_y, succ.x, succ.y,
                                        p1.x, p1.y, p2.x, p2.y):
                return True

            # Check minimum separation
            if self._segment_distance(pred.x, pred.y, new_x, new_y,
                                      p1.x, p1.y, p2.x, p2.y) < min_sep:
                return True
            if self._segment_distance(new_x, new_y, succ.x, succ.y,
                                      p1.x, p1.y, p2.x, p2.y) < min_sep:
                return True

        return False

    def _segments_intersect(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y) -> bool:
        denom = (p4y - p3y) * (p2x - p1x) - (p4x - p3x) * (p2y - p1y)
        if abs(denom) < 1e-10:
            return False
        ua = ((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)) / denom
        ub = ((p2x - p1x) * (p1y - p3y) - (p2y - p1y) * (p1x - p3x)) / denom
        return (0 < ua < 1) and (0 < ub < 1)

    def _point_segment_dist(self, px, py, ax, ay, bx, by) -> float:
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq < 1e-10:
            dx, dy = px - ax, py - ay
            return math.sqrt(dx * dx + dy * dy)
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
        cx, cy = ax + t * abx, ay + t * aby
        dx, dy = px - cx, py - cy
        return math.sqrt(dx * dx + dy * dy)

    def _segment_distance(self, a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y) -> float:
        return min(
            self._point_segment_dist(a1x, a1y, b1x, b1y, b2x, b2y),
            self._point_segment_dist(a2x, a2y, b1x, b1y, b2x, b2y),
            self._point_segment_dist(b1x, b1y, a1x, a1y, a2x, a2y),
            self._point_segment_dist(b2x, b2y, a1x, a1y, a2x, a2y)
        )

    def split_edges(self) -> int:
        nodes = self.nodes
        n = len(nodes)
        max_edge = self.max_edge_length
        min_sep = self.min_edge_length * 0.8

        # Find edges to split
        to_split = []
        for i in range(n):
            next_i = (i + 1) % n
            dx = nodes[next_i].x - nodes[i].x
            dy = nodes[next_i].y - nodes[i].y
            if dx * dx + dy * dy > max_edge * max_edge:
                mid_x = (nodes[i].x + nodes[next_i].x) * 0.5
                mid_y = (nodes[i].y + nodes[next_i].y) * 0.5
                if self._valid_split(mid_x, mid_y, i, min_sep):
                    to_split.append((i, mid_x, mid_y))

        # Insert in reverse order
        for i, mx, my in reversed(to_split):
            insert_idx = (i + 1) if (i + 1) < len(nodes) else len(nodes)
            nodes.insert(insert_idx, Node(mx, my))

        return len(to_split)

    def _valid_split(self, mx, my, skip_edge, min_sep) -> bool:
        nodes = self.nodes
        n = len(nodes)
        for i in range(n):
            if abs(i - skip_edge) <= 1 or abs(i - skip_edge) >= n - 1:
                continue
            next_i = (i + 1) % n
            dist = self._point_segment_dist(mx, my,
                nodes[i].x, nodes[i].y, nodes[next_i].x, nodes[next_i].y)
            if dist < min_sep:
                return False
        return True

    def step(self, step_num: int = 0, check_intersections: bool = True) -> None:
        # Optionally check intersections
        do_check = check_intersections and (step_num % 3 == 0)
        self.apply_forces(do_check)
        self.split_edges()

    def check_self_intersections(self) -> List[Tuple[int, int, float, float]]:
        """Check for self-intersections in the final curve.

        Returns list of (edge_i, edge_j, x, y) for each intersection found.
        Uses spatial hashing for efficiency.
        """
        nodes = self.nodes
        n = len(nodes)
        if n < 4:
            return []

        # Rebuild spatial hash for edges
        self._rebuild_spatial_hash()
        sh = self.spatial_hash

        intersections = []
        checked = set()

        for i in range(n):
            next_i = (i + 1) % n
            p1, p2 = nodes[i], nodes[next_i]

            # Query nearby edges
            mid_x = (p1.x + p2.x) * 0.5
            mid_y = (p1.y + p2.y) * 0.5
            edge_len = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            query_radius = max(self.repulsion_radius, edge_len)

            nearby = sh.query_edges_near(mid_x, mid_y, query_radius)

            for j in nearby:
                # Skip adjacent edges and already-checked pairs
                if abs(i - j) <= 1 or abs(i - j) >= n - 1:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)

                next_j = (j + 1) % n
                p3, p4 = nodes[j], nodes[next_j]

                # Check intersection
                if self._segments_intersect(p1.x, p1.y, p2.x, p2.y,
                                           p3.x, p3.y, p4.x, p4.y):
                    # Calculate intersection point
                    ix, iy = self._intersection_point(
                        p1.x, p1.y, p2.x, p2.y,
                        p3.x, p3.y, p4.x, p4.y
                    )
                    intersections.append((i, j, ix, iy))

        return intersections

    def _intersection_point(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y) -> Tuple[float, float]:
        """Calculate the intersection point of two line segments."""
        denom = (p4y - p3y) * (p2x - p1x) - (p4x - p3x) * (p2y - p1y)
        if abs(denom) < 1e-10:
            return (p1x + p2x) / 2, (p1y + p2y) / 2
        ua = ((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)) / denom
        ix = p1x + ua * (p2x - p1x)
        iy = p1y + ua * (p2y - p1y)
        return ix, iy

    def export_svg(self, filename: str) -> None:
        nodes = self.nodes
        n = len(nodes)
        if n == 0:
            return

        padding = 50
        min_x = min(node.x for node in nodes) - padding
        min_y = min(node.y for node in nodes) - padding
        max_x = max(node.x for node in nodes) + padding
        max_y = max(node.y for node in nodes) + padding

        svg = ET.Element('svg')
        svg.set('width', str(self.width))
        svg.set('height', str(self.height))
        svg.set('viewBox', f"{min_x:.2f} {min_y:.2f} {max_x - min_x:.2f} {max_y - min_y:.2f}")
        svg.set('xmlns', 'http://www.w3.org/2000/svg')

        # Catmull-Rom to Bezier for smooth curves
        points = [(node.x, node.y) for node in nodes]
        path_data = f"M {points[0][0]:.2f} {points[0][1]:.2f}"

        for i in range(n):
            p0 = points[(i - 1) % n]
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]

            # Catmull-Rom to Bezier conversion
            cp1x = p1[0] + (p2[0] - p0[0]) / 6
            cp1y = p1[1] + (p2[1] - p0[1]) / 6
            cp2x = p2[0] - (p3[0] - p1[0]) / 6
            cp2y = p2[1] - (p3[1] - p1[1]) / 6

            path_data += f" C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"

        path = ET.SubElement(svg, 'path')
        path.set('d', path_data)
        path.set('stroke', 'red')
        path.set('stroke-width', '3')
        path.set('fill', 'black')

        ET.ElementTree(svg).write(filename)
        self.logger.info(f"Exported {filename} ({n} nodes)")


def print_examples():
    """Print 30 example command lines with different patterns."""
    examples = [
        # Basic styles
        ("Dense branching", "--steps 2000 --growth 0.7 --repulsion 0.8 --noise 0.15 --alignment 0.5"),
        ("Sparse minimal", "--steps 2000 --growth 0.4 --repulsion 0.4 --noise 0.1 --alignment 0.5"),
        ("Regular symmetric", "--steps 2000 --growth 0.5 --repulsion 0.6 --noise 0.05 --alignment 0.7"),
        ("Organic natural", "--steps 2000 --growth 0.6 --repulsion 0.65 --noise 0.18 --alignment 0.5"),
        ("Chaotic wild", "--steps 1500 --growth 0.7 --repulsion 0.6 --noise 0.25 --alignment 0.3"),

        # High complexity
        ("Explosive growth", "--steps 2000 --growth 0.8 --repulsion 0.9 --noise 0.2 --alignment 0.4"),
        ("Fine tendrils", "--steps 3000 --growth 0.5 --repulsion 0.7 --noise 0.12 --alignment 0.6"),
        ("Coral reef", "--steps 2500 --growth 0.65 --repulsion 0.75 --noise 0.15 --alignment 0.45"),

        # Smooth curves
        ("Smooth flowing", "--steps 2000 --growth 0.5 --repulsion 0.6 --noise 0.08 --alignment 0.8"),
        ("Gentle waves", "--steps 1500 --growth 0.45 --repulsion 0.55 --noise 0.1 --alignment 0.75"),

        # Different starting shapes
        ("Rectangle start", "--steps 1500 --shape rectangle --growth 0.6 --repulsion 0.7 --noise 0.15"),
        ("Triangle start", "--steps 1500 --shape triangle --growth 0.6 --repulsion 0.7 --noise 0.15"),
        ("Star start", "--steps 1500 --shape star --growth 0.6 --repulsion 0.7 --noise 0.12"),
        ("Line start", "--steps 1500 --shape line --growth 0.55 --repulsion 0.7 --noise 0.12"),

        # Bounded patterns
        ("Circle bounded", "--steps 1500 --bounds 100 100 700 700 --bound-shape circle --boundary-repulsion 0.8 --growth 0.6 --repulsion 0.7"),
        ("Rectangle bounded", "--steps 1500 --bounds 100 100 700 700 --bound-shape rectangle --boundary-repulsion 0.8 --growth 0.6 --repulsion 0.7"),
        ("Tight circle", "--steps 2000 --bounds 200 200 600 600 --bound-shape circle --boundary-repulsion 1.0 --growth 0.7 --repulsion 0.8"),

        # Combined shape + bounds
        ("Triangle in circle", "--steps 1500 --shape triangle --bounds 100 100 700 700 --bound-shape circle --boundary-repulsion 0.8 --growth 0.6 --repulsion 0.7"),
        ("Star in rectangle", "--steps 1500 --shape star --bounds 100 100 700 700 --bound-shape rectangle --boundary-repulsion 0.8 --growth 0.6 --repulsion 0.7"),

        # Varying node counts
        ("Many initial nodes", "--steps 1000 --initial-nodes 40 --growth 0.5 --repulsion 0.6 --noise 0.1"),
        ("Few initial nodes", "--steps 2000 --initial-nodes 10 --growth 0.6 --repulsion 0.7 --noise 0.15"),

        # Edge length variations
        ("Fine detail", "--steps 2000 --min-edge-ratio 0.15 --max-edge-ratio 0.5 --growth 0.5 --repulsion 0.6"),
        ("Coarse strokes", "--steps 1500 --min-edge-ratio 0.4 --max-edge-ratio 1.0 --growth 0.6 --repulsion 0.7"),

        # Attraction/damping variations
        ("High attraction", "--steps 1500 --growth 0.6 --repulsion 0.7 --attraction 0.8 --noise 0.15"),
        ("Low damping", "--steps 1500 --growth 0.5 --repulsion 0.6 --damping 0.3 --noise 0.1"),
        ("High damping", "--steps 2000 --growth 0.6 --repulsion 0.7 --damping 0.8 --noise 0.15"),

        # Artistic presets
        ("Brain coral", "--steps 2500 --growth 0.55 --repulsion 0.7 --noise 0.1 --alignment 0.6"),
        ("Lightning bolt", "--steps 1000 --shape line --growth 0.8 --repulsion 0.5 --noise 0.3 --alignment 0.2"),
        ("Maze pattern", "--steps 2000 --growth 0.5 --repulsion 0.8 --noise 0.08 --alignment 0.7"),
        ("Organic blob", "--steps 1500 --growth 0.4 --repulsion 0.5 --noise 0.2 --alignment 0.4"),
    ]

    print("=" * 70)
    print("30 Example Command Lines for Different Patterns")
    print("=" * 70)
    print()
    print("Use with: pypy3 diffgrowth.py [options] --no-intersection-check --output NAME.svg")
    print()

    for i, (name, opts) in enumerate(examples, 1):
        seed = 100 + i
        print(f"# {i}. {name}")
        print(f"pypy3 diffgrowth.py {opts} --seed {seed} --no-intersection-check --output {name.lower().replace(' ', '_')}.svg")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Differential Growth Pattern Generator (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --steps 600 --seed 123
  %(prog)s --growth 0.7 --alignment 0.6 --output folded.svg
        """
    )

    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--initial-nodes', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--min-edge-ratio', type=float, default=0.25)
    parser.add_argument('--max-edge-ratio', type=float, default=0.75)
    parser.add_argument('--repulsion-radius-ratio', type=float, default=1.5)
    parser.add_argument('--attraction', type=float, default=0.5)
    parser.add_argument('--repulsion', type=float, default=0.5)
    parser.add_argument('--alignment', type=float, default=0.5)
    parser.add_argument('--growth', type=float, default=0.5)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--damping', type=float, default=0.5)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--shape', default='circle',
                        choices=['circle', 'rectangle', 'line', 'triangle', 'star'],
                        help='Initial shape to grow from')
    parser.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_X', 'MIN_Y', 'MAX_X', 'MAX_Y'),
                        help='Bounding region (e.g., --bounds 100 100 700 700)')
    parser.add_argument('--bound-shape', default='rectangle',
                        choices=['rectangle', 'circle', 'star'],
                        help='Shape of bounding constraint')
    parser.add_argument('--boundary-repulsion', type=float, default=0.0,
                        help='Force pushing nodes away from bounds (0-1)')
    parser.add_argument('--no-intersection-check', action='store_true',
                        help='Disable intersection checking (faster, requires balanced params)')
    parser.add_argument('--safe-mode', action='store_true',
                        help='Auto-adjust repulsion to ensure no intersections, disables checking')
    parser.add_argument('--output', default='growth.svg')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--examples', action='store_true',
                        help='Print 30 example command lines and exit')

    args = parser.parse_args()

    if args.examples:
        print_examples()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Handle safe mode: auto-adjust repulsion to guarantee no intersections
    repulsion = args.repulsion
    if args.safe_mode:
        safe_repulsion = DifferentialGrowth.get_safe_repulsion(args.growth, args.noise)
        if repulsion < safe_repulsion:
            repulsion = safe_repulsion
            logging.info(f"Safe mode: adjusted repulsion to {repulsion:.2f}")
        args.no_intersection_check = True

    # Warn if parameters are beyond safe constraints
    if not DifferentialGrowth.is_safe_params(args.growth, repulsion, args.noise):
        safe_r = DifferentialGrowth.get_safe_repulsion(args.growth, args.noise)
        logging.warning(
            f"Parameters beyond safe constraints (repulsion > 0.2*growth + 3*noise). "
            f"Current repulsion={repulsion:.2f}, recommended={safe_r:.2f}"
        )

    sim = DifferentialGrowth(
        width=args.width,
        height=args.height,
        initial_nodes=args.initial_nodes,
        min_edge_ratio=args.min_edge_ratio,
        max_edge_ratio=args.max_edge_ratio,
        repulsion_radius_ratio=args.repulsion_radius_ratio,
        attraction=args.attraction,
        repulsion=repulsion,
        alignment=args.alignment,
        growth=args.growth,
        noise=args.noise,
        damping=args.damping,
        random_seed=args.seed,
        shape=args.shape,
        bounds=tuple(args.bounds) if args.bounds else None,
        bound_shape=args.bound_shape,
        boundary_repulsion=args.boundary_repulsion
    )

    logging.info(f"Starting simulation: {args.steps} steps")

    check_intersections = not args.no_intersection_check
    for i in range(args.steps):
        sim.step(i, check_intersections=check_intersections)
        if (i + 1) % 50 == 0:
            logging.info(f"Step {i + 1}/{args.steps} - {len(sim.nodes)} nodes")

    # Always validate for self-intersections before export
    intersections = sim.check_self_intersections()
    if intersections:
        logging.warning(f"Intersection report: {len(intersections)} self-intersection(s) found")
        for i, j, x, y in intersections[:10]:  # Show first 10
            logging.warning(f"  Edges {i}-{j} intersect at ({x:.1f}, {y:.1f})")
        if len(intersections) > 10:
            logging.warning(f"  ... and {len(intersections) - 10} more")
        logging.warning("To prevent intersections, use --safe-mode")
    else:
        logging.info("Intersection report: clean (no self-intersections)")

    sim.export_svg(args.output)

    # Report statistics
    logging.info(f"Intersection checks: {sim.intersection_checks}")
    logging.info(f"Movements blocked: {sim.intersections_blocked}")
    if sim.intersection_checks > 0:
        block_rate = 100 * sim.intersections_blocked / sim.intersection_checks
        logging.info(f"Block rate: {block_rate:.1f}%")
    logging.info("Complete!")


if __name__ == "__main__":
    main()
