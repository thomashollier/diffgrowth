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


_NAMED_COLORS = {
    'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
    'green': (0, 128, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
    'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'orange': (255, 165, 0),
    'brown': (139, 90, 43), 'darkgreen': (0, 100, 0), 'forestgreen': (34, 139, 34),
    'limegreen': (50, 205, 50), 'darkred': (139, 0, 0), 'crimson': (220, 20, 60),
    'navy': (0, 0, 128), 'teal': (0, 128, 128), 'purple': (128, 0, 128),
    'gray': (128, 128, 128), 'grey': (128, 128, 128), 'silver': (192, 192, 192),
    'gold': (255, 215, 0), 'coral': (255, 127, 80), 'salmon': (250, 128, 114),
    'olive': (128, 128, 0), 'sienna': (160, 82, 45), 'tan': (210, 180, 140),
    'ivory': (255, 255, 240), 'khaki': (240, 230, 140),
}


def parse_color(color: str) -> Tuple[int, int, int]:
    """Parse a color string to (r, g, b). Supports named colors and #RRGGBB hex."""
    c = color.strip().lower()
    if c in _NAMED_COLORS:
        return _NAMED_COLORS[c]
    if c.startswith('#') and len(c) == 7:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    if c.startswith('rgb(') and c.endswith(')'):
        parts = c[4:-1].split(',')
        return int(parts[0]), int(parts[1]), int(parts[2])
    return _NAMED_COLORS.get(c, (255, 0, 0))


class Node:
    """Optimized node using __slots__ for faster attribute access."""
    __slots__ = ('x', 'y', 'vx', 'vy', 'fx', 'fy', 'birth_step')

    def __init__(self, x: float, y: float, birth_step: int = 0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.birth_step = birth_step


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


class SVGPathParser:
    """Parse SVG files and extract shape polygons (pure Python, no dependencies)."""

    logger = logging.getLogger(__name__)

    @classmethod
    def parse_file(cls, filepath: str, num_samples: int = 200) -> List[List[Tuple[float, float]]]:
        """Load SVG file and return list of polygons (each a list of (x,y) tuples)."""
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Handle SVG namespace
        ns = ''
        tag = root.tag
        if tag.startswith('{'):
            ns = tag[:tag.index('}') + 1]

        polygons = []

        for elem in root.iter():
            local = elem.tag.replace(ns, '') if ns else elem.tag
            pts = None

            if local == 'path':
                d = elem.get('d', '')
                if d:
                    pts = cls._parse_path_d(d, num_samples)
            elif local in ('rect', 'circle', 'ellipse', 'polygon', 'polyline'):
                pts = cls._parse_basic_shape(local, elem)

            if pts and len(pts) >= 3:
                polygons.append(pts)

        return polygons

    @classmethod
    def _tokenize_path(cls, d: str) -> List:
        """Tokenize SVG path d attribute into commands and numbers."""
        tokens = []
        i = 0
        n = len(d)
        while i < n:
            c = d[i]
            if c in ' ,\t\n\r':
                i += 1
                continue
            if c.isalpha():
                tokens.append(c)
                i += 1
                continue
            # Parse number (including negative, decimal, exponent)
            start = i
            if c in '+-':
                i += 1
            has_dot = False
            while i < n and (d[i].isdigit() or (d[i] == '.' and not has_dot)):
                if d[i] == '.':
                    has_dot = True
                i += 1
            # Exponent
            if i < n and d[i] in 'eE':
                i += 1
                if i < n and d[i] in '+-':
                    i += 1
                while i < n and d[i].isdigit():
                    i += 1
            if i > start:
                try:
                    tokens.append(float(d[start:i]))
                except ValueError:
                    i = start + 1  # skip bad char
            else:
                i += 1  # skip unrecognized
        return tokens

    @classmethod
    def _parse_path_d(cls, d: str, num_samples: int) -> List[Tuple[float, float]]:
        """Parse SVG path d attribute and return sampled points."""
        tokens = cls._tokenize_path(d)
        points: List[Tuple[float, float]] = []
        cx, cy = 0.0, 0.0  # current point
        sx, sy = 0.0, 0.0  # subpath start
        last_cp = None  # last control point for S/T smooth curves
        last_cmd = ''

        i = 0
        n = len(tokens)

        def take_float() -> float:
            nonlocal i
            if i < n and isinstance(tokens[i], float):
                val = tokens[i]
                i += 1
                return val
            return 0.0

        while i < n:
            tok = tokens[i]
            if isinstance(tok, str):
                cmd = tok
                i += 1
            else:
                # Implicit repeat of last command (L after M)
                cmd = last_cmd
                if cmd == 'M':
                    cmd = 'L'
                elif cmd == 'm':
                    cmd = 'l'

            if cmd in ('M', 'm'):
                x, y = take_float(), take_float()
                if cmd == 'm':
                    x += cx; y += cy
                cx, cy = x, y
                sx, sy = x, y
                points.append((cx, cy))
                last_cp = None
                last_cmd = cmd

            elif cmd in ('L', 'l'):
                x, y = take_float(), take_float()
                if cmd == 'l':
                    x += cx; y += cy
                cx, cy = x, y
                points.append((cx, cy))
                last_cp = None
                last_cmd = cmd

            elif cmd in ('H', 'h'):
                x = take_float()
                if cmd == 'h':
                    x += cx
                cx = x
                points.append((cx, cy))
                last_cp = None
                last_cmd = cmd

            elif cmd in ('V', 'v'):
                y = take_float()
                if cmd == 'v':
                    y += cy
                cy = y
                points.append((cx, cy))
                last_cp = None
                last_cmd = cmd

            elif cmd in ('C', 'c'):
                x1, y1 = take_float(), take_float()
                x2, y2 = take_float(), take_float()
                x, y = take_float(), take_float()
                if cmd == 'c':
                    x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
                pts = cls._sample_cubic_bezier((cx, cy), (x1, y1), (x2, y2), (x, y), 16)
                points.extend(pts[1:])
                last_cp = (x2, y2)
                cx, cy = x, y
                last_cmd = cmd

            elif cmd in ('S', 's'):
                # Smooth cubic: reflect last control point
                if last_cmd in ('C', 'c', 'S', 's') and last_cp:
                    x1 = 2 * cx - last_cp[0]
                    y1 = 2 * cy - last_cp[1]
                else:
                    x1, y1 = cx, cy
                x2, y2 = take_float(), take_float()
                x, y = take_float(), take_float()
                if cmd == 's':
                    x2 += cx; y2 += cy; x += cx; y += cy
                pts = cls._sample_cubic_bezier((cx, cy), (x1, y1), (x2, y2), (x, y), 16)
                points.extend(pts[1:])
                last_cp = (x2, y2)
                cx, cy = x, y
                last_cmd = cmd

            elif cmd in ('Q', 'q'):
                x1, y1 = take_float(), take_float()
                x, y = take_float(), take_float()
                if cmd == 'q':
                    x1 += cx; y1 += cy; x += cx; y += cy
                pts = cls._sample_quadratic_bezier((cx, cy), (x1, y1), (x, y), 12)
                points.extend(pts[1:])
                last_cp = (x1, y1)
                cx, cy = x, y
                last_cmd = cmd

            elif cmd in ('T', 't'):
                # Smooth quadratic: reflect last control point
                if last_cmd in ('Q', 'q', 'T', 't') and last_cp:
                    x1 = 2 * cx - last_cp[0]
                    y1 = 2 * cy - last_cp[1]
                else:
                    x1, y1 = cx, cy
                x, y = take_float(), take_float()
                if cmd == 't':
                    x += cx; y += cy
                pts = cls._sample_quadratic_bezier((cx, cy), (x1, y1), (x, y), 12)
                points.extend(pts[1:])
                last_cp = (x1, y1)
                cx, cy = x, y
                last_cmd = cmd

            elif cmd in ('A', 'a'):
                # Arc: consume 7 params, approximate as line (with warning)
                _rx = take_float(); _ry = take_float(); _rot = take_float()
                _large = take_float(); _sweep = take_float()
                x, y = take_float(), take_float()
                if cmd == 'a':
                    x += cx; y += cy
                cls.logger.warning("SVG arc commands approximated as straight lines")
                cx, cy = x, y
                points.append((cx, cy))
                last_cp = None
                last_cmd = cmd

            elif cmd in ('Z', 'z'):
                cx, cy = sx, sy
                last_cp = None
                last_cmd = cmd

            else:
                i += 1  # skip unknown
                last_cmd = cmd

        if len(points) < 3:
            return points
        return cls._resample_evenly(points, num_samples)

    @classmethod
    def _parse_basic_shape(cls, tag: str, elem) -> Optional[List[Tuple[float, float]]]:
        """Parse basic SVG shape elements into point lists."""

        def attr(name: str, default: float = 0.0) -> float:
            val = elem.get(name)
            return float(val) if val is not None else default

        if tag == 'rect':
            x, y = attr('x'), attr('y')
            w, h = attr('width'), attr('height')
            if w <= 0 or h <= 0:
                return None
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        elif tag == 'circle':
            cx, cy, r = attr('cx'), attr('cy'), attr('r')
            if r <= 0:
                return None
            pts = []
            n = 64
            for i in range(n):
                angle = 2 * math.pi * i / n
                pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            return pts

        elif tag == 'ellipse':
            cx, cy = attr('cx'), attr('cy')
            rx, ry = attr('rx'), attr('ry')
            if rx <= 0 or ry <= 0:
                return None
            pts = []
            n = 64
            for i in range(n):
                angle = 2 * math.pi * i / n
                pts.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
            return pts

        elif tag in ('polygon', 'polyline'):
            raw = elem.get('points', '')
            if not raw:
                return None
            # Parse space/comma separated coordinate pairs
            nums = []
            for tok in raw.replace(',', ' ').split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
            pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            return pts if len(pts) >= 3 else None

        return None

    @staticmethod
    def _sample_cubic_bezier(p0, p1, p2, p3, n: int) -> List[Tuple[float, float]]:
        """Sample n+1 points along a cubic Bezier curve."""
        pts = []
        for i in range(n + 1):
            t = i / n
            u = 1 - t
            uu, tt = u * u, t * t
            uuu, ttt = uu * u, tt * t
            x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
            y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
            pts.append((x, y))
        return pts

    @staticmethod
    def _sample_quadratic_bezier(p0, p1, p2, n: int) -> List[Tuple[float, float]]:
        """Sample n+1 points along a quadratic Bezier curve."""
        pts = []
        for i in range(n + 1):
            t = i / n
            u = 1 - t
            x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
            y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
            pts.append((x, y))
        return pts

    @staticmethod
    def _resample_evenly(points: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
        """Resample a polyline to n evenly-spaced points by arc length."""
        if len(points) < 2 or n < 2:
            return points

        # Compute cumulative arc lengths
        lengths = [0.0]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            lengths.append(lengths[-1] + math.sqrt(dx * dx + dy * dy))

        total = lengths[-1]
        if total < 1e-10:
            return points[:n]

        result = []
        seg = 0
        for i in range(n):
            target = total * i / n
            while seg < len(lengths) - 2 and lengths[seg + 1] < target:
                seg += 1
            seg_len = lengths[seg + 1] - lengths[seg]
            if seg_len < 1e-10:
                t = 0.0
            else:
                t = (target - lengths[seg]) / seg_len
            x = points[seg][0] + t * (points[seg + 1][0] - points[seg][0])
            y = points[seg][1] + t * (points[seg + 1][1] - points[seg][1])
            result.append((x, y))
        return result

    @staticmethod
    def remove_narrow_sections(points: List[Tuple[float, float]], min_width: float) -> List[Tuple[float, float]]:
        """Remove narrow protrusions (like antennae) from a polygon.

        Finds places where non-adjacent boundary points are closer than min_width,
        then shortcuts across, cutting off the narrow section.
        """
        n = len(points)
        if n < 10:
            return points

        min_gap = max(10, n // 10)  # minimum index distance to count as non-adjacent
        min_width_sq = min_width * min_width

        while True:
            n = len(points)
            best_cut = None
            best_removed = 0

            for i in range(n):
                px, py = points[i]
                for j in range(i + min_gap, n):
                    # Also check wrap-around adjacency
                    if (n - j + i) < min_gap:
                        continue
                    dx = points[j][0] - px
                    dy = points[j][1] - py
                    if dx * dx + dy * dy < min_width_sq:
                        # Found a bottleneck — cut the shorter side
                        forward = j - i
                        backward = n - forward
                        removed = min(forward, backward)
                        if removed > best_removed:
                            best_removed = removed
                            if forward <= backward:
                                best_cut = (i, j)
                            else:
                                best_cut = (j, i)

            if best_cut is None:
                break

            # Remove points between best_cut[0] and best_cut[1]
            a, b = best_cut
            if a < b:
                points = points[:a + 1] + points[b:]
            else:
                points = points[b:a + 1]

            if len(points) < 10:
                break

        return points

    @staticmethod
    def fit_to_canvas(
        points: List[Tuple[float, float]],
        width: float, height: float,
        scale: Optional[float] = None,
        margin: float = 50.0
    ) -> List[Tuple[float, float]]:
        """Center and scale polygon to fit within canvas dimensions."""
        if not points:
            return points

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pw, ph = max_x - min_x, max_y - min_y

        if pw < 1e-10 and ph < 1e-10:
            return [(width / 2, height / 2)] * len(points)

        if scale is None:
            # Auto-fit: scale to fill canvas minus margin
            avail_w = width - 2 * margin
            avail_h = height - 2 * margin
            scale = min(avail_w / max(pw, 1e-10), avail_h / max(ph, 1e-10))

        # Center of polygon
        pcx = (min_x + max_x) / 2
        pcy = (min_y + max_y) / 2
        # Center of canvas
        ccx, ccy = width / 2, height / 2

        result = []
        for x, y in points:
            rx = (x - pcx) * scale + ccx
            ry = (y - pcy) * scale + ccy
            result.append((rx, ry))
        return result


class PolygonBoundary:
    """Efficient polygon boundary for constraint testing using spatial hashing."""

    def __init__(self, points: List[Tuple[float, float]], cell_size: float = 30.0):
        self.points = points
        n = len(points)
        self.edges = []
        for i in range(n):
            j = (i + 1) % n
            self.edges.append((points[i][0], points[i][1], points[j][0], points[j][1]))

        # Build spatial hash of polygon edges
        self.cell_size = cell_size
        self.edge_grid: Dict[Tuple[int, int], List[int]] = {}
        for ei, (x1, y1, x2, y2) in enumerate(self.edges):
            min_cx = int(min(x1, x2) // cell_size)
            max_cx = int(max(x1, x2) // cell_size)
            min_cy = int(min(y1, y2) // cell_size)
            max_cy = int(max(y1, y2) // cell_size)
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    key = (cx, cy)
                    if key not in self.edge_grid:
                        self.edge_grid[key] = []
                    self.edge_grid[key].append(ei)

    def point_in_polygon(self, x: float, y: float) -> bool:
        """Ray casting algorithm (even-odd rule)."""
        inside = False
        pts = self.points
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def distance_to_boundary(self, x: float, y: float) -> Tuple[float, float, float]:
        """Return (distance, nearest_x, nearest_y) to polygon boundary.

        Uses spatial hash for efficient nearest-edge lookup.
        """
        cs = self.cell_size
        cx_i, cy_i = int(x // cs), int(y // cs)

        best_dist = float('inf')
        best_x, best_y = x, y

        # Search expanding rings until we find edges
        for ring in range(max(len(self.edge_grid), 10)):
            found_any = False
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if abs(dx) != ring and abs(dy) != ring:
                        continue  # Only check border cells of ring
                    key = (cx_i + dx, cy_i + dy)
                    edges = self.edge_grid.get(key)
                    if edges is None:
                        continue
                    found_any = True
                    for ei in edges:
                        ex1, ey1, ex2, ey2 = self.edges[ei]
                        nx, ny, d = self._point_to_segment(x, y, ex1, ey1, ex2, ey2)
                        if d < best_dist:
                            best_dist = d
                            best_x, best_y = nx, ny
            if found_any and best_dist <= (ring + 1) * cs * 1.5:
                break  # Close enough, no need to search further

        return best_dist, best_x, best_y

    @staticmethod
    def _point_to_segment(px, py, ax, ay, bx, by) -> Tuple[float, float, float]:
        """Return (nearest_x, nearest_y, distance) from point to segment."""
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq < 1e-10:
            dx, dy = px - ax, py - ay
            return ax, ay, math.sqrt(dx * dx + dy * dy)
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
        nx = ax + t * abx
        ny = ay + t * aby
        dx, dy = px - nx, py - ny
        return nx, ny, math.sqrt(dx * dx + dy * dy)

    def clamp_to_interior(self, x: float, y: float) -> Tuple[float, float]:
        """If point is outside polygon, project to nearest boundary edge."""
        if self.point_in_polygon(x, y):
            return x, y
        _, nx, ny = self.distance_to_boundary(x, y)
        return nx, ny

    def get_repulsion_force(self, x: float, y: float, margin: float, strength: float) -> Tuple[float, float]:
        """Return (fx, fy) repulsion force pushing away from boundary."""
        dist, bx, by = self.distance_to_boundary(x, y)

        if dist >= margin:
            return 0.0, 0.0

        force_mag = strength * (margin / max(1.0, dist) - 1)

        if dist < 1e-6:
            # Point is on boundary; push toward polygon center
            pts = self.points
            pcx = sum(p[0] for p in pts) / len(pts)
            pcy = sum(p[1] for p in pts) / len(pts)
            dx, dy = pcx - x, pcy - y
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                return 0.0, 0.0
            return (dx / d) * force_mag, (dy / d) * force_mag

        # Push away from nearest boundary point (toward interior)
        dx, dy = x - bx, y - by
        d = math.sqrt(dx * dx + dy * dy)
        if not self.point_in_polygon(x, y):
            # Outside: push inward (toward boundary point)
            return (-dx / d) * force_mag, (-dy / d) * force_mag
        # Inside: push away from boundary (toward interior)
        return (dx / d) * force_mag, (dy / d) * force_mag


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
        boundary_repulsion: float = 0.0,
        svg_polygon: Optional[List[Tuple[float, float]]] = None,
        svg_mode: str = 'grow',
        detail_scale: float = 1.0,
        directional_strength: float = 0.0,
        directional_angle: float = 270.0,
        twist_strength: float = 0.0,
        start_offset: Optional[Tuple[float, float]] = None
    ):
        self.width = width
        self.height = height
        self.base_length = min(width, height) / 40.0 * detail_scale

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

        # SVG polygon handling
        self.polygon_boundary: Optional[PolygonBoundary] = None
        self.svg_grow_points: Optional[List[Tuple[float, float]]] = None

        if svg_polygon is not None and svg_mode in ('constrain', 'fill'):
            self.polygon_boundary = PolygonBoundary(svg_polygon, cell_size=self.repulsion_radius)
            bound_shape = 'svg'
            xs = [p[0] for p in svg_polygon]
            ys = [p[1] for p in svg_polygon]
            bounds = (min(xs), min(ys), max(xs), max(ys))
            if svg_mode == 'fill':
                # Start nodes along the boundary edge, grow inward
                self.svg_grow_points = svg_polygon
                shape = 'svg'
        elif svg_polygon is not None and svg_mode == 'grow':
            self.svg_grow_points = svg_polygon
            shape = 'svg'

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

        # Directional force and twist
        self.dir_strength = directional_strength * self._GROWTH_SCALE
        self.dir_angle_rad = math.radians(directional_angle)
        self.twist_strength = twist_strength * self._GROWTH_SCALE

        # Statistics tracking
        self.intersections_blocked = 0
        self.intersection_checks = 0
        self.current_step = 0

        self.start_offset = start_offset or (0.0, 0.0)
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
        cx = self.width / 2 + self.start_offset[0]
        cy = self.height / 2 + self.start_offset[1]
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

        elif shape == 'svg':
            # Use pre-parsed SVG polygon points with slight jitter
            if self.svg_grow_points:
                for px, py in self.svg_grow_points:
                    x = px + random.uniform(-1, 1)
                    y = py + random.uniform(-1, 1)
                    self.nodes.append(Node(x, y))

        else:
            raise ValueError(f"Unknown shape: {shape}. Use: circle, rectangle, line, triangle, star, svg")

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

        # Directional + twist parameters
        dir_str = self.dir_strength
        dir_fx = math.cos(self.dir_angle_rad) * dir_str
        dir_fy = math.sin(self.dir_angle_rad) * dir_str
        twist_str = self.twist_strength
        center_x = self.width * 0.5
        center_y = self.height * 0.5

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

            # --- Directional force (uniform) ---
            node.fx += dir_fx
            node.fy += dir_fy

            # --- Twist (tangential force around center) ---
            if twist_str != 0.0:
                dx_c = nx - center_x
                dy_c = ny - center_y
                dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
                if dist_c > 1e-6:
                    node.fx += (-dy_c / dist_c) * twist_str
                    node.fy += (dx_c / dist_c) * twist_str

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

                elif self.bound_shape == 'svg' and self.polygon_boundary is not None:
                    fx, fy = self.polygon_boundary.get_repulsion_force(nx, ny, margin, br)
                    node.fx += fx
                    node.fy += fy

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

                elif self.bound_shape == 'svg' and self.polygon_boundary is not None:
                    node.x, node.y = self.polygon_boundary.clamp_to_interior(node.x, node.y)

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
            nodes.insert(insert_idx, Node(mx, my, birth_step=self.current_step))

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
        self.current_step = step_num
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

    def resolve_intersections(self) -> int:
        """Resolve self-intersections via 2-opt segment reversal.

        When edges (i, i+1) and (j, j+1) cross, reversing the node order
        between them reconnects the path so the crossing disappears.
        One reversal per iteration since indices shift after each swap.

        Returns the number of intersections resolved.
        """
        total_resolved = 0

        while True:
            intersections = self.check_self_intersections()
            if not intersections:
                if total_resolved > 0:
                    logging.info(f"All intersections resolved ({total_resolved} 2-opt swap(s))")
                return total_resolved

            # Resolve one crossing per pass (indices shift after reversal)
            edge_i, edge_j, ix, iy = intersections[0]
            n = len(self.nodes)
            i, j = min(edge_i, edge_j), max(edge_i, edge_j)

            # Reverse the shorter of the two arcs between the crossing edges
            forward_len = j - i  # nodes in segment [i+1 .. j]
            if forward_len <= n // 2:
                self.nodes[i + 1:j + 1] = self.nodes[i + 1:j + 1][::-1]
            else:
                # Reverse the wrap-around segment [j+1 .. n-1, 0 .. i]
                backward_len = n - forward_len
                indices = [(j + 1 + k) % n for k in range(backward_len)]
                for k in range(backward_len // 2):
                    a, b = indices[k], indices[backward_len - 1 - k]
                    self.nodes[a], self.nodes[b] = self.nodes[b], self.nodes[a]

            total_resolved += 1

    def _intersection_point(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y) -> Tuple[float, float]:
        """Calculate the intersection point of two line segments."""
        denom = (p4y - p3y) * (p2x - p1x) - (p4x - p3x) * (p2y - p1y)
        if abs(denom) < 1e-10:
            return (p1x + p2x) / 2, (p1y + p2y) / 2
        ua = ((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)) / denom
        ix = p1x + ua * (p2x - p1x)
        iy = p1y + ua * (p2y - p1y)
        return ix, iy

    def export_svg(self, filename: str, variable_stroke: bool = False,
                   stroke_curves: float = 6.0, stroke_straights: float = 0.5,
                   stroke_angle: float = 0.0, stroke_multiplier: float = 1.0,
                   stroke_color: str = 'red', fill_color: str = 'black',
                   stroke_tip: Optional[str] = None) -> None:
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

        points = [(node.x, node.y) for node in nodes]

        # Pre-compute per-segment age colors if stroke_tip is set
        seg_colors = None
        if stroke_tip is not None:
            max_step = max(nd.birth_step for nd in nodes)
            min_step = min(nd.birth_step for nd in nodes)
            step_range = max(1, max_step - min_step)
            old_r, old_g, old_b = parse_color(stroke_color)
            new_r, new_g, new_b = parse_color(stroke_tip)
            seg_colors = []
            for i in range(n):
                age1 = nodes[i].birth_step
                age2 = nodes[(i + 1) % n].birth_step
                avg_age = (age1 + age2) / 2.0
                t = (avg_age - min_step) / step_range  # 0=oldest, 1=youngest
                r = int(old_r + t * (new_r - old_r))
                g = int(old_g + t * (new_g - old_g))
                b = int(old_b + t * (new_b - old_b))
                seg_colors.append(f"rgb({r},{g},{b})")

        if not variable_stroke and seg_colors is None:
            # Single path with uniform stroke
            path_data = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
            for i in range(n):
                p0 = points[(i - 1) % n]
                p1 = points[i]
                p2 = points[(i + 1) % n]
                p3 = points[(i + 2) % n]
                cp1x = p1[0] + (p2[0] - p0[0]) / 6
                cp1y = p1[1] + (p2[1] - p0[1]) / 6
                cp2x = p2[0] - (p3[0] - p1[0]) / 6
                cp2y = p2[1] - (p3[1] - p1[1]) / 6
                path_data += f" C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"

            path = ET.SubElement(svg, 'path')
            path.set('d', path_data)
            path.set('stroke', stroke_color)
            path.set('stroke-width', '3')
            path.set('fill', fill_color)
        elif not variable_stroke and seg_colors is not None:
            # Age-colored: fill background, then per-segment colored strokes
            bg_data = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
            for i in range(n):
                p0 = points[(i - 1) % n]
                p1 = points[i]
                p2 = points[(i + 1) % n]
                p3 = points[(i + 2) % n]
                cp1x = p1[0] + (p2[0] - p0[0]) / 6
                cp1y = p1[1] + (p2[1] - p0[1]) / 6
                cp2x = p2[0] - (p3[0] - p1[0]) / 6
                cp2y = p2[1] - (p3[1] - p1[1]) / 6
                bg_data += f" C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"
            bg = ET.SubElement(svg, 'path')
            bg.set('d', bg_data)
            bg.set('stroke', 'none')
            bg.set('fill', fill_color)

            for i in range(n):
                p0 = points[(i - 1) % n]
                p1 = points[i]
                p2 = points[(i + 1) % n]
                p3 = points[(i + 2) % n]
                cp1x = p1[0] + (p2[0] - p0[0]) / 6
                cp1y = p1[1] + (p2[1] - p0[1]) / 6
                cp2x = p2[0] - (p3[0] - p1[0]) / 6
                cp2y = p2[1] - (p3[1] - p1[1]) / 6
                d = (f"M {p1[0]:.2f} {p1[1]:.2f} "
                     f"C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}")
                seg = ET.SubElement(svg, 'path')
                seg.set('d', d)
                seg.set('stroke', seg_colors[i])
                seg.set('stroke-width', '3')
                seg.set('fill', 'none')
                seg.set('stroke-linecap', 'round')
        else:
            # Per-segment paths with curvature-dependent stroke width
            # Compute curvature at each node (Menger curvature from 3 points)
            curvatures = []
            for i in range(n):
                px, py = points[(i - 1) % n]
                cx, cy = points[i]
                nx, ny = points[(i + 1) % n]
                # Triangle side lengths
                a = math.sqrt((cx - px)**2 + (cy - py)**2)
                b = math.sqrt((nx - cx)**2 + (ny - cy)**2)
                c = math.sqrt((nx - px)**2 + (ny - py)**2)
                # Twice the signed area
                area2 = abs((cx - px) * (ny - py) - (nx - px) * (cy - py))
                denom = a * b * c
                curvatures.append(area2 / denom if denom > 1e-10 else 0.0)

            # Map curvature to stroke width
            sorted_c = sorted(curvatures)
            ref_curv = sorted_c[len(sorted_c) // 2] * 3  # 3x median as "high"
            if ref_curv < 1e-10:
                ref_curv = 1.0

            # Compute per-node tangent directions, normals, and widths
            angle_rad = math.radians(stroke_angle)
            vec_x = math.cos(angle_rad)
            vec_y = math.sin(angle_rad)

            normals = []
            widths = []
            for i in range(n):
                tx = points[(i + 1) % n][0] - points[(i - 1) % n][0]
                ty = points[(i + 1) % n][1] - points[(i - 1) % n][1]
                tlen = math.sqrt(tx * tx + ty * ty)
                if tlen > 1e-10:
                    normals.append((-ty / tlen, tx / tlen))
                    # Directional multiplier: alignment of tangent with vector
                    alignment = abs((tx / tlen) * vec_x + (ty / tlen) * vec_y)
                    dir_mult = 1.0 + (stroke_multiplier - 1.0) * alignment
                else:
                    normals.append((0.0, 0.0))
                    dir_mult = 1.0

                # Curvature-based width * directional multiplier
                t = min(curvatures[i] / ref_curv, 1.0)
                w = stroke_straights + t * (stroke_curves - stroke_straights)
                widths.append(w * dir_mult)

            # Background fill
            bg_data = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]
                p0 = points[(i - 1) % n]
                p3 = points[(i + 2) % n]
                cp1x = p1[0] + (p2[0] - p0[0]) / 6
                cp1y = p1[1] + (p2[1] - p0[1]) / 6
                cp2x = p2[0] - (p3[0] - p1[0]) / 6
                cp2y = p2[1] - (p3[1] - p1[1]) / 6
                bg_data += f" C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"
            bg = ET.SubElement(svg, 'path')
            bg.set('d', bg_data)
            bg.set('stroke', 'none')
            bg.set('fill', fill_color)

            # Draw each segment as a filled tapered ribbon
            for i in range(n):
                i1 = (i + 1) % n
                p0 = points[(i - 1) % n]
                p1 = points[i]
                p2 = points[i1]
                p3 = points[(i + 2) % n]

                # Catmull-Rom to Bezier control points
                cp1x = p1[0] + (p2[0] - p0[0]) / 6
                cp1y = p1[1] + (p2[1] - p0[1]) / 6
                cp2x = p2[0] - (p3[0] - p1[0]) / 6
                cp2y = p2[1] - (p3[1] - p1[1]) / 6

                # Half-widths at start and end
                hw1 = widths[i] * 0.5
                hw2 = widths[i1] * 0.5
                n1x, n1y = normals[i]
                n2x, n2y = normals[i1]

                # Interpolate normal offset for control points (1/3 and 2/3)
                hw_cp1 = hw1 + (hw2 - hw1) / 3
                hw_cp2 = hw1 + (hw2 - hw1) * 2 / 3
                nc1x = n1x + (n2x - n1x) / 3
                nc1y = n1y + (n2y - n1y) / 3
                nc2x = n1x + (n2x - n1x) * 2 / 3
                nc2y = n1y + (n2y - n1y) * 2 / 3

                # Left side (positive normal offset)
                l1x, l1y = p1[0] + n1x * hw1, p1[1] + n1y * hw1
                lc1x, lc1y = cp1x + nc1x * hw_cp1, cp1y + nc1y * hw_cp1
                lc2x, lc2y = cp2x + nc2x * hw_cp2, cp2y + nc2y * hw_cp2
                l2x, l2y = p2[0] + n2x * hw2, p2[1] + n2y * hw2

                # Right side (negative normal offset)
                r1x, r1y = p1[0] - n1x * hw1, p1[1] - n1y * hw1
                rc1x, rc1y = cp1x - nc1x * hw_cp1, cp1y - nc1y * hw_cp1
                rc2x, rc2y = cp2x - nc2x * hw_cp2, cp2y - nc2y * hw_cp2
                r2x, r2y = p2[0] - n2x * hw2, p2[1] - n2y * hw2

                # Closed shape: left curve forward, right curve backward
                d = (f"M {l1x:.2f} {l1y:.2f} "
                     f"C {lc1x:.2f} {lc1y:.2f}, {lc2x:.2f} {lc2y:.2f}, {l2x:.2f} {l2y:.2f} "
                     f"L {r2x:.2f} {r2y:.2f} "
                     f"C {rc2x:.2f} {rc2y:.2f}, {rc1x:.2f} {rc1y:.2f}, {r1x:.2f} {r1y:.2f} Z")

                seg = ET.SubElement(svg, 'path')
                seg.set('d', d)
                seg.set('fill', seg_colors[i] if seg_colors else stroke_color)
                seg.set('stroke', 'none')

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
    parser.add_argument('--detail-scale', type=float, default=1.0,
                        help='Global scale for pattern detail (0.5=finer, 2.0=coarser)')
    parser.add_argument('--no-intersection-check', action='store_true',
                        help='Disable intersection checking (faster, requires balanced params)')
    parser.add_argument('--no-post-process', action='store_true',
                        help='Skip post-process intersection cleanup')
    parser.add_argument('--safe-mode', action='store_true',
                        help='Auto-adjust repulsion to ensure no intersections, disables checking')
    parser.add_argument('--variable-stroke', action='store_true',
                        help='Vary stroke width by local curvature and direction')
    parser.add_argument('--stroke-curves', type=float, default=6.0,
                        help='Stroke width at tight curves (default: 6.0)')
    parser.add_argument('--stroke-straights', type=float, default=0.5,
                        help='Stroke width at straight sections (default: 0.5)')
    parser.add_argument('--stroke-angle', type=float, default=0.0,
                        help='Direction angle in degrees for directional thickness (0-360)')
    parser.add_argument('--stroke-multiplier', type=float, default=1.0,
                        help='Thickness multiplier parallel to stroke-angle (perp=1.0)')
    parser.add_argument('--stroke-color', default='red',
                        help='Stroke/line color (default: red)')
    parser.add_argument('--fill-color', default='black',
                        help='Fill color for closed curve interior (default: black)')
    parser.add_argument('--stroke-tip', default=None,
                        help='Tip color for age gradient (old=stroke-color, young=stroke-tip)')
    parser.add_argument('--directional-strength', type=float, default=0.0,
                        help='Uniform directional pull force (like gravity/wind)')
    parser.add_argument('--directional-angle', type=float, default=270.0,
                        help='Direction angle in degrees (270=down, 0=right)')
    parser.add_argument('--twist-strength', type=float, default=0.0,
                        help='Tangential twist force around image center (creates spirals)')
    parser.add_argument('--start-offset', type=float, nargs=2, metavar=('X', 'Y'),
                        help='Offset starting position from center (pixels)')
    parser.add_argument('--output', default='growth.svg')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--examples', action='store_true',
                        help='Print 30 example command lines and exit')

    # SVG import arguments
    parser.add_argument('--svg-file', type=str, default=None,
                        help='SVG file to import as starting shape or boundary')
    parser.add_argument('--svg-mode', choices=['grow', 'constrain', 'fill'], default='grow',
                        help='How to use SVG: grow (outward), constrain (fill from center), fill (inward from edge)')
    parser.add_argument('--svg-scale', type=float, default=None,
                        help='Scale factor for SVG shape (default: auto-fit to canvas)')
    parser.add_argument('--svg-samples', type=int, default=None,
                        help='Number of points to sample along SVG path (default: auto)')
    parser.add_argument('--svg-min-width', type=float, default=None,
                        help='Remove narrow sections thinner than this (in pixels, filters antennae etc.)')

    args = parser.parse_args()

    if args.examples:
        print_examples()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse SVG file if provided
    svg_polygon = None
    if args.svg_file:
        # Auto-determine sample count
        if args.svg_samples is not None:
            num_samples = args.svg_samples
        elif args.svg_mode == 'grow':
            num_samples = args.initial_nodes
        else:
            num_samples = 500

        polygons = SVGPathParser.parse_file(args.svg_file, num_samples=num_samples)
        if not polygons:
            logging.error(f"No valid shapes found in {args.svg_file}")
            return

        # Select the largest polygon (by number of points, as proxy for complexity)
        svg_polygon = max(polygons, key=len)
        logging.info(f"Loaded SVG: {len(polygons)} shape(s), using largest ({len(svg_polygon)} points)")

        # Scale and center to canvas
        svg_polygon = SVGPathParser.fit_to_canvas(
            svg_polygon, args.width, args.height,
            scale=args.svg_scale, margin=50.0
        )

        # Filter narrow sections if requested
        if args.svg_min_width is not None:
            before = len(svg_polygon)
            svg_polygon = SVGPathParser.remove_narrow_sections(svg_polygon, args.svg_min_width)
            logging.info(f"Narrow filter ({args.svg_min_width}px): {before} -> {len(svg_polygon)} points")

        # Set sensible default for constrain/fill mode boundary repulsion
        if args.svg_mode in ('constrain', 'fill') and args.boundary_repulsion == 0.0:
            args.boundary_repulsion = 0.8
            logging.info(f"SVG {args.svg_mode} mode: auto-set boundary-repulsion to 0.8")

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
        boundary_repulsion=args.boundary_repulsion,
        svg_polygon=svg_polygon,
        svg_mode=args.svg_mode,
        detail_scale=args.detail_scale,
        directional_strength=args.directional_strength,
        directional_angle=args.directional_angle,
        twist_strength=args.twist_strength,
        start_offset=tuple(args.start_offset) if args.start_offset else None
    )

    logging.info(f"Starting simulation: {args.steps} steps")

    check_intersections = not args.no_intersection_check
    for i in range(args.steps):
        sim.step(i, check_intersections=check_intersections)
        if (i + 1) % 50 == 0:
            logging.info(f"Step {i + 1}/{args.steps} - {len(sim.nodes)} nodes")

    # Post-process: resolve self-intersections
    if not args.no_post_process:
        resolved = sim.resolve_intersections()
        if resolved:
            logging.info(f"Post-process resolved {resolved} intersection(s)")

    # Final intersection report
    intersections = sim.check_self_intersections()
    if intersections:
        logging.warning(f"Intersection report: {len(intersections)} self-intersection(s) remain")
        for i, j, x, y in intersections[:10]:  # Show first 10
            logging.warning(f"  Edges {i}-{j} intersect at ({x:.1f}, {y:.1f})")
        if len(intersections) > 10:
            logging.warning(f"  ... and {len(intersections) - 10} more")
        logging.warning("To prevent intersections, use --safe-mode")
    else:
        logging.info("Intersection report: clean (no self-intersections)")

    sim.export_svg(args.output, variable_stroke=args.variable_stroke,
                   stroke_curves=args.stroke_curves, stroke_straights=args.stroke_straights,
                   stroke_angle=args.stroke_angle, stroke_multiplier=args.stroke_multiplier,
                   stroke_color=args.stroke_color, fill_color=args.fill_color,
                   stroke_tip=args.stroke_tip)

    # Report statistics
    logging.info(f"Intersection checks: {sim.intersection_checks}")
    logging.info(f"Movements blocked: {sim.intersections_blocked}")
    if sim.intersection_checks > 0:
        block_rate = 100 * sim.intersections_blocked / sim.intersection_checks
        logging.info(f"Block rate: {block_rate:.1f}%")
    logging.info("Complete!")


if __name__ == "__main__":
    main()
