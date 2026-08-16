#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable MuJoCo MP4 renderer for the web viewer.

The standalone ``mujoco_view.py`` remains the feature-rich CLI. This module
extracts its ego-camera render path into a data-source-independent function so
the Flask viewer can compare MuJoCo output directly against the source video.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


_WEB_FLOOR_CLEARANCE = 0.85
_ROBOT_VIDEO_PRESET = os.environ.get("VIEWER_ROBOT_VIDEO_PRESET", "superfast")

try:
    _ENCODE_BUFFER_FRAMES = max(
        0, int(os.environ.get("VIEWER_ROBOT_ENCODE_BUFFER_FRAMES", "3")))
except ValueError:
    _ENCODE_BUFFER_FRAMES = 3


def _first_valid(mask: np.ndarray) -> int:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(indices[0]) if len(indices) else 0


def _rot_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source) + 1e-9
    target /= np.linalg.norm(target) + 1e-9
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if np.linalg.norm(cross) < 1e-8:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cosine))


def _gravity_axis(cam_c2w: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    cameras = np.asarray(cam_c2w, dtype=float)
    points = np.asarray(hand_points, dtype=float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    forward = cameras[:, :3, 2].mean(0)
    forward /= np.linalg.norm(forward) + 1e-9
    centers = cameras[:, :3, 3]
    span_all = (float(np.linalg.norm(
        np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)))
        + 1e-9)

    best, best_score = None, None
    for axis in np.concatenate([np.eye(3), -np.eye(3)], axis=0):
        pitch = -np.degrees(np.arcsin(np.clip(float(np.dot(forward, axis)), -1.0, 1.0)))
        if not 5.0 <= pitch <= 60.0:
            continue
        if float(np.dot(centers.mean(0), axis)) <= float(np.percentile(points @ axis, 95)):
            continue
        height = points @ axis
        score = -(float(np.percentile(height, 98) - np.percentile(height, 2)) / span_all)
        if best_score is None or score > best_score:
            best, best_score = axis, score
    if best is not None:
        return best

    # 没有可靠物理判据时也只选世界坐标轴，避免用手点云拟合出每段不同的斜地面。
    camera_up = (-cameras[:, :3, 1]).mean(0)
    axis_index = int(np.argmax(np.abs(camera_up)))
    up = np.zeros(3, dtype=float)
    up[axis_index] = 1.0 if camera_up[axis_index] >= 0 else -1.0
    return up


def _upright_rotation(cam_c2w: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    return _rot_align(_gravity_axis(cam_c2w, hand_points), np.array([0.0, 0.0, 1.0]))


def render_world_video(world: dict, cam_c2w: np.ndarray, kept: np.ndarray,
                       output: str | Path, *, fps: float,
                       intrinsics: np.ndarray | None = None,
                       image_size: tuple[int, int] | None = None,
                       width: int = 960, height: int | None = None,
                       view: str = "ego",
                       on_step=None) -> Path:
    """Render one GT or prediction from its source-video camera viewpoint."""
    from . import draw, mujoco_scene

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cameras = np.asarray(cam_c2w, dtype=np.float64).copy()
    validity = np.asarray(kept, dtype=bool)
    if validity.ndim != 2 or validity.shape[1] != 2:
        raise ValueError(f"kept 应为 [T,2]，实际为 {validity.shape}")

    copied = {}
    for side in ("left", "right"):
        vertices = world[side].get("verts")
        if vertices is None:
            raise RuntimeError("MuJoCo 需要 MANO 网格；当前结果只有 21 点关节")
        copied[side] = {
            "verts": np.asarray(vertices, dtype=np.float64).copy(),
            "joints": np.asarray(world[side]["joints"], dtype=np.float64).copy(),
        }
    frames = len(cameras)
    if frames <= 0 or validity.shape[0] != frames:
        raise ValueError("MuJoCo 相机、手部与有效掩码帧数不一致")
    K = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
    if K is not None and (K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError(f"相机内参无效: {K.shape}")
    if image_size is None:
        source_size = None
    else:
        source_size = tuple(map(int, image_size))
        if len(source_size) != 2 or min(source_size) <= 0:
            raise ValueError(f"原视频尺寸无效: {image_size}")
    if view == "ego" and (K is None or source_size is None):
        raise ValueError("视频视角 MuJoCo 渲染需要相机内参和原视频尺寸")
    render_width = max(2, int(width))
    render_width += render_width % 2
    if height is None:
        if source_size is None:
            render_height = 540
        else:
            render_height = max(2, int(round(render_width * source_size[1] / source_size[0])))
            render_height += render_height % 2
    else:
        render_height = max(2, int(height))

    hand_points = np.concatenate([
        copied["left"]["joints"].reshape(-1, 3),
        copied["right"]["joints"].reshape(-1, 3),
    ], axis=0)
    upright = _upright_rotation(cameras, hand_points)
    for side in ("left", "right"):
        copied[side]["verts"] = copied[side]["verts"] @ upright.T
        copied[side]["joints"] = copied[side]["joints"] @ upright.T
    cameras[:, :3, :3] = upright[None] @ cameras[:, :3, :3]
    cameras[:, :3, 3] = cameras[:, :3, 3] @ upright.T

    rotated_points = np.concatenate([
        copied["left"]["joints"].reshape(-1, 3),
        copied["right"]["joints"].reshape(-1, 3),
    ], axis=0)
    rotated_points = rotated_points[np.all(np.isfinite(rotated_points), axis=1)]
    origin = np.median(rotated_points, axis=0) if len(rotated_points) else np.zeros(3)
    for side in ("left", "right"):
        copied[side]["verts"] -= origin
        copied[side]["joints"] -= origin
    cameras[:, :3, 3] -= origin

    vertices_left = copied["left"]["verts"]
    vertices_right = copied["right"]["verts"]
    faces_right, faces_left = draw.get_faces()
    scene = mujoco_scene.HandWorldScene(
        faces_left, faces_right,
        vertices_left[_first_valid(validity[:, 0])],
        vertices_right[_first_valid(validity[:, 1])],
        width=render_width, height=render_height, floor=True,
        operation_mat=False, ego_floor_screen_level=True,
    )
    writer = None
    try:
        valid_vertices = []
        if validity[:, 0].any():
            valid_vertices.append(vertices_left[validity[:, 0]].reshape(-1, 3))
        if validity[:, 1].any():
            valid_vertices.append(vertices_right[validity[:, 1]].reshape(-1, 3))
        floor_points = (np.concatenate(valid_vertices, axis=0) if valid_vertices
                        else np.concatenate([vertices_left, vertices_right], axis=1).reshape(-1, 3))
        floor_points = floor_points[np.all(np.isfinite(floor_points), axis=1)]

        forward = cameras[:, :3, 2]
        forward = forward[np.all(np.isfinite(forward), axis=1)]
        mean_forward = forward.mean(0) if len(forward) else np.array([1.0, 0.0, 0.0])
        forward_xy = mean_forward[:2]
        forward_azimuth = (float(np.arctan2(forward_xy[1], forward_xy[0]))
                           if np.linalg.norm(forward_xy) > 1e-6 else 0.0)
        # Web 画面不放桌面/操作垫；地面固定为世界 XY 水平面，并与手留出距离。
        scene.place_floor(floor_points, margin=_WEB_FLOOR_CLEARANCE, fwd_az=None)

        camera_points = cameras[:, :3, 3]
        camera_points = camera_points[np.all(np.isfinite(camera_points), axis=1)]
        hand_finite = floor_points[np.all(np.isfinite(floor_points), axis=1)]
        mat_center, mat_half = scene._mat_center, scene._mat_half
        mat_corners = np.array([
            [mat_center[0] + sx * mat_half[0], mat_center[1] + sy * mat_half[1], mat_center[2]]
            for sx in (-1, 1) for sy in (-1, 1)
        ])
        key_points = np.concatenate([
            np.percentile(hand_finite, [1, 99], axis=0),
            np.percentile(camera_points, [1, 99], axis=0),
            mat_corners,
        ], axis=0)
        low, high = key_points.min(0), key_points.max(0)
        fit = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
        look_at = (low + high) / 2.0
        look_at[2] = float(hand_finite[:, 2].mean()) * 0.55 + float(camera_points[:, 2].mean()) * 0.45
        scene.autofit_camera(fit, azimuth=140.0, elevation=-22.0,
                             lookat=look_at, margin=1.25)

        base = float(np.degrees(forward_azimuth))
        views = {
            "front": (base + 180.0, -22.0),
            "back": (base, -22.0),
            "right": (base + 90.0, -22.0),
            "left": (base - 90.0, -22.0),
            "top": (base, -88.0),
        }
        if view not in views:
            if view != "ego":
                raise ValueError(f"未知 MuJoCo 视角: {view}")
            azimuth = elevation = 0.0
        else:
            azimuth, elevation = views[view]
        fovy_deg = (None if K is None or source_size is None else
                    float(np.degrees(2.0 * np.arctan(
                        (source_size[1] / 2.0) / float(K[1, 1])))))
        writer = draw.H264PipeWriter(
            output, float(fps), (render_width, render_height),
            preset=_ROBOT_VIDEO_PRESET,
            buffered_frames=_ENCODE_BUFFER_FRAMES)
        for frame in range(frames):
            scene.bake_frame(
                vertices_left[frame], vertices_right[frame],
                bool(validity[frame, 0]), bool(validity[frame, 1]),
            )
            if view == "ego":
                scene.hide_ghosts()
                scene.set_floor_ego_ground(cameras[frame], fovy_deg)
                scene.set_ego(cameras[frame], fovy_deg)
                image = scene.render_ego(K, source_size)
            else:
                scene.set_floor_ground()
                image = scene.render_free(azimuth, elevation, cameras, frame)
            writer.write(np.ascontiguousarray(image[:, :, ::-1]))
            if on_step is not None:
                on_step(frame + 1, frames)
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
        scene.close()
    return output
