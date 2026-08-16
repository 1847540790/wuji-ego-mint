
## 可视化观感（固定世界 3D / Wuji Hand，2026-08-05）
- 固定世界 3D 是**两套并行实现**：网页 `viewer/web/app.js:renderScene` 与导出 `render/fixed_world_video.py`。视角默认值与地面网格规格（`VIEW_AZ0/EL0`、`GRID_SPAN`、`GRID_TARGET_CELLS`、`GRID_MAJOR_EVERY`、`GROUND_CLEARANCE`）由 `test_viewer_frontend_contract.py` 逐条比对，改一端必须同步另一端。该坐标系无天然重力轴，「上」统一取**全段相机 +Y(下) 平均值取反**（Wuji 面板 `_gravity_up` 同口径）。
- 官方左右手 MJCF 可安全合成一个模型：名字都带 `left_/right_` 前缀不撞名，但两侧 `<compiler meshdir>` 冲突，必须把 `<mesh file>` 改写成绝对路径；视觉 geom 是 `group="1"`（各 26 个）、其余是重复碰撞网格。合并后靠 `mjvOption.geomgroup` 逐帧隐藏未检测的手（不出现也不投影），深度合成可整段删掉。
- ego 视角下**地面高度决定阴影是否进画面**：沿用 MuJoCo 面板的 0.85 m 会把地��和接触阴影推到画面外只剩天空，Wuji 面板取 0.35 m。MuJoCo 阴影本身在 EGL 下正常（探针验证过），弱是因为 headlight ambient 抢了对比度 —— 压 ambient、提 key diffuse 才看得出。
