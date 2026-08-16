import sys
import threading
from pathlib import Path


MODEL_EFFECT = Path(__file__).resolve().parents[2]
WEB_DIR = MODEL_EFFECT / "visualization" / "viewer" / "web"
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))


def test_robot_panels_wait_for_2d_and_share_playback_and_row():
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    style = (WEB_DIR / "style.css").read_text(encoding="utf-8")

    assert "function _wireServerVideoPanel(pan,badge,{autoStart=true}={})" in source
    assert "const primary2DReady=new Promise" in source
    assert "_wireServerVideoPanel(pan,badge,{autoStart:false})" in source
    assert "if(state.panels[pan.id]===pan) pan.releaseRender();" in source
    assert "const primarySceneIds = new Set(['mujoco_3d','wuji_retarget_3d']);" in source
    assert source.count("video.controls=false") >= 2
    assert source.count("video.classList.add('synced-follower-video')") == 2
    assert ".primary-scene-row" in style
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in style
    assert ".mujoco-wrap video.synced-follower-video { pointer-events: none; }" in style


def test_export_renders_overall_2d_before_robot_views_and_keeps_layout_order(tmp_path):
    from visualization.viewer.store import Store

    store = Store.__new__(Store)
    primary_done = threading.Event()
    calls = []
    composed = []

    def render_2d(*args, **kwargs):
        calls.append("both_2d")
        primary_done.set()
        return tmp_path / "2d.mp4"

    def render_robot(name, path):
        def render(*args, **kwargs):
            assert primary_done.is_set()
            calls.append(name)
            return tmp_path / path

        return render

    store.mp4 = render_2d
    store.mujoco_video = render_robot("mujoco_3d", "mujoco.mp4")
    store.retarget_video = render_robot("wuji_retarget_3d", "retarget.mp4")
    store.world_video = render_robot("world_motion_3d", "world.mp4")
    store._compose_export = lambda inputs: composed.extend(inputs) or tmp_path / "export.mp4"

    result = store.export_video(
        0,
        ["mujoco_3d", "both_2d", "wuji_retarget_3d"],
        mode="mesh_skel",
    )

    assert calls[0] == "both_2d"
    assert [source_id for source_id, _ in composed] == [
        "mujoco_3d", "both_2d", "wuji_retarget_3d",
    ]
    assert result == tmp_path / "export.mp4"


def test_export_without_2d_still_renders_selected_robot_views(tmp_path):
    from visualization.viewer.store import Store

    store = Store.__new__(Store)
    composed = []
    store.mujoco_video = lambda *args, **kwargs: tmp_path / "mujoco.mp4"
    store.retarget_video = lambda *args, **kwargs: tmp_path / "retarget.mp4"
    store._compose_export = lambda inputs: composed.extend(inputs) or tmp_path / "export.mp4"

    store.export_video(
        0,
        ["mujoco_3d", "wuji_retarget_3d"],
        mode="mesh_skel",
    )

    assert [source_id for source_id, _ in composed] == [
        "mujoco_3d", "wuji_retarget_3d",
    ]
