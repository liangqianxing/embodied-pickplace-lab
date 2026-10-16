"""MuJoCo rendering of an actually verified Panda policy replay, no synthetic motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont
from smoke import make_env
from study import OUT


def render(replay_path=None):
    index = json.loads((OUT / "replay_index.json").read_text())
    if replay_path is None:
        record = next(r for r in index if r["condition"] == "clean" and r["held_8_steps"])
        replay_path = OUT / record["path"]
    replay = json.loads(Path(replay_path).read_text())
    env = make_env(replay["seed"])
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=480, width=640)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1
    scene_option.sitegroup[:] = 0
    font_path = Path("/System/Library/Fonts/Supplemental/Georgia.ttf")
    font = (
        ImageFont.truetype(str(font_path), 18)
        if font_path.exists()
        else ImageFont.load_default(size=18)
    )
    images = []
    try:
        sampled_frames = replay["frames"][::2]
        if sampled_frames[-1]["step"] != replay["frames"][-1]["step"]:
            sampled_frames.append(replay["frames"][-1])
        for frame in sampled_frames:
            data.qpos[:] = frame["qpos"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="frontview", scene_option=scene_option)
            canvas = Image.new("RGB", (640, 540), "#f3f2ea")
            canvas.paste(Image.fromarray(renderer.render().copy()), (0, 60))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 8), "Panda Lift / trained state-based BC", font=font, fill="#263d30")
            draw.text(
                (16, 34),
                f"Scene {replay['seed']} | {frame['step'] / 20:.2f}s | {replay['condition']['name']}",
                font=font,
                fill="#64745c",
            )
            images.append(canvas)
    finally:
        renderer.close()
        env.close()
    destination = OUT / "media"
    destination.mkdir(exist_ok=True)
    images[-1].save(destination / "panda-lift.png")
    images[0].save(
        destination / "panda-lift.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    (destination / "provenance.json").write_text(
        json.dumps(
            {
                "replay": str(Path(replay_path).relative_to(OUT)),
                "seed": replay["seed"],
                "training_seed": replay["training_seed"],
                "official_success": replay["official_success"],
                "held_8_steps": replay["held_8_steps"],
                "display_fps": 10,
                "control_hz": 20,
            },
            indent=2,
        )
        + "\n"
    )
    print(destination / "panda-lift.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    render(parser.parse_args().replay)
