"""Actual MuJoCo side-by-side rendering of the deterministically selected repair case."""

import json
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from pickplace.env import PickPlaceEnv

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts/v1"


def render():
    pair = json.loads((OUT / "paired-delay_50ms.json").read_text())
    env = PickPlaceEnv(render_mode="rgb_array")
    env.reset(seed=pair["scenario_seed"])
    font_path = Path("/System/Library/Fonts/Supplemental/Georgia.ttf")
    font = (
        ImageFont.truetype(str(font_path), 20)
        if font_path.exists()
        else ImageFont.load_default(size=20)
    )
    small = (
        ImageFont.truetype(str(font_path), 15)
        if font_path.exists()
        else ImageFont.load_default(size=15)
    )
    total = max(len(pair[name]["frames"]) for name in ("identity", "ema_half"))
    images = []
    try:
        for index in range(0, total, 2):
            image = Image.new("RGB", (1280, 554), "#f3f2ea")
            draw = ImageDraw.Draw(image)
            for column, name in enumerate(("identity", "ema_half")):
                replay = pair[name]
                frame = replay["frames"][min(index, len(replay["frames"]) - 1)]
                env.data.qpos[:] = frame["qpos"]
                mujoco.mj_forward(env.model, env.data)
                image.paste(Image.fromarray(env.render()), (640 * column, 56))
                label = "Raw BC256" if name == "identity" else "Same weights + EMA(0.5)"
                draw.text((20 + 640 * column, 10), label, font=font, fill="#263d30")
                ended = index >= len(replay["frames"]) - 1
                status = ("SUCCESS" if replay["success"] else "TIMEOUT") if ended else "EXECUTING"
                draw.text(
                    (20 + 640 * column, 36),
                    f"50 ms command delay | {status} | t={frame['time']:.2f}s",
                    font=small,
                    fill="#5e6e55",
                )
            draw.text(
                (20, 535),
                f"Paired scene {pair['scenario_seed']} / training {pair['training_seed']} / actual MuJoCo state rendering",
                font=small,
                fill="#263d30",
            )
            images.append(image)
    finally:
        env.close()
    images[0].save(
        OUT / "paired-delay-50ms.gif", save_all=True, append_images=images[1:], duration=100, loop=0
    )
    images[len(images) // 2].save(OUT / "paired-delay-50ms.png")
    print(OUT / "paired-delay-50ms.gif")


if __name__ == "__main__":
    render()
