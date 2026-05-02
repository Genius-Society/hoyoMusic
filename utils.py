import os
import sys
import torch
import shutil
import subprocess

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

ZH2EN = {
    "Top-P 采样": "Top-P sample",
    "Top-K 采样 (0 为关闭)": "Top-K sample (0=closed)",
    "温度参数": "Temperature",
    "地区风格": "Region",
    "状态栏": "Status",
    "音频": "Audio",
    "下载 MIDI": "Download MIDI",
    "下载 PDF 乐谱": "Download PDF",
    "下载 MusicXML": "Download MusicXML",
    "下载 MXL": "Download MXL",
    "ABC 记谱": "ABC notation",
    "五线谱": "Staff",
    "原神音乐生成": "Genshin Music Generation",
    """欢迎使用此创空间, 此创空间基于 Tunesformer 开源项目制作，完全免费。当前模型还在调试中，计划在原神主线杀青后，所有国家地区角色全部开放后，二创音乐会齐全且样本均衡，届时重新微调模型并添加现实风格筛选辅助游戏各国家输出强化学习，以提升输出区分度与质量。注：崩铁方面数据工程正在运作中，未来也希望随主线杀青而基线化。<br>数据来源: <a href="https://musescore.org">MuseScore</a> 标签来源: <a href="https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki">Genshin Impact Wiki | Fandom</a> 模型基础: <a href="https://github.com/sander-wood/tunesformer">Tunesformer</a>""": """Welcome to this space based on the Tunesformer open source project, which is totally free! The current model is still in debugging, the plan is in the Genshin Impact after the main line is killed, all countries and regions after all the characters are open, the second creation of the concert will be complete and the sample is balanced, at that time to re-fine-tune the model and add the reality of the style of screening to assist in the game of each country's output to strengthen the learning in order to enhance the output differentiation and quality. Note: Data engineering on the Star Rail is in operation, and will hopefully be baselined in the future as well with the mainline kill.<br>Data source: <a href="https://musescore.org">MuseScore</a> Tags source: <a href="https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki">Genshin Impact Wiki | Fandom</a> Model base: <a href="https://github.com/sander-wood/tunesformer">Tunesformer</a>""",
}


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


TEYVAT = {
    "蒙德": "Mondstadt",
    "璃月": "Liyue",
    "稻妻": "Inazuma",
    "须弥": "Sumeru",
    "枫丹": "Fontaine",
    "纳塔": "Teyvat",  # Coming soon
}

if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "Genius-Society/hoyoMusic",
        cache_dir="./__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "Genius-Society/hoyoMusic",
        cache_dir="./__pycache__",
    )

WEIGHTS_PATH = f"{MODEL_DIR}/weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_DIR = "./__pycache__/tmp"
PATCH_LENGTH = 128  # Patch Length
PATCH_SIZE = 32  # Patch Size
PATCH_NUM_LAYERS = 9  # Number of layers in the encoder
CHAR_NUM_LAYERS = 3  # Number of layers in the decoder
PATCH_SAMPLING_BATCH_SIZE = 0  # Batchsize for patch during training, 0=full context
SHARE_WEIGHTS = False  # Whether to share weights between the encoder and decoder


if sys.platform.startswith("linux"):
    apkname = "MuseScore.AppImage"
    shutil.move(os.path.realpath(f"{MODEL_DIR}/{apkname}"), f"./{apkname}")
    extra_dir = "squashfs-root"
    if not os.path.exists(extra_dir):
        subprocess.run(["chmod", "+x", f"./{apkname}"])
        subprocess.run([f"./{apkname}", "--appimage-extract"])

    MSCORE = f"./{extra_dir}/AppRun"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

else:
    MSCORE = os.getenv("mscore")
    if not MSCORE:
        raise EnvironmentError("请配置好 mscore 环境变量!")
