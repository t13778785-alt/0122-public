import os
from PIL import Image
from shutil import  rmtree
import logging

# ====================== 环境变量配置 ======================
# 输入图片目录
INPUT_DIR = os.environ.get("INPUT_DIR", "/data/cleaning/source")
# 输出图片目录
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/cleaning/target")
# 旋转角度（度数），正数表示逆时针方向
ROTATE_ANGLE = float(os.environ.get("ROTATE_ANGLE", "90"))
# 是否保持原尺寸（True = 不裁切，图片会变大；False = 裁切到原尺寸）
EXPAND = os.environ.get("EXPAND", "false").lower() == "true"

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def rotate_image(image_path: str, angle: float, expand: bool):
    """
    旋转单张图片
    :param image_path: 图片文件路径
    :param angle: 旋转角度（度数）
    :param expand: 是否根据旋转结果调整图片尺寸
    :return: 旋转后的 PIL.Image 对象
    """
    with Image.open(image_path) as img:
        return img.rotate(angle, expand=expand)

def main():
    
    logging.info(f"Rotating images from {INPUT_DIR} to {OUTPUT_DIR}")
    logging.info(f"Rotation angle: {ROTATE_ANGLE} degrees, Expand: {EXPAND}")
    # 清空输出目录（可选）
    if os.path.exists(OUTPUT_DIR):
        # 只删除目录下的所有文件和子目录，不删除目录本身
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                rmtree(item_path)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Input directory ready: {INPUT_DIR}")
    logging.info(f"Output directory ready: {OUTPUT_DIR}")
    processed_count = 0
    error_count = 0

    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            rotated_img = rotate_image(filepath, ROTATE_ANGLE, EXPAND)
            rotated_img.save(os.path.join(OUTPUT_DIR, filename))
            processed_count += 1
            logging.info(f"Rotated image: {filename}")
        except Exception as e:
            logging.error(f"Error rotating {filename}: {e}")
            error_count += 1

    logging.info(f"Done. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
