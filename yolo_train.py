import os
import argparse
from ultralytics import YOLO

# 创建参数解析器
parser = argparse.ArgumentParser(description='YOLO training script')

# 添加命令行参数
parser.add_argument('--model_load_dir', type=str, default='/modeldir', help='Directory to load the model')
parser.add_argument('--train_model_out', type=str, default='/workspace/model-out', help='Output directory for the trained model')
parser.add_argument('--train_out', type=str, default='/workspace/out', help='Output directory for training results')
# 添加新的参数定义
parser.add_argument('--train_visualized_log', type=str, default='/workspace/visualizedlog', help='Directory for visualized training logs')
parser.add_argument('--data_url', type=str)
parser.add_argument('--gpu_num_per_node', type=int)


# 解析命令行参数
args = parser.parse_args()

# 获取参数值
model_dir = args.model_load_dir
output_dir = args.train_model_out
project = args.train_model_out
train_visualized_log = args.train_visualized_log
dataset = args.data_url


# 检查挂载路径是否存在
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found at: {model_dir}")

print(f"Using model directory: {model_dir}")

# 确定模型文件路径
yolo11n_path = os.path.join(model_dir, 'yolo11n.pt')

if not os.path.exists(yolo11n_path):
    raise FileNotFoundError(f"Model file not found at: {yolo11n_path}")

# 确定数据配置文件路径
yaml_file = os.path.join(dataset, 'data.yaml')

if not os.path.exists(yaml_file):
    raise FileNotFoundError(f"Data config file not found at: {yaml_file}")

# 加载 YOLO 模型
model = YOLO(yolo11n_path)

# 训练模型
model.train(
    data=yaml_file,
    imgsz=640,               # 图像尺寸
    batch=8,                 # CPU 模式下减小 batch size
    epochs=20,               # 训练周期
    device=0,             
    workers=4,                # CPU 线程数
    amp=False,                # 关闭混合精度训练
    project=project,  
    name="yolo-v11-cpu-experiment",
    save_period=10            # 每隔 10 轮保存一次
)

# 导出模型（ONNX）
export_path = model.export(
    format="onnx",
    imgsz=640,  # 可选的图像尺寸参数
    simplify=True  # 建议启用简化ONNX图
)
print(f"Model exported to: {export_path}")

# 你可以在这里使用 train_visualized_log 做一些操作，比如记录日志到该目录
print(f"Visualized training logs will be saved at: {train_visualized_log}")