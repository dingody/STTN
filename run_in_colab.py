"""
STTN Video Inpainting - Google Colab Setup Script
使用说明：
1. 将整个STTN项目文件夹上传到Colab
2. 运行此脚本进行环境配置和模型运行
"""

import os
import sys
import subprocess

def install_dependencies():
    """安装Colab所需的依赖包"""
    print("Installing dependencies for Colab...")
    
    # 升级pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # 安装主要依赖
    requirements = [
        "torch>=1.8.0",
        "torchvision>=0.9.0", 
        "opencv-python==4.6.0.66",
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.49.0",
        "imageio>=2.8.0",
        "scipy>=1.6.0"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")

def download_pretrained_model():
    """下载预训练模型"""
    print("Downloading pretrained model...")
    
    # 创建checkpoints目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 提示用户手动下载模型
    print("""
    请手动下载预训练模型：
    1. 访问：https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view?usp=sharing
    2. 下载 sttn.pth 文件
    3. 将文件放在 checkpoints/ 目录下
    
    或者使用以下命令（如果可以的话）：
    !gdown 1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv -O checkpoints/sttn.pth
    """)

def check_gpu():
    """检查GPU可用性"""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        return True
    else:
        print("✗ No GPU available")
        return False

def run_example():
    """运行示例"""
    print("Running video inpainting example...")
    
    # 检查必要文件是否存在
    required_files = [
        "examples/schoolgirls_orig.mp4",
        "examples/schoolgirls",
        "checkpoints/sttn.pth"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # 导入并运行
    try:
        from test_colab import run_video_inpainting
        result = run_video_inpainting()
        if result:
            print(f"✓ Video inpainting completed! Output: {result}")
            return True
        else:
            print("✗ Video inpainting failed!")
            return False
    except Exception as e:
        print(f"✗ Error running video inpainting: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("STTN Video Inpainting - Colab Setup")
    print("=" * 50)
    
    # 检查当前目录
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # 安装依赖
    install_dependencies()
    
    # 检查GPU
    check_gpu()
    
    # 下载模型提示
    download_pretrained_model()
    
    # 运行示例（如果文件齐全）
    print("\n" + "=" * 50)
    print("Ready to run! Use the following code:")
    print("=" * 50)
    print("""
# 基本用法
from test_colab import run_video_inpainting
result = run_video_inpainting()

# 自定义参数
result = run_video_inpainting(
    video_path="your_video.mp4",
    mask_path="your_mask_folder", 
    ckpt_path="checkpoints/sttn.pth"
)
    """)

if __name__ == "__main__":
    main()