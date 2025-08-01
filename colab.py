# -*- coding: utf-8 -*-
"""
STTN Video Inpainting for Google Colab
Single file solution for video inpainting in Colab environment

Usage:
1. Initialize environment: sttn = STTNColab()
2. Run video inpainting: sttn.run(video_path, mask_path)
"""

import os
import sys
import subprocess
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from core.utils import Stack, ToTorchFormatTensor


class STTNColab:
    def __init__(self, auto_setup=True):
        """
        初始化STTN Colab环境
        Args:
            auto_setup: 是否自动设置环境
        """
        self.w, self.h = 432, 240
        self.ref_length = 10
        self.neighbor_stride = 5
        self.default_fps = 24
        self.device = None
        self.model = None
        
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])
        
        if auto_setup:
            self.setup_environment()
            self.check_environment()
    
    def setup_environment(self):
        """设置Colab环境：安装依赖、下载模型等"""
        print("🔧 Setting up STTN environment...")
        
        # 1. 安装依赖
        self._install_dependencies()
        
        # 2. 设置设备
        self._setup_device()
        
        # 3. 下载预训练模型
        self._download_model()
        
    def _install_dependencies(self):
        """安装必要的依赖包"""
        print("📦 Installing dependencies...")
        
        requirements = [
            "torch>=1.8.0",
            "torchvision>=0.9.0", 
            "opencv-python==4.6.0.66",
            "Pillow>=8.0.0",
            "matplotlib>=3.3.0",
            "tqdm>=4.49.0"
        ]
        
        for req in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req, "-q"])
                print(f"  ✓ {req}")
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to install {req}")
    
    def _setup_device(self):
        """设置计算设备"""
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
        else:
            print("  ⚠️  No GPU available - will use CPU (very slow)")
    
    def _download_model(self):
        """下载预训练模型"""
        print("📥 Downloading pretrained model...")
        
        # 创建目录
        os.makedirs("checkpoints", exist_ok=True)
        
        model_path = "checkpoints/sttn.pth"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024*1024)
            print(f"  ✓ Model already exists ({size:.1f} MB)")
            return
        
        # 尝试使用gdown下载
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            subprocess.check_call([
                "gdown", "1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv", 
                "-O", model_path, "--quiet"
            ])
            
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024*1024)
                print(f"  ✓ Model downloaded ({size:.1f} MB)")
            else:
                raise Exception("Download failed")
                
        except:
            print("  ⚠️  Auto download failed. Please manually download:")
            print("     1. Visit: https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view")
            print("     2. Download sttn.pth")
            print("     3. Upload to checkpoints/ folder")
    
    def _load_model(self, model_path="checkpoints/sttn.pth"):
        """加载STTN模型"""
        if self.model is not None:
            return self.model
            
        print(f"🔄 Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 导入模型
        net = importlib.import_module('model.sttn')
        self.model = net.InpaintGenerator().to(self.device)
        
        # 加载权重
        data = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(data['netG'])
        self.model.eval()
        
        print("  ✓ Model loaded successfully")
        return self.model
    
    def check_environment(self):
        """检查环境配置"""
        print("🔍 Checking environment...")
        
        required_files = [
            "examples/schoolgirls_orig.mp4",
            "examples/schoolgirls",
            "model/sttn.py",
            "core/utils.py"
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            print("  ⚠️  Missing files:")
            for file in missing:
                print(f"     - {file}")
            print("  Please upload the complete STTN project folder")
        else:
            print("  ✓ All required files found")
        
        return len(missing) == 0
    
    def _get_ref_index(self, neighbor_ids, length):
        """获取参考帧索引"""
        ref_index = []
        for i in range(0, length, self.ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index
    
    def _read_mask(self, mask_path):
        """读取掩码"""
        masks = []
        mask_files = sorted(os.listdir(mask_path))
        
        for mask_file in mask_files:
            mask = Image.open(os.path.join(mask_path, mask_file))
            mask = mask.resize((self.w, self.h), Image.NEAREST)
            mask = np.array(mask.convert('L'))
            mask = np.array(mask > 0).astype(np.uint8)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            masks.append(Image.fromarray(mask * 255))
            
        return masks
    
    def _read_video(self, video_path):
        """读取视频帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame.resize((self.w, self.h)))
        
        cap.release()
        return frames
    
    def run(self, video_path="examples/schoolgirls_orig.mp4", 
            mask_path="examples/schoolgirls", 
            model_path="checkpoints/sttn.pth",
            output_name=None):
        """
        运行视频修复
        Args:
            video_path: 输入视频路径
            mask_path: 掩码文件夹路径
            model_path: 模型权重路径
            output_name: 输出文件名（可选）
        Returns:
            输出视频路径
        """
        print("🎬 Starting video inpainting...")
        
        # 检查文件
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask folder not found: {mask_path}")
        
        # 加载模型
        self._load_model(model_path)
        
        # 读取数据
        print("📖 Loading video and masks...")
        frames = self._read_video(video_path)
        masks = self._read_mask(mask_path)
        video_length = len(frames)
        
        print(f"  Video: {video_length} frames")
        print(f"  Masks: {len(masks)} files")
        
        # 预处理
        feats = self._to_tensors(frames).unsqueeze(0) * 2 - 1
        frames_np = [np.array(f).astype(np.uint8) for f in frames]
        
        binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
        masks_tensor = self._to_tensors(masks).unsqueeze(0)
        
        feats = feats.to(self.device)
        masks_tensor = masks_tensor.to(self.device)
        
        # 编码
        print("🔄 Encoding frames...")
        with torch.no_grad():
            feats = self.model.encoder((feats * (1 - masks_tensor).float()).view(video_length, 3, self.h, self.w))
            _, c, feat_h, feat_w = feats.size()
            feats = feats.view(1, video_length, c, feat_h, feat_w)
        
        # 修复
        print("🎨 Inpainting frames...")
        comp_frames = [None] * video_length
        
        for f in range(0, video_length, self.neighbor_stride):
            neighbor_ids = list(range(max(0, f - self.neighbor_stride), 
                                    min(video_length, f + self.neighbor_stride + 1)))
            ref_ids = self._get_ref_index(neighbor_ids, video_length)
            
            with torch.no_grad():
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids, :, :, :], 
                    masks_tensor[0, neighbor_ids + ref_ids, :, :, :]
                )
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                
                for i, idx in enumerate(neighbor_ids):
                    img = (pred_img[i].astype(np.uint8) * binary_masks[idx] + 
                           frames_np[idx] * (1 - binary_masks[idx]))
                    
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 + 
                                          img.astype(np.float32) * 0.5)
            
            if f % (self.neighbor_stride * 5) == 0:
                progress = min(100, (f + self.neighbor_stride) * 100 // video_length)
                print(f"  Progress: {progress}%")
        
        # 保存视频
        if output_name is None:
            output_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_result.mp4"
        
        print(f"💾 Saving result to {output_name}...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_name, fourcc, self.default_fps, (self.w, self.h))
        
        for f in range(video_length):
            comp = (comp_frames[f].astype(np.uint8) * binary_masks[f] + 
                   frames_np[f] * (1 - binary_masks[f]))
            writer.write(cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        writer.release()
        
        # 验证输出
        if os.path.exists(output_name):
            size = os.path.getsize(output_name) / (1024 * 1024)
            print(f"✅ Video inpainting completed!")
            print(f"   Output: {output_name} ({size:.1f} MB)")
            return output_name
        else:
            print("❌ Failed to save output video")
            return None


# 便捷函数
def quick_run(video_path="examples/schoolgirls_orig.mp4", mask_path="examples/schoolgirls"):
    """快速运行（自动初始化）"""
    sttn = STTNColab()
    return sttn.run(video_path, mask_path)


if __name__ == "__main__":
    # 示例用法
    sttn = STTNColab()
    result = sttn.run()
    print(f"Result: {result}")