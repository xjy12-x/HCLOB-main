from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'F:/contrast-11/runs/train/exp148/weights/best.pt')  
    # 验证模型
    metrics=model.val(
        val=True,  # (bool) 在训练期间进行验证/测试
        data=r'VisDrone.yaml',
        # split='val',  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
        batch=1,  # (int) 每批的图像数量（-1 为自动批处理）
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        device='',  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=8,  # 数据加载的工作线程数（每个DDP进程）
        save_json=False,  # 保存结果到JSON文件
        save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.001,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.6,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project='runs/val',  # 项目名称（可选）
        name='exp',  # 实验名称，结果保存在'project/name'目录下（可选）
        max_det=300,  # 每张图像的最大检测数
        half=False,  # 使用半精度 (FP16)
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
    )

    print(f"mAP50-95: {metrics.box.map}") # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50
    print(f"mAP75: {metrics.box.map75}")  # map75
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}") # FPS
# import torch
# import time
# import numpy as np
# from ultralytics import YOLO

# model = YOLO(r'F:/contrast-11/runs/train/exp148/weights/best.pt')

# # GPU 预热
# dummy = torch.randn(1, 3, 640, 640).cuda()
# for _ in range(20):
#     model.predict(source=dummy, verbose=False)

# # 正式计时（排除数据加载，只测纯推理）
# times = []
# for _ in range(200):
#     torch.cuda.synchronize()          # 等待 GPU 完成
#     start = time.perf_counter()
#     model.predict(source=dummy, verbose=False)
#     torch.cuda.synchronize()          # 等待 GPU 完成
#     times.append(time.perf_counter() - start)

# times_ms = np.array(times) * 1000
# print(f"平均推理时间: {times_ms.mean():.2f} ms")
# print(f"FPS: {1000 / times_ms.mean():.1f}")
# print(f"P95 延迟: {np.percentile(times_ms, 95):.2f} ms")
# print(f"最快: {times_ms.min():.2f} ms  最慢: {times_ms.max():.2f} ms")
