from ultralytics.models.yolo.model import YOLO




if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO('F:/contrast-11/ultralytics/cfg/models/11/yolo11-SCGA.yaml')
    # 加载上一次训练自动保存的last.pt权重
    #model = YOLO('F:/contrast-11/runs/train/exp160/weights/last.pt')

# # 继续训练
#model.train(resume=True)
model.train(data="F:/contrast-11/ultralytics/cfg/datasets/VisDrone.yaml",
                imgsz=640,
                epochs=400,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',                                                                                                         
                name='exp',
                single_cls=False,
                cache=False,
                amp=True,
                )
