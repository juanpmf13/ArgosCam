from ultralytics import YOLO
model = YOLO(r"D:\ArgosCam\runs_backup_cpu\detect\train\weights\versao1.pt")
model.export(format="onnx")