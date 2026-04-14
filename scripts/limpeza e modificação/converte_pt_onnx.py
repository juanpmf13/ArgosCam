from ultralytics import YOLO
model = YOLO(r"C:\ArgosCam\Runs\ArgosGate_Final_Windows\weights\best.pt")
model.export(format="onnx")