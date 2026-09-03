import cv2
from ultralytics import YOLO

model = YOLO('ball.pt')
cap = cv2.VideoCapture('sample3.mp4')
print('loaded')
count = 0
frames = 0

for i in range(40):
    ret, frame = cap.read()
    if not ret:
        break
    frames += 1
    r = model.predict(frame, conf=0.25, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        print('frame', i, 'boxes', 0)
        continue
    confs = [float(x) for x in r.boxes.conf.tolist()]
    best = int(r.boxes.conf.argmax().item())
    box = r.boxes.xyxy[best].cpu().numpy().tolist()
    print('frame', i, 'boxes', len(r.boxes), 'max_conf', max(confs), 'best_box', box)
    count += 1

cap.release()
print('frames_read', frames, 'detected_frames', count)
