import cv2
import numpy as np
import datetime
import os
from ultralytics import YOLO


# Setup and Initialization

VIDEO_PATH = "videos/traffic_light2.mp4"   

# Create captures folder
if not os.path.exists("captures"):
    os.makedirs("captures")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open the video file: {VIDEO_PATH}")
    exit()

model = YOLO("yolov8s.pt")

# Intervalele HSV pentru culorile semaforului
color_ranges = {
    "RED1":    [(0, 120, 80), (8, 255, 255)],
    "RED2":    [(170, 120, 80), (179, 255, 255)],
    "YELLOW":  [(15, 20, 100), (75, 255, 255)],
    "GREEN":   [(40, 30, 50), (90, 255, 255)]
}

previous_color = None
current_color = None



# 2. Detect color function

def detect_light_color(frame):
    #Returneaza culoarea dominanta
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_detected = None
    max_pixels = 0

    # Combin red1 si red2 intr-o singura masca
    lower_red1, upper_red1 = np.array(color_ranges["RED1"][0]), np.array(color_ranges["RED1"][1])
    lower_red2, upper_red2 = np.array(color_ranges["RED2"][0]), np.array(color_ranges["RED2"][1])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    count_red = cv2.countNonZero(mask_red)

    # YELLOW
    lower_yellow, upper_yellow = np.array(color_ranges["YELLOW"][0]), np.array(color_ranges["YELLOW"][1])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    count_yellow = cv2.countNonZero(mask_yellow)

    # GREEN
    lower_green, upper_green = np.array(color_ranges["GREEN"][0]), np.array(color_ranges["GREEN"][1])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    count_green = cv2.countNonZero(mask_green)

    # Debug vizual 
    cv2.imshow("Mask - RED", mask_red)
    cv2.imshow("Mask - YELLOW", mask_yellow)
    cv2.imshow("Mask - GREEN", mask_green)

    # Determina culoarea dominanta
    counts = {
        "RED": count_red,
        "YELLOW": count_yellow,
        "GREEN": count_green
    }

    for color_name, count in counts.items():
        if count > max_pixels and count > 500:  # prag minim pentru zgomot
            max_pixels = count
            color_detected = color_name

    return color_detected


def color_activity_score(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    r1 = cv2.inRange(hsv,
                     np.array(color_ranges["RED1"][0]),
                     np.array(color_ranges["RED1"][1]))
    r2 = cv2.inRange(hsv,
                     np.array(color_ranges["RED2"][0]),
                     np.array(color_ranges["RED2"][1]))
    red_score = cv2.countNonZero(cv2.bitwise_or(r1, r2))

    y = cv2.inRange(hsv,
                    np.array(color_ranges["YELLOW"][0]),
                    np.array(color_ranges["YELLOW"][1]))
    yellow_score = cv2.countNonZero(y)

    g = cv2.inRange(hsv,
                    np.array(color_ranges["GREEN"][0]),
                    np.array(color_ranges["GREEN"][1]))
    green_score = cv2.countNonZero(g)

    return max(red_score, yellow_score, green_score)



# 3. Salvare screenshot


def save_screenshot(frame, prev_color, curr_color):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"captures/change_{prev_color}_to_{curr_color}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Screenshot saved: {filename}")



# 4. Procesare frame-uri 


print("[INFO] Analizing the video to detect color changes...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of the video file.")
        break

    roi = None
    best_score = 0
    best_box = 0

    results = model(frame, conf=0.15, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "traffic light":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                roi_candidate = frame[y1:y2, x1:x2]
                score = color_activity_score(roi_candidate)

                if score > best_score:
                    best_score = score
                    roi = roi_candidate
                    best_box = (x1, y1, x2, y2)

                
    if roi is not None and best_score > 500:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    current_color = detect_light_color(roi)

    # afiseaza culoarea detectata pe ecran
    if current_color:
        cv2.putText(frame, f"Detected: {current_color}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Verifica schimbarea de culoare
    if current_color and previous_color and current_color != previous_color:
        print(f"[EVENT] Change detected: {previous_color} → {current_color}")
        save_screenshot(frame, previous_color, current_color)

    if current_color is not None:
        previous_color = current_color

    # afiseaza video-ul procesat
    cv2.imshow("Traffic Light Detection", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):  # apasa 'h' pentru a printa media HSV din ROI
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        print("[HSV mean]:", np.mean(hsv.reshape(-1, 3), axis=0))



cap.release()
cv2.destroyAllWindows()
print("[INFO] Finished.")
