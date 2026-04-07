import cv2
import mediapipe as mp

print("Finger tracking started...")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

points = []

def fingers_up(handLms):
    fingers = []

    # Index
    fingers.append(handLms.landmark[8].y < handLms.landmark[6].y)

    # Middle
    fingers.append(handLms.landmark[12].y < handLms.landmark[10].y)

    # Ring
    fingers.append(handLms.landmark[16].y < handLms.landmark[14].y)

    # Pinky
    fingers.append(handLms.landmark[20].y < handLms.landmark[18].y)

    # Thumb (simplified)
    fingers.append(handLms.landmark[4].x > handLms.landmark[3].x)

    return fingers

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    draw_mode = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            fingers = fingers_up(handLms)

            index_up = fingers[0]
            all_up = all(fingers)

            # ✏️ Draw ONLY when index is up (and others down)
            if index_up and not any(fingers[1:4]):
                draw_mode = True
                points.append((x, y))

            # 🖐 Clear when all fingers open
            if all_up:
                points = []

            # Draw current fingertip
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Draw path
    for i in range(1, len(points)):
        cv2.line(img, points[i - 1], points[i], (255, 0, 0), 3)

    # Show mode
    if draw_mode:
        cv2.putText(img, "DRAWING", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(img, "NOT DRAWING", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Finger Drawing", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()