import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    
    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)
    
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    
    if mag_ab * mag_cb == 0:
        return 0
    
    cos_angle = dot / (mag_ab * mag_cb)
    return math.degrees(math.acos(cos_angle))


cap = cv2.VideoCapture("agachamento.mp4")  # ou "video.mp4"

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            
            # Pegar pontos principais
            hip = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                   int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
            knee = (int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                    int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
            ankle = (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))
            shoulder = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                        int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Ângulo do joelho
            knee_angle = angle(hip, knee, ankle)
            cv2.putText(frame, f"Angulo joelho: {int(knee_angle)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            feedback = []

            # --- Análise do agachamento ---
            if knee_angle < 70:
                feedback.append("Agachamento muito baixo")
            elif knee_angle < 140:
                feedback.append("Agachando...")
            else:
                feedback.append("Em pe")

            # --- Inclinação da coluna ---
            coluna_ang = angle(shoulder, hip, knee)
            cv2.putText(frame, f"Coluna: {int(coluna_ang)}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if coluna_ang > 130:
                feedback.append("Coluna alinhada")
            elif coluna_ang > 60:
                feedback.append("Coluna na posição correta")
            else:
                feedback.append("Coluna muito para frente")

            # --- Posição do pé ---
            diff_x = (ankle[0] - hip[0]) / w

            if diff_x < -0.10:
                feedback.append("Pe muito para tras")
            elif diff_x > 0.10:
                feedback.append("Pe muito para frente")

            # --- Mostrar feedback na tela ---
            y_offset = 140
            for fb in feedback:
                cv2.putText(frame, fb, (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                y_offset += 40
        
        cv2.imshow("Analise de Postura", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
