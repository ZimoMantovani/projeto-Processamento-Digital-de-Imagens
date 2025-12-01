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


cap = cv2.VideoCapture("flexao2.mp4")  # ou "video.mp4"

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
            
            # PONTOS PRINCIPAIS
            shoulder = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                        int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            elbow = (int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
            wrist = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

            hip = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                   int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
            knee = (int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                    int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
            ankle = (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            feedback = []

            # --- Ângulo do cotovelo ---
            elbow_angle = angle(shoulder, elbow, wrist)
            cv2.putText(frame, f"Cotovelo: {int(elbow_angle)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if elbow_angle > 160:
                feedback.append("Posicao inicial (alta)")
            elif elbow_angle > 90:
                feedback.append("Executando")
            else:
                feedback.append("Flexao completa!")

            # --- Linha da coluna ---
            coluna_angle = angle(shoulder, hip, ankle)
            cv2.putText(frame, f"Coluna: {int(coluna_angle)}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if coluna_angle < 150:
                feedback.append("Coluna torta")
            elif coluna_angle < 170:
                feedback.append("Coluna quase reta")
            else:
                feedback.append("Coluna reta (OK)")

            # --- Quadril alto ou baixo ---
            # Medimos a posição vertical relativa do quadril
            hip_y = hip[1]
            shoulder_y = shoulder[1]
            ankle_y = ankle[1]

            if hip_y < shoulder_y - 20:
                feedback.append("Quadril muito alto")
            elif hip_y > ankle_y - 20:
                feedback.append("Quadril muito baixo")

            # Postura perfeita
            if ("Coluna reta (OK)" in feedback and 
                elbow_angle > 160 and 
                abs(hip_y - ((shoulder_y+ankle_y)//2)) < 20):
                feedback.append("Postura correta!")

            # --- Mostrar feedback ---
            y_offset = 140
            for fb in feedback:
                cv2.putText(frame, fb, (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                y_offset += 40
        
        cv2.imshow("Flexao - Analise de Postura", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


# Problema quadril muito baixo, 
# adicionar fundo na letra, para melhor visibilidade
#
#
#
