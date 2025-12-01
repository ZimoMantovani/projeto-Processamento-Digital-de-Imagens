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

def draw_text_with_background(frame, text, position, font_scale=0.8, color=(255,255,255), thickness=2):
    """Desenha texto com fundo escuro para melhor visibilidade"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Retângulo de fundo
    padding = 5
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


cap = cv2.VideoCapture("flexao2.mp4") 

# Variáveis para contar repetições
rep_count = 0
stage = "up"  # "up" ou "down"

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
            draw_text_with_background(frame, f"Cotovelo: {int(elbow_angle)}", (30, 50))

            # Contador de repetições
            if elbow_angle > 160:
                stage = "up"
                feedback.append("Posicao inicial (alta)")
            elif elbow_angle < 90 and stage == "up":
                stage = "down"
                rep_count += 1
                feedback.append("Flexao completa!")
            elif elbow_angle > 90 and elbow_angle <= 160:
                feedback.append("Executando")

            # --- Linha da coluna ---
            coluna_angle = angle(shoulder, hip, ankle)
            draw_text_with_background(frame, f"Coluna: {int(coluna_angle)}", (30, 90))

            if coluna_angle < 150:
                feedback.append("Coluna torta")
            elif coluna_angle < 170:
                feedback.append("Coluna quase reta")
            else:
                feedback.append("Coluna reta (OK)")

            # --- Quadril alto ou baixo ---
            # Calcula o ponto médio ideal entre ombro e tornozelo
            ideal_hip_y = (shoulder[1] + ankle[1]) // 2
            hip_deviation = hip[1] - ideal_hip_y
            
            # Tolerância de 50 pixels
            tolerance = 50
            
            if hip_deviation < -tolerance:  # Hip_y menor = mais alto na tela
                feedback.append("Quadril muito alto")
            elif hip_deviation > tolerance:  # Hip_y maior = mais baixo na tela
                feedback.append("Quadril muito baixo")

            # Postura perfeita (ajustada)
            if ("Coluna reta (OK)" in feedback and 
                abs(hip_deviation) < tolerance):
                feedback.append("Postura correta!")

            # --- Contador de repetições ---
            draw_text_with_background(frame, f"Repeticoes: {rep_count}", 
                                     (w - 250, 50), font_scale=1.0, 
                                     color=(0, 255, 255), thickness=2)

            # --- Mostrar feedback ---
            y_offset = 140
            for fb in feedback:
                # Cores diferentes para feedback
                if "OK" in fb or "correta" in fb or "completa" in fb:
                    cor = (0, 255, 0)  # Verde
                elif "torta" in fb or "muito" in fb:
                    cor = (0, 0, 255)  # Vermelho
                else:
                    cor = (255, 255, 255)  # Branco
                    
                draw_text_with_background(frame, fb, (30, y_offset), 
                                         font_scale=0.9, color=cor)
                y_offset += 40
        
        cv2.imshow("Flexao - Analise de Postura", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()