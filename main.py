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
    # Mistura o overlay com transparência
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Desenha o texto
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


cap = cv2.VideoCapture("agachamento.mp4")  

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

            feedback = []

            # --- Ângulo do joelho ---
            knee_angle = angle(hip, knee, ankle)
            draw_text_with_background(frame, f"Angulo joelho: {int(knee_angle)}", (30, 50))

            # Contador de repetições e análise do agachamento
            if knee_angle > 160:
                stage = "up"
                feedback.append("Em pe")
            elif knee_angle < 90 and stage == "up":
                stage = "down"
                rep_count += 1
                if knee_angle < 70:
                    feedback.append("Agachamento profundo!")
                else:
                    feedback.append("Agachamento completo!")
            elif knee_angle >= 90 and knee_angle <= 160:
                feedback.append("Agachando...")
                if knee_angle < 70:
                    feedback.append("(Muito baixo)")

            # --- Inclinação da coluna ---
            coluna_ang = angle(shoulder, hip, knee)
            draw_text_with_background(frame, f"Coluna: {int(coluna_ang)}", (30, 90))

            if coluna_ang > 130:
                feedback.append("Coluna muito ereta")
            elif coluna_ang > 90:
                feedback.append("Coluna alinhada (OK)")
            elif coluna_ang > 60:
                feedback.append("Coluna inclinada")
            else:
                feedback.append("Coluna muito para frente")

            # --- Posição do pé (horizontal) ---
            diff_x = (ankle[0] - knee[0]) / w
            
            # Tolerância ajustada
            tolerance = 0.08

            if diff_x < -tolerance:
                feedback.append("Joelho muito para frente")
            elif diff_x > tolerance:
                feedback.append("Joelho muito para tras")
            else:
                feedback.append("Joelho alinhado")

            # --- Profundidade ideal do agachamento ---
            if 80 <= knee_angle <= 100 and "Coluna alinhada (OK)" in feedback:
                feedback.append("Postura ideal!")

            # --- Contador de repetições ---
            draw_text_with_background(frame, f"Repeticoes: {rep_count}", 
                                     (w - 250, 50), font_scale=1.0, 
                                     color=(0, 255, 255), thickness=2)

            # --- Mostrar feedback na tela ---
            y_offset = 140
            for fb in feedback:
                # Cores diferentes para feedback
                if "OK" in fb or "ideal" in fb or "completo" in fb or "alinhado" in fb:
                    cor = (0, 255, 0)  # Verde
                elif "muito" in fb and "Muito baixo" not in fb:
                    cor = (0, 0, 255)  # Vermelho - avisos
                elif "profundo" in fb:
                    cor = (0, 255, 255)  # Amarelo - avançado
                else:
                    cor = (255, 255, 255)  # Branco - neutro
                    
                draw_text_with_background(frame, fb, (30, y_offset), 
                                         font_scale=0.9, color=cor)
                y_offset += 40
        
        cv2.imshow("Agachamento - Analise de Postura", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()