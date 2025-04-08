import cv2
import mediapipe as mp
import time

def main():
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam")
        return

    # Inicializa os modelos do MediaPipe
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Configurações dos modelos
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Variáveis para cálculo de FPS
    p_time = 0
    c_time = 0

    try:
        while True:
            # Captura o frame
            success, img = cap.read()
            if not success:
                print("Erro: Não foi possível capturar o frame")
                break

            # Espelha a imagem horizontalmente
            img = cv2.flip(img, 1)
            
            # Converte para RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Processa a imagem com os modelos
            pose_results = pose.process(img_rgb)
            face_results = face_mesh.process(img_rgb)
            
            # Desenha os landmarks da postura
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            # Desenha os landmarks faciais
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Desenha os contornos principais do rosto
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(200, 200, 200), 
                            thickness=1,
                            circle_radius=1
                        )
                    )
                    
                    # Destaca alguns pontos faciais importantes
                    h, w, _ = img.shape
                    for id_num in [1, 33, 61, 199, 263, 291]:  # Pontos de referência do rosto
                        if id_num < len(face_landmarks.landmark):
                            lm = face_landmarks.landmark[id_num]
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

            # Calcula e mostra o FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            
            cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            # Mostra a imagem resultante
            cv2.imshow("Postura e FaceMesh", img)
            
            # Sai do loop ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Libera os recursos
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        face_mesh.close()

if __name__ == "__main__":
    main()