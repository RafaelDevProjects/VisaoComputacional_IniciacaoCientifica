import cv2
import numpy as np

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis para suavização das detecções
prev_center = None
prev_radius = None
alpha = 0.7  # Fator de suavização

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Suavização da imagem
    frame_blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # Converte a imagem para o espaço de cor HSV
    hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

    # Define a faixa de cor para a detecção da bola (ajuste conforme necessário)
    lower_color = (20, 100, 100)
    upper_color = (30, 255, 255)

    # Cria uma máscara que isola os pixels dentro da faixa de cor
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Redução de ruído com operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Redução de ruído adicional
    mask = cv2.medianBlur(mask, 5)

    # Encontra os contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Combina todos os contornos encontrados
        combined_contour = np.concatenate(contours)
        hull = cv2.convexHull(combined_contour)

        # Encontra o círculo mínimo que envolve o contorno combinado
        (x, y), radius = cv2.minEnclosingCircle(hull)
        center = (int(x), int(y))
        radius = int(radius)

        # Suavização da posição e do raio do círculo
        if prev_center is not None:
            center = (
                int(alpha * prev_center[0] + (1 - alpha) * center[0]),
                int(alpha * prev_center[1] + (1 - alpha) * center[1]),
            )
            radius = int(alpha * prev_radius + (1 - alpha) * radius)

        # Desenha o círculo na imagem original
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Atualiza as variáveis para suavização
        prev_center = center
        prev_radius = radius

    # Exibe o resultado
    cv2.imshow("Frame", frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o recurso da câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
