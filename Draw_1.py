import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Warna default dan ukuran kuas
draw_color = (255, 105, 180)  # Pink (mengganti merah sebelumnya)
brush_size = 5
erase_size = 40  # Ukuran penghapus

# Warna-warna yang tersedia
colors = [(255, 105, 180), (0, 255, 0), (0, 0, 255)]  # Pink, Hijau, Biru

# Membuka kamera
cap = cv2.VideoCapture(0)

# Ukuran kanvas
canvas = None
prev_position = None  # Posisi sebelumnya (untuk menggambar)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Flip horizontal untuk pengalaman seperti cermin
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Konversi ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Jika belum ada kanvas, buat dengan ukuran frame
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Gambar tombol warna pada frame
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (10 + i * 60, 10), (60 + i * 60, 60), color, -1)

    # Jika ada tangan terdeteksi
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil posisi landmark jari
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[8]  # Titik ujung jari telunjuk
            thumb_tip = landmarks[4]        # Titik ujung ibu jari
            pinky_tip = landmarks[20]       # Titik ujung kelingking

            # Konversi ke koordinat piksel
            index_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            pinky_pos = (int(pinky_tip.x * w), int(pinky_tip.y * h))

            # Gesture Deteksi
            is_index_up = landmarks[8].y < landmarks[6].y  # Telunjuk terangkat
            is_thumb_up = landmarks[4].x < landmarks[3].x  # Ibu jari terbuka ke samping
            is_all_fingers_open = all(landmarks[finger].y < landmarks[finger - 2].y for finger in [8, 12, 16, 20])

            # Menghapus (semua jari terbuka)
            if is_all_fingers_open:
                cv2.circle(frame, index_pos, erase_size, (0, 0, 0), -1)  # Tampilkan penghapus
                cv2.circle(canvas, index_pos, erase_size, (0, 0, 0), -1)  # Hapus di kanvas

            # Menggambar (telunjuk dan ibu jari terbuka)
            elif is_index_up and is_thumb_up:
                if prev_position:  # Jika ada posisi sebelumnya, gambar garis
                    cv2.line(frame, prev_position, index_pos, draw_color, brush_size)
                    cv2.line(canvas, prev_position, index_pos, draw_color, brush_size)
                prev_position = index_pos  # Update posisi sebelumnya

            # Memilih warna (hanya telunjuk terbuka)
            elif is_index_up and not is_thumb_up:
                prev_position = None  # Reset posisi sebelumnya (tidak menggambar)

                # Deteksi memilih warna (telunjuk berada di area warna)
                if index_pos[1] < 60:
                    for i, color in enumerate(colors):
                        if 10 + i * 60 < index_pos[0] < 60 + i * 60:
                            draw_color = color

            # Gesture lainnya: Reset posisi sebelumnya
            else:
                prev_position = None

    # Gabungkan frame dan kanvas
    blended_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Tampilkan hasil
    cv2.imshow('Drawing', blended_frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup
cap.release()
cv2.destroyAllWindows()
