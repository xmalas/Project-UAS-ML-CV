import cv2
import mediapipe as mp
import numpy as np

meme_salah_fokus_path = "memes/salah_fokus1.jpg"


# ==============================
# Inisialisasi MediaPipe
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ==============================
# Buka Webcam
# ==============================
cap = cv2.VideoCapture(0)

def detect_pose(landmarks):
    """
    Menentukan pose wajah berdasarkan landmark
    (Rule-based, bukan CNN training)
    """
    # Landmark penting
    nose = landmarks[1]      # hidung
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[152]

    # Hitung perbedaan posisi
    dx = left_eye.x - right_eye.x
    dy = nose.y - chin.y

    if dx > 0.03:
        return "Menoleh Kanan"
    elif dx < -0.03:
        return "Menoleh Kiri"
    elif dy < -0.05:
        return "Kepala Atas"
    elif dy > 0.05:
        return "Kepala Bawah"
    else:
        return "Netral"

# ==============================
# Loop Real-Time
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip agar seperti kaca
    frame = cv2.flip(frame, 1)

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    pose_text = "Tidak Ada Wajah"
    meme_text = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Deteksi pose wajah
            pose_text = detect_pose(landmarks)

            # ==============================
            # Pemicu Meme
            # ==============================
            elif pose_text == "Menoleh Kiri":
    meme_img = cv2.imread(meme_salah_fokus_path)

    if meme_img is None:
        meme_text = "gambar meme tidak ditemukan"
    else:
        # Resize meme agar tidak terlalu besar
        meme_img = cv2.resize(meme_img, (200, 200))

        # Tampilkan meme di pojok kanan atas
        h, w, _ = meme_img.shape
        frame[20:20+h, frame.shape[1]-20-w:frame.shape[1]-20] = meme_img

        meme_text = "meme: salah fokus"


            # Gambar landmark wajah
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    # ==============================
    # Tampilkan Teks
    # ==============================
    cv2.putText(frame, f"Pose: {pose_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, meme_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Face Pose Meme Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
