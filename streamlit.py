import streamlit as st
import cv2
import mediapipe as mp

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: #1c1c1c;'>AI 體感穿搭與妝容推薦平台</h1>", unsafe_allow_html=True)

# 初始化 Mediapipe 模組
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# --- UI 設計開始 ---

with st.container():
    st.markdown("### 💄 妝容建議工具")
    st.markdown("使用臉部特徵（臉型、眉形、膚色等）推薦妝容與美妝產品。")
    st.markdown("*唇彩、粉底、修容、眼影、腮紅建議*")
    col1, col2 = st.columns(2)
    face_camera = col1.empty()
    face_scan = col2.empty()
    st.markdown(
        """
        <div style='background-color: #fffbea; border-left: 6px solid #ffcc00; padding: 1rem; border-radius: 8px; font-size: 1.1rem;'>
            😊 看起來您喜歡這個穿搭／妝容建議的，還有什麼需要我幫忙的嗎？
        </div>
        """,
        unsafe_allow_html=True
    )

with st.container():
    st.markdown("### 👗 穿搭建議神器")
    st.markdown("穿搭比例偵測與風格建議，提升整體造型美感。")
    col3, col4 = st.columns(2)
    pose_camera = col3.empty()
    pose_scan = col4.empty()
    st.markdown(
        """
        <div style='background-color: #fffbea; border-left: 6px solid #ffcc00; padding: 1rem; border-radius: 8px; font-size: 1.1rem;'>
            😊 看起來您喜歡這個穿搭／妝容建議的，還有什麼需要我幫忙的嗎？
        </div>
        """,
        unsafe_allow_html=True
    )


# --- 開始處理攝影機畫面 ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("❌ 無法讀取攝影機畫面")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 妝容掃描處理
    face_frame = frame.copy()
    face_result = face_mesh.process(rgb_frame)
    if face_result.multi_face_landmarks:
        for landmarks in face_result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=face_frame,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),  # 白色點
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 255, 200), thickness=1, circle_radius=1)  # 青綠色線
            )

    # 穿搭掃描處理
    pose_frame = frame.copy()
    pose_result = pose.process(rgb_frame)
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=pose_frame,
            landmark_list=pose_result.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # 顯示畫面
    face_camera.image(frame, channels="BGR", caption="原始攝影機畫面")
    face_scan.image(face_frame, channels="BGR", caption="臉部掃描結果")

    pose_camera.image(frame, channels="BGR", caption="原始攝影機畫面")
    pose_scan.image(pose_frame, channels="BGR", caption="身體骨架掃描結果")

cap.release()
cv2.destroyAllWindows()