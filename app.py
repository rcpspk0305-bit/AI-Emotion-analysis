import os
import sqlite3
import datetime
import queue
import threading
import traceback
from collections import Counter, deque

import cv2
import av
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from deepface import DeepFace
# FIX #2: Import VideoProcessorBase for proper lifecycle management
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="AI Emotion Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
DB_FILE = "emotion_sessions.db"
RECORDING_DIR = "secure_recordings"

os.makedirs(RECORDING_DIR, exist_ok=True)

# ==========================================
# 2. DATABASE FUNCTIONS
# ==========================================
def get_db_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    # FIX #4: Use context manager so connection closes even on exception
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT UNIQUE,
                video_filepath TEXT,
                dominant_emotions TEXT
            )
        """)
        conn.commit()

init_db()

def create_session_in_db(timestamp, filepath):
    # FIX #4: Use context manager
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (timestamp, video_filepath, dominant_emotions) VALUES (?, ?, ?)",
            (timestamp, filepath, "Recording...")
        )
        conn.commit()

def update_session_in_db(timestamp, emotions_list):
    # FIX #4: Use context manager
    with get_db_connection() as conn:
        summary = str(Counter(emotions_list).most_common(3)) if emotions_list else "No emotions detected"
        conn.execute(
            "UPDATE sessions SET dominant_emotions=? WHERE timestamp=?",
            (summary, timestamp)
        )
        conn.commit()

# ==========================================
# 3. WEBRTC VIDEO PROCESSOR
# ==========================================
# FIX #2: Inherit from VideoProcessorBase for proper streamlit-webrtc lifecycle
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.emotions_detected = []
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.video_filepath = os.path.join(RECORDING_DIR, f"session_{self.timestamp}.avi")
        self.writer = None
        self.session_saved = False
        self.db_saved_once = False

        self.current_emotion = "Detecting..."
        self.face_box = None
        self.emotion_queue = queue.Queue()

        # Smooth predictions over recent frames
        self.recent_predictions = deque(maxlen=5)

        # Background recording queue + thread
        self.recording_queue = queue.Queue(maxsize=200)
        self.recording_active = True
        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()

    def _record_loop(self):
        while self.recording_active:
            try:
                frame_to_write = self.recording_queue.get(timeout=1)

                if frame_to_write is None:
                    break

                if self.writer is None:
                    h, w, _ = frame_to_write.shape
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.writer = cv2.VideoWriter(self.video_filepath, fourcc, 15.0, (w, h))

                    if not self.writer.isOpened():
                        print("ERROR: VideoWriter failed to open")
                        self.writer = None
                    else:
                        print("Recording started:", self.video_filepath)

                        if not self.db_saved_once:
                            create_session_in_db(self.timestamp, self.video_filepath)
                            self.db_saved_once = True

                if self.writer is not None and self.writer.isOpened():
                    self.writer.write(frame_to_write)

            except queue.Empty:
                continue
            except Exception as e:
                print("Recording thread error:", e)
                traceback.print_exc()

    def recv(self, frame):
        raw_img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(raw_img, 1)   # mirrored preview
        self.frame_count += 1

        try:
            # Slight sharpening for better visible clarity
            sharpen_kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            img = cv2.filter2D(img, -1, sharpen_kernel)

            # ------------------------------------------
            # SEND FRAMES TO BACKGROUND RECORDER
            # ------------------------------------------
            if self.frame_count % 2 == 0:
                try:
                    frame_to_save = cv2.flip(raw_img.copy(), 1)
                    if not self.recording_queue.full():
                        self.recording_queue.put_nowait(frame_to_save)
                except Exception as e:
                    print("Queue record error:", e)

            # ------------------------------------------
            # EMOTION ANALYSIS (Improved stability)
            # ------------------------------------------
            if self.frame_count % 45 == 0:
                try:
                    print("Analyzing frame...")

                    small_img = cv2.resize(img, (640, 360))
                    results = DeepFace.analyze(
                        small_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )

                    if isinstance(results, list):
                        result = results[0]
                    else:
                        result = results

                    region = result.get("region", {})
                    probabilities = result.get("emotion", {})

                    original_h, original_w = img.shape[:2]
                    scale_x = original_w / 640
                    scale_y = original_h / 360

                    x = int(region.get("x", 0) * scale_x)
                    y = int(region.get("y", 0) * scale_y)
                    w = int(region.get("w", 0) * scale_x)
                    h = int(region.get("h", 0) * scale_y)

                    x = max(0, x)
                    y = max(0, y)
                    w = max(1, min(w, original_w - x))
                    h = max(1, min(h, original_h - y))

                    self.face_box = (x, y, w, h)

                    if probabilities:
                        top_emotion = max(probabilities, key=probabilities.get)
                        top_score = probabilities[top_emotion]

                        if top_score >= 45:
                            self.recent_predictions.append(top_emotion)
                            smoothed_emotion = Counter(self.recent_predictions).most_common(1)[0][0]

                            self.current_emotion = smoothed_emotion
                            self.emotions_detected.append(smoothed_emotion)

                            self.emotion_queue.put({
                                "dominant": smoothed_emotion,
                                "probabilities": probabilities
                            })

                            print(f"Emotion detected: {smoothed_emotion} ({top_score:.1f}%)")
                        else:
                            print(f"Weak prediction ignored: {top_emotion} ({top_score:.1f}%)")
                    else:
                        self.current_emotion = "Detecting..."

                except Exception as e:
                    print("DeepFace analysis error:", e)
                    traceback.print_exc()
                    self.face_box = None
                    self.current_emotion = "Detecting..."

            # ------------------------------------------
            # DRAW OVERLAY
            # ------------------------------------------
            annotated_img = img.copy()

            if self.face_box is not None:
                x, y, w, h = self.face_box
                h_img, w_img, _ = annotated_img.shape

                x = max(0, x)
                y = max(0, y)
                w = max(1, min(w, w_img - x))
                h = max(1, min(h, h_img - y))

                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 150), 3)

                label_y1 = max(0, y - 40)
                label_y2 = max(0, y)
                cv2.rectangle(annotated_img, (x, label_y1), (x + w, label_y2), (0, 255, 150), -1)

                cv2.putText(
                    annotated_img,
                    self.current_emotion.upper(),
                    (x + 5, max(25, y - 10)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )

            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        except Exception as e:
            print("Frame processing error:", e)
            traceback.print_exc()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def save_session(self):
        try:
            self.recording_active = False

            try:
                self.recording_queue.put(None, timeout=1)
            except:
                pass

            if self.recording_thread.is_alive():
                self.recording_thread.join(timeout=3)

            if self.writer is not None:
                self.writer.release()
                self.writer = None
                print("Recording saved:", self.video_filepath)

            if not self.db_saved_once:
                create_session_in_db(self.timestamp, self.video_filepath)
                self.db_saved_once = True

            if not self.session_saved:
                update_session_in_db(self.timestamp, self.emotions_detected)
                self.session_saved = True

        except Exception as e:
            print("Save session error:", e)

    def on_ended(self):
        self.save_session()

# ==========================================
# 4. SIDEBAR - ADMIN PANEL
# ==========================================
with st.sidebar:
    st.title("🔐 Admin Portal")
    st.markdown("---")

    entered_password = st.text_input("Admin Password:", type="password")

    if entered_password == ADMIN_PASSWORD:
        st.success("Authenticated")
        st.markdown("### 📼 Session Records")

        # FIX #4: Use context manager for admin DB reads too
        with get_db_connection() as conn:
            try:
                df_db = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", conn)

                if not df_db.empty:
                    session_options = df_db["timestamp"].tolist()
                    selected_session = st.selectbox("Select Session to Play:", session_options)

                    if selected_session:
                        session_data = df_db[df_db["timestamp"] == selected_session].iloc[0]
                        video_path = session_data["video_filepath"]

                        st.write("**Detected Emotions:**")
                        st.code(session_data["dominant_emotions"])

                        if os.path.exists(video_path):
                            with open(video_path, "rb") as video_file:
                                st.video(video_file.read())
                        else:
                            st.error("Video file not found on server.")
                else:
                    st.info("No sessions recorded yet.")

            except Exception as e:
                st.error(f"Database Error: {e}")

    elif entered_password != "":
        st.error("Incorrect Password.")

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================
st.title("🧠 Live Emotional Analysis Dashboard")
st.caption("Live facial expression estimation and real-time analytics")

# FIX #3: Pass limit=None explicitly to avoid accidental refresh storm
# and key collision with webrtc re-renders
st_autorefresh(interval=2000, limit=None, key="emotion-refresh")

col_video, col_graphs = st.columns([1.2, 1])

# ------------------------------------------
# LIVE VIDEO
# ------------------------------------------
with col_video:
    st.markdown("### 📸 Interactive Camera")
    webrtc_ctx = webrtc_streamer(
        key="emotion-system",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )

# ------------------------------------------
# LIVE TELEMETRY
# ------------------------------------------
with col_graphs:
    st.markdown("### 📊 Real-Time Telemetry")
    indicator_placeholder = st.empty()
    chart_placeholder = st.empty()
    debug_placeholder = st.empty()

    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=30)

    if "last_emotion" not in st.session_state:
        st.session_state.last_emotion = "Waiting..."

    indicator_placeholder.metric(
        label="Current Dominant Emotion",
        value=st.session_state.last_emotion
    )

# ==========================================
# 6. SAFE REAL-TIME UPDATE
# ==========================================
if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    try:
        got_new_data = False

        while not webrtc_ctx.video_processor.emotion_queue.empty():
            result = webrtc_ctx.video_processor.emotion_queue.get_nowait()
            st.session_state.last_emotion = result["dominant"].upper()
            st.session_state.history.append(result["probabilities"])
            got_new_data = True

        indicator_placeholder.metric(
            label="Current Dominant Emotion",
            value=st.session_state.last_emotion
        )

        if len(st.session_state.history) > 0:
            df = pd.DataFrame(st.session_state.history)

            fig = go.Figure()
            colors = {
                "happy": "#00CC96",
                "sad": "#636EFA",
                "angry": "#EF553B",
                "neutral": "#AB63FA",
                "fear": "#FFA15A",
                "surprise": "#19D3F3",
                "disgust": "#B6E880"
            }

            for emotion in df.columns:
                fig.add_trace(go.Scatter(
                    y=df[emotion],
                    mode="lines",
                    name=emotion.capitalize(),
                    line=dict(color=colors.get(emotion, "white"), width=3),
                    fill='tozeroy',
                    opacity=0.25
                ))

            fig.update_layout(
                title="Emotion Confidence Levels",
                xaxis_title="Time (Samples)",
                yaxis_title="Confidence (%)",
                yaxis=dict(range=[0, 100]),
                margin=dict(l=0, r=0, t=40, b=0),
                height=400,
                template="plotly_dark",
                hovermode="x unified"
            )

            # FIX #1: use_container_width=True is the correct Streamlit parameter
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        if got_new_data:
            debug_placeholder.success(f"Live detections: {len(st.session_state.history)}")
        else:
            debug_placeholder.info(f"Waiting for detection... | Samples: {len(st.session_state.history)}")

    except Exception as e:
        st.error(f"Live telemetry error: {e}")

# ==========================================
# 7. SAVE SESSION WHEN CAMERA STOPS
# ==========================================
if webrtc_ctx and not webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    if not webrtc_ctx.video_processor.session_saved:
        try:
            webrtc_ctx.video_processor.save_session()
        except Exception as e:
            print("Session save error:", e)