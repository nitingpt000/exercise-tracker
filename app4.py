import streamlit as st
import cv2
from pose_detector import PoseDetector
from exercise_analyzers import (
    SquatAnalyzer, PushUpAnalyzer, LateralRaiseAnalyzer, 
    BicepCurlAnalyzer, TricepExtensionAnalyzer, ShoulderPressAnalyzer,
    ForwardLungeAnalyzer, SideLungeAnalyzer, JumpingJackAnalyzer,
    HighKneesAnalyzer, PlankAnalyzer, BirdDogAnalyzer,
    WallSitAnalyzer, MountainClimberAnalyzer
)

def main():
    st.set_page_config(layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .big-number {
            font-size: 72px;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
        }
        .feedback-text {
            font-size: 24px;
            color: #424242;
            text-align: center;
            padding: 20px;
        }
        .exercise-title {
            font-size: 36px;
            color: #212121;
            text-align: center;
            padding: 20px;
        }
        .stButton>button {
            width: 100%;
            margin: 5px;
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            padding: 15px;
        }
        .stButton>button:hover {
            background-color: #e9ecef;
            border-color: #1E88E5;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title
    st.markdown('<h1 style="text-align: center; color: #212121; padding: 20px;">Exercise Form Analyzer</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = None
        st.session_state.analyzer = None

    # Create two columns for main layout
    left_col, right_col = st.columns([2, 3])

    with left_col:
        # Exercise selection section
        st.markdown('### Select Exercise')
        exercises = {
            "Squat": SquatAnalyzer,
            "Push-up": PushUpAnalyzer,
            "Lateral Raise": LateralRaiseAnalyzer,
            "Bicep Curl": BicepCurlAnalyzer,
            "Tricep Extension": TricepExtensionAnalyzer,
            "Shoulder Press": ShoulderPressAnalyzer,
            "Forward Lunge": ForwardLungeAnalyzer,
            "Side Lunge": SideLungeAnalyzer,
            "Jumping Jack": JumpingJackAnalyzer,
            "High Knees": HighKneesAnalyzer,
            "Plank": PlankAnalyzer,
            "Bird Dog": BirdDogAnalyzer,
            "Wall Sit": WallSitAnalyzer,
            "Mountain Climber": MountainClimberAnalyzer
        }

        # Create 2-column grid for exercise buttons
        button_cols = st.columns(2)
        for idx, (exercise_name, analyzer_class) in enumerate(exercises.items()):
            with button_cols[idx % 2]:
                if st.button(exercise_name):
                    st.session_state.selected_exercise = exercise_name
                    st.session_state.analyzer = analyzer_class()

        # Display metrics and feedback
        if st.session_state.selected_exercise:
            st.markdown(f'<div class="exercise-title">{st.session_state.selected_exercise}</div>', unsafe_allow_html=True)

            # Containers for metrics and feedback
            metrics_placeholder = st.empty()
            feedback_placeholder = st.empty()

    with right_col:
        # Video feed container with border
        st.markdown("""
            <div style="padding: 10px; border: 2px solid #ddd; border-radius: 10px;">
                <h3 style="text-align: center; color: #424242;">Live Feed</h3>
            </div>
        """, unsafe_allow_html=True)
        video_placeholder = st.empty()

        # Initialize pose detector and video capture
        pose_detector = PoseDetector()
        url = "http://192.168.1.4:4747/video"  # Use IP Webcam URL
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            st.error("Failed to access mobile camera feed. Please check your connection.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame from mobile camera.")
                    break

                frame = cv2.flip(frame, 1)  # Flip frame for better user experience
                keypoints = pose_detector.detect_pose(frame)
                frame = pose_detector.draw_keypoints(frame, keypoints)

                if st.session_state.analyzer is not None:
                    reps, feedback = st.session_state.analyzer.analyze(keypoints)

                    # Update metrics and feedback dynamically
                    if reps is not None:
                        metrics_placeholder.markdown(f'<div class="big-number">{reps}</div>', unsafe_allow_html=True)
                    if feedback:
                        feedback_placeholder.markdown(f'<div class="feedback-text">{feedback}</div>', unsafe_allow_html=True)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
