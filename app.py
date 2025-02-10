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
    st.title("Exercise Form Analyzer")
    
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = None
        st.session_state.analyzer = None

    # Create exercise buttons in a grid layout
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

    # Create a 3x5 grid for exercise buttons
    cols = st.columns(3)
    for idx, (exercise_name, analyzer_class) in enumerate(exercises.items()):
        with cols[idx % 3]:
            if st.button(exercise_name):
                st.session_state.selected_exercise = exercise_name
                st.session_state.analyzer = analyzer_class()

    if st.session_state.selected_exercise:
        st.markdown(f"### Analyzing: {st.session_state.selected_exercise}")

    pose_detector = PoseDetector()
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    feedback_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            frame = cv2.flip(frame, 1)
            keypoints = pose_detector.detect_pose(frame)
            frame = pose_detector.draw_keypoints(frame, keypoints)

            if st.session_state.analyzer is not None:
                reps, feedback = st.session_state.analyzer.analyze(keypoints)
                
                if reps is not None:
                    metrics_placeholder.markdown(f"### Reps: {reps}")
                if feedback:
                    feedback_placeholder.markdown(f"### Feedback: {feedback}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()