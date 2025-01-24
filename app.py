import streamlit as st
import cv2
from pose_detector import PoseDetector
from exercise_analyzers import SquatAnalyzer, PullUpAnalyzer, PushUpAnalyzer, LungeAnalyzer

def main():
    st.title("Exercise Form Analyzer")
    
    # Initialize session state for selected exercise if it doesn't exist
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = None
        st.session_state.analyzer = None
    
    # Exercise selection buttons in a horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Squats"):
            st.session_state.selected_exercise = "squats"
            st.session_state.analyzer = SquatAnalyzer()
    
    with col2:
        if st.button("Pull-ups"):
            st.session_state.selected_exercise = "pullups"
            st.session_state.analyzer = PullUpAnalyzer()
    
    with col3:
        if st.button("Push-ups"):
            st.session_state.selected_exercise = "pushups"
            st.session_state.analyzer = PushUpAnalyzer()
    
    with col4:
        if st.button("Lunges"):
            st.session_state.selected_exercise = "lunges"
            st.session_state.analyzer = LungeAnalyzer()
    
    # Display selected exercise
    if st.session_state.selected_exercise:
        st.markdown(f"### Currently analyzing: {st.session_state.selected_exercise.title()}")
    
    # Initialize pose detector
    pose_detector = PoseDetector()
    
    # Create placeholders for the webcam feed and metrics
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    feedback_placeholder = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            keypoints = pose_detector.detect_pose(frame)
            
            # Draw keypoints and connections
            frame = pose_detector.draw_keypoints(frame, keypoints)
            
            # Analyze exercise if one is selected
            if st.session_state.analyzer is not None:
                reps, feedback = st.session_state.analyzer.analyze(keypoints)
                
                if reps is not None:
                    metrics_placeholder.markdown(f"### Reps: {reps}")
                if feedback:
                    feedback_placeholder.markdown(f"### Feedback: {feedback}")
            
            # Convert BGR to RGB and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()