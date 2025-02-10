import streamlit as st
import cv2
from pose_detector import PoseDetector
from exercise_analyzers import (
    SquatAnalyzer, PushUpAnalyzer, LateralRaiseAnalyzer, 
    BicepCurlAnalyzer, TricepExtensionAnalyzer, ShoulderPressAnalyzer,
    ForwardLungeAnalyzer, SideLungeAnalyzer, JumpingJackAnalyzer,
    HighKneesAnalyzer, PlankAnalyzer, BirdDogAnalyzer,
    WallSitAnalyzer, MountainClimberAnalyzer, DeadliftAnalyzer,
    BurpeeAnalyzer, CalfRaiseAnalyzer, RussianTwistAnalyzer
)
import time
from datetime import datetime, timedelta

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    return str(timedelta(seconds=seconds))[2:7]

# Define the exercises dictionary globally
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
    "Mountain Climber": MountainClimberAnalyzer,
    "Deadlift": DeadliftAnalyzer,
    "Burpee": BurpeeAnalyzer,
    "Calf Raise": CalfRaiseAnalyzer,
    "Russian Twist": RussianTwistAnalyzer
}

def initialize_session_state():
    """Initialize all session state variables"""
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = None
        st.session_state.analyzer = None
    if 'workout_plan' not in st.session_state:
        st.session_state.workout_plan = []
    if 'current_exercise_index' not in st.session_state:
        st.session_state.current_exercise_index = 0
    if 'rest_timer' not in st.session_state:
        st.session_state.rest_timer = 0
    if 'is_resting' not in st.session_state:
        st.session_state.is_resting = False
    if 'workout_active' not in st.session_state:
        st.session_state.workout_active = False
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()

def create_workout_ui():
    """Create the workout programming interface"""
    st.markdown("### Program Workout")
    
    # Create columns for the workout program interface
    cols = st.columns([2, 1, 1, 1, 1])
    
    # Add new exercise button
    if cols[4].button("+ Add Exercise", key="add_exercise"):
        st.session_state.workout_plan.append({
            'exercise': 'Burpees',
            'reps': 12,
            'sets': 3,
            'rest': 60
        })

    # Display and edit workout plan
    for i, workout in enumerate(st.session_state.workout_plan):
        cols = st.columns([2, 1, 1, 1, 1])
        
        # Exercise selection
        workout['exercise'] = cols[0].selectbox(
            'Exercise',
            options=list(exercises.keys()),
            key=f'exercise_{i}',
            index=list(exercises.keys()).index(workout['exercise']) if workout['exercise'] in exercises else 0
        )
        
        # Reps selection
        workout['reps'] = cols[1].number_input(
            'Reps',
            min_value=1,
            max_value=50,
            value=workout['reps'],
            key=f'reps_{i}'
        )
        
        # Sets selection
        workout['sets'] = cols[2].number_input(
            'Sets',
            min_value=1,
            max_value=10,
            value=workout['sets'],
            key=f'sets_{i}'
        )
        
        # Rest time selection
        workout['rest'] = cols[3].number_input(
            'Rest (sec)',
            min_value=0,
            max_value=300,
            value=workout['rest'],
            key=f'rest_{i}'
        )
        
        # Remove exercise button
        if cols[4].button('🗑️', key=f'remove_{i}'):
            st.session_state.workout_plan.pop(i)
            st.experimental_rerun()

def update_workout_status():
    """Update workout timer and exercise progression"""
    if not st.session_state.workout_active:
        return

    current_time = time.time()
    elapsed = current_time - st.session_state.last_update_time
    st.session_state.last_update_time = current_time

    if st.session_state.is_resting:
        st.session_state.rest_timer -= elapsed
        if st.session_state.rest_timer <= 0:
            st.session_state.is_resting = False
            st.session_state.current_exercise_index += 1
            if st.session_state.current_exercise_index >= len(st.session_state.workout_plan):
                st.session_state.workout_active = False
                st.session_state.current_exercise_index = 0
    else:
        current_exercise = st.session_state.workout_plan[st.session_state.current_exercise_index]
        st.session_state.selected_exercise = current_exercise['exercise']
        st.session_state.analyzer = exercises[current_exercise['exercise']]()

def main():
    st.set_page_config(layout="wide")

    # Custom CSS (your existing CSS remains the same)
    st.markdown("""
        <style>
        .big-number { font-size: 72px; font-weight: bold; color: #1E88E5; text-align: center; }
        .feedback-text { font-size: 24px; color: #424242; text-align: center; padding: 20px; }
        .exercise-title { font-size: 36px; color: #212121; text-align: center; padding: 20px; }
        .stButton>button { width: 100%; margin: 5px; background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; }
        .stButton>button:hover { background-color: #e9ecef; border-color: #1E88E5; }
        </style>
    """, unsafe_allow_html=True)

    # Main title
    st.markdown('<h1 style="text-align: center; color: #212121; padding: 20px;">Exercise Form Analyzer</h1>', 
                unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Create two columns for main layout
    left_col, right_col = st.columns([2, 3])

    with left_col:
        # Tabs for workout plan and exercise selection
        tab1, tab2 = st.tabs(["Workout Program", "Single Exercise"])

        with tab1:
            create_workout_ui()
            
            # Workout control buttons
            cols = st.columns(3)
            if not st.session_state.workout_active:
                if cols[0].button("Start Workout", key="start_workout"):
                    st.session_state.workout_active = True
                    st.session_state.current_exercise_index = 0
                    st.session_state.is_resting = False
                    st.session_state.last_update_time = time.time()
            else:
                if cols[0].button("Pause Workout", key="pause_workout"):
                    st.session_state.workout_active = False
            
            if cols[1].button("Reset Workout", key="reset_workout"):
                st.session_state.workout_active = False
                st.session_state.current_exercise_index = 0
                st.session_state.is_resting = False
                st.session_state.rest_timer = 0

        with tab2:
            # Your existing exercise selection UI
            st.markdown('### Select Exercise')
            for exercise_name, analyzer_class in exercises.items():
                if st.button(exercise_name):
                    st.session_state.selected_exercise = exercise_name
                    st.session_state.analyzer = analyzer_class()

        # Display current exercise/rest status
        if st.session_state.workout_active:
            st.markdown("### Current Status")
            if st.session_state.is_resting:
                st.markdown(f"**Rest Time Remaining:** {format_time(int(st.session_state.rest_timer))}")
            else:
                current_exercise = st.session_state.workout_plan[st.session_state.current_exercise_index]
                st.markdown(f"**Current Exercise:** {current_exercise['exercise']}")
                st.markdown(f"**Target Reps:** {current_exercise['reps']}")
                st.markdown(f"**Sets Remaining:** {current_exercise['sets']}")

        # Display metrics and feedback
        if st.session_state.selected_exercise:
            st.markdown(f'<div class="exercise-title">{st.session_state.selected_exercise}</div>', 
                       unsafe_allow_html=True)
            metrics_placeholder = st.empty()
            feedback_placeholder = st.empty()

    with right_col:
        # Video feed container
        st.markdown("""
            <div style="padding: 10px; border: 2px solid #ddd; border-radius: 10px;">
                <h3 style="text-align: center; color: #424242;">Live Feed</h3>
            </div>
        """, unsafe_allow_html=True)
        video_placeholder = st.empty()

        # Initialize pose detector and video capture
        pose_detector = PoseDetector()
        cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            st.error("Failed to access webcam. Please check your camera connection.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                keypoints = pose_detector.detect_pose(frame)
                frame = pose_detector.draw_keypoints(frame, keypoints)

                if st.session_state.analyzer is not None:
                    try:
                        reps, feedback = st.session_state.analyzer.analyze(keypoints)
                    except Exception as e:
                        st.error(f"Error analyzing pose: {str(e)}")
                        reps, feedback = None, "Error analyzing pose"

                    if st.session_state.workout_active and not st.session_state.is_resting:
                        current_exercise = st.session_state.workout_plan[st.session_state.current_exercise_index]
                        if reps is not None and reps >= current_exercise['reps']:
                            current_exercise['sets'] -= 1
                            if current_exercise['sets'] <= 0:
                                # Move to next exercise
                                st.session_state.is_resting = True
                                st.session_state.rest_timer = current_exercise['rest']
                                # Reset rep counter for next exercise
                                st.session_state.analyzer.reset()
                            else:
                                # Rest between sets
                                st.session_state.rest_timer = current_exercise['rest']
                                st.session_state.is_resting = True
                                # Reset rep counter for next set
                                st.session_state.analyzer.reset()

                    if reps is not None:
                        metrics_placeholder.markdown(f'<div class="big-number">{reps}</div>', 
                                                  unsafe_allow_html=True)
                    if feedback:
                        feedback_placeholder.markdown(f'<div class="feedback-text">{feedback}</div>', 
                                                   unsafe_allow_html=True)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                update_workout_status()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()