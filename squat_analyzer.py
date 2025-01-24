import numpy as np
class SquatAnalyzer:
    def __init__(self):
        self.squat_state = "standing"  # Can be "standing", "squatting", "rising"
        self.rep_count = 0
        self.min_knee_angle = float('inf')
        self.previous_knee_angle = None

    def calculate_angle(self, point1, point2, point3):
        """Calculate the angle between three points"""
        a = np.array([point1[1], point1[0]])
        b = np.array([point2[1], point2[0]])
        c = np.array([point3[1], point3[0]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def analyze_squat(self, keypoints):
        # Get relevant keypoints
        hip = keypoints[11]  # Right hip
        knee = keypoints[13]  # Right knee
        ankle = keypoints[15]  # Right ankle
        
        # Check if keypoints are detected with enough confidence
        if hip[2] < 0.3 or knee[2] < 0.3 or ankle[2] < 0.3:
            return None, None

        # Calculate knee angle
        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        # Initialize previous_knee_angle if None
        if self.previous_knee_angle is None:
            self.previous_knee_angle = knee_angle
            return None, None

        # Determine squat state and count reps
        form_feedback = ""
        
        if self.squat_state == "standing" and knee_angle < 150:
            self.squat_state = "squatting"
            self.min_knee_angle = knee_angle
        
        elif self.squat_state == "squatting":
            if knee_angle < self.min_knee_angle:
                self.min_knee_angle = knee_angle
            
            if knee_angle > 150:
                self.squat_state = "standing"
                self.rep_count += 1
                
                # Analyze form
                if self.min_knee_angle > 100:
                    form_feedback = "Squat deeper!"
                elif self.min_knee_angle < 60:
                    form_feedback = "Don't go too deep!"
                else:
                    form_feedback = "Good form!"
                
                self.min_knee_angle = float('inf')
        
        self.previous_knee_angle = knee_angle
        return self.rep_count, form_feedback