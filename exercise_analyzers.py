import numpy as np

class ExerciseAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.state = "starting"

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

class SquatAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]  # Right hip
        knee = keypoints[13]  # Right knee
        ankle = keypoints[15]  # Right ankle
        
        if hip[2] < 0.3 or knee[2] < 0.3 or ankle[2] < 0.3:
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 150:
            self.state = "squatting"
        elif self.state == "squatting" and knee_angle > 150:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Good squat!"

        return self.rep_count, None

class PullUpAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]  # Right shoulder
        elbow = keypoints[7]    # Right elbow
        wrist = keypoints[9]    # Right wrist
        
        if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
            return None, "Cannot detect key points"

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and elbow_angle < 90:
            self.state = "pulling"
        elif self.state == "pulling" and elbow_angle > 150:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Good pull-up!"

        return self.rep_count, None

class PushUpAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]  # Right shoulder
        elbow = keypoints[7]    # Right elbow
        wrist = keypoints[9]    # Right wrist
        
        if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
            return None, "Cannot detect key points"

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and elbow_angle < 90:
            self.state = "pushing"
        elif self.state == "pushing" and elbow_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Good push-up!"

        return self.rep_count, None

class LungeAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]    # Right hip
        knee = keypoints[13]   # Right knee
        ankle = keypoints[15]  # Right ankle
        
        if hip[2] < 0.3 or knee[2] < 0.3 or ankle[2] < 0.3:
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 100:
            self.state = "lunging"
        elif self.state == "lunging" and knee_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Good lunge!"

        return self.rep_count, None