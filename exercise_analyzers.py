import numpy as np

class ExerciseAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.state = "starting"
        self.previous_angles = None

    def calculate_angle(self, point1, point2, point3):
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
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 120:
            self.state = "squatting"
        elif self.state == "squatting" and knee_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep back straight"
        return self.rep_count, None

class PushUpAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        
        if not all(k[2] > 0.3 for k in [shoulder, elbow, wrist]):
            return None, "Cannot detect key points"

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and elbow_angle < 90:
            self.state = "pushing"
        elif self.state == "pushing" and elbow_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep core tight"
        return self.rep_count, None

class LateralRaiseAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        
        if not all(k[2] > 0.3 for k in [shoulder, elbow, wrist]):
            return None, "Cannot detect key points"

        arm_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and arm_angle > 150:
            self.state = "raising"
        elif self.state == "raising" and arm_angle < 60:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Control the movement"
        return self.rep_count, None

class BicepCurlAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        
        if not all(k[2] > 0.3 for k in [shoulder, elbow, wrist]):
            return None, "Cannot detect key points"

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and elbow_angle < 60:
            self.state = "curling"
        elif self.state == "curling" and elbow_angle > 150:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep elbows still"
        return self.rep_count, None

class TricepExtensionAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        
        if not all(k[2] > 0.3 for k in [shoulder, elbow, wrist]):
            return None, "Cannot detect key points"

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and elbow_angle > 150:
            self.state = "extending"
        elif self.state == "extending" and elbow_angle < 60:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep upper arms still"
        return self.rep_count, None

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        
        if not all(k[2] > 0.3 for k in [shoulder, elbow, wrist]):
            return None, "Cannot detect key points"

        press_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.state == "starting" and press_angle > 150:
            self.state = "pressing"
        elif self.state == "pressing" and press_angle < 90:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Full range of motion"
        return self.rep_count, None

class ForwardLungeAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 100:
            self.state = "lunging"
        elif self.state == "lunging" and knee_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep torso upright"
        return self.rep_count, None

class SideLungeAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 110:
            self.state = "lunging"
        elif self.state == "lunging" and knee_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Push knee out"
        return self.rep_count, None

class JumpingJackAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder_l = keypoints[5]
        hip_l = keypoints[11]
        ankle_l = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [shoulder_l, hip_l, ankle_l]):
            return None, "Cannot detect key points"

        arm_angle = self.calculate_angle(shoulder_l, hip_l, ankle_l)
        
        if self.state == "starting" and arm_angle > 150:
            self.state = "jumping"
        elif self.state == "jumping" and arm_angle < 60:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Full extension"
        return self.rep_count, None

class HighKneesAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 45:
            self.state = "raising"
        elif self.state == "raising" and knee_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Drive knees high"
        return self.rep_count, None

class PlankAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        hip = keypoints[11]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [shoulder, hip, ankle]):
            return None, "Cannot detect key points"

        body_angle = self.calculate_angle(shoulder, hip, ankle)
        
        if 160 <= body_angle <= 180:
            return self.rep_count, "Good form"
        return self.rep_count, "Keep body straight"

class BirdDogAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        shoulder = keypoints[5]
        hip = keypoints[11]
        knee = keypoints[13]
        
        if not all(k[2] > 0.3 for k in [shoulder, hip, knee]):
            return None, "Cannot detect key points"

        body_angle = self.calculate_angle(shoulder, hip, knee)
        
        if self.state == "starting" and body_angle > 160:
            self.state = "extending"
        elif self.state == "extending" and body_angle < 120:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep back straight"
        return self.rep_count, None

class WallSitAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if 85 <= knee_angle <= 95:
            return self.rep_count, "Good form"
        return self.rep_count, "Adjust to 90 degrees"

class MountainClimberAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"

        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 60:
            self.state = "climbing"
        elif self.state == "climbing" and knee_angle > 150:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Keep hips low"
        return self.rep_count, None
class DeadliftAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        # Using shoulder, hip, and knee to check hip hinge movement
        shoulder = keypoints[5]
        hip = keypoints[11]
        knee = keypoints[13]
        
        if not all(k[2] > 0.3 for k in [shoulder, hip, knee]):
            return None, "Cannot detect key points"
        
        hinge_angle = self.calculate_angle(shoulder, hip, knee)
        
        if self.state == "starting" and hinge_angle < 90:
            self.state = "lifting"
        elif self.state == "lifting" and hinge_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Maintain neutral spine"
        return self.rep_count, None
class BurpeeAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        # Using hip, knee, and ankle for a squat–jump transition
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        
        if not all(k[2] > 0.3 for k in [hip, knee, ankle]):
            return None, "Cannot detect key points"
        
        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        if self.state == "starting" and knee_angle < 110:
            self.state = "burpee"
        elif self.state == "burpee" and knee_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Explosive jump"
        return self.rep_count, None
class CalfRaiseAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        # Using knee, ankle, and hip to get an approximation of the ankle angle
        knee = keypoints[13]
        ankle = keypoints[15]
        hip = keypoints[11]
        
        if not all(k[2] > 0.3 for k in [knee, ankle, hip]):
            return None, "Cannot detect key points"
        
        ankle_angle = self.calculate_angle(knee, ankle, hip)
        
        if self.state == "starting" and ankle_angle < 150:
            self.state = "raising"
        elif self.state == "raising" and ankle_angle > 160:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Focus on calf contraction"
        return self.rep_count, None
class RussianTwistAnalyzer(ExerciseAnalyzer):
    def analyze(self, keypoints):
        # Using shoulder, hip, and wrist as a rough proxy for trunk rotation
        shoulder = keypoints[5]
        hip = keypoints[11]
        wrist = keypoints[9]
        
        if not all(k[2] > 0.3 for k in [shoulder, hip, wrist]):
            return None, "Cannot detect key points"
        
        twist_angle = self.calculate_angle(shoulder, hip, wrist)
        
        if self.state == "starting" and twist_angle > 150:
            self.state = "twisting"
        elif self.state == "twisting" and twist_angle < 90:
            self.state = "starting"
            self.rep_count += 1
            return self.rep_count, "Engage obliques"
        return self.rep_count, None