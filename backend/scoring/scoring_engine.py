class InterviewScorer:

    def __init__(self):
        self.score = {
            "emotion": 0,
            "eye_contact": 0,
            "head_pose": 0,
            "speech": 0,
            "background": 0
        }

    # ----------------------
    # Emotion
    # ----------------------

    def emotion_score(self, emotion):

        positive = [
            "Happy",
            "Neutral"
        ]

        if emotion in positive:
            self.score["emotion"] = 25
        else:
            self.score["emotion"] = 15

    # ----------------------
    # Eye Contact
    # ----------------------

    def eye_score(self, direction):

        if direction == "Looking Center":
            self.score["eye_contact"] = 25

        elif direction in [
            "Looking Left",
            "Looking Right"
        ]:
            self.score["eye_contact"] = 18

        else:
            self.score["eye_contact"] = 10

    # ----------------------
    # Head Pose
    # ----------------------

    def head_score(self, pose):

        if pose == "Center":
            self.score["head_pose"] = 20
        else:
            self.score["head_pose"] = 12

    # ----------------------
    # Speech
    # ----------------------

    def speech_score(self, speech_score):

        self.score["speech"] = min(
            speech_score,
            20
        )

    # ----------------------
    # Background
    # ----------------------

    def background_score(self, natural):

        if natural:
            self.score["background"] = 10
        else:
            self.score["background"] = 5

    # ----------------------
    # Final Score
    # ----------------------

    def total(self):

        total = sum(self.score.values())

        return total