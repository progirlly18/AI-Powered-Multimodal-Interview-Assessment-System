import threading

from backend.vision.vision_service import analyze_interview
from backend.audio.speech_analyzer import analyze_speech
from backend.scoring.scoring_engine import InterviewScorer


class InterviewService:

    def __init__(self):

        self.vision_result = None
        self.speech_result = None

    # -----------------------------
    # THREAD FUNCTIONS
    # -----------------------------

    def run_vision(self, duration):

        self.vision_result = analyze_interview(duration)

    def run_speech(self, duration):

        self.speech_result = analyze_speech(duration)

    # -----------------------------
    # MAIN
    # -----------------------------

    def run(self, duration):

        vision_thread = threading.Thread(
            target=self.run_vision,
            args=(duration,)
        )

        speech_thread = threading.Thread(
            target=self.run_speech,
            args=(duration,)
        )

        print("Starting vision...")
        vision_thread.start()

        print("Starting speech...")
        speech_thread.start()

        vision_thread.join()
        print("Vision finished")

        speech_thread.join()
        print("Speech finished")

        scorer = InterviewScorer()

        scorer.emotion_score(
            self.vision_result["emotion"]
        )

        scorer.eye_score(
            self.vision_result["eye"]
        )

        scorer.head_score(
            self.vision_result["head"]
        )

        scorer.speech_score(
            self.speech_result["speech_score"]
        )

        scorer.background_score(
            self.speech_result["background_natural"]
        )

        total = scorer.total()

        recommendations = []

        if self.vision_result["eye"] != "Looking Center":
            recommendations.append(
                "Maintain better eye contact."
            )

        if self.vision_result["head"] != "Center":
            recommendations.append(
                "Keep your head facing the interviewer."
            )

        if self.speech_result["total_fillers"] > 3:
            recommendations.append(
                "Reduce filler words."
            )

        if self.speech_result["wpm"] < 90:
            recommendations.append(
                "Speak slightly faster."
            )

        if self.speech_result["wpm"] > 170:
            recommendations.append(
                "Slow down your speaking pace."
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Excellent interview performance!"
            )

        return {

            "emotion": self.vision_result["emotion"],

            "eye_contact": self.vision_result["eye"],

            "head_pose": self.vision_result["head"],

            "speech_score": self.speech_result["speech_score"],

            "overall_score": total,

            "recommendations": recommendations,

            "transcript": self.speech_result["transcript"],

            "background": self.speech_result["background"],

            "wpm": self.speech_result["wpm"],

            "fillers": self.speech_result["total_fillers"]

        }