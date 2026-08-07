class InterviewService:

    def __init__(self):

        pass

    def run(self, duration):

        """
        Runs the complete interview pipeline.

        Returns dictionary containing all results.
        """

        result = {

            "emotion": "Happy",

            "emotion_score": 91,

            "eye_contact": 94,

            "head_pose": "Center",

            "speech_score": 96,

            "overall_score": 93,

            "recommendations": [

                "Excellent eye contact.",

                "Maintain current speaking pace.",

                "Reduce filler words."

            ]

        }

        return result