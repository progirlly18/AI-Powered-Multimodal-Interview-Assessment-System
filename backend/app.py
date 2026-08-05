from scoring.scoring_engine import InterviewScorer

print("=" * 50)
print("      AI INTERVIEW ASSESSMENT SYSTEM")
print("=" * 50)

# -------------------------
# Dummy outputs for testing
# (We'll replace these with the real AI outputs next.)
# -------------------------

emotion = "Happy"
eye = "Looking Center"
head = "Center"

speech_score = 18        # out of 20

natural_background = True

# -------------------------
# Score Interview
# -------------------------

scorer = InterviewScorer()

scorer.emotion_score(emotion)
scorer.eye_score(eye)
scorer.head_score(head)
scorer.speech_score(speech_score)
scorer.background_score(natural_background)

# -------------------------
# Results
# -------------------------

print("\nInterview Results")
print("-" * 50)

print(f"Emotion        : {emotion}")
print(f"Eye Contact    : {eye}")
print(f"Head Pose      : {head}")
print(f"Speech Score   : {speech_score}/20")
print(f"Background     : {'Natural' if natural_background else 'Suspicious'}")

print("\n" + "-" * 50)

print(f"Overall Score  : {scorer.total()}/100")

print("=" * 50)