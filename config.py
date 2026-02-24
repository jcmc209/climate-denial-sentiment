"""
Configuration for Climate Change Denialism Sentiment Analysis.
"""

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DENIAL_KEYWORDS = [
    "hoax", "denier", "deniers", "denial", "denying", "deny", "denies",
    "fake", "scam", "myth", "not real", "isn't real", "isnt real",
    "no evidence", "fraud", "lie", "manipulated", "conspiracy",
    "alarmist", "alarmism", "scare", "fear mongering", "propaganda",
    "pseudoscience", "junk science", "not man made", "natural cycle",
    "sun cycle", "not caused by humans", "carbon is good",
    "climate skeptic", "climate sceptic", "don't believe",
    "doesn't exist", "does not exist",
]
