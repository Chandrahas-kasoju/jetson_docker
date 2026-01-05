# Posture.msg
# Defines the detected human posture with an integer class and a score.

# Posture Classes (Constants for clarity)
int32 UNKNOWN=0
int32 LYING_DOWN=1
int32 SITTING=2
int32 STANDING=3
int32 UPRIGHT=3

# The detected posture class, using one of the constants above
int32 posture_class

# A score from 0-100 indicating uprightness
int32 uprightness_score
