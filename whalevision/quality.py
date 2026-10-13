"""Frame quality rules from Christiansen et al. (2018).

Seven attributes, each scored 1 (good) / 2 (medium) / 3 (poor). A frame is
rejected if any attribute scores 3, or if at least two of the three body
posture attributes (roll, arch, pitch) score 2.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityAttribute:
    key: str
    label: str


ATTRIBUTES = (
    QualityAttribute("camera_focus", "Camera focus"),
    QualityAttribute("body_straightness", "Body straightness"),
    QualityAttribute("body_roll", "Body roll"),
    QualityAttribute("body_arch", "Body arch"),
    QualityAttribute("body_pitch", "Body pitch"),
    QualityAttribute("length_measurability", "Total length measurability"),
    QualityAttribute("width_measurability", "Width measurability"),
)

_POSTURE_KEYS = ("body_roll", "body_arch", "body_pitch")


def evaluate(scores: dict) -> tuple[bool, str]:
    """Apply the rejection rules. Returns (passed, reason)."""
    for attribute in ATTRIBUTES:
        if scores.get(attribute.key) == 3:
            return False, f"poor score (3) on {attribute.label}"

    medium_posture = [key for key in _POSTURE_KEYS if scores.get(key) == 2]
    if len(medium_posture) >= 2:
        labels = ", ".join(
            attribute.label
            for attribute in ATTRIBUTES
            if attribute.key in medium_posture
        )
        return False, f"medium score (2) on two posture attributes: {labels}"

    return True, "passed"


def _self_check():
    good = {attribute.key: 1 for attribute in ATTRIBUTES}
    assert evaluate(good) == (True, "passed")

    one_poor = dict(good, camera_focus=3)
    assert evaluate(one_poor)[0] is False

    two_medium_posture = dict(good, body_roll=2, body_arch=2)
    assert evaluate(two_medium_posture)[0] is False

    one_medium_posture = dict(good, body_roll=2)
    assert evaluate(one_medium_posture) == (True, "passed")

    print("quality.py self-check OK")


if __name__ == "__main__":
    _self_check()
