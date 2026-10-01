import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ml.gestures import decide, WORDS, ASL_WORDS, ASL_LETTERS, open_palm_landmarks, live_prediction, words_for_gesture, geometric_gesture, asl_number
from ml.asl_letters import MotionLetterTracker, LETTERS


def hand_points(folded=()):
    points = [SimpleNamespace(x=.5, y=.7) for _ in range(21)]
    points[0] = SimpleNamespace(x=.5, y=.95)
    for x, (mcp, pip, tip) in zip((.35, .45, .55, .65), ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        points[mcp] = SimpleNamespace(x=x, y=.65)
        points[pip] = SimpleNamespace(x=x, y=.48)
        points[tip] = SimpleNamespace(x=x, y=.24 if mcp not in folded else .67)
    return points


class GestureDecisionTests(unittest.TestCase):
    def test_two_different_hands_are_returned_as_one_pair(self):
        thumbs_up = [SimpleNamespace(category_name="Thumb_Up", score=.9)]
        low_palm = [SimpleNamespace(category_name="Open_Palm", score=.55)]
        folded = hand_points(folded=(13, 17))
        open_hand = hand_points()
        first = SimpleNamespace(hand_landmarks=[folded, open_hand], gestures=[thumbs_up, low_palm])
        swapped = SimpleNamespace(hand_landmarks=[open_hand, folded], gestures=[low_palm, thumbs_up])
        for result in (first, swapped):
            prediction = live_prediction(result, .65)
            self.assertEqual(prediction["gesture"], "Open_Palm|Thumb_Up")
            self.assertEqual(prediction["text"], "Stop + Yes")
        self.assertIsNone(prediction["confidence"])

    def test_partial_two_hand_result_is_not_a_false_single_hand_result(self):
        hand = hand_points(folded=(13, 17))
        result = SimpleNamespace(hand_landmarks=[hand, hand], gestures=[
            [SimpleNamespace(category_name="Thumb_Up", score=.9)],
            [SimpleNamespace(category_name="None", score=.8)]])
        self.assertIsNone(live_prediction(result, .65)["text"])

    def test_recorded_pair_can_be_decided_without_allowing_unknown_labels(self):
        pair = "Open_Palm|Thumb_Up"
        self.assertEqual(decide([(pair, .7)] * 12), pair)
        self.assertEqual(words_for_gesture(pair), "Stop + Yes")
        with self.assertRaises(ValueError):
            decide([("Open_Palm|Unknown", .9)] * 12)

    def test_open_palm_geometry_recovers_low_model_score(self):
        result = SimpleNamespace(hand_landmarks=[hand_points()], gestures=[[SimpleNamespace(category_name="Open_Palm", score=.55)]])
        self.assertEqual(live_prediction(result, .65)["text"], "Stop")

    def test_open_palm_rule_does_not_accept_two_fingers_or_no_hand(self):
        self.assertFalse(open_palm_landmarks(hand_points(folded=(13, 17))))
        self.assertFalse(open_palm_landmarks([]))
        result = SimpleNamespace(hand_landmarks=[], gestures=[])
        self.assertIsNone(live_prediction(result, .65)["text"])
    def test_each_supported_held_gesture(self):
        for label in WORDS:
            self.assertEqual(decide([(None, 0)] * 5 + [(label, .9)] * 12 + [(None, 0)] * 5), label)

    def test_empty_clip_cannot_invent_words(self):
        with self.assertRaises(ValueError):
            decide([(None, 0)] * 24)

    def test_low_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            decide([("Thumb_Up", .4)] * 24)

    def test_brief_false_positive_is_rejected(self):
        with self.assertRaises(ValueError):
            decide([(None, 0)] * 20 + [("Thumb_Up", .99)] * 3)

    def test_conflicting_gestures_are_rejected(self):
        with self.assertRaises(ValueError):
            decide([("Thumb_Up", .9)] * 12 + [("Thumb_Down", .9)] * 12)

    def test_unsupported_gesture_cannot_become_yes(self):
        self.assertEqual(decide([("Closed_Fist", .99)] * 24), "Closed_Fist")
        with self.assertRaises(ValueError):
            decide([("Unknown", .99)] * 24)

    def test_mode_vocabulary_is_isolated(self):
        self.assertEqual(len(WORDS), 26)
        self.assertEqual(len(ASL_WORDS), 9)
        self.assertEqual(len(ASL_LETTERS), 26)
        self.assertEqual(len(LETTERS), 24)
        self.assertEqual(words_for_gesture("ASL_3", "asl"), "Three")
        self.assertEqual(decide([("ASL_3", .7)] * 12, "asl"), "ASL_3")
        with self.assertRaises(ValueError):
            decide([("ASL_3", .9)] * 12)
        with self.assertRaises(ValueError):
            decide([("Closed_Fist", .9)] * 12, "asl")
        self.assertEqual(words_for_gesture("LETTER_W", "asl_letters"), "W")
        with self.assertRaises(ValueError):
            decide([("LETTER_W", .9)] * 12, "asl")

    def test_extra_handshape_words_explain_finger_count(self):
        patterns = {
            "Thumb_Middle": 2, "Thumb_Ring": 2, "Thumb_Middle_Ring": 3,
            "Thumb_Middle_Pinky": 3, "Thumb_Ring_Pinky": 3,
            "Index_Ring": 2, "Index_Middle_Pinky": 3,
            "Index_Ring_Pinky": 3, "Middle_Ring_Pinky": 3,
            "Middle_Pinky": 2,
        }
        for gesture, count in patterns.items():
            with self.subTest(gesture=gesture):
                word = words_for_gesture(gesture)
                self.assertTrue(word.startswith("Two fingers:" if count == 2 else "Three fingers:"))
                for finger in gesture.lower().split("_"):
                    self.assertIn(finger, word)

    def test_motion_letters_need_a_path_not_a_static_pose(self):
        def points(tip_index, x, y):
            hand = hand_points()
            hand[0] = SimpleNamespace(x=.5, y=.8)
            hand[9] = SimpleNamespace(x=.5, y=.6)
            hand[tip_index] = SimpleNamespace(x=x, y=y)
            return hand
        tracker = MotionLetterTracker()
        j_state = (False, False, False, False, True)
        self.assertIsNone(tracker.observe(points(20, .40, .25), j_state, 0))
        self.assertIsNone(tracker.observe(points(20, .40, .37), j_state, .3))
        self.assertIsNone(tracker.observe(points(20, .41, .50), j_state, .6))
        self.assertEqual(tracker.observe(points(20, .53, .42), j_state, .9), "J")
        self.assertEqual(tracker.observe(points(20, .53, .42), j_state, 1.2), "J")
        tracker.reset()
        z_state = (False, True, False, False, False)
        for time, x, y in ((0, .30, .30), (.3, .45, .31), (.6, .29, .40)):
            self.assertIsNone(tracker.observe(points(8, x, y), z_state, time))
        self.assertEqual(tracker.observe(points(8, .48, .41), z_state, .9), "Z")
        tracker.reset()
        for time, x, y in ((0, .30, .30), (.3, .34, .31), (.6, .36, .33), (.9, .38, .34)):
            self.assertIsNone(tracker.observe(points(8, x, y), z_state, time))

    def test_new_handshapes_and_asl_numbers(self):
        fist = hand_points(folded=(5, 9, 13, 17))
        self.assertEqual(geometric_gesture(fist), "Closed_Fist")
        one = hand_points(folded=(9, 13, 17))
        self.assertEqual(asl_number(one), "ASL_1")
        two = hand_points(folded=(13, 17))
        self.assertEqual(asl_number(two), "ASL_2")
        four = hand_points()
        four[4] = SimpleNamespace(x=.46, y=.64)
        self.assertEqual(asl_number(four), "ASL_4")
        self.assertEqual(geometric_gesture(four), "Four")
        six = hand_points()
        six[4] = SimpleNamespace(x=six[20].x, y=six[20].y)
        self.assertEqual(asl_number(six), "ASL_6")

    def test_ten_added_simple_handshapes_have_distinct_rules(self):
        cases = [
            ("Closed_Fist", (5, 9, 13, 17), "folded"),
            ("ASL_Three", (13, 17), "thumb"),
            ("Four", (), "tucked"),
            ("Okay", (5,), "touch"),
            ("Call_Me", (5, 9, 13), "thumb"),
            ("Rock_On", (9, 13), "folded"),
            ("Letter_L", (9, 13, 17), "thumb"),
            ("Pinky", (5, 9, 13), "folded"),
            ("Pinch", (5, 9, 13, 17), "touch"),
            ("Three_Fingers", (17,), "folded"),
        ]
        for expected, folded, thumb in cases:
            with self.subTest(expected=expected):
                hand = hand_points(folded=folded)
                if thumb == "thumb":
                    hand[2] = SimpleNamespace(x=.32, y=.75)
                    hand[4] = SimpleNamespace(x=.13, y=.54)
                elif thumb == "tucked":
                    hand[4] = SimpleNamespace(x=.46, y=.64)
                elif thumb == "touch":
                    hand[4] = SimpleNamespace(x=hand[8].x, y=hand[8].y)
                self.assertEqual(geometric_gesture(hand), expected)

    def test_asl_number_contact_positions_six_through_nine(self):
        for number, tip in ((6, 20), (7, 16), (8, 12), (9, 8)):
            with self.subTest(number=number):
                hand = hand_points()
                hand[4] = SimpleNamespace(x=hand[tip].x, y=hand[tip].y)
                self.assertEqual(asl_number(hand), f"ASL_{number}")

    def test_ten_additional_simple_patterns(self):
        cases = {
            "Thumb_Middle": (1, 0, 1, 0, 0),
            "Thumb_Ring": (1, 0, 0, 1, 0),
            "Thumb_Middle_Ring": (1, 0, 1, 1, 0),
            "Thumb_Middle_Pinky": (1, 0, 1, 0, 1),
            "Thumb_Ring_Pinky": (1, 0, 0, 1, 1),
            "Index_Ring": (0, 1, 0, 1, 0),
            "Index_Middle_Pinky": (0, 1, 1, 0, 1),
            "Index_Ring_Pinky": (0, 1, 0, 1, 1),
            "Middle_Ring_Pinky": (0, 0, 1, 1, 1),
            "Middle_Pinky": (0, 0, 1, 0, 1),
        }
        mcps = (5, 9, 13, 17)
        for expected, state in cases.items():
            with self.subTest(expected=expected):
                hand = hand_points(folded=tuple(mcp for mcp, is_up in zip(mcps, state[1:]) if not is_up))
                if state[0]:
                    hand[2] = SimpleNamespace(x=.32, y=.75)
                    hand[4] = SimpleNamespace(x=.13, y=.54)
                self.assertEqual(geometric_gesture(hand), expected)
                mirrored = [SimpleNamespace(x=1 - point.x, y=point.y) for point in hand]
                resized = [SimpleNamespace(x=.1 + .8 * point.x, y=.1 + .8 * point.y) for point in hand]
                self.assertEqual(geometric_gesture(mirrored), expected)
                self.assertEqual(geometric_gesture(resized), expected)

    def test_letter_prediction_is_separate_from_number_mode(self):
        hand = hand_points(folded=(5, 9, 13))
        result = SimpleNamespace(hand_landmarks=[hand], gestures=[[]])
        with patch("ml.asl_letters.predict", return_value=("Y", .97)):
            self.assertEqual(live_prediction(result, .65, "asl_letters")["text"], "Y")
        self.assertIsNone(live_prediction(result, .65, "asl")["text"])
