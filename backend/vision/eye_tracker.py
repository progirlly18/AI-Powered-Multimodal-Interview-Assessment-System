from collections import Counter


class EyeTracker:

    def __init__(self):

        self.directions = []

        self.current_direction = None

        self.current_length = 0

        self.longest_left = 0
        self.longest_right = 0

        self.left_events = 0
        self.right_events = 0

    def update(self, direction):

        if direction == "No Face":
            return

        self.directions.append(direction)

        # -----------------------------
        # Track continuous glances
        # -----------------------------

        if direction == self.current_direction:

            self.current_length += 1

        else:

            # Save previous streak

            if self.current_direction == "Looking Left":

                self.longest_left = max(
                    self.longest_left,
                    self.current_length
                )

                self.left_events += 1

            elif self.current_direction == "Looking Right":

                self.longest_right = max(
                    self.longest_right,
                    self.current_length
                )

                self.right_events += 1

            self.current_direction = direction
            self.current_length = 1

    def get_report(self):

        # Save last streak

        if self.current_direction == "Looking Left":

            self.longest_left = max(
                self.longest_left,
                self.current_length
            )

        elif self.current_direction == "Looking Right":

            self.longest_right = max(
                self.longest_right,
                self.current_length
            )

        if len(self.directions) == 0:

            return {
                "dominant": "No Face"
            }

        total = len(self.directions)

        counts = Counter(self.directions)

        center = counts.get("Looking Center", 0)
        left = counts.get("Looking Left", 0)
        right = counts.get("Looking Right", 0)

        return {

            "dominant":
                counts.most_common(1)[0][0],

            "center_percentage":
                round(center * 100 / total, 1),

            "left_percentage":
                round(left * 100 / total, 1),

            "right_percentage":
                round(right * 100 / total, 1),

            "left_events":
                self.left_events,

            "right_events":
                self.right_events,

            "longest_left_frames":
                self.longest_left,

            "longest_right_frames":
                self.longest_right
        }