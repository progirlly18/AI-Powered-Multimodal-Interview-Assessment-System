import time


class Timeline:

    def __init__(self):

        self.start_time = time.time()

        self.events = []

    def add_event(self, event_type, value):

        self.events.append({

            "time": round(time.time() - self.start_time, 2),

            "type": event_type,

            "value": value

        })

    def get_events(self):

        return self.events