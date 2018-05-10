class Annealing(object):
    def __init__(self, initial, step, cooling_rate, target_value=0):
        self.value        = initial
        self.step         = step
        self.cooling_rate = cooling_rate
        # Anneling process will stop at this value
        self.target_value = target_value
        self.reset_counter()

    def reset_counter(self):
        self.counter = 0

    def decay(self):
        """
        Step decaying
        """
        self.counter += 1
        # Update value if reach steps and above the target value
        if self.counter == self.step and self.value > self.target_value:
            self.value = self.value * self.cooling_rate
            self.reset_counter()
        return self.value

