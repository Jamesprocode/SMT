import time

frames = ["(-_-)", "(-o-)", "(O_o)", "(^_^)", "(-_-)", "(x_x)"]

for i in range(20):
    print(frames[i % len(frames)], end="\r")
    time.sleep(0.3)
