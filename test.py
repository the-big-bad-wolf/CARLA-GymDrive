import math
import matplotlib.pyplot as plt


angles = [i * 360 / 60 for i in range(60)]


vehicle_width = 10.0
vehicle_length = 20.0


center_x, center_y = 0.0, 0.0
sensors = []

for angle in angles:
    rad = math.radians(angle)
    if math.cos(rad) != 0:
        slope = math.sin(rad) / math.cos(rad)
        if abs(slope) <= vehicle_length / vehicle_width:
            x = vehicle_width / 2 if math.cos(rad) > 0 else -vehicle_width / 2
            y = slope * x
        else:
            y = vehicle_length / 2 if math.sin(rad) > 0 else -vehicle_length / 2
            x = y / slope
    else:
        x = 0
        y = vehicle_length / 2 if math.sin(rad) > 0 else -vehicle_length / 2

    sensors.append(((x, y), angle))
    plt.plot([center_x, x], [center_y, y], "b-")

print(sensors)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(-vehicle_width, vehicle_width)
plt.ylim(-vehicle_length, vehicle_length)
plt.show()
