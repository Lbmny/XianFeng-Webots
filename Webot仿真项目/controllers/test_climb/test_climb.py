"""Minimal climbing test controller — drives robot toward obstacle, logs position."""
from controller import Robot, Node
import sys

robot = Robot()
timestep = int(robot.getBasicTimeStep())
name = robot.getName()
print(f"[TestClimb] Robot: {name}")

# Init GPS
gps = robot.getDevice("gps")
if gps:
    gps.enable(timestep)
else:
    print("[TestClimb] WARNING: No GPS found")

# Init all motors
motor_names = []
for i in range(robot.getNumberOfDevices()):
    dev = robot.getDeviceByIndex(i)
    if dev.getNodeType() == Node.ROTATIONAL_MOTOR:
        motor_names.append(dev.getName())

print(f"[TestClimb] Motors found: {motor_names}")
motors = []
for mn in motor_names:
    m = robot.getDevice(mn)
    m.setPosition(float('inf'))
    m.setVelocity(0.0)
    motors.append(m)

# Set speed: all motors at +8 rad/s
SPEED = 8.0
for m in motors:
    m.setVelocity(SPEED)
print(f"[TestClimb] All motors set to {SPEED} rad/s")

step = 0
while robot.step(timestep) != -1:
    step += 1
    if step % 25 == 0:
        if gps:
            pos = gps.getValues()
            print(f"[{name}] step={step} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.3f})")
        else:
            print(f"[{name}] step={step}")

    if step > 2000:
        print(f"[{name}] Test timeout — stopping")
        break
