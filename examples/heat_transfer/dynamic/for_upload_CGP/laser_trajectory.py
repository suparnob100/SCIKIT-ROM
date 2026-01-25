import numpy as np
from numba import njit

class BaseTrajectory:
    """Abstract base class for laser trajectories using forward Euler integration or analytic formulae."""
    def __init__(self, dt, feed_rate):
        self.dt = dt
        self.feed_rate = feed_rate

    def position(self, t):
        """Return (x, y) at time t."""
        raise NotImplementedError

@njit
def _circular_pos(center_x, center_y, radius, feed_rate, t):
    # Analytical circular motion: angular speed = feed_rate / radius
    omega = feed_rate / radius
    x = center_x + radius * np.cos(omega * t)
    y = center_y + radius * np.sin(omega * t)
    return x, y

class CircularTrajectory(BaseTrajectory):
    def __init__(self, dt, feed_rate, center_x=10, center_y=10, radius=5):
        super().__init__(dt, feed_rate)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.name = 'Circular'

    def position(self, t):
        return _circular_pos(self.center_x, self.center_y, self.radius, self.feed_rate, t)

@njit
def _sinusoidal_pos(x0, y0, A, feed_rate, dt, t):
    x = x0
    y = y0
    nsteps = int(t / dt)
    for _ in range(nsteps):
        inv = 1.0 / np.sqrt(1 + (A**2) * np.cos(x) ** 2)
        x += dt * feed_rate * inv
        y += dt * feed_rate * A * np.cos(x) * inv
    remainder = t - nsteps * dt
    if remainder > 0:
        inv = 1.0 / np.sqrt(1 + A**2 * np.cos(x) ** 2)
        x += remainder * feed_rate * inv
        y += remainder * feed_rate * A * np.cos(x) * inv
    return x, y

class SinusoidalTrajectory(BaseTrajectory):
    def __init__(self, dt, feed_rate, x0, y0, A=5.0):
        super().__init__(dt, feed_rate)
        self.x0 = x0
        self.y0 = y0
        self.A = A
        self.name = 'Sinusoidal'

    def position(self, t):
        return _sinusoidal_pos(self.x0, self.y0, self.A, self.feed_rate, self.dt, t)

@njit
def _spiral_pos(center_x, center_y, spiral_rate, feed_rate, dt, t):
    theta = 0.0
    x = center_x
    y = center_y
    nsteps = int(t / dt)
    for _ in range(nsteps):
        theta_dot = feed_rate / (spiral_rate * np.sqrt(1 + theta ** 2))
        x_dot = (feed_rate * (np.cos(theta) - theta * np.sin(theta))) / np.sqrt(1 + theta ** 2)
        y_dot = (feed_rate * (np.sin(theta) + theta * np.cos(theta))) / np.sqrt(1 + theta ** 2)
        theta += dt * theta_dot
        x += dt * x_dot
        y += dt * y_dot
    remainder = t - nsteps * dt
    if remainder > 0:
        theta_dot = feed_rate / (spiral_rate * np.sqrt(1 + theta ** 2))
        x_dot = (feed_rate * (np.cos(theta) - theta * np.sin(theta))) / np.sqrt(1 + theta ** 2)
        y_dot = (feed_rate * (np.sin(theta) + theta * np.cos(theta))) / np.sqrt(1 + theta ** 2)
        theta += remainder * theta_dot
        x += remainder * x_dot
        y += remainder * y_dot
    return x, y

class SpiralTrajectory(BaseTrajectory):
    def __init__(self, dt, feed_rate, center_x=10, center_y=10, spiral_rate=5/(6*np.pi)):
        super().__init__(dt, feed_rate)
        self.center_x = center_x
        self.center_y = center_y
        self.spiral_rate = spiral_rate
        self.name = 'Spiral'

    def position(self, t):
        return _spiral_pos(self.center_x, self.center_y, self.spiral_rate, self.feed_rate, self.dt, t)

@njit
def _straight_line_pos(x0, y0, dx, dy, feed_rate, t):
    x = x0 + t * feed_rate * dx
    y = y0 + t * feed_rate * dy
    return x, y

class AnyStraightLineTrajectory(BaseTrajectory):
    """Linear trajectory along a specified direction with analytic position."""
    def __init__(self, dt, feed_rate, x0, y0, direction_x=1, direction_y=0):
        super().__init__(dt, feed_rate)
        norm = np.hypot(direction_x, direction_y)
        self.dx = direction_x / norm
        self.dy = direction_y / norm
        self.x0 = x0
        self.y0 = y0
        self.name = 'StraightLine'

    def position(self, t):
        return _straight_line_pos(self.x0, self.y0, self.dx, self.dy, self.feed_rate, t)

class HorizontalLineTrajectory(BaseTrajectory):
    """Linear trajectory along a specified direction using forward Euler integration."""
    def __init__(self, dt, feed_rate, x0, y0):
        super().__init__(dt, feed_rate)
        # Normalize direction
        self.x0 = x0
        self.y0 = y0
        self.name = 'HorizontalLine'

    def position(self, t):
        x = self.x0 + self.feed_rate * t
        y = self.y0
        # Constant derivatives
        return x, y