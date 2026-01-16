"""Physical engine package - dynamics, numerical integration, maps, obstacles and ocean currents."""

from .integrator import RK4Integrator, EulerIntegrator
from .dynamics_2d import PointMassDynamics2D
from .map_2d import Map2D
from .obstacle_2d import Obstacle, CircleObstacle, RectangleObstacle, create_random_obstacles
from .ocean_current_2d import (
    OceanCurrent,
    UniformCurrent,
    VortexCurrent,
    GradientCurrent,
    OscillatingCurrent,
    TurbulentCurrent,
    CompositeOceanCurrent,
    CurrentType,
    create_ocean_current
)

__all__ = [
    "RK4Integrator", "EulerIntegrator", "PointMassDynamics2D",
    "Map2D", "Obstacle", "CircleObstacle", "RectangleObstacle", "create_random_obstacles",
    "OceanCurrent", "UniformCurrent", "VortexCurrent", "GradientCurrent",
    "OscillatingCurrent", "TurbulentCurrent", "CompositeOceanCurrent",
    "CurrentType", "create_ocean_current"
]
