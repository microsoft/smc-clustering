# Licensed under the MIT license.

import torch


def generate_circles(
    num_circles: int,
    min_radius: float = 0.1,
    max_radius: float = 1.5,
    min_points: int = 3,
    max_points: int = 20,
    min_x: float = -5.0,
    max_x: float = 5.0,
    min_y: float = -5.0,
    max_y: float = 5.0,
):
    circles = []
    masks = []
    for _ in range(num_circles):
        radius = torch.rand(1) * (max_radius - min_radius) + min_radius
        x = torch.rand(1) * (max_x - min_x) + min_x
        y = torch.rand(1) * (max_y - min_y) + min_y
        num_points = torch.randint(min_points, max_points, (1,))
        theta = torch.rand(num_points) * 2 * 3.14159
        x = radius * torch.cos(theta) + x
        y = radius * torch.sin(theta) + y
        circle = torch.stack([x, y], dim=-1)

        # do some padding with zeros
        num_pad = max_points - num_points
        circle = torch.cat([circle, torch.zeros(num_pad, 2)], dim=0)
        mask = torch.ones(max_points, dtype=bool)
        mask[num_points:] = False

        circles.append(circle)
        masks.append(mask)

    return circles, masks
