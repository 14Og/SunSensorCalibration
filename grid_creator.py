import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import yaml
from typing import List, Tuple, Iterable
from collections import namedtuple

# cartesian_coordinates = namedtuple("cartesian_coordinates", ["all", "min", "max"])


class GridCreator:
    def __init__(
        self,
        pathname: str = "",
        filename: str = "",
        azimuth_scope: List = [0, 45],
        azimuth_step: int = 5,
        elevation_scope: List = [0, 360],
        elevation_step: int = 5,
        cartesian_linspace_size=20,
    ):
        azimuth_len = (azimuth_scope[1] - azimuth_scope[0]) // azimuth_step + 1
        elevation_len = (elevation_scope[1] - elevation_scope[0]) // elevation_step

        self.azimuth_linspace: np.ndarray = np.linspace(
            azimuth_scope[0], azimuth_scope[1], azimuth_len, dtype=int
        )
        self.elevation_linspace: np.ndarray = np.linspace(
            elevation_scope[0],
            elevation_scope[1],
            elevation_len,
            dtype=int,
            endpoint=False,
        )
        self.spherical_sample_space, s_grid1, s_grid2 = self._get_sample_space(
            self.azimuth_linspace, self.elevation_linspace
        )
        self.pathname = pathname or "grid_data"
        self.filename = (
            filename
            or f"{azimuth_scope[0]}_{azimuth_scope[1]}_step_{azimuth_step}"
            + f"_{elevation_scope[0]}_{elevation_scope[1]}_step_{elevation_step}"
        )
        self.x_linspace, self.x_parameters = self._get_x_linspace(
            self.spherical_sample_space, cartesian_linspace_size
        )

        self.y_linspace, self.y_parameters = self._get_y_linspace(
            self.spherical_sample_space, cartesian_linspace_size
        )

        self.cartesian_sample_space, c_grid1, c_grid2 = self._get_sample_space(
            self.x_linspace, self.y_linspace
        )

        self.matched_cartesian_grid, self.unmatchedcarte_grid = (
            self.create_stand_calibration_data(self.cartesian_sample_space)
        )
        self.plot_grids()

    @staticmethod
    def _get_sample_space(
        linspace1: np.ndarray, linspace2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # sample_space, grid1, grid2
        grid1, grid2 = np.meshgrid(linspace1, linspace2)
        spherical_sample_space = list(
            map(tuple, np.column_stack((grid1.ravel(), grid2.ravel())))
        )
        return np.array(spherical_sample_space), grid1, grid2

    @staticmethod
    def _get_x_linspace(
        spherical_space: np.ndarray,
        size: int,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, float, float]]:

        x_all = np.array(
            list(
                set(
                    [
                        math.sin(math.radians(sample[0]))
                        * math.cos(math.radians(sample[1]))
                        for sample in spherical_space
                    ]
                )
            )
        )
        x_max = max(x_all)
        x_min = min(x_all)
        x_range = np.linspace(x_min, x_max, size)
        return x_range, (x_all, x_min, x_max)

    @staticmethod
    def _get_y_linspace(
        spherical_space: np.ndarray, size: int
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, float, float]]:
        y_all = np.array(
            list(
                set(
                    [
                        math.sin(math.radians(sample[0]))
                        * math.sin(math.radians(sample[1]))
                        for sample in spherical_space
                    ]
                )
            )
        )
        y_max = max(y_all)
        y_min = min(y_all)

        y_range = np.linspace(y_min, y_max, size)
        return y_range, (y_all, y_min, y_max)

    @staticmethod
    def _cartesian_grid_to_spherical(
        x, y, z, resolution=0.1125, match_to_resolution=False
    ) -> List[float]:
        azimuth = math.degrees(math.acos(z))
        elevation = math.atan2(y, x)
        elevation = math.degrees(
            2 * math.pi + elevation if elevation < 0 else elevation
        )

        if match_to_resolution:
            if azimuth % resolution != 0:
                azimuth = (
                    azimuth // resolution
                    if azimuth % resolution < resolution / 2
                    else (azimuth // resolution) + 1
                ) * resolution
            if elevation % resolution != 0:
                elevation = (
                    elevation // resolution
                    if elevation % resolution < resolution / 2
                    else (elevation // resolution) + 1
                ) * resolution

        azimuth = round(azimuth, 4)
        elevation = round(elevation, 4)
        return [azimuth, elevation]

    def plot_grids(self):
        fig, (matched_ax, unmatched_ax, both) = plt.subplots(nrows=1, ncols=3)
        matched_ax.set_title("matched coordinates")
        unmatched_ax.set_title("unmatched coordinates")
        both.set_title("combined grids")
        matched_ax.scatter(
            self.matched_cartesian_grid[:, 0],
            self.matched_cartesian_grid[:, 1],
            marker=".",
            color="red",
        )
        unmatched_ax.scatter(
            self.unmatchedcarte_grid[:, 0],
            self.unmatchedcarte_grid[:, 1],
            marker=".",
            color="blue",
        )
        both.scatter(
            self.matched_cartesian_grid[:, 0],
            self.matched_cartesian_grid[:, 1],
            marker=".",
            color="red",
        )
        both.scatter(
            self.unmatchedcarte_grid[:, 0],
            self.unmatchedcarte_grid[:, 1],
            marker=".",
            color="blue",
        )
        plt.show()

    def create_stand_calibration_data(
        self, cartesian_sample_space: np.ndarray, mode: str = "yaml"
    ) -> Tuple[np.ndarray, np.ndarray]:
        radius = self.x_parameters[-1]
        spherical_grid_matched = []
        spherical_grid_unmatched = []
        cartesian_grid_unmatched = []
        for pair in cartesian_sample_space:
            x = pair[0]
            y = pair[1]
            if math.sqrt(x**2 + y**2) <= radius:
                z = math.sqrt(1 - x**2 - y**2)
                cartesian_grid_unmatched.append([x, y, z])
                spherical_grid_matched.append(
                    self._cartesian_grid_to_spherical(x, y, z, match_to_resolution=True)
                )
                spherical_grid_unmatched.append(
                    self._cartesian_grid_to_spherical(
                        x, y, z, match_to_resolution=False
                    )
                )

        spherical_grid_matched.sort(key=lambda h: (h[0], h[1]))
        spherical_grid_matched_dict = [
            {"azimuth": value[0], "elevation": value[1]}
            for value in spherical_grid_matched
        ]
        spherical_grid_unmatched.sort(key=lambda h: (h[0], h[1]))
        spherical_grid_unmatched_dict = [
            {"azimuth": value[0], "elevation": value[1]}
            for value in spherical_grid_unmatched
        ]
        with open(
            os.path.join(self.pathname, self.filename + "_matched" + f".{mode}"), "w"
        ) as matched_grid_file, open(
            os.path.join(self.pathname, self.filename + "_unmatched" + f".{mode}"), "w"
        ) as unmatched_grid_file:
            matched_grid_file.write(
                yaml.dump(data=spherical_grid_matched_dict, default_flow_style=False)
            )
            unmatched_grid_file.write(
                yaml.dump(data=spherical_grid_unmatched_dict, default_flow_style=False)
            )
        cartesian_grid_unmatched = np.array(cartesian_grid_unmatched)
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            cartesian_grid_unmatched[:, 0],
            cartesian_grid_unmatched[:, 1],
            cartesian_grid_unmatched[:, 2],
        )
        plt.show()
        return np.array(spherical_grid_matched), np.array(spherical_grid_unmatched)


if __name__ == "__main__":
    gc = GridCreator(azimuth_scope=[0, 50])
