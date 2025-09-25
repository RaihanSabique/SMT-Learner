import pandas as pd
import ast
import numpy as np
class Preprocessing:
    def __init__(self, data_path):
        """Initialize analysis with data path"""
        self.df = pd.read_csv(data_path)
        print("\nDataset loaded successfully.")
        print(f"Initial number of records: {len(self.df)}")
        self.preprocess_data()

    def getPreprocessedData(self):
        return self.df

    def preprocess_data(self):
        print("Preprocessing start")
        self.df['path'] = self.df.apply(lambda x: ast.literal_eval(x.path), axis=1)
        self.df['time_diff_ms'] = self.df.apply(lambda x: ast.literal_eval(x.time_diff_ms), axis=1)
        self.df['distances'] = self.df.apply(lambda x: self.distance_from_target(x.target_point_x, x.target_point_y, x.path), axis=1)
        self.df['rmsd'] = self.df.apply(lambda x: self.calculate_path_rmsd(x.path, (x.start_point_x, x.start_point_y),
                                                            (x.target_point_x, x.target_point_y)), axis=1)
        print("Preprocessing end")

    def distance_from_target(self, target_x, target_y, path_array):
        path = np.array(path_array)
        target = np.array((target_x, target_y))
        distances = np.sqrt(np.sum((target - path) ** 2, axis=1))
        return distances

    def calculate_path_rmsd(self, path_array, start_point, target_point):
        """
        Calculate Root Mean Square Deviation of a path from the straight line
        between start and target points.

        Parameters:
        path_array: numpy array of shape (n, 2) containing path coordinates
        start_point: tuple or array (x, y) of start position
        target_point: tuple or array (x, y) of target position

        Returns:
        float: RMSD value representing deviation from straight line
        """
        # Convert inputs to numpy arrays if they aren't already
        path = np.array(path_array)
        start = np.array(start_point)
        target = np.array(target_point)

        def point_to_line_distance(point, line_start, line_end):
            """
            Calculate the perpendicular distance from a point to a line defined by two points.
            Uses the formula: distance = |ax + by + c| / sqrt(a² + b²)
            where ax + by + c = 0 is the line equation
            """
            # Line equation coefficients: ax + by + c = 0
            a = line_end[1] - line_start[1]  # y2 - y1
            b = line_start[0] - line_end[0]  # x1 - x2
            c = line_end[0] * line_start[1] - line_start[0] * line_end[1]  # x2*y1 - x1*y2

            # Calculate perpendicular distance
            numerator = np.abs(a * point[0] + b * point[1] + c)
            denominator = np.sqrt(a ** 2 + b ** 2)

            return numerator / denominator if denominator != 0 else 0

        # Calculate distances from each path point to the straight line
        distances = np.array([point_to_line_distance(point, start, target)
                              for point in path])

        # Calculate RMSD
        rmsd = np.sqrt(np.mean(distances ** 2))

        return rmsd



