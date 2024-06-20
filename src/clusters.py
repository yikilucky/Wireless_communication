import numpy as np


def create_clusters():
    # Simulation window parameters
    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 5

    # Parameters for the parent and daughter point processes
    # lambda_parent = 0.3  # density of parent Poisson point process
    lambda_daughter = 100  # mean number of points in each cluster
    radius_cluster = 0.5  # radius of cluster disk (for daughter points)

    # rectangle dimensions
    x_delta = x_max - x_min
    y_delta = y_max - y_min
    area_total = x_delta * y_delta  # area of  rectangle

    # Simulate Poisson point process for the parents
    # num_parent_points = np.random.poisson(area_total * lambda_parent)  # Poisson number of points
    num_parent_points = 9
    # x and y coordinates of Poisson points for the parent
    xx_parent = x_min + x_delta * np.random.uniform(0, 1, num_parent_points)
    yy_parent = y_min + y_delta * np.random.uniform(0, 1, num_parent_points)

    xx_parent = np.append(xx_parent, [0])
    yy_parent = np.append(yy_parent, [0])
    num_parent_points += 1

    # Simulate Poisson point process for the daughters (ie final poiint process)
    # num_daughter_points = np.random.poisson(lambda_daughter, num_parent_points-1)
    num_daughter_points = np.full((num_parent_points - 1,), lambda_daughter, dtype=int)
    num_daughter_points = np.append(num_daughter_points, [30])
    sum_daughters = sum(num_daughter_points)  # total number of points


    # Generate the (relative) locations in polar coordinates by
    # simulating independent variables.
    theta = 2 * np.pi * np.random.uniform(0, 1, sum_daughters)  # angular coordinates
    rho = radius_cluster * np.sqrt(np.random.uniform(0, 1, sum_daughters))  # radial coordinates

    # Convert from polar to Cartesian coordinates
    xx0 = rho * np.cos(theta)
    yy0 = rho * np.sin(theta)

    # replicate parent points (ie centres of disks/clusters)
    xx = np.repeat(xx_parent, num_daughter_points)
    yy = np.repeat(yy_parent, num_daughter_points)

    # translate points (ie parents points are the centres of cluster disks)
    xx = xx + xx0  # one-dimension
    yy = yy + yy0  # one-dimension

    xx_new = np.split(xx, np.cumsum(num_daughter_points)[:-1])
    yy_new = np.split(yy, np.cumsum(num_daughter_points)[:-1])

    xx_new = np.array(xx_new, dtype="object")  # two-dimension
    yy_new = np.array(yy_new, dtype="object")  # two-dimension

    # thin points if outside the simulation window

    def check_array(array1, array2, lower, upper):
        result = []
        for ii in range(len(array1)):
            row_result = []
            for jj in range(len(array1[ii])):
                if array1[ii][jj] <= lower or array1[ii][jj] >= upper or array2[ii][jj] <= lower or array2[ii][jj] >= upper:
                    row_result.append(False)
                else:
                    row_result.append(True)
            result.append(row_result)
        return result

    boole_results = check_array(xx_new, yy_new, x_min, x_max)  # two-dimension

    devices_x = {}
    for i in range(len(xx_new)):
        devices_x[i] = xx_new[i][np.array(boole_results[i])]

    devices_y = {}
    for i in range(len(yy_new)):
        devices_y[i] = yy_new[i][np.array(boole_results[i])]

    return num_parent_points, xx_parent, yy_parent, devices_x, devices_y
