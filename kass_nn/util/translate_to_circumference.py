import math


def get_scp_point_A( minute, radius):
    one_min = math.pi / 720 # constant
    rad = minute * one_min
    x = math.sqrt(radius / (1 + math.tan(rad) ** 2))
    y = math.tan(rad) * x
    return [x ,y]


def get_scp_point_B( minute, radius):
    one_min = math.pi / 720
    rad = minute * one_min
    x = math.sqrt(radius / (1 + math.tan(rad) ** 2))
    x = -x
    y = math.tan(rad) * x
    return [x ,y]


def parse_sc_to_scp(data_pandas, charac):
    points = []
    col1 = charac.columns[0]
    col2 = charac.columns[1]
    for index, row in data_pandas.iterrows():
        new_point = []
        if 0 <= row[col1] < 360 or 1080 < row[1] <= 1440:
            new_point = get_scp_point_A(row[col1], charac.radius1 + charac.radius2 * row[col2])
        elif 360 < row[col1] < 1080:
            new_point = get_scp_point_B(row[col1], charac.radius1 + charac.radius2 * row[col2])
        elif row[1] == 360:
            new_point = [0, math.sqrt(charac.radius1 + charac.radius2 * row[col2])]
        elif row[1] == 1080:
            new_point = [0, -math.sqrt(charac.radius1 + charac.radius2 * row[col2])]
        for i in range (2, len(charac.columns)):
            new_point.append(row[charac.columns[i]])
        points.append(new_point)
    return points
