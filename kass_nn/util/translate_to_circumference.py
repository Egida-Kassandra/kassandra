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


def parse_sc_to_scp(data_pandas, columns, radius1, radius2):
    points = []
    col1 = columns[0]
    col2 = columns[1]
    for index, row in data_pandas.iterrows():
        if 0 <= row[col1] < 360 or 1080 < row[1] <= 1440:
            points.append(get_scp_point_A(row[col1], radius1 + radius2 * row[col2]))
        elif 360 < row[col1] < 1080:
            points.append(get_scp_point_B(row[col1], radius1 + radius2 * row[col2]))
        elif row[1] == 360:
            points.append([0, math.sqrt(radius1 + radius2 * row[col2])])
        elif row[1] == 1080:
            points.append([0, -math.sqrt(radius1 + radius2 * row[col2])])
    return points
