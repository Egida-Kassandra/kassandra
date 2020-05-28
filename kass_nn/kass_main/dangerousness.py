import yaml

border_0 = 0.50
border_1 = 0.55
border_2 = 0.60
border_3 = 0.65
border_4 = 0.70



def get_dangerousness_int(anomaly_value):
    if 0 <= anomaly_value <= border_0:
        return 0
    elif border_0 <= anomaly_value <= border_1:
        return 1
    elif border_1 <= anomaly_value <= border_2:
        return 2
    elif border_2 <= anomaly_value <= border_3:
        return 3
    elif border_3 <= anomaly_value <= border_4:
        return 4
    elif border_4 <= anomaly_value:
        return 5

def get_dangerousness_label(anomaly_values):
    anom_value = get_dangerousness(anomaly_values)
    return "Full anomaly value: {}\nDangerousness in range [0-5]: {}".format(anom_value, get_dangerousness_int(anom_value))

def get_dangerousness(anomaly_values):
    yaml_document = open("../config/config.yml")
    danger_values = yaml.safe_load(yaml_document)
    result = anomaly_values[0] * danger_values["value_min_meth"] + anomaly_values[1] * danger_values["value_min_dir"] \
             + anomaly_values[2] * danger_values["value_min_file_ext"] + anomaly_values[3] * danger_values["value_min_long"]

    return result

