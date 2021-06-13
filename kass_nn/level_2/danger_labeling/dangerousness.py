import yaml

border_0 = 0.50
border_1 = 0.55
border_2 = 0.60
border_3 = 0.65
border_4 = 0.70

dict_ponds = {0: "value_min_meth", 1: "value_min_dir", 2: "value_min_file_ext", 3: "value_min_long"}


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


def get_dangerousness_label(anomaly_values,config_file):
    anom_value = get_dangerousness(anomaly_values, config_file)
    return "Full anomaly value: {}\nDangerousness in range [0-5]: {}".format(anom_value,
                                                                             get_dangerousness_int(anom_value))


def get_dangerousness(anomaly_values, config_file):
    yaml_document = open(config_file)
    danger_values = yaml.safe_load(yaml_document)
    dang_pond_extra = danger_values["dangerous_value_extra"]

    none_values_rest = 0
    none_values_num = 0
    i = 0
    # Share value ponderation between the rest of characteristics if one is None
    for val in anomaly_values:
        if val is None:
            none_values_rest += danger_values[dict_ponds[i]]
            none_values_num += 1
        i += 1
    none_value_pond = none_values_rest / (len(anomaly_values) - none_values_num)

    anomaly_values = [r for r in anomaly_values if r is not None]
    dang_num = len([r for r in anomaly_values if r >= border_3])

    dang_pond = 0
    if dang_num > 0 and len(anomaly_values) is not dang_num:
        dang_pond = dang_pond_extra
        dang_value_pond = (dang_pond * dang_num) / (len(anomaly_values) - dang_num)
    else:
        dang_value_pond = 0
    result = 0
    i = 0
    for val in anomaly_values:
        add_value = danger_values[dict_ponds[i]] + none_value_pond
        if val >= border_3:
            result += val * (add_value + dang_pond)
        else:
            result += val * (add_value - dang_value_pond)

        i += 1
    return result
