import yaml


def is_characteristic(charac, logpar_dict):
    return charac in logpar_dict


def is_in_training_single(test, prev_logpar):
    is_ip = is_characteristic(test[0], prev_logpar.dict_ip)
    is_meth = is_characteristic(test[1], prev_logpar.dict_req_meth)
    is_url = is_characteristic(test[2], prev_logpar.dict_req_url)
    presence_list = [is_ip, is_meth, is_url]
    get_results_single(presence_list)
    return presence_list


def get_results_single(presence_list):
    print("Sublevel Single")
    print("\tIP is in training: {}".format(presence_list[0]))
    print("\tMethod is in training: {}".format(presence_list[1]))
    print("\tURL is in training: {}".format(presence_list[2]))


def is_in_training_combined(test, prev_logpar):
    ip_meth = is_characteristic(test[0] + test[1], prev_logpar.comb_list_ip_meth)
    meth_url = is_characteristic(test[1] + test[2], prev_logpar.comb_list_meth_url)
    ip_url = is_characteristic(test[0] + test[2], prev_logpar.comb_list_ip_url)
    ip_meth_url = is_characteristic(test[0] + test[1] + test[2], prev_logpar.comb_list_ip_meth_url)
    presence_list = [ip_meth, meth_url, ip_url, ip_meth_url]
    get_results_combined(presence_list)
    return presence_list


def get_results_combined(presence_list):
    print("Sublevel Combined")
    print("\tIP and Method are in training: {}".format(presence_list[0]))
    print("\tMethod and URL are in training: {}".format(presence_list[1]))
    print("\tIP and URL are in training: {}".format(presence_list[2]))
    print("\tIP, Method and URL are in training: {}".format(presence_list[3]))
    return presence_list


def get_danger_value(presence_list_single, presence_list_combined, config_file):
    presence_list = presence_list_single + presence_list_combined
    dict_danger_values = {0: "danger_ip", 1: "danger_meth", 2: "danger_url", 3: "danger_ip_meth", 4: "danger_meth_url",
                          5: "danger_ip_url", 6: "danger_ip_meth_url"}
    yaml_document = open(config_file)
    danger_values = yaml.safe_load(yaml_document)
    result = 0
    for i in range(0, len(presence_list)):
        if not presence_list[i]:
            result += danger_values[dict_danger_values[i]]
    return result * 100
