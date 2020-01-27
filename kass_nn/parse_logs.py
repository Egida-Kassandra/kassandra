from _weakref import ref

from dateutil.parser import parse
from multiprocessing import Pool
import numpy as np
import re
dict_ip = {}
dict_req_meth = {}
dict_req_url = {}
dict_req_protocol = {}
dict_ref_url = {}
dict_user_agent = {}


def parse_file(filename, array_data):
    lines_number = 60000
    lines = open(filename).read().splitlines()
    pool = Pool(16)
    result = pool.map(parse_line, lines)
    pool.close()
    pool.join()
    result = [r for r in result if r is not None]

    return result


def parse_line(line):
    single_data = []
    if len(line) == 0:
        return None
    try:
        line = line.strip()
        ## IP ##
        str_line = line.split(' - - ')
        ip = str_line[0]
        ip = parse_str_to_dict(dict_ip, ip)
        single_data.append(ip)

        ## Timestamp ##
        time = str_line[1].split("]")[0].strip("[")
        date = parse(time[:11] + " " + time[12:])
        single_data.append(int(date.day))
        single_data.append(int(date.month))
        single_data.append(int(date.year))
        single_data.append(int(date.hour))
        single_data.append(int(date.minute))
        single_data.append(int(date.second))
        ## time zone offset ????

        ## Request ##
        #request = str_line[1].split("\"")[1]
        #request = request.split(" ")
        request = re.split('" | "| ', str_line[1])
        method = request[2]
        url = request[3]
        protocol = request[4]
       # print(method, url, protocol)
        single_data.append(parse_str_to_dict(dict_req_meth, method))
        single_data.append(parse_str_to_dict(dict_req_url, url))
        single_data.append(parse_str_to_dict(dict_req_protocol, protocol))
        ## Status code and bytes ##
        #status_code_and_bytes = str_line[1].split("\"")[2].strip().split(" ")
        status_code = request[5]
        bytes_transf = request[6]
        if bytes_transf == '-':
            bytes_transf = 0
        single_data.append(int(status_code))
        single_data.append(int(bytes_transf))
        ## Referrer URL ##
        ref_url = str_line[1].split("\"")[3]
        single_data.append(parse_str_to_dict(dict_ref_url, ref_url))
        ## User agent ##
        user_agent = str_line[1].split("\"")[5]
        single_data.append(parse_str_to_dict(dict_user_agent, user_agent))

    except IndexError as e:
        print("fuck this line: ", line)
        return None
    return single_data


def parse_str_to_dict(dictionary, word):
    if word in dictionary:
        return dictionary[word]
    else:
        dictionary[word] = len(dictionary)
        return dictionary[word]


if __name__ == '__main__':
    train_data = []
    train_labels = []
    train_data = parse_file('access_news.log', train_data)
    print(len(train_data))