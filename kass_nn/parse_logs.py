from _weakref import ref

from dateutil.parser import parse
from multiprocessing import Pool
import numpy as np
import re
import threading
from datetime import datetime

class LogParser:

    def __init__(self):
        self.dict_ip = {}
        self.dict_req_meth = {}
        self.dict_req_url = {}
        self.dict_req_protocol = {}
        self.dict_ref_url = {}
        self.dict_user_agent = {}
        self.dict_status_code = {}
        self.dict_calendar_ids = {
            0: [datetime.strptime('09:00:00', '%H:%M:%S'),  datetime.strptime('14:00:00', '%H:%M:%S')], # work
            500: [datetime.strptime('16:00:00', '%H:%M:%S'),  datetime.strptime('20:00:00', '%H:%M:%S')], # work
            1000: [datetime.strptime('00:00:00', '%H:%M:%S'),  datetime.strptime('08:59:59', '%H:%M:%S')],
            1500: [datetime.strptime('14:00:01', '%H:%M:%S'),  datetime.strptime('15:59:59', '%H:%M:%S')],
            2000: [datetime.strptime('20:00:01', '%H:%M:%S'),  datetime.strptime('23:59:59', '%H:%M:%S')]
        }
        self.weights_train = [1,500,1,200,1,1,1]
        self.weights_test = [1,500,1,200,1,1,1]
        #weights = []

    def parse_calendar_get_id(self, date_string):
        current_hour = datetime.strptime(date_string, '%H:%M:%S')
        for key in self.dict_calendar_ids:
            if self.is_hour_in_range(current_hour, self.dict_calendar_ids[key]):
                return key
        return len(self.dict_calendar_ids)


    def is_hour_in_range(self, current_hour, range):
        if(range[0] <= current_hour and current_hour <= range[1]):
            return True
        return False

    def parse_file(self, filename, is_train):
        lines = open(filename).read().splitlines()
        #pool = Pool(1)

        www = self.weights_test
        if is_train:
            www = self.weights_train
        #result = pool.map(partial(parse_line, weights=www), lines)
        result = []
        for line in lines:
            result.append(self.parse_line(line, www, is_train))
        #pool.close()
        #pool.join()
        result = [r for r in result if r is not None]
        #print(self.dict_ip)
        return result

    def parse_line(self, line, weights, is_train):
        single_data = []
        if len(line) == 0:
            return None
        try:
            if not is_train:
                print(line)
            line = line.strip()
            ## IP ##
            str_line = line.split(' - - ')
            ip = str_line[0]
            self.parse_str_to_dict(self.dict_ip, ip, weights[0])
            single_data.append(self.dict_ip[ip])

            ## Timestamp ##
            time = str_line[1].split("]")[0].strip("[")
            #date = parse(time[:11] + " " + time[12:])
            calendar_id = self.parse_calendar_get_id(time[12:].split(' ')[0])
            single_data.append(calendar_id)
            #self.parse_calendar_get_id(date)
            """
            single_data.append(int(date.day))
            single_data.append(int(date.month))
            single_data.append(int(date.year))
            single_data.append(int(date.hour))
            single_data.append(int(date.minute))
            single_data.append(int(date.second))
            """
            ## time zone offset ????

            ## Request ##
            #request = str_line[1].split("\"")[1]
            #request = request.split(" ")
            request = re.split('" | "| ', str_line[1])
            request = [r for r in request if r is not '']
            method = request[2]
            url = request[3]
            protocol = request[4]
           # print(method, url, protocol)
            self.parse_str_to_dict(self.dict_req_meth, method, weights[1])
            single_data.append(self.dict_req_meth[method])

            self.parse_str_to_dict(self.dict_req_url, url, weights[2])
            single_data.append(self.dict_req_url[url])

            #self. parse_str_to_dict(self.dict_req_protocol, protocol, weights[3])
            #single_data.append(self.dict_req_protocol[protocol])
            ## Status code and self.bytes ##
            #status_code_and_bytes = str_line[1].split("\"")[2].strip().split(" ")
            status_code = request[5]
            bytes_transf = request[6]
            if bytes_transf == '-':
                bytes_transf = 0
            self.parse_str_to_dict(self.dict_status_code, status_code, weights[3])
            single_data.append(self.dict_status_code[status_code])
            #single_data.append(int(bytes_transf))
            ## Referrer URL ##
            ref_url = str_line[1].split("\"")[3]
            #self.parse_str_to_dict(self.dict_ref_url, ref_url, weights[5])
            #single_data.append(self.dict_ref_url[ref_url])
            ## User agent ##
            user_agent = str_line[1].split("\"")[5]
            #self.parse_str_to_dict(self.dict_user_agent, user_agent, weights[6])
            #single_data.append(self.dict_user_agent[user_agent])
            #print(dict_req_meth)
            #print(single_data)
            if not is_train:

                #print(line)
                print(single_data)
                #print()
        except Exception as e:
            print(e)
            print("fuck this line: ", line)

            return None
        return single_data

    """============================== TEST DATA ================================"""


    def parse_str_to_dict(self, dictionary, word, step):

        try:
            if word not in dictionary:
                dictionary[word] = len(dictionary)*step
        finally:
            return dictionary


"""
if __name__ == '__main__':
    train_data = []
    train_labels = []
    lg = LogParser()
    train_data = lg.parse_file('testdata3.log', True)
    print(len(train_data))
"""