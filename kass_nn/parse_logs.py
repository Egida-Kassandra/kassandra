from _weakref import ref
from datetime import datetime
from dateutil.parser import parse
from multiprocessing import Pool

import numpy as np
import re
import threading
import os


class LogParser:
    """LogParser

    Clase que se encarga de transformar los archivos de log en arrays numericos
    """

    def __init__(self):
        """Constructor"""
        self.dict_ip = {}
        self.dict_req_meth = {}
        self.dict_req_url = {}
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

    def parse_calendar_get_id(self, date_string):
        """
        Parses the timestamp and returns id of hour range
        :param date_string: timestamp
        """
        current_hour = datetime.strptime(date_string, '%H:%M:%S')
        for key in self.dict_calendar_ids:
            if self.is_hour_in_range(current_hour, self.dict_calendar_ids[key]):
                return key
        return len(self.dict_calendar_ids)


    def is_hour_in_range(self, current_hour, range):
        """
        Returns if a determined hour is in range hour passed as param
        :param current_hour: datetime
        :param range: list of datetimes
        """
        if(range[0] <= current_hour and current_hour <= range[1]):
            return True
        return False

    def parse_file(self, filename, is_train):
        """
        Parse file
        :param filename: name of the file
        :param is_train: boolean
        """
        #cur_path = os.path.dirname(__file__)
        #lines = open(os.path.relpath(filename, cur_path)).read().splitlines()
        cur_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(cur_path, filename)
        lines = open(path).read().splitlines()

        www = self.weights_test
        if is_train:
            www = self.weights_train
        result = []
        for line in lines:
            new_line = self.parse_line(line, www, is_train)
            result.append(new_line)
    
        result = [r for r in result if r is not None]
        return result

    def parse_line(self, line, weights, is_train):
        """
        Parses a log line to a list of ints
        :param line: line to parse
        :param weights: list of ints
        :param is_train: boolean
        """
        single_data = []
        if len(line) == 0:
            return None
        try:
            line = line.strip()
            # IP #
            single_data.append(self.get_ip(line, weights[0]))
            # Timestamp #
            single_data.append(self.get_calendar(line))
            # Request #
            request = self.get_request(line)
            # Method #
            single_data.append(self.get_method(request, weights[1]))
            # URL #
            single_data.append(self.get_url(request, weights[2]))
            # Status code #
            single_data.append(self.get_status_code(request, weights[3]))
            # Referred URL #
            # User agent #
            if not is_train:
                print(line)
                print(single_data)
        except Exception as e:
            print(e)
            print("Parse error in line ", line)
            return None
        return single_data

    """============================== GET VARIABLES ================================"""

    def get_ip(self, line, weight):
        """
        Returns IP of log
        @param line: line to parse
        @param weight: int
        """
        str_line = line.split(' - - ')
        ip = str_line[0]
        self.parse_str_to_dict(self.dict_ip, ip, weight)
        return self.dict_ip[ip]

    def get_calendar(self, line):
        """
        Returns calendar ID of log
        @param line: line to parse
        """
        time = line.split(' - - ')[1].split("]")[0].strip("[")
        return self.parse_calendar_get_id(time[12:].split(' ')[0])

    def get_request(self, line):
        """
        This method is just for Paya to not be disturbed. Returns request of log.
        @param line: line to parse
        """
        request = re.split('" | "| ', line.split(' - - ')[1])
        return [r for r in request if r is not '']

    def get_method(self, request, weight):
        """
        Returns method of log
        @param request: line to parse
        @param weight: int
        """
        method = request[2]
        url = request[3]
        self.parse_str_to_dict(self.dict_req_meth, method, weight)
        return self.dict_req_meth[method]

    def get_url(self, request, weight):
        """
        Returns url of log
        @param request: line to parse
        @param weight: int
        """
        url = request[3]
        self.parse_str_to_dict(self.dict_req_url, url, weight)
        return self.dict_req_url[url]

    def get_status_code(self, request, weight):
        """
        Returns status code of log
        @param request: line to parse
        @param weight: int
        """
        status_code = request[5]
        self.parse_str_to_dict(self.dict_status_code, status_code, weight)
        return self.dict_status_code[status_code]

    def get_bytes(self, request):
        """
        Returns transferred bytes of log
        @param request: line to parse
        """
        bytes_transf = request[6]
        if bytes_transf == '-':
            bytes_transf = 0
        return int(bytes_transf)

    def get_referred_url(self, line):
        """
        Returns referred url of log
        @param line: line to parse
        """
        ref_url = line.split(' - - ')[1].split("\"")[3]
        self.parse_str_to_dict(self.dict_ref_url, ref_url, 1)
        return self.dict_ref_url[ref_url]

    def get_user_agent(self, line):
        """
        Returns user agent of log
        @param line: line to parse
        """
        user_agent = line.split(' - - ')[1].split("\"")[5]
        self.parse_str_to_dict(self.dict_user_agent, user_agent, 1)
        return self.dict_user_agent[user_agent]

    """============================== TEST DATA ================================"""

    def parse_str_to_dict(self, dictionary, word, step):
        """
        Adds new key to the dictionary passed by parameter
        and calculates the value of the key multiplying the step by
        the length of the dictionary
        Returns dictionary with new key.
        @param dictionary: dictionary
        @param word: str
        @param step: int
        """
        try:
            if word not in dictionary:
                dictionary[word] = len(dictionary)*step
        finally:
            return dictionary
