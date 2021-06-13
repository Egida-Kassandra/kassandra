from builtins import len
from ctypes import c_ubyte
from datetime import datetime
import re
import os


class LogParser:
    """LogParser

    Clase que se encarga de transformar los archivos de log en arrays numericos
    """

    def __init__(self, filename):
        """Constructor"""
        self.dict_ip = {}
        self.dict_req_meth = {}
        self.dict_req_meth_freq = {}
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

        self.dict_req_url_freq = {}
        self.dict_file_ext = {}
        self.dict_file_ext_freq = {}
        self.dict_req_len = {}
        self.dict_req_len_freq = {}

        self.comb_list_ip_meth = []
        self.comb_list_meth_url = []
        self.comb_list_ip_url = []
        self.comb_list_ip_meth_url = []

        self.weights_train = [1,1,1,1,1,1,1]
        self.weights_test = [1,1,1,1,1,1,1]

        self.parsed_train_data = []
        self.parsed_train_data = self.parse_file(filename, True)

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

    def get_minute(self, date_string):
        current_hour = datetime.strptime(date_string, '%H:%M:%S')
        return current_hour.hour*60 + current_hour.minute

    def is_hour_in_range(self, current_hour, range):
        """
        Returns if a determined hour is in range hour passed as param
        :param current_hour: datetime
        :param range: list of datetimes
        """
        if range[0] <= current_hour <= range[1]:
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
        lines = open(filename).read().splitlines()
        if is_train:
            self.pre_parse_file(lines)

        www = self.weights_test
        if is_train:
            www = self.weights_train
        result = []
        generated_file = None #open("../sint_data_url_50freq.txt", "a")
        for line in lines:
            new_line = self.parse_line(line, www, is_train, generated_file)
            result.append(new_line)
    
        result = [r for r in result if r is not None]
        return result

    def parse_line(self, line, weights, is_train, generated_file):
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
            url = self.get_url(request, weights[2])
            #############################
            #self.generate_synt_data(generated_file, url, line)
            #############################
            single_data.append(url)
            # Status code #
            single_data.append(self.get_status_code(request, weights[3]))
            # Referred URL #
            # User agent #

            # File extension
            single_data.append(self.get_file_extension(request))

            # Request length
            single_data.append(self.get_req_len(request))

            # Combined dicts
            combined_str = self.get_combined_strings(line)
            if combined_str[0] not in self.comb_list_meth_url:
                self.comb_list_ip_meth.append(combined_str[0])
                self.comb_list_meth_url.append(combined_str[1])
                self.comb_list_ip_url.append(combined_str[2])
                self.comb_list_ip_meth_url.append(combined_str[3])
        except Exception as e:
            print(e)
            print("Parse error in line ", line)
            return None
        return single_data


    def public_parse_line(self, line, is_train):
        www = self.weights_test
        if is_train:
            www = self.weights_train
        return [self.parse_line(line, www, is_train, None)]


    def generate_synt_data(self, file, url, line):
        if (url < 50):
            file.write(line + "\n")

    def pre_parse_line(self, line):
        """
        Pre-Parses a log line to a list of ints
        :param line: line to parse
        """
        if len(line) != 0:

            try:
                line = line.strip()
                # IP #
                # Timestamp #
                # Request #
                request = self.get_request(line)
                # Method #
                self.get_method_pre_parse(request)
                # URL #
                self.get_url_pre_parse(request)
                # Status code #
                # Referred URL #
                # User agent #

                # File extension
                self.get_file_ext_pre_parse(request)

                # Request length
                self.get_req_len_preparse(request)

            except Exception as e:
                print(e)
                print("Pre-Parse error in line ", line)


    def pre_parse_file(self, lines):
        """
                Parse file
                :param lines: lines of the file
                """
        for line in lines:
            self.pre_parse_line(line)
        ## Parse URL
        self.dict_req_meth_freq = self.parse_frequencies(self.dict_req_meth_freq)
        self.dict_req_url_freq = self.parse_frequencies(self.dict_req_url_freq)
        self.dict_file_ext_freq = self.parse_frequencies(self.dict_file_ext_freq)
        self.dict_req_len_freq = self.parse_frequencies(self.dict_req_len_freq)

    def parse_frequencies(self, dict):
        dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
        value = 0
        for key in dict:
            dict[key] = value
            value += 1
        return dict

    """=========================== GET STRING VARIABLES ============================"""

    def get_string_variables(self, filename):
        """
        Parse file
        :param filename: name of the file
        """
        lines = open(filename).read().splitlines()
        result = []
        for line in lines:
            line = line.strip()
            new_line = self.get_string_log_list(line)
            result.append(new_line)
        result = [r for r in result if r is not None]
        return result

    def get_string_log_list(self, log):
        # IP
        str_line = log.split(' - - ')
        ip = str_line[0]
        # Meth
        request = re.split('" | "| ', log.split(' - - ')[1])
        request = [r for r in request if r is not '']
        method = request[2]
        # URL dir
        url = request[3] # here
        url_list = url.split("/")
        directory = ''
        if len(url_list) < 3 and len(url_list[1].split(".")) > 1:
            directory = url_list[0]
        else:
            directory = url_list[1]
        #print(url_list)
        #print(directory)
        return [ip, method, directory]

    def get_combined_strings(self, log):
        strlist = self.get_string_log_list(log)
        ip_meth = strlist[0]+strlist[1]
        meth_url = strlist[1] + strlist[2]
        ip_url = strlist[0] + strlist[2]
        ip_meth_url = strlist[0] + strlist[1] + strlist[2]
        return [ip_meth, meth_url, ip_url, ip_meth_url]

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
        #return self.parse_calendar_get_id(time[12:].split(' ')[0])
        return self.get_minute(time[12:].split(' ')[0])

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
        self.parse_str_to_dict_freq(self.dict_req_meth, self.dict_req_meth_freq, method, weight)
        return self.dict_req_meth[method]

    def get_method_pre_parse(self, request):
        """
        Returns meth of log
        @param request: line to parse
        """
        method = request[2]
        url = request[3]
        self.parse_str_to_dict_pre_parse(self.dict_req_meth_freq, method)
        return self.dict_req_meth_freq[method]


    def get_url_pre_parse(self, request):
        """
        Returns url of log
        @param request: line to parse
        """
        url = request[3]
        url_list = url.split("/")
        directory = ''
        if len(url_list) < 3 and len(url_list[1].split(".")) > 1:
            directory = url_list[0]
        else:
            directory = url_list[1]
        self.parse_str_to_dict_pre_parse(self.dict_req_url_freq, directory)
        return self.dict_req_url_freq[directory]

    def get_url(self, request, weight):
        """
        Returns url of log
        @param request: line to parse
        @param weight: int
        """
        url = request[3]
        url_list = url.split("/")
        directory = ''
        if len(url_list) < 3 and len(url_list[1].split(".")) > 1:
            directory = url_list[0]
        else:
            directory = url_list[1]
        self.parse_str_to_dict_freq(self.dict_req_url, self.dict_req_url_freq, directory, weight)
        return self.dict_req_url[directory]



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


    def get_file_ext_pre_parse(self, request):
        """
        Returns url of log
        @param request: line to parse
        """
        url = request[3]
        try:
            possible_extension = re.split("[? | %]", url)[0].split(".")
            if len(possible_extension) > 1:
                extension = possible_extension[-1].strip("/").lower()
                self.parse_str_to_dict_pre_parse(self.dict_file_ext_freq, extension)
                return self.dict_file_ext_freq[extension]
            else:
                return -1
        except Exception as e:
            return -1

    def get_file_extension(self, request):
        """

        @param request: line to parse
        """
        url = request[3]
        try:
            possible_extension = re.split("[? | %]", url)[0].split(".")
            if len(possible_extension) > 1:
                extension = possible_extension[-1].strip("/").lower()
                self.parse_str_to_dict_freq(self.dict_file_ext, self.dict_file_ext_freq, extension, 0)
                return self.dict_file_ext[extension]
            else:
                return -1
        except Exception as e:
            return -1

    def get_req_len_preparse(self, request):
        """
        Returns url of log
        @param request: line to parse
        """
        url = request[3]
        length = len(url)
        str_len = str(length)
        self.parse_str_to_dict_pre_parse(self.dict_req_len_freq, str_len)
        return self.dict_req_len_freq[str_len]

    def get_req_len(self, request):
        """

        @param request: line to parse
        """
        """
        url = request[3]
        length = len(url)
        self.parse_str_to_dict_freq(self.dict_req_len, self.dict_req_len_freq, length, 0)
        """
        url = request[3]
        length = len(url)
        str_len = str(length)
        self.parse_str_to_dict_freq(self.dict_req_len, self.dict_req_len_freq, str_len, 0)

        """
        try:
            if str_len not in self.dict_req_len:
                self.dict_req_len[str_len] = length
        except Exception as e:
            print(e)
            return self.dict_req_len
        """
        try:
            return self.dict_req_len[str_len]
        except:
            return len(self.dict_req_len)


    """============================== PARSE TO DICT ================================"""

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
                if len(dictionary) == 0:
                    dictionary[word] = 0
                else:
                    #dictionary[word] = (int(dictionary[list(dictionary)[len(dictionary)-1]]/step)+1)*step
                    dictionary[word] = int(dictionary[list(dictionary)[len(dictionary)-1]])+1
        finally:
            return dictionary

    def parse_str_to_dict_pre_parse(self, dict_freq, word):
        """
        Returns dictionary with new key.
        @param dictionary: dictionary
        @param word: str
        """
        try:
            if word not in dict_freq:
                if len(dict_freq) == 0:
                    dict_freq[word] = 1
                else:
                    dict_freq[word] = 1
            else:
                dict_freq[word] = dict_freq[word] + 1
        finally:
            return dict_freq

    def parse_str_to_dict_freq(self, dictionary, dict_freq, word, step):
        """
        Returns dictionary with new key.
        @param dictionary: dictionary
        @param word: str
        """
        try:
            if word not in dictionary:
                dictionary[word] = dict_freq[word]
        finally:
            return dictionary
