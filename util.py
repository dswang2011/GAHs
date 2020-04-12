
import re
import numpy as np
import configparser
import argparse


# read three data types: string, int, and float. 
def parse_parameters(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    config_common = config['COMMON']
    dictionary = {}
    for key,value in config_common.items():
        array = value.split(',')
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        new_array = []
    
        for value in array:
            value = value.strip()
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else :
                    value= float(value)
            new_array.append(value)
        dictionary[key] = new_array
    return dictionary

def parse_and_set(file_path, opt):
    config = configparser.ConfigParser()
    config.read(file_path)
    config_common = config['COMMON']
    for key,value in config_common.items():
        array = value.split(',')
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        new_array = []
    
        for value in array:
            value = value.strip()
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else :
                    value= float(value)
            new_array.append(value)
        setattr(opt,key, value)

