#!/usr/bin/env python2
# -*- coding: utf8 -*-

import pymysql
import time

def getConn():
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='777', db='StudentManager', charset='utf8')
    return conn

def getNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

def replaceStudents(data):
    conn = getConn()
    cur = conn.cursor()
    try:
        cur.execute("USE StudentManager")
        cur.execute("INSERT INTO students (Name, Class, Age, Sex, Math, Computer, Physics, Registry) values (%s,%s,%s,%s,%s,%s,%s,%s)",
        (data['Name'], data['Class'], data['Age'], data['Sex'], data['Math'], data['Computer'], data['Physics'], getNowTime()) 
        )
        conn.commit()
        if cur and conn:
            cur.close()
            conn.close()
    except ValueError as e:
        print e
    except pymysql.err.InternalError as e:
        print e
    except UnicodeEncodeError as e:
        print e


import numpy as np
import random
def find_interval(x, partition):
    '''
    返回x所在区间的左边界索引；
    如果 partition[i] < x < partition[i+1]， 则返回 i; 否则返回 -1；
    '''
    for i in range(0, len(partition)):
        if x < partition[i]:
            return i - 1
    return -1

def weighted_choice(sequence, 
                    weights,
                    secure=True):
    """ 
    weighted_choice selects a random element of 
    the sequence according to the list of weights
    """
    
    if secure:
        crypto = random.SystemRandom()
        x = crypto.random()
    else:
        x = np.random.random()
    cum_weights = [0] + list(np.cumsum(weights))
    index = find_interval(x, cum_weights)
    return sequence[index]
def cartesian_choice(*iterables):
    res = []
    for population in iterables:
        lst = random.choice(population)
        res.append(lst)
    return res

def weighted_cartesian_choice(*iterables):
    """
    A list with weighted random choices from each iterable of iterables 
    is being created in respective order
    """
    res = []
    for population, weight in iterables:
        lst = weighted_choice(population, weight)
        res.append(lst)
    return res

def getName():
    weighted_firstnames = [ ("John", 80), ("Eve", 70), ("Jane", 2), 
                        ("Paul", 8), ("Frank", 20), ("Laura", 6), 
                        ("Robert", 17), ("Zoe", 3), ("Roger", 8), 
                        ("Simone", 9), ("Bernard", 8), ("Sarah", 7),
                        ("Yvonne", 11), ("Bill", 12), ("Bernd", 10)]
    weighted_surnames = [('Singer', 2), ('Miles', 2), ('Moore', 5), 
                        ('Looper', 1), ('Rampman', 1), ('Chopman', 1), 
                        ('Smiley', 1), ('Bychan', 1), ('Smith', 150), 
                        ('Baker', 144), ('Miller', 87), ('Cook', 5),
                        ('Joyce', 1), ('Bush', 5), ('Shorter', 6), 
                        ('Klein', 1)]
    firstnames, weights = zip(*weighted_firstnames)
    wsum = sum(weights)
    weights_firstnames = [ 1.0 * x / wsum for x in weights]
    surnames, weights = zip(*weighted_surnames)
    wsum = sum(weights)
    weights_surnames = [ 1.0 * x / wsum for x in weights]
    weights = (weights_firstnames, weights_surnames)
    #print (firstnames, surnames)
    #print weights

    def synthesizer( data, weights=None, format_func=None, repeats=True):
        """
        "data" is a tuple or list of lists or tuples containing the 
        data.
        
        "weights" is a list or tuple of lists or tuples with the 
        corresponding weights of the data lists or tuples.
        
        "format_func" is a reference to a function which defines
        how a random result of the creator function will be formated. 
        If None,the generator "synthesize" will yield the list "res".
        
        If "repeats" is set to True, the output values yielded by 
        "synthesize" will not be unique.
        """
        
        if not repeats:
            memory = set()
            
        def choice(data, weights):
            if weights:
                return weighted_cartesian_choice(*zip(data, weights))
            else:
                return cartesian_choice(*data)
        def synthesize():
            while True:
                res = choice(data, weights)
                if not repeats:
                    sres = str(res)
                    while sres in memory:
                        res = choice(data, weights)
                        sres = str(res)
                    memory.add(sres)
                if format_func:
                    yield format_func(res)
                else:
                    yield res
        return synthesize
            
    recruit_employee = synthesizer( (firstnames, surnames), 
                                    weights = weights,
                                    format_func=lambda x: " ".join(x),
                                    repeats=False)
    employee = recruit_employee()
    return next(employee)


def getClass():
    return random.choice([1,2,3,4,5])

def getAge():
    return random.randint(22, 45)

def getSex():
    return random.choice(['男', '女'])

def getMath():
    return random.randint(0, 100)

def getComputer():
    return random.randint(0, 100)

def getPhysics():
    return random.randint(0, 100)

if __name__ == '__main__':
    data = dict()
    i = 0
    while i < 100:
        data['Name'] = getName()
        data['Class'] = getClass()
        data['Age'] = getAge()
        data['Sex'] = getSex()
        data['Math'] = getMath()
        data['Computer'] = getComputer()
        data['Physics'] = getPhysics()
        if i % 2 == 0:
            time.sleep(2)
        print data
        replaceStudents(data)
        i += 1
    print i
