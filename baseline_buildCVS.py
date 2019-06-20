"""
baseline code system- build new dataset cvs
PAN-AP'19 training: fiu6Jaershaem3Oh
database:Twitter feed data--label: bot, male, female.
Using the english data
Author: Aaron Lee
Date: 28/04/2019
Student Id: 300422249
"""

import os,csv
from xml.etree import ElementTree as ET

#read all .xml file
def file_name(file_dir):
    filelist=[]
    for root, dirs, files in os.walk(file_dir):
        for i in files:
            if os.path.splitext(i)[1] == '.xml':
                filelist.append(i)
    return filelist

def readTxt(path):
    # read the txt file
    training_set = open(path)
    training_data_line = training_set.readlines()
    truthList=[]
    for line in training_data_line:
            truthList.append(line.split(':::'))
    return truthList


def traversalDir_XMLFile(path,name,truthList):
    # read the xml file
    path= "pan_database/en/"+path
    tree = ET.parse(path)
    root = tree.getroot()
    childs = root.getchildren()

    datalist=[];
    for instance in truthList:
        if (instance[0] == name):
            #Xtype = instance[1]   label bot human
            Ytype = instance[2]  #Ytype label:bot, male, female.

    for child0 in childs:
        for child00 in child0.getchildren():
            datalist.append((child00.text,Ytype))
    return datalist

def data_write_csv(file_name, datas):
    #save the data to csv file
    csvFile = open(file_name, "w+")
    writer = csv.writer(csvFile)
    name_attribute = ['text', 'label']
    writer.writerow(name_attribute)
    for data in datas:
        for node in data:
            writer.writerow(node)
    csvFile.close()

if __name__ == '__main__':
        mydir = "pan_database/en"
        c = file_name(mydir)
        truthFile="pan_database/en/truth.txt"
        truthList=readTxt(truthFile)
        datalist=[]
        for i in c:
            datalist.append((traversalDir_XMLFile(i,i.replace(".xml",""),truthList)))

        download_dir = "PanNewData.csv"
        data_write_csv(download_dir,datalist)