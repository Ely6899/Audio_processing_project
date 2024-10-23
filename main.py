from GeneralUtils import DirPaths, Exceptions, FileUtils
from GeneralUtils.Exceptions import FileNotSupportedException
from Preprocessing import AudioPreprocess
from GeneralUtils.ObjectUtils import LibriDataObject

if __name__ == '__main__':
    obj = LibriDataObject.construct_object_from_json("LibriSpeechObjects/19-198-0000.json")
    print(obj)
