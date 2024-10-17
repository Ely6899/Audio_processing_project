from GeneralUtils import DirPaths, Exceptions, FileUtils
from GeneralUtils.Exceptions import FileNotSupportedException
from Preprocessing import AudioPreprocess
from GeneralUtils.ObjectUtils import LibriDataObject, construct_json_from_audio_file

if __name__ == '__main__':
    print(construct_json_from_audio_file(LibriDataObject("19-198-0000.flac")))
