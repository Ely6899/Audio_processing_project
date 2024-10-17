from GeneralUtils import DirPaths, Exceptions, FileUtils
from GeneralUtils.Exceptions import FileNotSupportedException
from Preprocessing import AudioPreprocess

if __name__ == '__main__':
    try:
        file_data, sr = AudioPreprocess.read_audio_file_as_waveform("hello.flac")
    except FileNotSupportedException as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
