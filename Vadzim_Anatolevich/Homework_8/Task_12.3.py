# -*- coding: utf-8 -*-

import os;
import ffmpeg;

start_dir = os.getcwd();

def convert_to_mp4(avi_file):
    name, ext = os.path.splitext(avi_file);
    result_name = name + ".mp4";
    ffmpeg.input(avi_file).output(result_name).run();
    print("Конвертация завершена: {}".format(avi_file));
    
for path, folder, files in os.walk(start_dir):
    for file in files:
        if file.endswith('(reversed).avi'):
            print("Найден файл: %s" % file);
            convert_to_mp4(os.path.join(start_dir, file));
        else:
            pass;