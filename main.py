# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import keras
import tensorflow as tf


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(torch.__version__)

    print(torch.cuda.is_available())
    print(keras.__version__)
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
