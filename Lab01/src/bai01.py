"""Bài 01: Kiểm tra môi trường Python."""

import sys
import gymnasium as gym

def main():
    print(f"Python Version: {sys.version.split(' ')[0]}")
    print(f"Gymnasium Version: {gym.__version__}")

if __name__ == "__main__":
    main()
