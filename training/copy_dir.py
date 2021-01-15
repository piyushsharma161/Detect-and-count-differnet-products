import os, shutil
import argparse
'''
Creating the output directory if not already exists
Doing the copy directory by recursively calling my own method.
When we come to actually copying the file I check if the file is modified then only we should copy.

'''

parser = argparse.ArgumentParser()


def copy_dir(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_dir(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def main():
    parser.add_argument('src', help='description for source file')
    parser.add_argument('dst', help='description for destination file')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        os.makedirs(args.src)
    shutil.rmtree(args.dst)
    copy_dir(args.src, args.dst)
    shutil.rmtree(args.src)
    os.mkdir(args.src)

if __name__ == '__main__':
    main()