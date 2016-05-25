# Script used to generate CSV files
import sys
import os.path
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print "usage: create_csv <base_path>"
        sys.exit(1)
    
    ROOT_DIR=sys.argv[1]
    NEW_LINE=";"

    label = 0
    for dirname, dirnames, filenames in os.walk(ROOT_DIR):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print "%s%s%d" % (abs_path, NEW_LINE, label)
            label = label + 1
