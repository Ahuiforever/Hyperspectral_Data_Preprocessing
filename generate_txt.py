'''
To generate the corresponding text file according to the specified folder, which is the manually divided training or testing dataset.
'''
import glob
import os


def main(root_path, xml_path):
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')
    train_txt = os.path.join(root_path, 'train.txt')
    test_txt = os.path.join(root_path, 'test.txt')
    with open(train_txt, 'w') as f:
        for filename in glob.glob(os.path.join(train_path, '*.jpg')):
            file = filename.split(os.sep)[-1].replace('.jpg', '')
            if os.path.isfile(os.path.join(xml_path, file + '.xml')):
                f.write(file)
                f.write('\n')
    with open(test_txt, 'w') as f:
        for filename in glob.glob(os.path.join(test_path, '*.jpg')):
            file = filename.split(os.sep)[-1].replace('.jpg', '')
            if os.path.isfile(os.path.join(xml_path, file + '.xml')):
                f.write(file)
                f.write('\n')


if __name__ == '__main__':
    root_path = r'F:\works\recurrent--base_line_model\data\manually partition'
    xml_path = r'F:\works\recurrent--base_line_model\mmdetection-master\data\VOCdevkit\VOC2007\Annotations'
    main(root_path, xml_path)
