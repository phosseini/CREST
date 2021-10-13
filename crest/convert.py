import sys
from crest import Converter


def main(argv):
    usage_msg = 'Usage: python convert.py [DATASET_ID_0] [DATASET_ID_1] ... [DATASET_ID_n] [OUTPUT_FILE_NAME]'
    if argv[1] == '-i':
        print('\n{}'.format(usage_msg))
        print('--------------------')
        print('Current dataset ids:')
        for dataset_name, dataset_id in Converter().namexid.items():
            print('--> {}: {}'.format(dataset_name, dataset_id))
        print('--------------------')
        exit(1)
    elif len(argv) >= 3:
        dataset_ids = []
        for dataset_id in argv[1:-1]:
            if dataset_id.isdigit():
                dataset_ids.append(int(dataset_id))
            else:
                print('error: dataset id should be an integer')
                exit(1)

        if argv[-1].endswith('.xlsx'):
            file_path = argv[-1]
        else:
            print('error: OUTPUT_FILE_NAME should be *.xlsx')
            exit(1)

        print("Dataset ids: {}".format(dataset_ids))
        print("Output file: {}".format(file_path))

        converter = Converter()
        df, _ = converter.convert2crest(dataset_ids=dataset_ids)
        df.to_excel(file_path)
    elif len(argv) < 3:
        print(usage_msg)
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
