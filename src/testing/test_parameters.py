import sys

sys.path.append('.')
import config


"""Must be run from `src` directory: python3 testing/test_parameters.py"""
if __name__ == '__main__':
  args = config.build_config()

  args.write_to_json_file('commandline_args.txt')
