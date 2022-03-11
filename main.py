import pandas as pd


class Wrangler:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.dfs = []

    def test(self):
        print(directory_path)


if __name__ == '__main__':
    directory_path = 'BigSupplyCo_Data_CSVs/'

    test_class = Wrangler(directory_path)
    test_class.test()

    ## forktree 2022-03-11 11:19
    print("Test Github fork process")
