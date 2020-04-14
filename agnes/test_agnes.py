import pandas as pd

from agnes import Agnes

def main():
    agnes = Agnes()

    # df = pd.DataFrame([[0,0,0],[3,4,0], [5,12,0]])
    df = pd.read_csv("csv/iris_no_label.csv")
    # print(df)

    agnes.agnes(df, 'single')



if __name__ == "__main__":
    main()