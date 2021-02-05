import Exercise_4.PytorchChallengeTests as tester

def main():

    var1 = tester.TestDataset()
    var1.setUp()
    var1.test_shape()
    var1.test_normalization()



if __name__ == '__main__':
    main()