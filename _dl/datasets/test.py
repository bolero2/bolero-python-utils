if __name__ == "__main__":

    a = [1, 2, 3, 4]
    a = '1234'

    try:
        print("try")
        aa = sum(a)

    except:
        print("except")
        aa = a.split()

    print('result :', aa)
            
