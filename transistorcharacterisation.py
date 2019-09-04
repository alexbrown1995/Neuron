def readin(name):
    with open(name, 'r') as csvfile:
        voltages = csv.reader(csvfile, delimiter=',')
        print(voltages)
        x = []
        y = []

        linecount = 0
        for row in voltages:
            if linecount == 0:
                linecount += 1
            else:  # if (float(row[0]) >= 0):

                x.append(float(row[0]))
                y.append(float(row[1]))

    return x,y

vbe,I=readin('trans_2n2222data.csv')