def read_head(head):
    lb = []
    # file = open(head,"r")
    lines = head[15]
    tmp = (lines[5:-1])
    if "," in tmp:
        tmp2 = tmp.split(",")
        for i in range(0,len(tmp2)):
            if (tmp2[i] == "AF"):
                lb.append((1))
            elif (tmp2[i] == "I-AVB"):
                lb.append(2)
            elif (tmp2[i] == "LBBB"):
                lb.append(3)
            elif (tmp2[i] == "Normal"):
                lb.append(4)
            elif (tmp2[i] == "PAC"):
                lb.append(5)
            elif (tmp2[i] == "PVC"):
                lb.append(6)
            elif (tmp2[i] == "RBBB"):
                lb.append(7)
            elif (tmp2[i] == "STD"):
                lb.append(8)
            elif (tmp2[i] == "STE"):
                lb.append(9)
                    
    else:
        if (tmp == "AF"):
            lb.append((1))
        elif (tmp == "I-AVB"):
            lb.append(2)
        elif (tmp == "LBBB"):
            lb.append(3)
        elif (tmp == "Normal"):
            lb.append(4)
        elif (tmp == "PAC"):
            lb.append(5)
        elif (tmp == "PVC"):
            lb.append(6)
        elif (tmp == "RBBB"):
            lb.append(7)
        elif (tmp == "STD"):
            lb.append(8)
        elif (tmp == "STE"):
            lb.append(9)
    return lb