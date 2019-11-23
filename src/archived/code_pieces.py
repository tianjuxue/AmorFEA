def sparse_representation(A):
    # seems stupid... first compute the size
    counter = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j]**2 > 1e-10:
                counter += 1
    index = torch.zeros((2, counter), dtype=torch.long)
    value = torch.zeros(counter, dtype=torch.float)

    counter = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j]**2 > 1e-10:
                index[0, counter] = i
                index[1, counter] = j
                value[counter] = A[i][j]
                counter += 1

    A_sp = torch.sparse.FloatTensor(index, value, torch.Size([len(A), len(A)]))
    print("Create sparse representation of A")
    return A_sp