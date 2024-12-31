# Współbieżna eliminacja Gaussa

## Wstęp teoretyczny

Eliminacja Gaussa służy do obliczania układów równań liniowych postaci $M \times x = y$.
Układ równań można przedstawić jako:

![](./img/system1.png)

Dla wygody obliczeń, algorytm może potraktować macierz M i wektor y jako całość:

![](./img/system2.png)

Algorytm korzysta z 3 elementarnych, niepodzielnych operacji:
```py
def A(i, k):
    return M[k][i] / M[i][i]

m[k][i] = A(i, k)  # pierwsza operacja

def B(i, j, k):
    return M[i][j] * m[k][i]

n[i][j][k] = B(i, j, k)  # druga operacja

def C(i, j, k):
    return M[k][j] - n[i][j][k]

M[k][j] = C(i, j, k)  # trzecia operacja
```

Sam algorytm ma postać:

```py
for i in range(N-1):
    for k in range(i+1, N):
        A(i, k)
        for j in range(i, N+1):
            B(i, j, k)
            C(i, j, k)
```

