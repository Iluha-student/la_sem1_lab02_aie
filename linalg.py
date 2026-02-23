from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    # Преобразуем CSC-матрицу в плотный формат
    dense_A = A.to_dense()
    n = len(dense_A)

    # Инициализация матриц L и U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Вычисление U
        for k in range(i, n):
            sum_u = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = dense_A[i][k] - sum_u

        # Вычисление L
        for k in range(i, n):
            # Проверка на вырожденную матрицу
            if U[i][i] == 0:
                return None
            
            sum_l = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (dense_A[k][i] - sum_l) / U[i][i]

    # Устанавливаем диагональные элементы L в 1
    for i in range(n):
        L[i][i] = 1.0

    # Преобразуем L и U обратно в CSC формат
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)

    return (L_csc, U_csc)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    # Получаем LU-разложение матрицы A
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result
    n = len(b)

    # Прямая подстановка (Ly = b)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L.to_dense()[i][j] * y[j] for j in range(i))
        if L.to_dense()[i][i] == 0:
            return None

    # Обратная подстановка (Ux = y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U.to_dense()[i][j] * x[j] for j in range(i + 1, n))) / U.to_dense()[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    # Получаю LU-разложение матрицы A
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result
    dense_U = U.to_dense()

    # Определитель L всегда равен 1, так как на диагонали единицы
    det_U = 1.0
    n = len(dense_U)
    for i in range(n):
        det_U *= dense_U[i][i]

    return det_U

