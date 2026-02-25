from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSC import CSCMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        rows, _ = shape # Количество строк

        # Проверки корректности CSR формата
        if len(indptr) != rows + 1:
            raise ValueError()
        if indptr[0] != 0:
            raise ValueError()
        if indptr[-1] != len(data):
            raise ValueError()
        if len(data) != len(indices):
            raise ValueError()
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        m, n = self.shape  # m=строки, n=столбцы
        dense = [[0.0] * n for _ in range(m)] # Плотная матрица

        # Прохожу построчно
        for i in range(m):
            # Беру все ненулевые элементы этой строки
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for p in range(start, end):
                j = self.indices[p] # Индекс столбца
                dense[i][j] = self.data[p]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()
        
        m, _ = self.shape # Количество строк
        result_data: CSRData = []
        result_indices: CSRIndices = []
        result_indptr: CSRIndptr = [0] * (m + 1)
        
        for i in range(m): # По строкам
            # Границы для строки i в обеих матрицах
            a_start = self.indptr[i]
            a_end = self.indptr[i + 1]
            b_start = other.indptr[i]
            b_end = other.indptr[i + 1]
            pa = a_start
            pb = b_start
            
            # Слияние двух отсортированных списков индексов столбцов
            while pa < a_end and pb < b_end:
                col_a = self.indices[pa] # Индекс столбца в первой матрице
                col_b = other.indices[pb] # Индекс столбца во второй
                if col_a == col_b:
                    val = self.data[pa] + other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(col_a)
                        result_data.append(val)
                    pa += 1
                    pb += 1
                elif col_a < col_b:
                    # Только в первой
                    val = self.data[pa]
                    if abs(val) > 1e-14:
                        result_indices.append(col_a)
                        result_data.append(val)
                    pa += 1
                else:
                    # Только во второй
                    val = other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(col_b)
                        result_data.append(val)
                    pb += 1
            
            # Остатки из первой строки
            while pa < a_end:
                col_a = self.indices[pa]
                val = self.data[pa]
                if abs(val) > 1e-14:
                    result_indices.append(col_a)
                    result_data.append(val)
                pa += 1
            
            # Остатки из второй строки
            while pb < b_end:
                col_b = other.indices[pb]
                val = other.data[pb]
                if abs(val) > 1e-14:
                    result_indices.append(col_b)
                    result_data.append(val)
                pb += 1
            result_indptr[i + 1] = len(result_data)
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix

        m, n = self.shape
        total_nnz = len(self.data)

        transposed_data = [0.0] * total_nnz
        transposed_indices = [0] * total_nnz
        transposed_indptr = [0] * (n + 1)

        # Подсчет ненулевых элементов в каждом столбце
        col_counts = [0] * n
        for i in range(m):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[p]
                col_counts[j] += 1

        # Заполнение transposed_indptr
        for j in range(n):
            transposed_indptr[j + 1] = transposed_indptr[j] + col_counts[j]

        # Заполнение временных массивов
        next_pos = transposed_indptr.copy()
        for i in range(m):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[p]
                pos = next_pos[j]
                transposed_data[pos] = self.data[p]
                transposed_indices[pos] = i
                next_pos[j] += 1

        return CSCMatrix(transposed_data, transposed_indices, transposed_indptr, (n, m))


    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()

        if self.shape[1] != other.shape[0]:
            raise ValueError("Размеры матриц не совпадают для умножения")

        m, n = self.shape # A: m×n
        _, p = other.shape # B: n×p

        result_data = []
        result_indices = []
        result_indptr = [0]

        # По строкам результирующей матрицы
        for i in range(m):
            row_data = {}
            # Беру все ненулевые элементы строки i матрицы A
            for k1 in range(self.indptr[i], self.indptr[i + 1]):
                k = self.indices[k1]
                val1 = self.data[k1]
                # Умножаю на все ненулевые элементы столбца k матрицы B
                for k2 in range(other.indptr[k], other.indptr[k + 1]):
                    j = other.indices[k2] # Индекс столбца j
                    val2 = other.data[k2]
                    # Накапливаю C[i,j] += A[i,k] * B[k,j]
                    if j in row_data:
                        row_data[j] += val1 * val2
                    else:
                        row_data[j] = val1 * val2

            # Сортирую по столбцам и добавляем ненулевые
            sorted_keys = sorted(row_data.keys())
            for j in sorted_keys:
                if abs(row_data[j]) > 1e-14: # Пропускаю малые значения
                    result_data.append(row_data[j])
                    result_indices.append(j)

            result_indptr.append(len(result_data))

        return CSRMatrix(result_data, result_indices, result_indptr, (m, p))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix or not dense_matrix[0]:
            return cls([], [], [0], (0, 0))

        m = len(dense_matrix)
        n = len(dense_matrix[0])

        data = []
        indices = []
        indptr = [0]

        # Прохоу по трокам
        for i in range(m):
            row_nnz = 0
            for j in range(n):
                if abs(dense_matrix[i][j]) > 1e-14:
                    data.append(dense_matrix[i][j])
                    indices.append(j)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)

        return cls(data, indices, indptr, (m, n))
    
    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        data = []
        rows = []
        cols = []

        # Прохожу по всем строкам и собираю координаты
        for i in range(len(self.indptr) - 1):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[p])
                rows.append(i) # Индекс строки
                cols.append(self.indices[p]) # Индекс столбца

        return COOMatrix(data, rows, cols, self.shape)