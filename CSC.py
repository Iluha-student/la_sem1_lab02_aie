from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        rows, cols = shape
        
        if len(indptr) != cols + 1:
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
        """Преобразует CSC в плотную матрицу."""
        m, n = self.shape
        dense = [[0.0] * n for _ in range(m)]
        for j in range(n):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]
        return dense

    def _add_impl(self, other: "Matrix") -> "Matrix":
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()
        rows, cols = self.shape
        result_data: CSCData = []
        result_indices: CSCIndices = []
        result_indptr: CSCIndptr = [0] * (cols + 1)
        for j in range(cols):
            a_start = self.indptr[j]
            a_end = self.indptr[j + 1]
            b_start = other.indptr[j]
            b_end = other.indptr[j + 1]
            pa = a_start
            pb = b_start
            while pa < a_end and pb < b_end:
                row_a = self.indices[pa]
                row_b = other.indices[pb]
                if row_a == row_b:
                    val = self.data[pa] + other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(row_a)
                        result_data.append(val)
                    pa += 1
                    pb += 1
                elif row_a < row_b:
                    val = self.data[pa]
                    if abs(val) > 1e-14:
                        result_indices.append(row_a)
                        result_data.append(val)
                    pa += 1
                else:
                    val = other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(row_b)
                        result_data.append(val)
                    pb += 1
            while pa < a_end:
                row_a = self.indices[pa]
                val = self.data[pa]
                if abs(val) > 1e-14:
                    result_indices.append(row_a)
                    result_data.append(val)
                pa += 1
            while pb < b_end:
                row_b = other.indices[pb]
                val = other.data[pb]
                if abs(val) > 1e-14:
                    result_indices.append(row_b)
                    result_data.append(val)
                pb += 1
            result_indptr[j + 1] = len(result_data)
        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> "Matrix":
        """Умножение CSC на скаляр."""
        rows, cols = self.shape
        if scalar == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)
        new_data = [v * scalar for v in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> "Matrix":
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        rows, cols = self.shape
        new_rows, new_cols = cols, rows
        row_counts: list[int] = [0] * new_rows
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            row_counts[j] = end - start
        new_indptr: CSCIndptr = [0] * (new_rows + 1)
        for i in range(new_rows):
            new_indptr[i + 1] = new_indptr[i] + row_counts[i]
        new_data: CSCData = [0.0] * len(self.data)
        new_indices: CSCIndices = [0] * len(self.indices)
        row_positions = new_indptr.copy()
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = row_positions[j]
                new_data[pos] = self.data[idx]
                new_indices[pos] = i
                row_positions[j] += 1
        return CSRMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: "Matrix") -> "Matrix":
        """Умножение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        result_data: CSCData = []
        result_indices: CSCIndices = []
        result_indptr: CSCIndptr = [0] * (cols_B + 1)
        
        row_entries_B: list[list[tuple[int, float]]] = [[] for _ in range(rows_B)]
        for col in range(cols_B):
            start = other.indptr[col]
            end = other.indptr[col + 1]
            for idx in range(start, end):
                row = other.indices[idx]
                val = other.data[idx]
                row_entries_B[row].append((col, val))
        temp_row: list[float] = [0.0] * rows_A
        for j in range(cols_B):
            for i in range(rows_A):
                temp_row[i] = 0.0
            for i in range(rows_B):
                for col_b, val_b in row_entries_B[i]:
                    if col_b == j:
                        col_start = self.indptr[i]
                        col_end = self.indptr[i + 1]
                        for a_idx in range(col_start, col_end):
                            row_a = self.indices[a_idx]
                            val_a = self.data[a_idx]
                            temp_row[row_a] += val_a * val_b
            for i in range(rows_A):
                if abs(temp_row[i]) > 1e-14:
                    result_data.append(temp_row[i])
                    result_indices.append(i)
            result_indptr[j + 1] = len(result_data)
        return CSCMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSCMatrix":
        """Создание CSC из плотной матрицы."""
        if not dense_matrix or not dense_matrix[0]:
            return cls([], [], [0, 0], (0, 0))
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        data: CSCData = []
        indices: CSCIndices = []
        col_counts: list[int] = [0] * cols
        for j in range(cols):
            for i in range(rows):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    indices.append(i)
                    col_counts[j] += 1
        indptr: CSCIndptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> "CSRMatrix":
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        m, n = self.shape
        row_counts = [0] * m
        for row_idx in self.indices:
            row_counts[row_idx] += 1
        indptr: CSCIndptr = [0] * (m + 1)
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]
        data: CSCData = [0.0] * len(self.data)
        indices: CSCIndices = [0] * len(self.indices)
        current_pos = indptr.copy()
        for j in range(n):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                i = self.indices[k]
                val = self.data[k]
                pos = current_pos[i]
                data[pos] = val
                indices[pos] = j
                current_pos[i] += 1
        return CSRMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> "COOMatrix":
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows, cols = self.shape
        data_list: list[float] = []
        row_indices: list[int] = []
        col_indices: list[int] = []
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                data_list.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(j)
        return COOMatrix(data_list, row_indices, col_indices, self.shape)