from typing import Dict, List, Tuple, TYPE_CHECKING

from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix


class COOMatrix(Matrix):

    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape) -> None:
        super().__init__(shape)
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Количество значений, строк и столбцов должно совпадать")
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        m, n = self.shape
        dense = [[0.0] * n for _ in range(m)]
        for value, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = value
        return dense

    def _add_impl(self, other: Matrix) -> Matrix:
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            other = other._to_coo()

        sum_dict: Dict[Tuple[int, int], float] = {}
        
        for v, r, c in zip(self.data, self.row, self.col):
            sum_dict[(r, c)] = sum_dict.get((r, c), 0.0) + v
        
        for v, r, c in zip(other.data, other.row, other.col):
            sum_dict[(r, c)] = sum_dict.get((r, c), 0.0) + v

        
        res_data: COOData = []
        res_row: COORows = []
        res_col: COOCols = []
        for (r, c), v in sum_dict.items():
            if abs(v) > 1e-14:
                res_data.append(v)
                res_row.append(r)
                res_col.append(c)
        return COOMatrix(res_data, res_row, res_col, self.shape)

    def _mul_impl(self, scalar: float) -> Matrix:
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        new_data = [v * scalar for v in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> Matrix:
        """Транспонирование COO матрицы."""
        m, n = self.shape
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), (n, m))

    def _matmul_impl(self, other: Matrix) -> Matrix:
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError()
        m, n = self.shape[0], other.shape[1]
        other_csr = other._to_csr()
        result: Dict[Tuple[int, int], float] = {}
        for val_a, row_a, col_a in zip(self.data, self.row, self.col):
            row_start = other_csr.indptr[col_a]
            row_end = other_csr.indptr[col_a + 1]
            for k in range(row_start, row_end):
                col_b = other_csr.indices[k]
                val_b = other_csr.data[k]
                key = (row_a, col_b)
                result[key] = result.get(key, 0.0) + val_a * val_b
        res_data: COOData = []
        res_row: COORows = []
        res_col: COOCols = []
        for (r, c), v in result.items():
            if abs(v) > 1e-14:
                res_data.append(v)
                res_row.append(r)
                res_col.append(c)
        return COOMatrix(res_data, res_row, res_col, (m, n))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "COOMatrix":
        """Создание COO из плотной матрицы."""
        if not dense_matrix or not dense_matrix[0]:
            return cls([], [], [], (0, 0))
        m = len(dense_matrix)
        n = len(dense_matrix[0])
        data: COOData = []
        rows: COORows = []
        cols: COOCols = []
        for i, row_list in enumerate(dense_matrix):
            for j, value in enumerate(row_list):
                if value != 0:
                    data.append(value)
                    rows.append(i)
                    cols.append(j)
        return cls(data, rows, cols, (m, n))

    def _to_csc(self) -> "CSCMatrix":
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        m, n = self.shape
        
        triples: List[Tuple[int, int, float]] = list(zip(self.col, self.row, self.data))
        triples.sort()
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0] * (n + 1)
        for c, r, v in triples:
            data.append(v)
            indices.append(r)
            indptr[c + 1] += 1
        
        for j in range(n):
            indptr[j + 1] += indptr[j]
        return CSCMatrix(data, indices, indptr, (m, n))

    def _to_csr(self) -> "CSRMatrix":
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        m, n = self.shape
        
        triples: List[Tuple[int, int, float]] = list(zip(self.row, self.col, self.data))
        triples.sort()
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0] * (m + 1)
        for r, c, v in triples:
            data.append(v)
            indices.append(c)
            indptr[r + 1] += 1
        for i in range(m):
            indptr[i + 1] += indptr[i]
        return CSRMatrix(data, indices, indptr, (m, n))