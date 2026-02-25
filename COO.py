from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        # Проверяю, что все списки одинаковой длины
        if not (len(data) == len(row) == len(col)):
            raise ValueError("количество значений в строках и столбцах должно совпадать")
        
        self.data = data[:]  # Копируем списки
        self.row = row[:]
        self.col = col[:]

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        # Прохожу по всем ненулевым элементам и ставлю их на место
        for val, i, j in zip(self.data, self.row, self.col):
            dense[i][j] = val # Перезапись при совпадении

        return dense
        

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""

        if not isinstance(other, COOMatrix):
           other = other._to_coo()
        
        # Создаём словарь для суммирования значений с одинаковыми координатами
        sum_dict = {}

        # Обрабатываем текущую матрицу
        for val, i, j in zip(self.data, self.row, self.col):
            key = (i, j)
            if key in sum_dict:
                sum_dict[key] += val
            else:
                sum_dict[key] = val
        
        # Обрабатываем другую матрицу
        for val, i, j in zip(other.data, other.row, other.col):
            key = (i, j)
            if key in sum_dict:
                sum_dict[key] += val
            else:
                sum_dict[key] = val
        
        sum_dict = {k: v for k, v in sum_dict.items() if abs(v) > 1e-14}
        
        # Извлекаем данные из словаря
        new_data = []
        new_rows = []
        new_cols = []

        # Сортируем ключи для сохранения порядка (по строкам и столбцам)
        for (i, j) in sorted(sum_dict.keys()):
            new_data.append(sum_dict[(i, j)])
            new_rows.append(i)
            new_cols.append(j)

         # Возвращаем новую матрицу
        return COOMatrix(new_data, new_rows, new_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        new_data = [v * scalar for v in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        a,b = self.shape
        # Меняю shape и подменяем списки координат
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), (a,b))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Размеры двух матриц не совпадают")
        
        if not isinstance(other, COOMatrix):
            other = other._to_coo()
        
        # Словарь для хранения результатов умножения
        result_dict = {}

        # Умножаем каждую ненулевую тройку из первой матрицы на каждую ненулевую тройку из второй
        for val1, i, k in zip(self.data, self.row, self.col):
            for val2, k2, j in zip(other.data, other.row, other.col):
                if k == k2:
                    key = (i, j)
                    if key in result_dict:
                        result_dict[key] += val1 * val2
                    else:
                        result_dict[key] = val1 * val2
        
        # Извлекаем данные из словаря
        new_data = []
        new_rows = []
        new_cols = []

        for (i, j) in sorted(result_dict.keys()):
            new_data.append(result_dict[(i, j)])
            new_rows.append(i)
            new_cols.append(j)

        return COOMatrix(new_data, new_rows, new_cols, (self.shape[0], other.shape[1]))
    
    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
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

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        rows, cols = self.shape
    
        # Группируем элементы ПО СТОЛБЦАМ
        col_elements = [[] for _ in range(cols)]  # [(row, val), ...] для каждого столбца
        
        for val, i, j in zip(self.data, self.row, self.col):
            if abs(val) > 1e-14:
                col_elements[j].append((i, val))
        
        # Строим data, indices, indptr
        data = []
        indices = []
        indptr = [0]
        pos = 0
        
        for j in range(cols):
            # Сортируем элементы столбца j ПО СТРОКАМ (обязательно!)
            col_elements[j].sort(key=lambda x: x[0])
            
            for row_idx, val in col_elements[j]:
                indices.append(row_idx)
                data.append(val)
                pos += 1
            
            indptr.append(pos)

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        m, n = self.shape

         # Сортирую тройки: сначала по строкам, затем столбцам
        triples: List[Tuple[int, int, float]] = list(zip(self.row, self.col, self.data))
        triples.sort()

        data: List[float] = [v for _, _, v in triples]
        indices: List[int] = [c for _, c, _ in triples]  # В CSR indices = столбцы в строке

        # Строю indptr по количеству элементов в каждой строке
        indptr: List[int] = [0] * (m + 1)
        for i in range(1, m + 1):
            # Считаю элементы в строке i-1
            indptr[i] = indptr[i - 1] + sum(1 for r, _, _ in triples if r == i - 1)
        
        return CSRMatrix(data, indices, indptr, (m, n))
