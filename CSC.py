from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        rows, cols = shape  # Извлекаю количество строк и столбцов из shape

        if len(indptr) != cols + 1:
            raise ValueError("Некорректная длина indptr")
        if indptr[0] != 0:
            raise ValueError("Первый элемент indptr должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError("Последний элемент indptr должен быть равен длине data")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")
        
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        # Создаю транспонированную матрицу для удобства
        dense = [[0.0] * rows for _ in range(cols)]
        
        # Прохожу по всем мтолбцам в поисках ненулевых элементов
        for col in range(cols):
            for idx in range(self.indptr[col], self.indptr[col + 1]):
                row = self.indices[idx]
                dense[col][row] = self.data[idx]

        # Транспонирую обратно
        dense_matrix = list(zip(*dense))
        return [list(row) for row in dense_matrix]

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc() # Привожу вторую матрицу к CSC
        
        rows, cols = self.shape
        result_data: CSCData = []
        result_indices: CSCIndices = []
        result_indptr: CSCIndptr = [0] * (cols + 1)

        for j in range(cols):
            # Границы ненулевых элементов в столбце j для обеих матриц
            a_start = self.indptr[j]
            a_end = self.indptr[j + 1]
            b_start = other.indptr[j]
            b_end = other.indptr[j + 1]

            pa = a_start # Указатель по первой матрице
            pb = b_start # Указатель по второй матрице

            # Слияние двух отсортированных списков индексов строк
            while pa < a_end and pb < b_end:
                row_a = self.indices[pa]
                row_b = other.indices[pb]
                # Складываю значения совпадающих строк
                if row_a == row_b:
                    val = self.data[pa] + other.data[pb]
                    # Игнорируем нули
                    if abs(val) > 1e-14:
                        result_indices.append(row_a)
                        result_data.append(val)
                    pa += 1
                    pb += 1
                
                # Для первой матрицы
                elif row_a < row_b:
                    val = self.data[pa]
                    if abs(val) > 1e-14:
                        result_indices.append(row_a)
                        result_data.append(val)
                    pa += 1
                # Для второй матрицы
                else:
                    val = other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(row_b)
                        result_data.append(val)
                    pb += 1
            
            # Дописываем остатки из первой матрицы
            while pa < a_end:
                row_a = self.indices[pa]
                val = self.data[pa]
                if abs(val) > 1e-14:
                    result_indices.append(row_a)
                    result_data.append(val)
                pa += 1
            
            # Дописываю остатки из второй матрицы
            while pb < b_end:
                row_b = other.indices[pb]
                val = other.data[pb]
                if abs(val) > 1e-14:
                    result_indices.append(row_b)
                    result_data.append(val)
                pb += 1
            
            result_indptr[j + 1] = len(result_data)
        
        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        rows, cols = self.shape
        new_rows, new_cols = cols, rows

        # Считаю количество ненулевых элементов в каждой новой строке
        row_counts: list[int] = [0] * new_rows
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            row_counts[j] = end - start # Кол-во ненулевых в столбце j
        
        # Строю новый indptr для CSR формата
        new_indptr: CSCIndptr = [0] * (new_rows + 1)
        for i in range(new_rows):
            new_indptr[i + 1] = new_indptr[i] + row_counts[i]
        
        # Создаю новые массивы и заполняю их
        new_data: CSCData = [0.0] * len(self.data)
        new_indices: CSCIndices = [0] * len(self.indices)
        row_positions = new_indptr.copy() # Текущие позиции для каждой строки

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
        

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()

        if self.shape[1] != other.shape[0]:
            raise ValueError("Размеры матриц не совпадают для умножения")

        # Преобразуем в CSR для удобства умножения
        from CSR import CSRMatrix
        self_csr = self._to_csr()
        other_csr = other._to_csr()

        # Умножение CSR матриц
        result_csr = self_csr._matmul_impl(other_csr)

        # Преобразуем результат обратно в CSC
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix or not dense_matrix[0]:
            return cls([], [], [0], (0, 0))
    
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])

        data = []
        indices = []
        indptr = [0]

        # Прохожу по столбцам
        for j in range(cols):
            col_count = 0
            for i in range(rows):
                if abs(dense_matrix[i][j]) > 1e-14:  # Проверка на ненулевые элементы
                    data.append(dense_matrix[i][j])
                    indices.append(i)
                    col_count += 1
            indptr.append(indptr[-1] + col_count)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        m, n = self.shape # m=строки, n=столбцы

        # Считаю количество ненулевых в каждой строке
        row_counts = [0] * m
        for row_idx in self.indices:
            row_counts[row_idx] += 1
        
        # Строим indptr для строк
        indptr: CSCIndptr = [0] * (m + 1)
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        # Заполняю данные (нужно отсортировать по строкам)
        data: CSCData = [0.0] * len(self.data)
        indices: CSCIndices = [0] * len(self.indices)
        current_pos = indptr.copy()

        for j in range(n):
            col_start = self.indptr[j] # По столбцам
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                i = self.indices[k] # Строка
                val = self.data[k]
                pos = current_pos[i] # Позиция в строке i
                data[pos] = val
                indices[pos] = j # В CSR индекс колонки
                current_pos[i] += 1
        
        return CSRMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        rows, cols = self.shape

        data_list: list[float] = []
        row_indices: list[int] = []
        col_indices: list[int] = []

        # Прохожу по всем колонкам и собираю координаты
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                data_list.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(j)
        
        return COOMatrix(data_list, row_indices, col_indices, self.shape)
