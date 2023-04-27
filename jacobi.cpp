#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <locale.h>
#include "mpi.h"

using namespace std;

// Параметр уравнения
const double a = 10.0e5;

// Порог сходимости
const double eps = 1.0e-8;

// Начальное приближение
const double phi = 0.0;

// Границы области моделирования
const double X1 = -1.0;
const double X2 = 1.0;
const double Y1 = -1.0;
const double Y2 = 1.0;
const double Z1 = -1.0;
const double Z2 = 1.0;

// Размеры области
const double Dx = X2 - X1;
const double Dy = Y2 - Y1;
const double Dz = Z2 - Z1;

// Количество узлов сетки
const int Nx = 10;
const int Ny = 10;
const int Nz = 15;

// Размеры шага (расстояния между соседними узлами) на сетке
const double hx = Dx / (Nx - 1);
const double hy = Dy / (Ny - 1);
const double hz = Dz / (Nz - 1);

// Теги сообщений
const int msg_tag1 = 100;
const int msg_tag2 = 101;
const int msg_tag3 = 102;
const int msg_tag4 = 103;
const int msg_tag5 = 104;
const int msg_tag6 = 105;

// Функция фи
double phi_func(double x, double y, double z) {
	return pow(x, 2) + pow(y, 2) + pow(z, 2);
}

// Начальные значения
void setGrid(double***& grid, int size_x, double start_x, int num_rank, int proc_count)
{
	grid = new double** [size_x];

	// Начальное приближение для некраевых узлов 
	for (int i = 0; i < size_x; i++) {
		grid[i] = new double* [Ny];

		for (int j = 0; j < Ny; j++) {
			grid[i][j] = new double[Nz];

			for (int k = 1; k < Nz-1; k++) {
				grid[i][j][k] = phi;
			}
		}
	}

	// Краевые значения для плоскостей XY и XZ

	for (int i = 0; i < size_x; i++) {
		double current_x = start_x + hx * i;

		// обрабатываем верхнюю и нижнию плоскости XY подобласти
		for (int k = 0; k < Nz; k++) {
			// При Z = 0
			grid[i][0][k] = phi_func(current_x, Y1, Z1 + hz * k);
			// При Z = Ny - 1
			grid[i][Ny - 1][k] = phi_func(current_x, Y2, Z1 + hz * k);
		}

		// обрабатываем заднюю и переденюю плоскости XZ подобласти
		for (int j = 1; j < Ny - 1; j++) {

			double current_y = Y1 + hy * j;
			// При Z = 0
			grid[i][j][0] = phi_func(current_x, current_y, Z1);
			// При Z = Nz - 1
			grid[i][j][Nz - 1] = phi_func(current_x, current_y, Z2);
		}
	}

	// Часть процессов не содержит краевых плоскостей ZY
	// Обработаем краевые плоскости ZY, которые содержит первый и последний процесс

	// Первый процесс
	if (num_rank == 0) {
		for (int j = 1; j < Ny - 1; j++) {
			for (int k = 1; k < Nz - 1; k++) {
				grid[0][j][k] = phi_func(X1, Y1 + hy * j, Z1 + hz * k);
			}
		}
	}

	// Последний процесс
	if (num_rank == proc_count - 1) {
		for (int j = 1; j < Ny - 1; j++) {
			for (int k = 1; k < Nz - 1; k++) {
				grid[size_x - 1][j][k] = phi_func(X2, Y1 + hy * j, Z1 + hz * k);
			}
		}
	}

}

// Очищаем память
void deleteGrid(double*** grid, int size_x) {
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < Ny; j++) {
			delete[] grid[i][j];
		}
		delete[] grid[i];
	}
	delete[] grid;
}

// Нахождение максимального расхождения полученного приближенного решения и точного решения
double Get_Accuracy(double*** grid, int size_x, double start_x, int rootProcRank)
{
	
	double err;					// Значение ошибки на узле
	double max_err = 0.0;			// Максимальное значение ошибки в текущем процессе

	for (int i = 1; i < size_x - 1; i++) {
		for (int j = 1; j < Ny - 1; j++) {
			for (int k = 1; k < Nz - 1; k++) {

				// Модуль разности текущего значения и точного решения
				err = abs(grid[i][j][k] - phi_func(start_x + hx * i, Y1 + hy * j, Z1 + hz * k));

				if (err > max_err) max_err = err;
			}
		}
	}

	// Получаем маскимальное значение ошибки по всем потокам и отправляем его в выходной буфер главного процесса
	double Max = -1;
	MPI_Reduce((void*)&max_err, (void*)&Max, 1, MPI_DOUBLE, MPI_MAX, rootProcRank, MPI_COMM_WORLD);

	return Max;
}

// Метод Якоби
void Jacobi(double***& grid_1, int size_x, int start_x, int proc_lower, int proc_upper)
{

	int rank;				// номер потока
	int size;				// количество потоков
	double conv;			// значение сходимости на узле
	double max_conv;		// маскимальное значение сходимости в процессе на итерации
	char end_flag;				// вспомогательный флаг

	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	double*** grid_2;					// вспомогательный массив (содержит значения предыдущих итераций)
	grid_2 = new double** [size_x];
	for (int i = 0; i < size_x; i++) {
		grid_2[i] = new double* [Ny];

		for (int j = 0; j < Ny; j++) {
			grid_2[i][j] = new double[Nz];

			for (int k = 0; k < Nz; k++) {
				grid_2[i][j][k] = grid_1[i][j][k];
			}
		}
	}

	double*** curr_node = grid_1;	// указатель на текущие значения массива
	double*** past_node = grid_2;		// указатель на значения массива на предыдущей итерации
	double*** temp_node;				// указатель для перемены указателей на массивы

	int message_len = (Ny - 2) * (Nz - 2);		// размер буфера сообщения (при обмене информацией)
	double* message = new double[message_len]; // буфер сообщения (при обмене информацией)
	char flag = 1;								// вспомогательный флаг
	
	// Константы для уравнений
	const double hX2 = pow(hx, 2);		
	const double hY2 = pow(hy, 2);
	const double hZ2 = pow(hz, 2);
	double c = 1 / ((2 / hX2) + (2 / hY2) + (2 / hZ2) + a);
	double temp;

	while (flag) {
		max_conv = 0.0;
		// Вычислим граничные значения своей подобласти для отправки соседним процессам

		// Левая граница
		for (int j = 1; j < Ny - 1; j++) {
			for (int k = 1; k < Nz - 1; k++) {

				// Вычисляем значения на узле (формула 3 методических указаний)
				temp = (curr_node[2][j][k] + curr_node[0][j][k]) / hX2;
				temp += (curr_node[1][j + 1][k] + curr_node[1][j - 1][k]) / hY2;
				temp += (curr_node[1][j][k + 1] + curr_node[1][j][k - 1]) / hZ2;
				temp -= 6.0 - a *curr_node[1][j][k];
				temp *= c;
				past_node[1][j][k] = temp;

				// Вычисляем сходимость для данного узла
				conv = abs(past_node[1][j][k] - curr_node[1][j][k]);
				if (conv > max_conv) max_conv = conv;
			}
		}

		// Отправляем сообщение младшему процессу
		if (proc_lower != -1) {
			for (int j = 0; j < Ny - 2; j++) {
				for (int k = 0; k < Nz - 2; k++) {
					message[(Ny - 2) * j + k] = past_node[1][j + 1][k + 1];
				}
			}
			MPI_Send((void*)message, message_len, MPI_DOUBLE, proc_lower, msg_tag6, MPI_COMM_WORLD);
		}

		// Побрабатываем случай, когда только один процесс
		if (size_x != 3) {

			// Правая граница
			for (int j = 1; j < Ny - 1; j++) {
				for (int k = 1; k < Nz - 1; k++) {

					// Вычисляем значения на узле (формула 3 методических указаний)
					temp = (curr_node[size_x - 1][j][k] + curr_node[size_x - 3][j][k]) / hX2;
					temp += (curr_node[size_x - 2][j + 1][k] + curr_node[size_x - 2][j - 1][k]) / hY2;
					temp += (curr_node[size_x - 2][j][k + 1] + curr_node[size_x - 2][j][k - 1]) / hZ2;
					temp -= 6.0 - a *curr_node[size_x - 2][j][k];
					temp *= c;
					past_node[size_x - 2][j][k] = temp;

					// Вычисляем сходимость для данного узла
					conv = abs(past_node[size_x - 2][j][k] - curr_node[size_x - 2][j][k]);
					if (conv > max_conv) max_conv = conv;
				}
			}
		}

		// Отправляем сообщение старшему процессу
		if (proc_upper != -1) {
			for (int j = 0; j < Ny - 2; j++) {
				for (int k = 0; k < Nz - 2; k++) {
					message[(Ny - 2) * j + k] = past_node[size_x - 2][j + 1][k + 1];
				}
			}
			MPI_Send((void*)message, message_len, MPI_DOUBLE, proc_upper, msg_tag5, MPI_COMM_WORLD);
		}

		// Вычисляем значения внутренней области

		for (int i = 2; i < size_x - 2; i++) {
			for (int j = 1; j < Ny - 1; j++) {
				for (int k = 1; k < Nz - 1; k++) {

					// Вычисляем значения на узле (формула 3 методических указаний)
					temp = (curr_node[i + 1][j][k] + curr_node[i - 1][j][k]) / hX2;
					temp += (curr_node[i][j + 1][k] + curr_node[i][j - 1][k]) / hY2;
					temp += (curr_node[i][j][k + 1] + curr_node[i][j][k - 1]) / hZ2;
					temp -= 6.0 - a *curr_node[i][j][k];
					temp *= c;
					past_node[i][j][k] = temp;

					// Вычисляем сходимость для данного узла
					conv = abs(past_node[i][j][k] - curr_node[i][j][k]);
					if (conv > max_conv) max_conv = conv;
				}
			}
		}

		// Получаем информацию от младшего процесса
		if (proc_lower != -1) {
			MPI_Recv((void*)message, message_len, MPI_DOUBLE, proc_lower, msg_tag5, MPI_COMM_WORLD, &status);
			for (int j = 0; j < Ny - 2; j++) {
				for (int k = 0; k < Nz - 2; k++) {
					past_node[0][j + 1][k + 1] = message[(Ny - 2) * j + k];
				}
			}
		}

		// Получаем информацию от старшего процесса
		if (proc_upper != -1) {
			MPI_Recv((void*)message, message_len, MPI_DOUBLE, proc_upper, msg_tag6, MPI_COMM_WORLD, &status);
			for (int j = 0; j < Ny - 2; j++) {
				for (int k = 0; k < Nz - 2; k++) {
					past_node[size_x - 1][j + 1][k + 1] = message[(Ny - 2) * j + k];
				}
			}
		}

		// Проверяем, выполнилось ли условие прекращения вычислений 
		if (eps < max_conv) end_flag = 1;
		else end_flag = 0;
		
		// Проверяем, во всех ли процессах выполнилось условия прекращения вычислений. Если нет - продолжаем вычисления
		MPI_Reduce((void*)&end_flag, (void*)&flag, 1, MPI_CHAR, MPI_BOR, 0, MPI_COMM_WORLD);
		MPI_Bcast((void*)&flag, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

		temp_node = curr_node;
		curr_node = past_node;
		past_node = temp_node;
	}

	if (curr_node == grid_1) deleteGrid(grid_2, size_x);
	else 
	{
		temp_node = grid_1;
		grid_1 = curr_node;
		deleteGrid(temp_node, size_x);
	}

	delete[] message;
}

// Основная функция
int main(int argc, char** argv)
{
    // Количество процессов
	int proc_count;

    // Ранг текущего процесса
	int num_rank;

    // номер процесса, младшего для текущего (-1 - такого процесса нет)
	int low_proc_rank;

    // номер процесса, старшего для текущего (-1 - такого процесса нет)
	int upper_proc_rank;

    // Глобальный индекс по x, с которого начинается область текущего процесса (включительно)
	int proc_start_x;

    // Глобальный индекс по x, на котором заканчивается область текущего процесса (включительно)		
	int proc_end_x;

    // Cтатус возврата
	MPI_Status status;			

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &num_rank);		// номер текущего проесса
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);		// число параллельных процессов
	
	
	// Обработка работы главного процесса
	if (num_rank == 0) {

		// Определяем размеры подобластей процессов в зависимости от количества узлов (с учётом дублирующих слоёв)
		int batch_size = (Nx - 2) / proc_count;
		int remainder = (Nx - 2) % proc_count;		// Остаток, если нельзя сделать все блоки одинаковой длины
		int area_size = batch_size;		

		// Дополняем размер обрабатываемого блока
		if (remainder != 0) area_size++;

        // Начало области для процесса-получателя
		proc_start_x = 1;

        // Конец области для процесса-получателя 
		proc_end_x = area_size;

		low_proc_rank = -1;

		// Обрабатываем ситуацию, когда используется только один процесс (нет обмена граничными значениями)
		if (proc_count == 1) upper_proc_rank = -1;
		else upper_proc_rank = 1;

		// Переменные для передачи информации в порождаемые процессы
		int proc_lower;			// Младший процесс для процесса-получателя
		int proc_upper;			// Старший процесс для процесса-получателя
		int start_x = 1;	// Начало области для процесса-получателя
		int end_x;			// Конец области для процесса-получателя

		// Рассылаем информацию по процессам
		for (int i = 1; i < proc_count; i++) {
			
			// Отправляем номер младшего процесса
			proc_lower = i - 1;
			MPI_Send((void*)&proc_lower, 1, MPI_INT, i, msg_tag1, MPI_COMM_WORLD);

			// Если это последний по номеру процесс, то у него нет старшего процесса
			if (i == proc_count - 1) proc_upper = -1; 
			else proc_upper = i + 1;

			MPI_Send((void*)&proc_upper, 1, MPI_INT, i, msg_tag2, MPI_COMM_WORLD);

			start_x += area_size;
			area_size = batch_size;
			// Дополняем размер обрабатываемого блока, если остаток от деления был не нулевой
			if (i < remainder) area_size++;

			MPI_Send((void*)&start_x, 1, MPI_INT, i, msg_tag3, MPI_COMM_WORLD);
			// cout << start_x;

			// Отправляем конец рабочей области
			end_x = start_x + area_size - 1;
			MPI_Send((void*)&end_x, 1, MPI_INT, i, msg_tag4, MPI_COMM_WORLD);
			// cout << end_x << endl;
		}
	}
	else {
		// Процессы получают информацию для своей работы через сообщения 
		MPI_Recv((void*)&low_proc_rank, 1, MPI_INT, 0, msg_tag1, MPI_COMM_WORLD, &status);
		MPI_Recv((void*)&upper_proc_rank, 1, MPI_INT, 0, msg_tag2, MPI_COMM_WORLD, &status);
		MPI_Recv((void*)&proc_start_x, 1, MPI_INT, 0, msg_tag3, MPI_COMM_WORLD, &status);
		MPI_Recv((void*)&proc_end_x, 1, MPI_INT, 0, msg_tag4, MPI_COMM_WORLD, &status);
	}

    // Добавляем дублирующие граничные слои соседних подобластей
	int size_x = proc_end_x - proc_start_x + 3;		
	double*** grid;								
	
	// Заполняем начальные значения в узлах сетки
	setGrid(grid, size_x, X1 + hx * (proc_start_x - 1), num_rank, proc_count);

	// Ожидаем завершения инициализации всех процессов
	MPI_Barrier(MPI_COMM_WORLD);

	double start = MPI_Wtime();
	Jacobi(grid, size_x, proc_start_x, low_proc_rank, upper_proc_rank);
	double end = MPI_Wtime();

	double diff_time = end - start;

	// Считаем максимальную точность полученных значений
	double accuracy = Get_Accuracy(grid, size_x, X1 + hx * (proc_start_x - 1), 0);

	if (num_rank == 0) {
		cout << endl << "Program running time: " << diff_time;
		cout << endl << "Accuracy: " << accuracy << endl << endl;
	}

	// Конец параллельной области
	MPI_Finalize();

	// Очищаем память
	deleteGrid(grid, size_x);
}
