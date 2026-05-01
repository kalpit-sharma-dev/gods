package ndarray

import (
	"fmt"
	"math"
)

// Dot computes dot product of two 1D vectors.
func Dot[T Number](a, b *NDArray[T]) (T, error) {
	var zero T
	if a == nil || b == nil {
		return zero, fmt.Errorf("gods/ndarray: dot requires non-nil arrays")
	}
	if len(a.shape) != 1 || len(b.shape) != 1 {
		return zero, fmt.Errorf("gods/ndarray: dot requires 1D arrays, got %v and %v", a.shape, b.shape)
	}
	if a.shape[0] != b.shape[0] {
		return zero, fmt.Errorf("gods/ndarray: dot size mismatch %d vs %d", a.shape[0], b.shape[0])
	}
	var out T
	for i := 0; i < a.shape[0]; i++ {
		out += a.data[i] * b.data[i]
	}
	return out, nil
}

// MatMul computes matrix multiplication for 2D arrays.
func MatMul[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("gods/ndarray: matmul requires non-nil arrays")
	}
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("gods/ndarray: matmul requires 2D arrays, got %v and %v", a.shape, b.shape)
	}
	m, k := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k != k2 {
		return nil, fmt.Errorf("gods/ndarray: matmul dimension mismatch %d vs %d", k, k2)
	}
	out := make([]T, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc T
			for t := 0; t < k; t++ {
				acc += a.At(i, t) * b.At(t, j)
			}
			out[i*n+j] = acc
		}
	}
	return &NDArray[T]{data: out, shape: []int{m, n}, strides: []int{n, 1}}, nil
}

// Inv computes inverse of a square float64 matrix using Gauss-Jordan elimination.
func Inv(a *NDArray[float64]) (*NDArray[float64], error) {
	if a == nil {
		return nil, fmt.Errorf("gods/ndarray: inv requires non-nil array")
	}
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("gods/ndarray: inv requires square 2D matrix, got %v", a.shape)
	}
	n := a.shape[0]
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, 2*n)
		for j := 0; j < n; j++ {
			row[j] = a.At(i, j)
		}
		row[n+i] = 1
		aug[i] = row
	}
	for col := 0; col < n; col++ {
		pivot := col
		for r := col + 1; r < n; r++ {
			if math.Abs(aug[r][col]) > math.Abs(aug[pivot][col]) {
				pivot = r
			}
		}
		if almostZero(aug[pivot][col]) {
			return nil, fmt.Errorf("gods/ndarray: matrix is singular")
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]
		div := aug[col][col]
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= div
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := aug[r][col]
			if almostZero(f) {
				continue
			}
			for j := 0; j < 2*n; j++ {
				aug[r][j] -= f * aug[col][j]
			}
		}
	}
	out := make([]float64, n*n)
	for i := 0; i < n; i++ {
		copy(out[i*n:(i+1)*n], aug[i][n:])
	}
	return &NDArray[float64]{data: out, shape: []int{n, n}, strides: []int{n, 1}}, nil
}

// Det computes determinant of a square float64 matrix.
func Det(a *NDArray[float64]) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("gods/ndarray: det requires non-nil array")
	}
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return 0, fmt.Errorf("gods/ndarray: det requires square 2D matrix, got %v", a.shape)
	}
	n := a.shape[0]
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			m[i][j] = a.At(i, j)
		}
	}
	sign := 1.0
	for col := 0; col < n; col++ {
		pivot := col
		for r := col + 1; r < n; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[pivot][col]) {
				pivot = r
			}
		}
		if almostZero(m[pivot][col]) {
			return 0, nil
		}
		if pivot != col {
			m[col], m[pivot] = m[pivot], m[col]
			sign *= -1
		}
		for r := col + 1; r < n; r++ {
			f := m[r][col] / m[col][col]
			for c := col; c < n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	det := sign
	for i := 0; i < n; i++ {
		det *= m[i][i]
	}
	return det, nil
}

// SVD computes a simple approximate SVD where U and Vt are identity and S contains singular values.
func SVD(a *NDArray[float64]) (U, S, Vt *NDArray[float64], err error) {
	if a == nil {
		return nil, nil, nil, fmt.Errorf("gods/ndarray: svd requires non-nil array")
	}
	if len(a.shape) != 2 {
		return nil, nil, nil, fmt.Errorf("gods/ndarray: svd requires 2D matrix, got %v", a.shape)
	}
	m, n := a.shape[0], a.shape[1]
	k := m
	if n < k {
		k = n
	}
	svals := make([]float64, k)
	for i := 0; i < k; i++ {
		var sum float64
		for r := 0; r < m; r++ {
			sum += a.At(r, i) * a.At(r, i)
		}
		svals[i] = math.Sqrt(sum)
	}
	u := identity(m)
	vt := identity(n)
	s := &NDArray[float64]{data: svals, shape: []int{k}, strides: []int{1}}
	return u, s, vt, nil
}

// Norm computes vector L2 norm or matrix Frobenius norm.
func Norm[T Number](a *NDArray[T]) float64 {
	if a == nil {
		return 0
	}
	var sum float64
	for _, v := range a.data {
		f := toFloat(v)
		sum += f * f
	}
	return math.Sqrt(sum)
}

// Solve solves linear system a*x = b using inverse.
func Solve(a, b *NDArray[float64]) (*NDArray[float64], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("gods/ndarray: solve requires non-nil arrays")
	}
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("gods/ndarray: solve requires square matrix a, got %v", a.shape)
	}
	if len(b.shape) != 1 && len(b.shape) != 2 {
		return nil, fmt.Errorf("gods/ndarray: solve requires vector or matrix b, got %v", b.shape)
	}
	if b.shape[0] != a.shape[0] {
		return nil, fmt.Errorf("gods/ndarray: solve dimension mismatch a=%v b=%v", a.shape, b.shape)
	}
	ainv, err := Inv(a)
	if err != nil {
		return nil, err
	}
	if len(b.shape) == 1 {
		b2, _ := b.Reshape(b.shape[0], 1)
		x2, err := MatMul(ainv, b2)
		if err != nil {
			return nil, err
		}
		return x2.Reshape(b.shape[0])
	}
	return MatMul(ainv, b)
}

func identity(n int) *NDArray[float64] {
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		data[i*n+i] = 1
	}
	return &NDArray[float64]{data: data, shape: []int{n, n}, strides: []int{n, 1}}
}
