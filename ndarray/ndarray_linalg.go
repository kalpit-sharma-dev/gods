package ndarray

import (
	"fmt"
	"math"
	"sort"
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
	// Compute eigendecomposition of A^T A to derive V and singular values.
	ata, err := matMulTransposeLeft(a)
	if err != nil {
		return nil, nil, nil, err
	}
	evals, evecs, err := jacobiEigenSym(ata)
	if err != nil {
		return nil, nil, nil, err
	}

	n := a.shape[1]
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return evals[order[i]] > evals[order[j]] })

	svals := make([]float64, n)
	vData := make([]float64, n*n)
	for col := 0; col < n; col++ {
		src := order[col]
		lambda := evals[src]
		if lambda < 0 {
			lambda = 0
		}
		svals[col] = math.Sqrt(lambda)
		for row := 0; row < n; row++ {
			vData[row*n+col] = evecs[row*n+src]
		}
	}
	v := &NDArray[float64]{data: vData, shape: []int{n, n}, strides: []int{n, 1}}

	// U = A * V * Sigma^{-1}
	m := a.shape[0]
	uData := make([]float64, m*n)
	for j := 0; j < n; j++ {
		sigma := svals[j]
		if sigma <= 1e-12 {
			continue
		}
		for i := 0; i < m; i++ {
			var acc float64
			for k := 0; k < n; k++ {
				acc += a.At(i, k) * v.At(k, j)
			}
			uData[i*n+j] = acc / sigma
		}
	}
	u := &NDArray[float64]{data: uData, shape: []int{m, n}, strides: []int{n, 1}}
	vt := v.T()
	s := &NDArray[float64]{data: svals, shape: []int{n}, strides: []int{1}}
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
	n := a.shape[0]
	m := 1
	if len(b.shape) == 2 {
		m = b.shape[1]
	}

	// Build augmented RHS with 2D view.
	b2 := make([]float64, n*m)
	if len(b.shape) == 1 {
		for i := 0; i < n; i++ {
			b2[i] = b.At(i)
		}
	} else {
		copy(b2, b.data)
	}

	lu, piv, err := luDecompose(a)
	if err != nil {
		return nil, err
	}
	x := make([]float64, n*m)
	work := make([]float64, n)
	for col := 0; col < m; col++ {
		for i := 0; i < n; i++ {
			work[i] = b2[piv[i]*m+col]
		}
		// Forward solve Ly = Pb.
		for i := 0; i < n; i++ {
			for j := 0; j < i; j++ {
				work[i] -= lu[i*n+j] * work[j]
			}
		}
		// Backward solve Ux = y.
		for i := n - 1; i >= 0; i-- {
			for j := i + 1; j < n; j++ {
				work[i] -= lu[i*n+j] * work[j]
			}
			diag := lu[i*n+i]
			if almostZero(diag) {
				return nil, fmt.Errorf("gods/ndarray: singular matrix in solve")
			}
			work[i] /= diag
		}
		for i := 0; i < n; i++ {
			x[i*m+col] = work[i]
		}
	}

	if len(b.shape) == 1 {
		return &NDArray[float64]{data: x[:n], shape: []int{n}, strides: []int{1}}, nil
	}
	return &NDArray[float64]{data: x, shape: []int{n, m}, strides: []int{m, 1}}, nil
}

func luDecompose(a *NDArray[float64]) ([]float64, []int, error) {
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return nil, nil, fmt.Errorf("gods/ndarray: LU requires square matrix, got %v", a.shape)
	}
	n := a.shape[0]
	lu := make([]float64, len(a.data))
	copy(lu, a.data)
	piv := make([]int, n)
	for i := range piv {
		piv[i] = i
	}
	for k := 0; k < n; k++ {
		pivot := k
		maxVal := math.Abs(lu[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := math.Abs(lu[i*n+k]); v > maxVal {
				maxVal = v
				pivot = i
			}
		}
		if almostZero(maxVal) {
			return nil, nil, fmt.Errorf("gods/ndarray: singular matrix")
		}
		if pivot != k {
			piv[k], piv[pivot] = piv[pivot], piv[k]
			for j := 0; j < n; j++ {
				lu[k*n+j], lu[pivot*n+j] = lu[pivot*n+j], lu[k*n+j]
			}
		}
		for i := k + 1; i < n; i++ {
			lu[i*n+k] /= lu[k*n+k]
			for j := k + 1; j < n; j++ {
				lu[i*n+j] -= lu[i*n+k] * lu[k*n+j]
			}
		}
	}
	return lu, piv, nil
}

func matMulTransposeLeft(a *NDArray[float64]) (*NDArray[float64], error) {
	if len(a.shape) != 2 {
		return nil, fmt.Errorf("gods/ndarray: expected 2D matrix, got %v", a.shape)
	}
	m, n := a.shape[0], a.shape[1]
	out := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var acc float64
			for r := 0; r < m; r++ {
				acc += a.At(r, i) * a.At(r, j)
			}
			out[i*n+j] = acc
			out[j*n+i] = acc
		}
	}
	return &NDArray[float64]{data: out, shape: []int{n, n}, strides: []int{n, 1}}, nil
}

func jacobiEigenSym(a *NDArray[float64]) ([]float64, []float64, error) {
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return nil, nil, fmt.Errorf("gods/ndarray: jacobi expects square matrix, got %v", a.shape)
	}
	n := a.shape[0]
	mat := make([]float64, len(a.data))
	copy(mat, a.data)
	vecs := make([]float64, n*n)
	for i := 0; i < n; i++ {
		vecs[i*n+i] = 1
	}
	const maxIter = 128
	for iter := 0; iter < maxIter; iter++ {
		p, q := 0, 1
		maxOff := 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				v := math.Abs(mat[i*n+j])
				if v > maxOff {
					maxOff = v
					p, q = i, j
				}
			}
		}
		if maxOff < 1e-12 {
			break
		}
		app := mat[p*n+p]
		aqq := mat[q*n+q]
		apq := mat[p*n+q]
		phi := 0.5 * math.Atan2(2*apq, aqq-app)
		c := math.Cos(phi)
		s := math.Sin(phi)

		for k := 0; k < n; k++ {
			mkp := mat[k*n+p]
			mkq := mat[k*n+q]
			mat[k*n+p] = c*mkp - s*mkq
			mat[k*n+q] = s*mkp + c*mkq
		}
		for k := 0; k < n; k++ {
			mpk := mat[p*n+k]
			mqk := mat[q*n+k]
			mat[p*n+k] = c*mpk - s*mqk
			mat[q*n+k] = s*mpk + c*mqk
		}
		mat[p*n+q] = 0
		mat[q*n+p] = 0

		for k := 0; k < n; k++ {
			vkp := vecs[k*n+p]
			vkq := vecs[k*n+q]
			vecs[k*n+p] = c*vkp - s*vkq
			vecs[k*n+q] = s*vkp + c*vkq
		}
	}
	evals := make([]float64, n)
	for i := 0; i < n; i++ {
		evals[i] = mat[i*n+i]
	}
	return evals, vecs, nil
}

func identity(n int) *NDArray[float64] {
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		data[i*n+i] = 1
	}
	return &NDArray[float64]{data: data, shape: []int{n, n}, strides: []int{n, 1}}
}
