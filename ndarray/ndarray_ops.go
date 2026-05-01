package ndarray

import (
	"fmt"
	"math"
)

// AddScalar adds a scalar to each element.
func (a *NDArray[T]) AddScalar(s T) *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	for i := range a.data {
		out[i] = addAny(a.data[i], s)
	}
	return &NDArray[T]{data: out, shape: a.Shape(), strides: a.Strides()}
}

// SubScalar subtracts a scalar from each element.
func (a *NDArray[T]) SubScalar(s T) *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	for i := range a.data {
		out[i] = subAny(a.data[i], s)
	}
	return &NDArray[T]{data: out, shape: a.Shape(), strides: a.Strides()}
}

// MulScalar multiplies each element by a scalar.
func (a *NDArray[T]) MulScalar(s T) *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	for i := range a.data {
		out[i] = mulAny(a.data[i], s)
	}
	return &NDArray[T]{data: out, shape: a.Shape(), strides: a.Strides()}
}

// DivScalar divides each element by a scalar.
func (a *NDArray[T]) DivScalar(s T) *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	for i := range a.data {
		out[i] = divAny(a.data[i], s)
	}
	return &NDArray[T]{data: out, shape: a.Shape(), strides: a.Strides()}
}

// PowScalar raises each element to a scalar power.
func (a *NDArray[T]) PowScalar(p float64) *NDArray[float64] {
	if a == nil {
		return nil
	}
	out := make([]float64, len(a.data))
	for i := range a.data {
		out[i] = math.Pow(toFloat(a.data[i]), p)
	}
	return &NDArray[float64]{data: out, shape: a.Shape(), strides: computeStrides(a.shape)}
}

// Add performs element-wise addition with broadcasting.
func Add[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return binaryOp(a, b, func(x, y T) T { return x + y })
}

// Sub performs element-wise subtraction with broadcasting.
func Sub[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return binaryOp(a, b, func(x, y T) T { return x - y })
}

// Mul performs element-wise multiplication with broadcasting.
func Mul[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return binaryOp(a, b, func(x, y T) T { return x * y })
}

// Div performs element-wise division with broadcasting.
func Div[T Number](a, b *NDArray[T]) (*NDArray[T], error) {
	return binaryOp(a, b, func(x, y T) T { return x / y })
}

// Abs applies absolute value element-wise.
func (a *NDArray[T]) Abs() *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	for i := range a.data {
		out[i] = absAny(a.data[i])
	}
	return &NDArray[T]{data: out, shape: a.Shape(), strides: a.Strides()}
}

// Neg negates all elements.
func (a *NDArray[T]) Neg() *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	for i := range a.data {
		out[i] = negAny(a.data[i])
	}
	return &NDArray[T]{data: out, shape: a.Shape(), strides: a.Strides()}
}

// Sqrt computes element-wise square root.
func (a *NDArray[T]) Sqrt() *NDArray[float64] {
	if a == nil {
		return nil
	}
	out := make([]float64, len(a.data))
	for i, v := range a.data {
		out[i] = math.Sqrt(toFloat(v))
	}
	return &NDArray[float64]{data: out, shape: a.Shape(), strides: computeStrides(a.shape)}
}

// Exp computes element-wise exponent.
func (a *NDArray[T]) Exp() *NDArray[float64] {
	if a == nil {
		return nil
	}
	out := make([]float64, len(a.data))
	for i, v := range a.data {
		out[i] = math.Exp(toFloat(v))
	}
	return &NDArray[float64]{data: out, shape: a.Shape(), strides: computeStrides(a.shape)}
}

// Log computes element-wise natural logarithm.
func (a *NDArray[T]) Log() *NDArray[float64] {
	if a == nil {
		return nil
	}
	out := make([]float64, len(a.data))
	for i, v := range a.data {
		out[i] = math.Log(toFloat(v))
	}
	return &NDArray[float64]{data: out, shape: a.Shape(), strides: computeStrides(a.shape)}
}

// Sum reduces along an axis (-1 means all axes).
func (a *NDArray[T]) Sum(axis int) *NDArray[T] {
	if a == nil {
		return nil
	}
	if axis == -1 || len(a.shape) == 1 {
		var total T
		for _, v := range a.data {
			total = addAny(total, v)
		}
		return &NDArray[T]{data: []T{total}, shape: []int{1}, strides: []int{1}}
	}
	if len(a.shape) == 2 && axis >= 0 && axis <= 1 {
		r, c := a.shape[0], a.shape[1]
		if axis == 0 {
			out := make([]T, c)
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					out[j] = addAny(out[j], a.At(i, j))
				}
			}
			return &NDArray[T]{data: out, shape: []int{c}, strides: []int{1}}
		}
		out := make([]T, r)
		for i := 0; i < r; i++ {
			var acc T
			for j := 0; j < c; j++ {
				acc = addAny(acc, a.At(i, j))
			}
			out[i] = acc
		}
		return &NDArray[T]{data: out, shape: []int{r}, strides: []int{1}}
	}
	return a.Sum(-1)
}

// Min reduces to minimum values along axis.
func (a *NDArray[T]) Min(axis int) *NDArray[T] {
	return reduceMinMax(a, axis, true)
}

// Max reduces to maximum values along axis.
func (a *NDArray[T]) Max(axis int) *NDArray[T] {
	return reduceMinMax(a, axis, false)
}

// Mean reduces to mean values along axis.
func (a *NDArray[T]) Mean(axis int) *NDArray[float64] {
	if a == nil {
		return nil
	}
	if axis == -1 || len(a.shape) == 1 {
		if len(a.data) == 0 {
			return &NDArray[float64]{shape: []int{0}, strides: []int{1}}
		}
		var sum float64
		for _, v := range a.data {
			sum += toFloat(v)
		}
		return &NDArray[float64]{data: []float64{sum / float64(len(a.data))}, shape: []int{1}, strides: []int{1}}
	}
	if len(a.shape) == 2 && axis >= 0 && axis <= 1 {
		r, c := a.shape[0], a.shape[1]
		if axis == 0 {
			out := make([]float64, c)
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					out[j] += toFloat(a.At(i, j))
				}
			}
			for j := range out {
				out[j] /= float64(r)
			}
			return &NDArray[float64]{data: out, shape: []int{c}, strides: []int{1}}
		}
		out := make([]float64, r)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				out[i] += toFloat(a.At(i, j))
			}
			out[i] /= float64(c)
		}
		return &NDArray[float64]{data: out, shape: []int{r}, strides: []int{1}}
	}
	return a.Mean(-1)
}

// Eq compares element-wise for equality.
func (a *NDArray[T]) Eq(b *NDArray[T]) (*NDArray[bool], error) {
	return compareOp(a, b, func(x, y T) bool { return compareRaw(x, y) == 0 })
}

// Gt compares element-wise for greater-than.
func (a *NDArray[T]) Gt(b *NDArray[T]) (*NDArray[bool], error) {
	return compareOp(a, b, func(x, y T) bool { return compareRaw(x, y) > 0 })
}

// Lt compares element-wise for less-than.
func (a *NDArray[T]) Lt(b *NDArray[T]) (*NDArray[bool], error) {
	return compareOp(a, b, func(x, y T) bool { return compareRaw(x, y) < 0 })
}

func binaryOp[T Number](a, b *NDArray[T], fn func(T, T) T) (*NDArray[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("gods/ndarray: binary op requires non-nil arrays")
	}
	outShape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	outSize := product(outShape)
	out := make([]T, outSize)
	outStrides := computeStrides(outShape)
	for i := 0; i < outSize; i++ {
		oidx := unflattenIndex(i, outShape, outStrides)
		ai := broadcastFlatIndex(oidx, outShape, a.shape, a.strides)
		bi := broadcastFlatIndex(oidx, outShape, b.shape, b.strides)
		out[i] = fn(a.data[ai], b.data[bi])
	}
	return &NDArray[T]{data: out, shape: outShape, strides: outStrides}, nil
}

func compareOp[T any](a, b *NDArray[T], fn func(T, T) bool) (*NDArray[bool], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("gods/ndarray: comparison requires non-nil arrays")
	}
	outShape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	outSize := product(outShape)
	out := make([]bool, outSize)
	outStrides := computeStrides(outShape)
	for i := 0; i < outSize; i++ {
		oidx := unflattenIndex(i, outShape, outStrides)
		ai := broadcastFlatIndex(oidx, outShape, a.shape, a.strides)
		bi := broadcastFlatIndex(oidx, outShape, b.shape, b.strides)
		out[i] = fn(a.data[ai], b.data[bi])
	}
	return &NDArray[bool]{data: out, shape: outShape, strides: outStrides}, nil
}

func broadcastShape(a, b []int) ([]int, error) {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		ai, bi := 1, 1
		if len(a)-1-i >= 0 {
			ai = a[len(a)-1-i]
		}
		if len(b)-1-i >= 0 {
			bi = b[len(b)-1-i]
		}
		if ai != bi && ai != 1 && bi != 1 {
			return nil, fmt.Errorf("gods/ndarray: shapes %v and %v are not broadcastable", a, b)
		}
		if ai > bi {
			out[n-1-i] = ai
		} else {
			out[n-1-i] = bi
		}
	}
	return out, nil
}

func broadcastFlatIndex(outIdx, outShape, inShape, inStrides []int) int {
	offset := len(outShape) - len(inShape)
	flat := 0
	for axis := range inShape {
		idx := outIdx[axis+offset]
		if inShape[axis] == 1 {
			idx = 0
		}
		flat += idx * inStrides[axis]
	}
	return flat
}

func reduceMinMax[T any](a *NDArray[T], axis int, isMin bool) *NDArray[T] {
	if a == nil || len(a.data) == 0 {
		return &NDArray[T]{shape: []int{0}, strides: []int{1}}
	}
	if axis == -1 || len(a.shape) == 1 {
		best := a.data[0]
		for _, v := range a.data[1:] {
			cmp := compareRaw(v, best)
			if (isMin && cmp < 0) || (!isMin && cmp > 0) {
				best = v
			}
		}
		return &NDArray[T]{data: []T{best}, shape: []int{1}, strides: []int{1}}
	}
	if len(a.shape) == 2 && axis >= 0 && axis <= 1 {
		r, c := a.shape[0], a.shape[1]
		if axis == 0 {
			out := make([]T, c)
			for j := 0; j < c; j++ {
				out[j] = a.At(0, j)
				for i := 1; i < r; i++ {
					v := a.At(i, j)
					cmp := compareRaw(v, out[j])
					if (isMin && cmp < 0) || (!isMin && cmp > 0) {
						out[j] = v
					}
				}
			}
			return &NDArray[T]{data: out, shape: []int{c}, strides: []int{1}}
		}
		out := make([]T, r)
		for i := 0; i < r; i++ {
			out[i] = a.At(i, 0)
			for j := 1; j < c; j++ {
				v := a.At(i, j)
				cmp := compareRaw(v, out[i])
				if (isMin && cmp < 0) || (!isMin && cmp > 0) {
					out[i] = v
				}
			}
		}
		return &NDArray[T]{data: out, shape: []int{r}, strides: []int{1}}
	}
	return reduceMinMax(a, -1, isMin)
}

func compareRaw[T any](a, b T) int {
	switch av := any(a).(type) {
	case int:
		bv := any(b).(int)
		switch {
		case av < bv:
			return -1
		case av > bv:
			return 1
		default:
			return 0
		}
	case int64:
		bv := any(b).(int64)
		switch {
		case av < bv:
			return -1
		case av > bv:
			return 1
		default:
			return 0
		}
	case float64:
		bv := any(b).(float64)
		switch {
		case av < bv:
			return -1
		case av > bv:
			return 1
		default:
			return 0
		}
	case float32:
		bv := any(b).(float32)
		switch {
		case av < bv:
			return -1
		case av > bv:
			return 1
		default:
			return 0
		}
	default:
		as := fmt.Sprint(a)
		bs := fmt.Sprint(b)
		if as < bs {
			return -1
		}
		if as > bs {
			return 1
		}
		return 0
	}
}

func toFloat[T any](v T) float64 {
	switch x := any(v).(type) {
	case int:
		return float64(x)
	case int8:
		return float64(x)
	case int16:
		return float64(x)
	case int32:
		return float64(x)
	case int64:
		return float64(x)
	case uint:
		return float64(x)
	case uint8:
		return float64(x)
	case uint16:
		return float64(x)
	case uint32:
		return float64(x)
	case uint64:
		return float64(x)
	case float32:
		return float64(x)
	case float64:
		return x
	default:
		return 0
	}
}

func addAny[T any](a, b T) T {
	switch av := any(a).(type) {
	case int:
		return any(av + any(b).(int)).(T)
	case int8:
		return any(av + any(b).(int8)).(T)
	case int16:
		return any(av + any(b).(int16)).(T)
	case int32:
		return any(av + any(b).(int32)).(T)
	case int64:
		return any(av + any(b).(int64)).(T)
	case uint:
		return any(av + any(b).(uint)).(T)
	case uint8:
		return any(av + any(b).(uint8)).(T)
	case uint16:
		return any(av + any(b).(uint16)).(T)
	case uint32:
		return any(av + any(b).(uint32)).(T)
	case uint64:
		return any(av + any(b).(uint64)).(T)
	case float32:
		return any(av + any(b).(float32)).(T)
	case float64:
		return any(av + any(b).(float64)).(T)
	default:
		return a
	}
}

func subAny[T any](a, b T) T {
	switch av := any(a).(type) {
	case int:
		return any(av - any(b).(int)).(T)
	case int8:
		return any(av - any(b).(int8)).(T)
	case int16:
		return any(av - any(b).(int16)).(T)
	case int32:
		return any(av - any(b).(int32)).(T)
	case int64:
		return any(av - any(b).(int64)).(T)
	case uint:
		return any(av - any(b).(uint)).(T)
	case uint8:
		return any(av - any(b).(uint8)).(T)
	case uint16:
		return any(av - any(b).(uint16)).(T)
	case uint32:
		return any(av - any(b).(uint32)).(T)
	case uint64:
		return any(av - any(b).(uint64)).(T)
	case float32:
		return any(av - any(b).(float32)).(T)
	case float64:
		return any(av - any(b).(float64)).(T)
	default:
		return a
	}
}

func mulAny[T any](a, b T) T {
	switch av := any(a).(type) {
	case int:
		return any(av * any(b).(int)).(T)
	case int8:
		return any(av * any(b).(int8)).(T)
	case int16:
		return any(av * any(b).(int16)).(T)
	case int32:
		return any(av * any(b).(int32)).(T)
	case int64:
		return any(av * any(b).(int64)).(T)
	case uint:
		return any(av * any(b).(uint)).(T)
	case uint8:
		return any(av * any(b).(uint8)).(T)
	case uint16:
		return any(av * any(b).(uint16)).(T)
	case uint32:
		return any(av * any(b).(uint32)).(T)
	case uint64:
		return any(av * any(b).(uint64)).(T)
	case float32:
		return any(av * any(b).(float32)).(T)
	case float64:
		return any(av * any(b).(float64)).(T)
	default:
		return a
	}
}

func divAny[T any](a, b T) T {
	switch av := any(a).(type) {
	case int:
		return any(av / any(b).(int)).(T)
	case int8:
		return any(av / any(b).(int8)).(T)
	case int16:
		return any(av / any(b).(int16)).(T)
	case int32:
		return any(av / any(b).(int32)).(T)
	case int64:
		return any(av / any(b).(int64)).(T)
	case uint:
		return any(av / any(b).(uint)).(T)
	case uint8:
		return any(av / any(b).(uint8)).(T)
	case uint16:
		return any(av / any(b).(uint16)).(T)
	case uint32:
		return any(av / any(b).(uint32)).(T)
	case uint64:
		return any(av / any(b).(uint64)).(T)
	case float32:
		return any(av / any(b).(float32)).(T)
	case float64:
		return any(av / any(b).(float64)).(T)
	default:
		return a
	}
}

func absAny[T any](a T) T {
	switch av := any(a).(type) {
	case int:
		if av < 0 {
			av = -av
		}
		return any(av).(T)
	case int8:
		if av < 0 {
			av = -av
		}
		return any(av).(T)
	case int16:
		if av < 0 {
			av = -av
		}
		return any(av).(T)
	case int32:
		if av < 0 {
			av = -av
		}
		return any(av).(T)
	case int64:
		if av < 0 {
			av = -av
		}
		return any(av).(T)
	case float32:
		return any(float32(math.Abs(float64(av)))).(T)
	case float64:
		return any(math.Abs(av)).(T)
	default:
		return a
	}
}

func negAny[T any](a T) T {
	switch av := any(a).(type) {
	case int:
		return any(-av).(T)
	case int8:
		return any(-av).(T)
	case int16:
		return any(-av).(T)
	case int32:
		return any(-av).(T)
	case int64:
		return any(-av).(T)
	case float32:
		return any(-av).(T)
	case float64:
		return any(-av).(T)
	default:
		return a
	}
}
