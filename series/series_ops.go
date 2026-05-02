package series

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/kalpit-sharma-dev/gods/internal/bitmap"
	"github.com/kalpit-sharma-dev/gods/internal/util"
)

// Map applies fn to every non-null element and propagates nulls.
func (s *Series[T]) Map(fn func(T) T) *Series[T] {
	if s == nil {
		return nil
	}
	out := s.cloneWith(s.values, s.validity)
	for i := range out.values {
		if !out.validity.IsValid(i) {
			continue
		}
		out.values[i] = fn(out.values[i])
	}
	return out
}

// MapErr applies fn to every non-null element; errors become nulls.
func (s *Series[T]) MapErr(fn func(T) (T, error)) (*Series[T], []error) {
	if s == nil {
		return nil, nil
	}
	out := s.cloneWith(s.values, s.validity)
	errs := make([]error, 0)
	for i := range out.values {
		if !out.validity.IsValid(i) {
			continue
		}
		v, err := fn(out.values[i])
		if err != nil {
			out.validity.Set(i, false)
			errs = append(errs, fmt.Errorf("gods/series: map error at index %d: %w", i, err))
			continue
		}
		out.values[i] = v
	}
	return out, errs
}

// Filter returns a new Series with valid elements where fn returns true.
func (s *Series[T]) Filter(fn func(T) bool) *Series[T] {
	if s == nil {
		return nil
	}
	values := make([]T, 0, s.Len())
	nullMask := make([]bool, 0, s.Len())
	for i := range s.values {
		if !s.validity.IsValid(i) {
			continue
		}
		if fn(s.values[i]) {
			values = append(values, s.values[i])
			nullMask = append(nullMask, false)
		}
	}
	return WithNulls(s.name, values, nullMask)
}

// Reduce folds valid values left-to-right, skipping nulls.
func (s *Series[T]) Reduce(initial T, fn func(acc, val T) T) T {
	acc := initial
	if s == nil {
		return acc
	}
	for i := range s.values {
		if !s.validity.IsValid(i) {
			continue
		}
		acc = fn(acc, s.values[i])
	}
	return acc
}

// Apply mutates non-null elements in place and returns self.
func (s *Series[T]) Apply(fn func(T) T) *Series[T] {
	if s == nil {
		return nil
	}
	for i := range s.values {
		if !s.validity.IsValid(i) {
			continue
		}
		s.values[i] = fn(s.values[i])
	}
	return s
}

// Head returns the first n rows.
func (s *Series[T]) Head(n int) *Series[T] {
	if s == nil {
		return nil
	}
	if n < 0 {
		n = 0
	}
	if n > s.Len() {
		n = s.Len()
	}
	values := s.values[:n]
	valid := bitmap.New(n)
	for i := 0; i < n; i++ {
		valid.Set(i, s.validity.IsValid(i))
	}
	return &Series[T]{name: s.name, values: append([]T(nil), values...), validity: valid, dtype: s.dtype}
}

// Tail returns the last n rows.
func (s *Series[T]) Tail(n int) *Series[T] {
	if s == nil {
		return nil
	}
	if n < 0 {
		n = 0
	}
	if n > s.Len() {
		n = s.Len()
	}
	start := s.Len() - n
	values := s.values[start:]
	valid := bitmap.New(n)
	for i := 0; i < n; i++ {
		valid.Set(i, s.validity.IsValid(start+i))
	}
	return &Series[T]{name: s.name, values: append([]T(nil), values...), validity: valid, dtype: s.dtype}
}

// Concat merges two same-typed series.
func Concat[T any](a, b *Series[T]) *Series[T] {
	if a == nil && b == nil {
		return nil
	}
	if a == nil {
		return b.cloneWith(b.values, b.validity)
	}
	if b == nil {
		return a.cloneWith(a.values, a.validity)
	}
	outVals := make([]T, 0, a.Len()+b.Len())
	outVals = append(outVals, a.values...)
	outVals = append(outVals, b.values...)
	outValid := bitmap.New(a.Len() + b.Len())
	for i := 0; i < a.Len(); i++ {
		outValid.Set(i, a.validity.IsValid(i))
	}
	for i := 0; i < b.Len(); i++ {
		outValid.Set(a.Len()+i, b.validity.IsValid(i))
	}
	return &Series[T]{
		name:     a.name,
		values:   outVals,
		validity: outValid,
		dtype:    a.dtype,
	}
}

// Cast converts a series to a target dtype.
func (s *Series[T]) Cast(dtype util.Dtype) (any, error) {
	if s == nil {
		return nil, fmt.Errorf("gods/series: cannot cast nil series")
	}
	switch dtype {
	case util.DtypeInt64:
		values := make([]int64, s.Len())
		mask := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			v, ok := s.At(i)
			if !ok {
				mask[i] = true
				continue
			}
			iv, err := toInt64(any(v))
			if err != nil {
				return nil, fmt.Errorf("gods/series: cast to int64 at index %d: %w", i, err)
			}
			values[i] = iv
		}
		return WithNulls[int64](s.name, values, mask), nil
	case util.DtypeFloat64:
		values := make([]float64, s.Len())
		mask := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			v, ok := s.At(i)
			if !ok {
				mask[i] = true
				continue
			}
			fv, err := toFloat64(any(v))
			if err != nil {
				return nil, fmt.Errorf("gods/series: cast to float64 at index %d: %w", i, err)
			}
			values[i] = fv
		}
		return WithNulls[float64](s.name, values, mask), nil
	case util.DtypeBool:
		values := make([]bool, s.Len())
		mask := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			v, ok := s.At(i)
			if !ok {
				mask[i] = true
				continue
			}
			bv, err := toBool(any(v))
			if err != nil {
				return nil, fmt.Errorf("gods/series: cast to bool at index %d: %w", i, err)
			}
			values[i] = bv
		}
		return WithNulls[bool](s.name, values, mask), nil
	case util.DtypeString:
		values := make([]string, s.Len())
		mask := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			v, ok := s.At(i)
			if !ok {
				mask[i] = true
				continue
			}
			values[i] = fmt.Sprint(v)
		}
		return WithNulls[string](s.name, values, mask), nil
	case util.DtypeTime:
		values := make([]time.Time, s.Len())
		mask := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			v, ok := s.At(i)
			if !ok {
				mask[i] = true
				continue
			}
			tv, err := toTime(any(v))
			if err != nil {
				return nil, fmt.Errorf("gods/series: cast to time at index %d: %w", i, err)
			}
			values[i] = tv
		}
		return WithNulls[time.Time](s.name, values, mask), nil
	case util.DtypeAny:
		values := make([]any, s.Len())
		mask := make([]bool, s.Len())
		for i := 0; i < s.Len(); i++ {
			v, ok := s.At(i)
			if !ok {
				mask[i] = true
				continue
			}
			values[i] = any(v)
		}
		return WithNulls[any](s.name, values, mask), nil
	default:
		return nil, fmt.Errorf("gods/series: unsupported dtype %v", dtype)
	}
}

func toInt64(v any) (int64, error) {
	switch x := v.(type) {
	case int:
		return int64(x), nil
	case int8:
		return int64(x), nil
	case int16:
		return int64(x), nil
	case int32:
		return int64(x), nil
	case int64:
		return x, nil
	case uint:
		return int64(x), nil
	case uint8:
		return int64(x), nil
	case uint16:
		return int64(x), nil
	case uint32:
		return int64(x), nil
	case uint64:
		if x > math.MaxInt64 {
			return 0, fmt.Errorf("uint64 overflow for int64 conversion")
		}
		return int64(x), nil
	case float32:
		return int64(x), nil
	case float64:
		return int64(x), nil
	case string:
		n, err := strconv.ParseInt(x, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("parse int64: %w", err)
		}
		return n, nil
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("cannot convert %T to int64", v)
	}
}

func toFloat64(v any) (float64, error) {
	switch x := v.(type) {
	case int:
		return float64(x), nil
	case int8:
		return float64(x), nil
	case int16:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case int64:
		return float64(x), nil
	case uint:
		return float64(x), nil
	case uint8:
		return float64(x), nil
	case uint16:
		return float64(x), nil
	case uint32:
		return float64(x), nil
	case uint64:
		return float64(x), nil
	case float32:
		return float64(x), nil
	case float64:
		return x, nil
	case string:
		n, err := strconv.ParseFloat(x, 64)
		if err != nil {
			return 0, fmt.Errorf("parse float64: %w", err)
		}
		return n, nil
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", v)
	}
}

func toBool(v any) (bool, error) {
	switch x := v.(type) {
	case bool:
		return x, nil
	case string:
		b, err := strconv.ParseBool(x)
		if err != nil {
			return false, fmt.Errorf("parse bool: %w", err)
		}
		return b, nil
	case int, int8, int16, int32, int64:
		return fmt.Sprint(v) != "0", nil
	case uint, uint8, uint16, uint32, uint64:
		return fmt.Sprint(v) != "0", nil
	case float32:
		return x != 0, nil
	case float64:
		return x != 0, nil
	default:
		return false, fmt.Errorf("cannot convert %T to bool", v)
	}
}

func toTime(v any) (time.Time, error) {
	switch x := v.(type) {
	case time.Time:
		return x, nil
	case string:
		for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02"} {
			t, err := time.Parse(layout, x)
			if err == nil {
				return t, nil
			}
		}
		return time.Time{}, fmt.Errorf("cannot parse time from %q", x)
	case int64:
		return time.Unix(x, 0).UTC(), nil
	case float64:
		return time.Unix(int64(x), 0).UTC(), nil
	default:
		return time.Time{}, fmt.Errorf("cannot convert %T to time", v)
	}
}
