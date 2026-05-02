package series

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"golang.org/x/exp/constraints"
)

var errEmptyNumericSeries = errors.New("gods/series: series is empty or all-null")

func numericValues[T constraints.Integer | constraints.Float](s *Series[T]) ([]float64, []int) {
	if s == nil {
		return nil, nil
	}
	values := make([]float64, 0, s.Len())
	indices := make([]int, 0, s.Len())
	for i := 0; i < s.Len(); i++ {
		v, ok := s.At(i)
		if !ok {
			continue
		}
		values = append(values, float64(v))
		indices = append(indices, i)
	}
	return values, indices
}

// Sum computes the sum of all non-null numeric values.
func Sum[T constraints.Integer | constraints.Float](s *Series[T]) (T, error) {
	var zero T
	if s == nil {
		return zero, errEmptyNumericSeries
	}
	var acc T
	count := 0
	for i := 0; i < s.Len(); i++ {
		v, ok := s.At(i)
		if !ok {
			continue
		}
		acc += v
		count++
	}
	if count == 0 {
		return zero, errEmptyNumericSeries
	}
	return acc, nil
}

// Mean computes the arithmetic mean of non-null values.
func Mean[T constraints.Integer | constraints.Float](s *Series[T]) (float64, error) {
	values, _ := numericValues(s)
	if len(values) == 0 {
		return 0, errEmptyNumericSeries
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values)), nil
}

// Median computes the median of non-null values.
func Median[T constraints.Integer | constraints.Float](s *Series[T]) (float64, error) {
	return Quantile(s, 0.5)
}

// Var computes sample variance (n-1 denominator) of non-null values.
func Var[T constraints.Integer | constraints.Float](s *Series[T]) (float64, error) {
	values, _ := numericValues(s)
	if len(values) < 2 {
		return 0, errEmptyNumericSeries
	}
	m, _ := Mean(s)
	var acc float64
	for _, v := range values {
		d := v - m
		acc += d * d
	}
	return acc / float64(len(values)-1), nil
}

// Std computes sample standard deviation of non-null values.
func Std[T constraints.Integer | constraints.Float](s *Series[T]) (float64, error) {
	v, err := Var(s)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(v), nil
}

// Min returns the minimum non-null value.
func Min[T constraints.Integer | constraints.Float](s *Series[T]) (T, error) {
	var zero T
	if s == nil {
		return zero, errEmptyNumericSeries
	}
	found := false
	min := zero
	for i := 0; i < s.Len(); i++ {
		v, ok := s.At(i)
		if !ok {
			continue
		}
		if !found || v < min {
			min = v
			found = true
		}
	}
	if !found {
		return zero, errEmptyNumericSeries
	}
	return min, nil
}

// Max returns the maximum non-null value.
func Max[T constraints.Integer | constraints.Float](s *Series[T]) (T, error) {
	var zero T
	if s == nil {
		return zero, errEmptyNumericSeries
	}
	found := false
	max := zero
	for i := 0; i < s.Len(); i++ {
		v, ok := s.At(i)
		if !ok {
			continue
		}
		if !found || v > max {
			max = v
			found = true
		}
	}
	if !found {
		return zero, errEmptyNumericSeries
	}
	return max, nil
}

// ArgMin returns the index of the smallest non-null value.
func ArgMin[T constraints.Integer | constraints.Float](s *Series[T]) (int, error) {
	if s == nil {
		return -1, errEmptyNumericSeries
	}
	found := false
	var min T
	idx := -1
	for i := 0; i < s.Len(); i++ {
		v, ok := s.At(i)
		if !ok {
			continue
		}
		if !found || v < min {
			min = v
			idx = i
			found = true
		}
	}
	if !found {
		return -1, errEmptyNumericSeries
	}
	return idx, nil
}

// ArgMax returns the index of the largest non-null value.
func ArgMax[T constraints.Integer | constraints.Float](s *Series[T]) (int, error) {
	if s == nil {
		return -1, errEmptyNumericSeries
	}
	found := false
	var max T
	idx := -1
	for i := 0; i < s.Len(); i++ {
		v, ok := s.At(i)
		if !ok {
			continue
		}
		if !found || v > max {
			max = v
			idx = i
			found = true
		}
	}
	if !found {
		return -1, errEmptyNumericSeries
	}
	return idx, nil
}

// Quantile computes the q-quantile for q in [0,1].
func Quantile[T constraints.Integer | constraints.Float](s *Series[T], q float64) (float64, error) {
	if q < 0 || q > 1 {
		return 0, fmt.Errorf("gods/series: quantile q must be in [0,1], got %v", q)
	}
	values, _ := numericValues(s)
	if len(values) == 0 {
		return 0, errEmptyNumericSeries
	}
	sort.Float64s(values)
	if len(values) == 1 {
		return values[0], nil
	}
	pos := q * float64(len(values)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return values[lo], nil
	}
	frac := pos - float64(lo)
	return values[lo]*(1-frac) + values[hi]*frac, nil
}

// Describe returns a numeric summary map.
func Describe[T constraints.Integer | constraints.Float](s *Series[T]) map[string]float64 {
	out := map[string]float64{
		"count": 0,
		"nulls": 0,
	}
	if s == nil {
		return out
	}
	values, _ := numericValues(s)
	out["count"] = float64(len(values))
	out["nulls"] = float64(s.NullCount())
	if len(values) == 0 {
		return out
	}
	if m, err := Mean(s); err == nil {
		out["mean"] = m
	}
	if med, err := Median(s); err == nil {
		out["median"] = med
	}
	if st, err := Std(s); err == nil {
		out["std"] = st
	}
	if v, err := Var(s); err == nil {
		out["var"] = v
	}
	if min, err := Min(s); err == nil {
		out["min"] = float64(min)
	}
	if max, err := Max(s); err == nil {
		out["max"] = float64(max)
	}
	if p25, err := Quantile(s, 0.25); err == nil {
		out["q25"] = p25
	}
	if p75, err := Quantile(s, 0.75); err == nil {
		out["q75"] = p75
	}
	return out
}
