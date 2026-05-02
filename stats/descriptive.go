package stats

import (
	"errors"
	"math"
	"sort"
)

// Mean returns the arithmetic mean.
func Mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	var sum float64
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// Median returns the median.
func Median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	cp := append([]float64(nil), data...)
	sort.Float64s(cp)
	n := len(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

// Mode returns one or more most frequent values.
func Mode(data []float64) []float64 {
	if len(data) == 0 {
		return nil
	}
	freq := map[float64]int{}
	maxCount := 0
	for _, v := range data {
		freq[v]++
		if freq[v] > maxCount {
			maxCount = freq[v]
		}
	}
	out := make([]float64, 0)
	for v, c := range freq {
		if c == maxCount {
			out = append(out, v)
		}
	}
	sort.Float64s(out)
	return out
}

// Variance returns sample variance.
func Variance(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}
	m := Mean(data)
	var acc float64
	for _, v := range data {
		d := v - m
		acc += d * d
	}
	return acc / float64(len(data)-1)
}

// Std returns sample standard deviation.
func Std(data []float64) float64 {
	return math.Sqrt(Variance(data))
}

// Skewness returns moment-based skewness.
func Skewness(data []float64) float64 {
	if len(data) < 3 {
		return 0
	}
	m := Mean(data)
	s := Std(data)
	if s == 0 {
		return 0
	}
	var acc float64
	for _, v := range data {
		z := (v - m) / s
		acc += z * z * z
	}
	return (float64(len(data)) / float64((len(data)-1)*(len(data)-2))) * acc
}

// Kurtosis returns excess kurtosis.
func Kurtosis(data []float64) float64 {
	if len(data) < 4 {
		return 0
	}
	m := Mean(data)
	s2 := Variance(data)
	if s2 == 0 {
		return 0
	}
	n := float64(len(data))
	var acc4 float64
	for _, v := range data {
		d := v - m
		acc4 += d * d * d * d
	}
	return (n*(n+1)*acc4)/(float64((len(data)-1)*(len(data)-2)*(len(data)-3))*s2*s2) -
		(3*float64((len(data)-1)*(len(data)-1)))/float64((len(data)-2)*(len(data)-3))
}

// Covariance returns sample covariance.
func Covariance(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}
	mx := Mean(x)
	my := Mean(y)
	var acc float64
	for i := range x {
		acc += (x[i] - mx) * (y[i] - my)
	}
	return acc / float64(len(x)-1)
}

// Correlation returns Pearson correlation coefficient.
func Correlation(x, y []float64) (float64, error) {
	if len(x) != len(y) || len(x) < 2 {
		return 0, errors.New("gods/stats: correlation requires equal-length slices with at least 2 elements")
	}
	stdX := Std(x)
	stdY := Std(y)
	if stdX == 0 || stdY == 0 {
		return 0, errors.New("gods/stats: correlation undefined for zero-variance data")
	}
	return Covariance(x, y) / (stdX * stdY), nil
}

// Quantile returns q-quantile with linear interpolation.
func Quantile(data []float64, q float64) float64 {
	if len(data) == 0 || q < 0 || q > 1 {
		return 0
	}
	cp := append([]float64(nil), data...)
	sort.Float64s(cp)
	if len(cp) == 1 {
		return cp[0]
	}
	pos := q * float64(len(cp)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return cp[lo]
	}
	w := pos - float64(lo)
	return cp[lo]*(1-w) + cp[hi]*w
}

// IQR returns interquartile range.
func IQR(data []float64) float64 {
	return Quantile(data, 0.75) - Quantile(data, 0.25)
}

// ZScore returns standardized values.
func ZScore(data []float64) []float64 {
	if len(data) == 0 {
		return nil
	}
	m := Mean(data)
	s := Std(data)
	out := make([]float64, len(data))
	if s == 0 {
		return out
	}
	for i, v := range data {
		out[i] = (v - m) / s
	}
	return out
}

// CumSum returns cumulative sum.
func CumSum(data []float64) []float64 {
	out := make([]float64, len(data))
	var acc float64
	for i, v := range data {
		acc += v
		out[i] = acc
	}
	return out
}

// CumProd returns cumulative product.
func CumProd(data []float64) []float64 {
	out := make([]float64, len(data))
	acc := 1.0
	for i, v := range data {
		acc *= v
		out[i] = acc
	}
	return out
}

// RollingMean returns rolling window means.
func RollingMean(data []float64, window int) []float64 {
	if window <= 0 || len(data) == 0 {
		return nil
	}
	out := make([]float64, len(data))
	var sum float64
	for i := range data {
		sum += data[i]
		if i >= window {
			sum -= data[i-window]
		}
		if i+1 >= window {
			out[i] = sum / float64(window)
		}
	}
	return out
}

// RollingStd returns rolling window standard deviations.
func RollingStd(data []float64, window int) []float64 {
	if window <= 1 || len(data) == 0 {
		return nil
	}
	out := make([]float64, len(data))
	for i := range data {
		if i+1 < window {
			continue
		}
		out[i] = Std(data[i-window+1 : i+1])
	}
	return out
}
