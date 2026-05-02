package stats

import (
	"math"
	"math/rand"
	"time"
)

// Distribution defines common distribution operations.
type Distribution interface {
	PDF(x float64) float64
	CDF(x float64) float64
	PPF(p float64) float64
	Sample(n int) []float64
	Mean() float64
	Variance() float64
}

type normalDist struct {
	mean float64
	std  float64
}

// Normal creates a normal distribution.
func Normal(mean, std float64) Distribution {
	if std <= 0 {
		std = 1
	}
	return normalDist{mean: mean, std: std}
}

func (d normalDist) PDF(x float64) float64 {
	z := (x - d.mean) / d.std
	return math.Exp(-0.5*z*z) / (d.std * math.Sqrt(2*math.Pi))
}

func (d normalDist) CDF(x float64) float64 {
	z := (x - d.mean) / (d.std * math.Sqrt2)
	return 0.5 * (1 + math.Erf(z))
}

func (d normalDist) PPF(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	lo, hi := d.mean-10*d.std, d.mean+10*d.std
	for i := 0; i < 80; i++ {
		mid := (lo + hi) / 2
		if d.CDF(mid) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

func (d normalDist) Sample(n int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	out := make([]float64, n)
	for i := range out {
		out[i] = d.mean + d.std*r.NormFloat64()
	}
	return out
}

func (d normalDist) Mean() float64     { return d.mean }
func (d normalDist) Variance() float64 { return d.std * d.std }

type uniformDist struct {
	low  float64
	high float64
}

// Uniform creates a uniform distribution.
func Uniform(low, high float64) Distribution {
	if high <= low {
		high = low + 1
	}
	return uniformDist{low: low, high: high}
}

func (d uniformDist) PDF(x float64) float64 {
	if x < d.low || x > d.high {
		return 0
	}
	return 1 / (d.high - d.low)
}

func (d uniformDist) CDF(x float64) float64 {
	if x <= d.low {
		return 0
	}
	if x >= d.high {
		return 1
	}
	return (x - d.low) / (d.high - d.low)
}

func (d uniformDist) PPF(p float64) float64 {
	if p <= 0 {
		return d.low
	}
	if p >= 1 {
		return d.high
	}
	return d.low + p*(d.high-d.low)
}

func (d uniformDist) Sample(n int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	out := make([]float64, n)
	for i := range out {
		out[i] = d.low + r.Float64()*(d.high-d.low)
	}
	return out
}

func (d uniformDist) Mean() float64     { return (d.low + d.high) / 2 }
func (d uniformDist) Variance() float64 { return math.Pow(d.high-d.low, 2) / 12 }

type poissonDist struct{ lambda float64 }

// Poisson creates a Poisson distribution.
func Poisson(lambda float64) Distribution {
	if lambda <= 0 {
		lambda = 1
	}
	return poissonDist{lambda: lambda}
}

func (d poissonDist) PDF(x float64) float64 {
	k := int(math.Round(x))
	if k < 0 {
		return 0
	}
	return math.Exp(-d.lambda) * math.Pow(d.lambda, float64(k)) / gamma(float64(k+1))
}

func (d poissonDist) CDF(x float64) float64 {
	k := int(math.Floor(x))
	if k < 0 {
		return 0
	}
	var sum float64
	for i := 0; i <= k; i++ {
		sum += d.PDF(float64(i))
	}
	if sum > 1 {
		return 1
	}
	return sum
}

func (d poissonDist) PPF(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return math.Inf(1)
	}
	sum := 0.0
	for k := 0; k < 10000; k++ {
		sum += d.PDF(float64(k))
		if sum >= p {
			return float64(k)
		}
	}
	return math.Inf(1)
}

func (d poissonDist) Sample(n int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	out := make([]float64, n)
	for i := range out {
		L := math.Exp(-d.lambda)
		k := 0
		p := 1.0
		for p > L {
			k++
			p *= r.Float64()
		}
		out[i] = float64(k - 1)
	}
	return out
}

func (d poissonDist) Mean() float64     { return d.lambda }
func (d poissonDist) Variance() float64 { return d.lambda }

type exponentialDist struct{ rate float64 }

// Exponential creates an exponential distribution.
func Exponential(rate float64) Distribution {
	if rate <= 0 {
		rate = 1
	}
	return exponentialDist{rate: rate}
}

func (d exponentialDist) PDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return d.rate * math.Exp(-d.rate*x)
}

func (d exponentialDist) CDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1 - math.Exp(-d.rate*x)
}

func (d exponentialDist) PPF(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return math.Inf(1)
	}
	return -math.Log(1-p) / d.rate
}

func (d exponentialDist) Sample(n int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	out := make([]float64, n)
	for i := range out {
		out[i] = d.PPF(r.Float64())
	}
	return out
}

func (d exponentialDist) Mean() float64     { return 1 / d.rate }
func (d exponentialDist) Variance() float64 { return 1 / (d.rate * d.rate) }

type binomialDist struct {
	n int
	p float64
}

// Binomial creates a binomial distribution.
func Binomial(n int, p float64) Distribution {
	if n < 1 {
		n = 1
	}
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	return binomialDist{n: n, p: p}
}

func (d binomialDist) PDF(x float64) float64 {
	k := int(math.Round(x))
	if k < 0 || k > d.n {
		return 0
	}
	return comb(d.n, k) * math.Pow(d.p, float64(k)) * math.Pow(1-d.p, float64(d.n-k))
}

func (d binomialDist) CDF(x float64) float64 {
	k := int(math.Floor(x))
	if k < 0 {
		return 0
	}
	if k >= d.n {
		return 1
	}
	var sum float64
	for i := 0; i <= k; i++ {
		sum += d.PDF(float64(i))
	}
	return sum
}

func (d binomialDist) PPF(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return float64(d.n)
	}
	sum := 0.0
	for k := 0; k <= d.n; k++ {
		sum += d.PDF(float64(k))
		if sum >= p {
			return float64(k)
		}
	}
	return float64(d.n)
}

func (d binomialDist) Sample(n int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	out := make([]float64, n)
	for i := range out {
		c := 0
		for j := 0; j < d.n; j++ {
			if r.Float64() < d.p {
				c++
			}
		}
		out[i] = float64(c)
	}
	return out
}

func (d binomialDist) Mean() float64     { return float64(d.n) * d.p }
func (d binomialDist) Variance() float64 { return float64(d.n) * d.p * (1 - d.p) }

type tDist struct{ df float64 }

// TDist creates a Student's t distribution.
func TDist(df float64) Distribution {
	if df <= 0 {
		df = 1
	}
	return tDist{df: df}
}

func (d tDist) PDF(x float64) float64 {
	num := gamma((d.df + 1) / 2)
	den := math.Sqrt(d.df*math.Pi) * gamma(d.df/2)
	return (num / den) * math.Pow(1+(x*x)/d.df, -(d.df+1)/2)
}

func (d tDist) CDF(x float64) float64 {
	// numeric integration fallback
	lo := -20.0
	hi := x
	if hi < lo {
		lo, hi = hi, lo
	}
	n := 2000
	h := (hi - lo) / float64(n)
	sum := 0.0
	for i := 0; i < n; i++ {
		x0 := lo + float64(i)*h
		x1 := x0 + h
		sum += 0.5 * (d.PDF(x0) + d.PDF(x1)) * h
	}
	if x < -20 {
		return 0
	}
	if x > 20 {
		return 1
	}
	if x >= 0 {
		return 0.5 + sum - (0.5 - d.CDF(0))
	}
	return 0.5 - (d.CDF(0) - sum)
}

func (d tDist) PPF(p float64) float64 {
	return bisectionPPF(d, p, -50, 50)
}

func (d tDist) Sample(n int) []float64 {
	normal := Normal(0, 1)
	chi := ChiSquared(d.df)
	out := make([]float64, n)
	nSamples := normal.Sample(n)
	cSamples := chi.Sample(n)
	for i := range out {
		out[i] = nSamples[i] / math.Sqrt(cSamples[i]/d.df)
	}
	return out
}

func (d tDist) Mean() float64 {
	if d.df > 1 {
		return 0
	}
	return math.NaN()
}

func (d tDist) Variance() float64 {
	if d.df > 2 {
		return d.df / (d.df - 2)
	}
	if d.df > 1 {
		return math.Inf(1)
	}
	return math.NaN()
}

type chiSquaredDist struct{ df float64 }

// ChiSquared creates a chi-squared distribution.
func ChiSquared(df float64) Distribution {
	if df <= 0 {
		df = 1
	}
	return chiSquaredDist{df: df}
}

func (d chiSquaredDist) PDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	k := d.df / 2
	return math.Pow(x, k-1) * math.Exp(-x/2) / (math.Pow(2, k) * gamma(k))
}

func (d chiSquaredDist) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	n := 2000
	h := x / float64(n)
	sum := 0.0
	for i := 0; i < n; i++ {
		x0 := float64(i) * h
		x1 := x0 + h
		sum += 0.5 * (d.PDF(x0) + d.PDF(x1)) * h
	}
	if sum > 1 {
		return 1
	}
	return sum
}

func (d chiSquaredDist) PPF(p float64) float64 {
	return bisectionPPF(d, p, 0, d.df*10+50)
}

func (d chiSquaredDist) Sample(n int) []float64 {
	// Sum of squares of standard normals.
	k := int(math.Round(d.df))
	if k < 1 {
		k = 1
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	out := make([]float64, n)
	for i := range out {
		sum := 0.0
		for j := 0; j < k; j++ {
			z := r.NormFloat64()
			sum += z * z
		}
		out[i] = sum
	}
	return out
}

func (d chiSquaredDist) Mean() float64     { return d.df }
func (d chiSquaredDist) Variance() float64 { return 2 * d.df }

func bisectionPPF(d Distribution, p, lo, hi float64) float64 {
	if p <= 0 {
		return lo
	}
	if p >= 1 {
		return hi
	}
	for i := 0; i < 80; i++ {
		mid := (lo + hi) / 2
		if d.CDF(mid) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

func comb(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}
	res := 1.0
	for i := 1; i <= k; i++ {
		res *= float64(n-k+i) / float64(i)
	}
	return res
}

func gamma(x float64) float64 {
	return math.Gamma(x)
}
