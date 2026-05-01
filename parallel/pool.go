package parallel

import "runtime"

// Pool controls parallel worker counts for operations.
type Pool struct {
	workers int
}

// NewPool creates a pool. workers=0 uses runtime.NumCPU().
func NewPool(workers int) *Pool {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	return &Pool{workers: workers}
}

// Workers returns configured worker count.
func (p *Pool) Workers() int {
	if p == nil || p.workers <= 0 {
		return runtime.NumCPU()
	}
	return p.workers
}
