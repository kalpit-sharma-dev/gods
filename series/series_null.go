package series

// IsNull returns a bool series indicating null positions.
func (s *Series[T]) IsNull() *Series[bool] {
	if s == nil {
		return nil
	}
	out := make([]bool, s.Len())
	for i := range out {
		out[i] = s.validity.IsNull(i)
	}
	return New("is_null", out)
}

// IsValid returns a bool series indicating valid positions.
func (s *Series[T]) IsValid() *Series[bool] {
	if s == nil {
		return nil
	}
	out := make([]bool, s.Len())
	for i := range out {
		out[i] = s.validity.IsValid(i)
	}
	return New("is_valid", out)
}

// FillNa replaces null values with fillValue.
func (s *Series[T]) FillNa(fillValue T) *Series[T] {
	if s == nil {
		return nil
	}
	out := s.cloneWith(s.values, s.validity)
	for i := range out.values {
		if out.validity.IsNull(i) {
			out.values[i] = fillValue
			out.validity.Set(i, true)
		}
	}
	return out
}

// FillNaForward forward-fills null values from the previous valid row.
func (s *Series[T]) FillNaForward() *Series[T] {
	if s == nil {
		return nil
	}
	out := s.cloneWith(s.values, s.validity)
	var last T
	hasLast := false
	for i := range out.values {
		if out.validity.IsValid(i) {
			last = out.values[i]
			hasLast = true
			continue
		}
		if hasLast {
			out.values[i] = last
			out.validity.Set(i, true)
		}
	}
	return out
}

// FillNaBackward backward-fills null values from the next valid row.
func (s *Series[T]) FillNaBackward() *Series[T] {
	if s == nil {
		return nil
	}
	out := s.cloneWith(s.values, s.validity)
	var next T
	hasNext := false
	for i := len(out.values) - 1; i >= 0; i-- {
		if out.validity.IsValid(i) {
			next = out.values[i]
			hasNext = true
			continue
		}
		if hasNext {
			out.values[i] = next
			out.validity.Set(i, true)
		}
	}
	return out
}

// DropNa removes null rows from the series.
func (s *Series[T]) DropNa() *Series[T] {
	if s == nil {
		return nil
	}
	values := make([]T, 0, s.Len()-s.NullCount())
	for i := range s.values {
		if s.validity.IsValid(i) {
			values = append(values, s.values[i])
		}
	}
	return New(s.name, values)
}

// Mask sets entries to null where condition is true.
func (s *Series[T]) Mask(condition *Series[bool]) *Series[T] {
	if s == nil {
		return nil
	}
	out := s.cloneWith(s.values, s.validity)
	if condition == nil {
		return out
	}
	n := s.Len()
	if condition.Len() < n {
		n = condition.Len()
	}
	for i := 0; i < n; i++ {
		v, ok := condition.At(i)
		if ok && v {
			out.validity.Set(i, false)
		}
	}
	return out
}
