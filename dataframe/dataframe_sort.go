package dataframe

import (
	"fmt"
	"math"
	"sort"

	"github.com/kalpit-sharma-dev/gods/series"
)

// SortKey describes one sort level.
type SortKey struct {
	Column    string
	Ascending bool
	NullsLast bool
}

// Sort returns a sorted DataFrame by one or more keys.
func (df *DataFrame) Sort(keys ...SortKey) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if len(keys) == 0 || df.rowLen == 0 {
		return df.Select(df.Columns()...)
	}
	order := make([]int, df.rowLen)
	for i := range order {
		order[i] = i
	}
	for _, key := range keys {
		if _, err := df.Col(key.Column); err != nil {
			return nil, err
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		ri, rj := order[i], order[j]
		for _, key := range keys {
			col, _ := df.Col(key.Column)
			vi, okI := col.at(ri)
			vj, okJ := col.at(rj)
			if okI != okJ {
				if key.NullsLast {
					return okI
				}
				return !okI
			}
			if !okI && !okJ {
				continue
			}
			cmp := compareValues(vi, vj)
			if cmp == 0 {
				continue
			}
			if key.Ascending {
				return cmp < 0
			}
			return cmp > 0
		}
		return ri < rj
	})
	return df.selectRows(order)
}

// Rank returns element ranks for a column.
func (df *DataFrame) Rank(column, method string) (*series.Series[float64], error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	col, err := df.Col(column)
	if err != nil {
		return nil, err
	}
	type pair struct {
		idx   int
		value any
		valid bool
	}
	items := make([]pair, df.rowLen)
	for i := 0; i < df.rowLen; i++ {
		v, ok := col.at(i)
		items[i] = pair{idx: i, value: v, valid: ok}
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].valid != items[j].valid {
			return items[i].valid
		}
		if !items[i].valid {
			return items[i].idx < items[j].idx
		}
		return compareValues(items[i].value, items[j].value) < 0
	})
	ranks := make([]float64, df.rowLen)
	nullMask := make([]bool, df.rowLen)
	for i := 0; i < len(items); {
		if !items[i].valid {
			nullMask[items[i].idx] = true
			i++
			continue
		}
		j := i + 1
		for j < len(items) && items[j].valid && compareValues(items[i].value, items[j].value) == 0 {
			j++
		}
		switch method {
		case "average":
			r := (float64(i+1) + float64(j)) / 2
			for k := i; k < j; k++ {
				ranks[items[k].idx] = r
			}
		case "min":
			r := float64(i + 1)
			for k := i; k < j; k++ {
				ranks[items[k].idx] = r
			}
		case "max":
			r := float64(j)
			for k := i; k < j; k++ {
				ranks[items[k].idx] = r
			}
		case "dense":
			denseRank := 1.0
			if i > 0 {
				prev := i - 1
				for prev >= 0 && !items[prev].valid {
					prev--
				}
				if prev >= 0 {
					denseRank = ranks[items[prev].idx]
					if compareValues(items[prev].value, items[i].value) != 0 {
						denseRank++
					}
				}
			}
			for k := i; k < j; k++ {
				ranks[items[k].idx] = denseRank
			}
		default:
			return nil, fmt.Errorf("gods/dataframe: unsupported rank method %q", method)
		}
		i = j
	}
	return series.WithNulls("rank_"+column, ranks, nullMask), nil
}

func compareValues(a, b any) int {
	if ai, ok := toInt64Exact(a); ok {
		if bi, ok := toInt64Exact(b); ok {
			switch {
			case ai < bi:
				return -1
			case ai > bi:
				return 1
			default:
				return 0
			}
		}
	}
	if au, ok := toUint64Exact(a); ok {
		if bu, ok := toUint64Exact(b); ok {
			switch {
			case au < bu:
				return -1
			case au > bu:
				return 1
			default:
				return 0
			}
		}
	}
	switch av := a.(type) {
	case int:
		return compareFloats(float64(av), toF64(b))
	case int8:
		return compareFloats(float64(av), toF64(b))
	case int16:
		return compareFloats(float64(av), toF64(b))
	case int32:
		return compareFloats(float64(av), toF64(b))
	case int64:
		return compareFloats(float64(av), toF64(b))
	case uint:
		return compareFloats(float64(av), toF64(b))
	case uint8:
		return compareFloats(float64(av), toF64(b))
	case uint16:
		return compareFloats(float64(av), toF64(b))
	case uint32:
		return compareFloats(float64(av), toF64(b))
	case uint64:
		return compareFloats(float64(av), toF64(b))
	case float32:
		return compareFloats(float64(av), toF64(b))
	case float64:
		return compareFloats(av, toF64(b))
	case string:
		bv, ok := b.(string)
		if !ok {
			return 1
		}
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
		return 0
	case bool:
		bv, ok := b.(bool)
		if !ok {
			return 1
		}
		if av == bv {
			return 0
		}
		if !av && bv {
			return -1
		}
		return 1
	default:
		if a == nil && b == nil {
			return 0
		}
		if a == nil {
			return -1
		}
		if b == nil {
			return 1
		}
		if fmt.Sprint(a) == fmt.Sprint(b) {
			return 0
		}
		if fmt.Sprint(a) < fmt.Sprint(b) {
			return -1
		}
		return 1
	}
}

func toInt64Exact(v any) (int64, bool) {
	switch x := v.(type) {
	case int:
		return int64(x), true
	case int8:
		return int64(x), true
	case int16:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	default:
		return 0, false
	}
}

func toUint64Exact(v any) (uint64, bool) {
	switch x := v.(type) {
	case uint:
		return uint64(x), true
	case uint8:
		return uint64(x), true
	case uint16:
		return uint64(x), true
	case uint32:
		return uint64(x), true
	case uint64:
		return x, true
	default:
		return 0, false
	}
}

func toF64(v any) float64 {
	switch x := v.(type) {
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
		return math.NaN()
	}
}

func compareFloats(a, b float64) int {
	if math.IsNaN(a) && math.IsNaN(b) {
		return 0
	}
	if math.IsNaN(a) {
		return -1
	}
	if math.IsNaN(b) {
		return 1
	}
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}
