package dataframe

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/kalpit-sharma-dev/gods/series"
)

// AggFunc is an aggregation function identifier.
type AggFunc string

const (
	// AggSum computes group sum.
	AggSum AggFunc = "sum"
	// AggMean computes group mean.
	AggMean AggFunc = "mean"
	// AggMin computes group minimum.
	AggMin AggFunc = "min"
	// AggMax computes group maximum.
	AggMax AggFunc = "max"
	// AggCount computes valid-value count.
	AggCount AggFunc = "count"
	// AggStd computes sample standard deviation.
	AggStd AggFunc = "std"
	// AggFirst takes first valid value.
	AggFirst AggFunc = "first"
	// AggLast takes last valid value.
	AggLast AggFunc = "last"
)

// GroupBy represents grouped DataFrame rows by one or more keys.
type GroupBy struct {
	df     *DataFrame
	byKeys []string
	groups map[string][]int
	keys   map[string]compositeKey
}

// GroupBy groups DataFrame rows by key columns.
func (df *DataFrame) GroupBy(columns ...string) (*GroupBy, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if len(columns) == 0 {
		return nil, fmt.Errorf("gods/dataframe: groupby requires at least one column")
	}
	byCols := make([]column, len(columns))
	for i, col := range columns {
		resolved, err := df.Col(col)
		if err != nil {
			return nil, err
		}
		byCols[i] = resolved
	}
	groups := make(map[string][]int)
	keys := make(map[string]compositeKey)
	for i := 0; i < df.rowLen; i++ {
		keyParts := make(compositeKey, len(columns))
		for j, col := range byCols {
			v, ok := col.at(i)
			if !ok {
				keyParts[j] = nil
				continue
			}
			keyParts[j] = v
		}
		key := encodeCompositeKey(keyParts)
		if _, seen := keys[key]; !seen {
			keys[key] = keyParts
		}
		groups[key] = append(groups[key], i)
	}
	return &GroupBy{
		df:     df,
		byKeys: append([]string(nil), columns...),
		groups: groups,
		keys:   keys,
	}, nil
}

// Agg aggregates columns based on the provided specification.
func (g *GroupBy) Agg(spec map[string]AggFunc) (*DataFrame, error) {
	if g == nil || g.df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil groupby")
	}
	if len(spec) == 0 {
		return nil, fmt.Errorf("gods/dataframe: empty aggregation spec")
	}
	groupKeys := g.orderedGroupKeys()
	byCols := make([]column, len(g.byKeys))
	for i, by := range g.byKeys {
		col, err := g.df.Col(by)
		if err != nil {
			return nil, err
		}
		byCols[i] = col
	}
	specNames := make([]string, 0, len(spec))
	for name := range spec {
		specNames = append(specNames, name)
	}
	sort.Strings(specNames)
	specCols := make([]column, len(specNames))
	specFns := make([]AggFunc, len(specNames))
	for i, colName := range specNames {
		col, err := g.df.Col(colName)
		if err != nil {
			return nil, err
		}
		specCols[i] = col
		specFns[i] = spec[colName]
	}
	totalCols := len(g.byKeys) + len(specNames)
	columnNames := make([]string, 0, totalCols)
	columnNames = append(columnNames, g.byKeys...)
	columnNames = append(columnNames, specNames...)
	columnValues := make([][]any, totalCols)
	columnNulls := make([][]bool, totalCols)
	for i := range columnValues {
		columnValues[i] = make([]any, len(groupKeys))
		columnNulls[i] = make([]bool, len(groupKeys))
	}
	for rowIdx, gk := range groupKeys {
		indices := g.groups[gk]
		if len(indices) > 0 {
			anchor := indices[0]
			for colIdx, col := range byCols {
				v, ok := col.at(anchor)
				if !ok {
					columnNulls[colIdx][rowIdx] = true
					continue
				}
				columnValues[colIdx][rowIdx] = v
			}
		} else {
			for colIdx := range byCols {
				columnNulls[colIdx][rowIdx] = true
			}
		}
		base := len(g.byKeys)
		for i := range specCols {
			val, valid := aggregateColumn(specCols[i], indices, specFns[i])
			if !valid {
				columnNulls[base+i][rowIdx] = true
				continue
			}
			columnValues[base+i][rowIdx] = val
		}
	}
	builtCols := make([]column, 0, totalCols)
	for i, name := range columnNames {
		col, err := inferSeriesColumn(name, columnValues[i], columnNulls[i])
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: infer agg column %q: %w", name, err)
		}
		builtCols = append(builtCols, col)
	}
	return New(builtCols...)
}

// Apply applies a function to each group and concatenates outputs.
func (g *GroupBy) Apply(fn func(group *DataFrame) (*DataFrame, error)) (*DataFrame, error) {
	if g == nil || g.df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil groupby")
	}
	if fn == nil {
		return nil, fmt.Errorf("gods/dataframe: nil apply function")
	}
	groupKeys := g.orderedGroupKeys()
	dfs := make([]*DataFrame, 0, len(groupKeys))
	for _, k := range groupKeys {
		sub, err := g.df.selectRows(g.groups[k])
		if err != nil {
			return nil, err
		}
		out, err := fn(sub)
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: group apply for key %q: %w", k, err)
		}
		dfs = append(dfs, out)
	}
	return Concat(dfs, true)
}

// Size returns counts for each group.
func (g *GroupBy) Size() (*series.Series[int64], error) {
	if g == nil || g.df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil groupby")
	}
	groupKeys := g.orderedGroupKeys()
	values := make([]int64, len(groupKeys))
	for i, k := range groupKeys {
		values[i] = int64(len(g.groups[k]))
	}
	return series.New("size", values), nil
}

// Transform applies a group function and aligns output to original row order.
func (g *GroupBy) Transform(column string, fn func(*DataFrame) any) (*series.Series[any], error) {
	if g == nil || g.df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil groupby")
	}
	if fn == nil {
		return nil, fmt.Errorf("gods/dataframe: nil transform function")
	}
	if _, err := g.df.Col(column); err != nil {
		return nil, err
	}
	values := make([]any, g.df.rowLen)
	nullMask := make([]bool, g.df.rowLen)
	for _, indices := range g.groups {
		sub, err := g.df.selectRows(indices)
		if err != nil {
			return nil, err
		}
		result := fn(sub)
		switch r := result.(type) {
		case *series.Series[any]:
			if r.Len() != len(indices) {
				return nil, fmt.Errorf("gods/dataframe: transform result length %d does not match group length %d", r.Len(), len(indices))
			}
			for i, rowIdx := range indices {
				v, ok := r.At(i)
				if !ok {
					nullMask[rowIdx] = true
					continue
				}
				values[rowIdx] = v
			}
		case []any:
			if len(r) != len(indices) {
				return nil, fmt.Errorf("gods/dataframe: transform result length %d does not match group length %d", len(r), len(indices))
			}
			for i, rowIdx := range indices {
				if r[i] == nil {
					nullMask[rowIdx] = true
					continue
				}
				values[rowIdx] = r[i]
			}
		default:
			for _, rowIdx := range indices {
				if result == nil {
					nullMask[rowIdx] = true
					continue
				}
				values[rowIdx] = result
			}
		}
	}
	return series.WithNulls("transform_"+column, values, nullMask), nil
}

func aggregateColumn(col column, indices []int, fn AggFunc) (any, bool) {
	switch s := col.toSeriesAny().(type) {
	case *series.Series[int64]:
		return aggregateInt64Series(s, indices, fn)
	case *series.Series[float64]:
		return aggregateFloat64Series(s, indices, fn)
	}

	switch fn {
	case AggCount:
		var count int64
		for _, idx := range indices {
			_, ok := col.at(idx)
			if ok {
				count++
			}
		}
		return count, true
	case AggFirst:
		for _, idx := range indices {
			v, ok := col.at(idx)
			if ok {
				return v, true
			}
		}
		return nil, false
	case AggLast:
		for i := len(indices) - 1; i >= 0; i-- {
			v, ok := col.at(indices[i])
			if ok {
				return v, true
			}
		}
		return nil, false
	case AggMin:
		var min any
		found := false
		for _, idx := range indices {
			v, ok := col.at(idx)
			if !ok {
				continue
			}
			if !found || compareValues(v, min) < 0 {
				min = v
				found = true
			}
		}
		if !found {
			return nil, false
		}
		return min, true
	case AggMax:
		var max any
		found := false
		for _, idx := range indices {
			v, ok := col.at(idx)
			if !ok {
				continue
			}
			if !found || compareValues(v, max) > 0 {
				max = v
				found = true
			}
		}
		if !found {
			return nil, false
		}
		return max, true
	case AggSum, AggMean, AggStd:
		var count int
		var sum float64
		var sumSq float64
		for _, idx := range indices {
			v, ok := col.at(idx)
			if !ok {
				continue
			}
			if n, ok := numericToFloat64(v); ok {
				count++
				sum += n
				sumSq += n * n
				continue
			}
			s, ok := v.(string)
			if ok {
				n, err := strconv.ParseFloat(s, 64)
				if err == nil {
					count++
					sum += n
					sumSq += n * n
				}
			}
		}
		if count == 0 {
			return nil, false
		}
		switch fn {
		case AggSum:
			return sum, true
		case AggMean:
			return sum / float64(count), true
		case AggStd:
			if count < 2 {
				return float64(0), true
			}
			// Sample variance using stable one-pass sums.
			variance := (sumSq - (sum*sum)/float64(count)) / float64(count-1)
			if variance < 0 {
				variance = 0
			}
			return math.Sqrt(variance), true
		}
	}
	return nil, false
}

func (g *GroupBy) orderedGroupKeys() []string {
	groupKeys := make([]string, 0, len(g.groups))
	for k := range g.groups {
		groupKeys = append(groupKeys, k)
	}
	sort.Slice(groupKeys, func(i, j int) bool {
		ki, iok := g.keys[groupKeys[i]]
		kj, jok := g.keys[groupKeys[j]]
		if iok && jok {
			if cmp := compareCompositeKey(ki, kj); cmp != 0 {
				return cmp < 0
			}
		}
		return groupKeys[i] < groupKeys[j]
	})
	return groupKeys
}

func aggregateInt64Series(s *series.Series[int64], indices []int, fn AggFunc) (any, bool) {
	var count int
	var first int64
	var last int64
	var min int64
	var max int64
	var sum int64
	var sumF float64
	var sumSq float64
	for _, idx := range indices {
		v, ok := s.At(idx)
		if !ok {
			continue
		}
		if count == 0 {
			first = v
			min = v
			max = v
		}
		last = v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
		fv := float64(v)
		sumF += fv
		sumSq += fv * fv
		count++
	}
	if count == 0 {
		if fn == AggCount {
			return int64(0), true
		}
		return nil, false
	}
	switch fn {
	case AggCount:
		return int64(count), true
	case AggFirst:
		return first, true
	case AggLast:
		return last, true
	case AggMin:
		return min, true
	case AggMax:
		return max, true
	case AggSum:
		return sum, true
	case AggMean:
		return sumF / float64(count), true
	case AggStd:
		if count < 2 {
			return float64(0), true
		}
		variance := (sumSq - (sumF*sumF)/float64(count)) / float64(count-1)
		if variance < 0 {
			variance = 0
		}
		return math.Sqrt(variance), true
	default:
		return nil, false
	}
}

func aggregateFloat64Series(s *series.Series[float64], indices []int, fn AggFunc) (any, bool) {
	var count int
	var first float64
	var last float64
	var min float64
	var max float64
	var sum float64
	var sumSq float64
	for _, idx := range indices {
		v, ok := s.At(idx)
		if !ok {
			continue
		}
		if count == 0 {
			first = v
			min = v
			max = v
		}
		last = v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
		sumSq += v * v
		count++
	}
	if count == 0 {
		if fn == AggCount {
			return int64(0), true
		}
		return nil, false
	}
	switch fn {
	case AggCount:
		return int64(count), true
	case AggFirst:
		return first, true
	case AggLast:
		return last, true
	case AggMin:
		return min, true
	case AggMax:
		return max, true
	case AggSum:
		return sum, true
	case AggMean:
		return sum / float64(count), true
	case AggStd:
		if count < 2 {
			return float64(0), true
		}
		variance := (sumSq - (sum*sum)/float64(count)) / float64(count-1)
		if variance < 0 {
			variance = 0
		}
		return math.Sqrt(variance), true
	default:
		return nil, false
	}
}
