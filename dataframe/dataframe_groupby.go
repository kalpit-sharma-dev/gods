package dataframe

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/yourusername/gods/series"
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
}

// GroupBy groups DataFrame rows by key columns.
func (df *DataFrame) GroupBy(columns ...string) (*GroupBy, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if len(columns) == 0 {
		return nil, fmt.Errorf("gods/dataframe: groupby requires at least one column")
	}
	for _, col := range columns {
		if _, err := df.Col(col); err != nil {
			return nil, err
		}
	}
	groups := make(map[string][]int)
	for i := 0; i < df.rowLen; i++ {
		keyParts := make([]string, len(columns))
		for j, colName := range columns {
			col, _ := df.Col(colName)
			v, ok := col.at(i)
			if !ok {
				keyParts[j] = "<null>"
				continue
			}
			keyParts[j] = fmt.Sprintf("%v", v)
		}
		key := strings.Join(keyParts, "||")
		groups[key] = append(groups[key], i)
	}
	return &GroupBy{df: df, byKeys: append([]string(nil), columns...), groups: groups}, nil
}

// Agg aggregates columns based on the provided specification.
func (g *GroupBy) Agg(spec map[string]AggFunc) (*DataFrame, error) {
	if g == nil || g.df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil groupby")
	}
	if len(spec) == 0 {
		return nil, fmt.Errorf("gods/dataframe: empty aggregation spec")
	}
	groupKeys := make([]string, 0, len(g.groups))
	for k := range g.groups {
		groupKeys = append(groupKeys, k)
	}
	sort.Strings(groupKeys)
	rows := make([]map[string]any, 0, len(groupKeys))
	for _, gk := range groupKeys {
		indices := g.groups[gk]
		row := map[string]any{}
		for i, by := range g.byKeys {
			parts := strings.Split(gk, "||")
			if i < len(parts) {
				p := parts[i]
				if p == "<null>" {
					row[by] = nil
				} else {
					row[by] = p
				}
			}
		}
		for colName, fn := range spec {
			col, err := g.df.Col(colName)
			if err != nil {
				return nil, err
			}
			val, valid := aggregateColumn(col, indices, fn)
			if !valid {
				row[colName] = nil
			} else {
				row[colName] = val
			}
		}
		rows = append(rows, row)
	}
	return dataFrameFromRows(rows)
}

// Apply applies a function to each group and concatenates outputs.
func (g *GroupBy) Apply(fn func(group *DataFrame) (*DataFrame, error)) (*DataFrame, error) {
	if g == nil || g.df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil groupby")
	}
	if fn == nil {
		return nil, fmt.Errorf("gods/dataframe: nil apply function")
	}
	groupKeys := make([]string, 0, len(g.groups))
	for k := range g.groups {
		groupKeys = append(groupKeys, k)
	}
	sort.Strings(groupKeys)
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
	groupKeys := make([]string, 0, len(g.groups))
	for k := range g.groups {
		groupKeys = append(groupKeys, k)
	}
	sort.Strings(groupKeys)
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
	validVals := make([]any, 0, len(indices))
	for _, idx := range indices {
		v, ok := col.at(idx)
		if ok {
			validVals = append(validVals, v)
		}
	}
	switch fn {
	case AggCount:
		return int64(len(validVals)), true
	case AggFirst:
		if len(validVals) == 0 {
			return nil, false
		}
		return validVals[0], true
	case AggLast:
		if len(validVals) == 0 {
			return nil, false
		}
		return validVals[len(validVals)-1], true
	case AggMin:
		if len(validVals) == 0 {
			return nil, false
		}
		min := validVals[0]
		for _, v := range validVals[1:] {
			if compareValues(v, min) < 0 {
				min = v
			}
		}
		return min, true
	case AggMax:
		if len(validVals) == 0 {
			return nil, false
		}
		max := validVals[0]
		for _, v := range validVals[1:] {
			if compareValues(v, max) > 0 {
				max = v
			}
		}
		return max, true
	case AggSum, AggMean, AggStd:
		if len(validVals) == 0 {
			return nil, false
		}
		nums := make([]float64, 0, len(validVals))
		for _, v := range validVals {
			if n, ok := numericToFloat64(v); ok {
				nums = append(nums, n)
				continue
			}
			if s, ok := v.(string); ok {
				n, err := strconv.ParseFloat(s, 64)
				if err == nil {
					nums = append(nums, n)
				}
			}
		}
		if len(nums) == 0 {
			return nil, false
		}
		var sum float64
		for _, n := range nums {
			sum += n
		}
		switch fn {
		case AggSum:
			return sum, true
		case AggMean:
			return sum / float64(len(nums)), true
		case AggStd:
			if len(nums) < 2 {
				return float64(0), true
			}
			mean := sum / float64(len(nums))
			var acc float64
			for _, n := range nums {
				d := n - mean
				acc += d * d
			}
			return math.Sqrt(acc / float64(len(nums)-1)), true
		}
	}
	return nil, false
}
