package dataframe

import (
	"fmt"
	"sort"
)

// JoinType specifies DataFrame join strategy.
type JoinType string

const (
	// InnerJoin keeps only rows with matching keys in both inputs.
	InnerJoin JoinType = "inner"
	// LeftJoin keeps all rows from left and matching rows from right.
	LeftJoin JoinType = "left"
	// RightJoin keeps all rows from right and matching rows from left.
	RightJoin JoinType = "right"
	// OuterJoin keeps all rows from either side.
	OuterJoin JoinType = "outer"
)

// Join performs a relational join on common key columns.
func Join(left, right *DataFrame, on []string, how JoinType, suffixes [2]string) (*DataFrame, error) {
	if left == nil || right == nil {
		return nil, fmt.Errorf("gods/dataframe: join requires non-nil inputs")
	}
	if len(on) == 0 {
		return nil, fmt.Errorf("gods/dataframe: join requires at least one key column")
	}
	if suffixes[0] == "" && suffixes[1] == "" {
		suffixes = [2]string{"_x", "_y"}
	}
	for _, key := range on {
		if _, err := left.Col(key); err != nil {
			return nil, fmt.Errorf("gods/dataframe: join key %q missing in left: %w", key, err)
		}
		if _, err := right.Col(key); err != nil {
			return nil, fmt.Errorf("gods/dataframe: join key %q missing in right: %w", key, err)
		}
	}

	leftIndex := map[string][]int{}
	for i := 0; i < left.rowLen; i++ {
		key := joinKey(left, i, on)
		encoded := encodeCompositeKey(key)
		leftIndex[encoded] = append(leftIndex[encoded], i)
	}
	rightIndex := map[string][]int{}
	for i := 0; i < right.rowLen; i++ {
		key := joinKey(right, i, on)
		encoded := encodeCompositeKey(key)
		rightIndex[encoded] = append(rightIndex[encoded], i)
	}

	keys := map[string]struct{}{}
	switch how {
	case InnerJoin:
		for k := range leftIndex {
			if _, ok := rightIndex[k]; ok {
				keys[k] = struct{}{}
			}
		}
	case LeftJoin:
		for k := range leftIndex {
			keys[k] = struct{}{}
		}
	case RightJoin:
		for k := range rightIndex {
			keys[k] = struct{}{}
		}
	case OuterJoin:
		for k := range leftIndex {
			keys[k] = struct{}{}
		}
		for k := range rightIndex {
			keys[k] = struct{}{}
		}
	default:
		return nil, fmt.Errorf("gods/dataframe: unsupported join type %q", how)
	}

	keyOrder := make([]string, 0, len(keys))
	keyValues := make(map[string]compositeKey, len(keys))
	for k := range keys {
		keyOrder = append(keyOrder, k)
		if li := leftIndex[k]; len(li) > 0 {
			keyValues[k] = joinKey(left, li[0], on)
			continue
		}
		if ri := rightIndex[k]; len(ri) > 0 {
			keyValues[k] = joinKey(right, ri[0], on)
			continue
		}
		keyValues[k] = compositeKey{}
	}
	sort.Slice(keyOrder, func(i, j int) bool {
		return compareCompositeKey(keyValues[keyOrder[i]], keyValues[keyOrder[j]]) < 0
	})

	rightNameMap := buildJoinedNameMap(right, on, suffixes[1], left.index)
	leftNameMap := buildJoinedNameMap(left, on, suffixes[0], right.index)

	rows := make([]map[string]any, 0)
	for _, k := range keyOrder {
		ls := leftIndex[k]
		rs := rightIndex[k]
		if len(ls) == 0 {
			ls = []int{-1}
		}
		if len(rs) == 0 {
			rs = []int{-1}
		}
		for _, li := range ls {
			for _, ri := range rs {
				if li == -1 && ri == -1 {
					continue
				}
				if how == InnerJoin && (li == -1 || ri == -1) {
					continue
				}
				if how == LeftJoin && li == -1 {
					continue
				}
				if how == RightJoin && ri == -1 {
					continue
				}
				row := map[string]any{}
				fillJoinRow(row, left, li, leftNameMap)
				fillJoinRow(row, right, ri, rightNameMap)
				rows = append(rows, row)
			}
		}
	}

	return dataFrameFromRows(rows)
}

// Concat stacks DataFrames vertically.
func Concat(dfs []*DataFrame, ignoreIndex bool) (*DataFrame, error) {
	_ = ignoreIndex
	if len(dfs) == 0 {
		return Empty(), nil
	}
	colSet := map[string]struct{}{}
	for _, df := range dfs {
		if df == nil {
			continue
		}
		for _, c := range df.Columns() {
			colSet[c] = struct{}{}
		}
	}
	colNames := make([]string, 0, len(colSet))
	for c := range colSet {
		colNames = append(colNames, c)
	}
	sort.Strings(colNames)

	totalRows := 0
	for _, df := range dfs {
		if df == nil {
			continue
		}
		totalRows += df.rowLen
	}
	if totalRows == 0 {
		return Empty(), nil
	}

	built := make([]column, 0, len(colNames))
	for _, name := range colNames {
		values := make([]any, 0, totalRows)
		nullMask := make([]bool, 0, totalRows)
		for _, df := range dfs {
			if df == nil {
				continue
			}
			c, err := df.Col(name)
			if err != nil {
				for i := 0; i < df.rowLen; i++ {
					var zero any
					values = append(values, zero)
					nullMask = append(nullMask, true)
				}
				continue
			}
			for i := 0; i < df.rowLen; i++ {
				v, ok := c.at(i)
				if !ok {
					var zero any
					values = append(values, zero)
					nullMask = append(nullMask, true)
					continue
				}
				values = append(values, v)
				nullMask = append(nullMask, false)
			}
		}
		col, err := inferSeriesColumn(name, values, nullMask)
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: concat infer column %q: %w", name, err)
		}
		built = append(built, col)
	}
	return New(built...)
}

func joinKey(df *DataFrame, row int, on []string) compositeKey {
	values := make([]any, len(on))
	for i, k := range on {
		c, _ := df.Col(k)
		v, ok := c.at(row)
		if !ok {
			values[i] = nil
		} else {
			values[i] = v
		}
	}
	return compositeKey(values)
}

func buildJoinedNameMap(df *DataFrame, on []string, suffix string, other map[string]int) map[string]string {
	keySet := map[string]struct{}{}
	for _, k := range on {
		keySet[k] = struct{}{}
	}
	out := map[string]string{}
	for _, name := range df.Columns() {
		if _, isKey := keySet[name]; isKey {
			out[name] = name
			continue
		}
		if _, exists := other[name]; exists {
			out[name] = name + suffix
			continue
		}
		out[name] = name
	}
	return out
}

func fillJoinRow(dst map[string]any, df *DataFrame, row int, nameMap map[string]string) {
	for _, name := range df.Columns() {
		outName := nameMap[name]
		if _, exists := dst[outName]; exists && row < 0 {
			continue
		}
		if row < 0 {
			if _, exists := dst[outName]; !exists {
				dst[outName] = nil
			}
			continue
		}
		col, _ := df.Col(name)
		v, ok := col.at(row)
		if !ok {
			dst[outName] = nil
			continue
		}
		dst[outName] = v
	}
}
