package dataframe

import (
	"fmt"

	"github.com/kalpit-sharma-dev/gods/expr"
	"github.com/kalpit-sharma-dev/gods/series"
)

// Row is passed to filter and transformation predicates.
// Callback-based APIs may reuse row storage across iterations for performance.
// Copy the map if values need to be retained after callback returns.
type Row map[string]any

type rowView struct {
	cols []column
	row  Row
}

func newRowView(cols []column) *rowView {
	row := make(Row, len(cols))
	for _, c := range cols {
		row[c.Name()] = nil
	}
	return &rowView{cols: cols, row: row}
}

func (v *rowView) at(i int) Row {
	for _, c := range v.cols {
		val, ok := c.at(i)
		if !ok {
			v.row[c.Name()] = nil
			continue
		}
		v.row[c.Name()] = val
	}
	return v.row
}

// Filter returns rows where predicate returns true.
func (df *DataFrame) Filter(predicate func(Row) bool) *DataFrame {
	if df == nil || predicate == nil {
		return Empty()
	}
	mask := make([]bool, df.rowLen)
	view := newRowView(df.cols)
	for i := 0; i < df.rowLen; i++ {
		mask[i] = predicate(view.at(i))
	}
	return df.filterWithMask(mask)
}

// FilterExpr filters rows using an expression predicate.
func (df *DataFrame) FilterExpr(e expr.Expr) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if e == nil {
		return nil, fmt.Errorf("gods/dataframe: nil expression")
	}
	return df.Filter(func(r Row) bool {
		return e.Eval(map[string]any(r))
	}), nil
}

// BoolMask filters rows using a boolean mask series.
func (df *DataFrame) BoolMask(mask *series.Series[bool]) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if mask == nil {
		return nil, fmt.Errorf("gods/dataframe: nil mask")
	}
	if mask.Len() != df.rowLen {
		return nil, fmt.Errorf("gods/dataframe: mask length %d does not match row count %d", mask.Len(), df.rowLen)
	}
	keep := make([]bool, df.rowLen)
	for i := 0; i < df.rowLen; i++ {
		v, ok := mask.At(i)
		keep[i] = ok && v
	}
	return df.filterWithMask(keep), nil
}

// DropNullRows drops rows with nulls in selected columns.
func (df *DataFrame) DropNullRows(columns ...string) *DataFrame {
	if df == nil {
		return Empty()
	}
	colNames := columns
	if len(colNames) == 0 {
		colNames = df.Columns()
	}
	keep := make([]bool, df.rowLen)
	for i := 0; i < df.rowLen; i++ {
		valid := true
		for _, name := range colNames {
			c, err := df.Col(name)
			if err != nil {
				valid = false
				break
			}
			_, ok := c.at(i)
			if !ok {
				valid = false
				break
			}
		}
		keep[i] = valid
	}
	return df.filterWithMask(keep)
}

// FillNa fills nulls in a column with value.
func (df *DataFrame) FillNa(column string, value any) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	return df.Apply(column, func(v any, ok bool) (any, bool) {
		if ok {
			return v, true
		}
		return value, true
	})
}

// Apply applies fn to every value in the specified column.
func (df *DataFrame) Apply(columnName string, fn func(any, bool) (any, bool)) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if fn == nil {
		return nil, fmt.Errorf("gods/dataframe: nil apply function")
	}
	idx, ok := df.index[columnName]
	if !ok {
		return nil, fmt.Errorf("gods/dataframe: column %q not found", columnName)
	}
	values := make([]any, df.rowLen)
	nullMask := make([]bool, df.rowLen)
	for i := 0; i < df.rowLen; i++ {
		v, valid := df.cols[idx].at(i)
		nv, nvalid := fn(v, valid)
		if !nvalid {
			nullMask[i] = true
			continue
		}
		values[i] = nv
	}
	newCol, err := inferSeriesColumn(columnName, values, nullMask)
	if err != nil {
		return nil, err
	}
	cols := make([]column, len(df.cols))
	for i, c := range df.cols {
		if i == idx {
			cols[i] = newCol
			continue
		}
		cols[i] = c.clone()
	}
	return New(cols...)
}

// Assign creates or replaces a column from a row-wise function.
func (df *DataFrame) Assign(newColName string, fn func(Row) (any, bool)) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if fn == nil {
		return nil, fmt.Errorf("gods/dataframe: nil assign function")
	}
	values := make([]any, df.rowLen)
	nullMask := make([]bool, df.rowLen)
	view := newRowView(df.cols)
	for i := 0; i < df.rowLen; i++ {
		v, ok := fn(view.at(i))
		if !ok {
			nullMask[i] = true
			continue
		}
		values[i] = v
	}
	col, err := inferSeriesColumn(newColName, values, nullMask)
	if err != nil {
		return nil, err
	}
	if _, exists := df.index[newColName]; exists {
		cols := make([]column, len(df.cols))
		for i, existing := range df.cols {
			if existing.Name() == newColName {
				cols[i] = col
				continue
			}
			cols[i] = existing.clone()
		}
		return New(cols...)
	}
	return df.AddColumn(col)
}

// Mutate replaces or creates a column in-place using row-wise function.
func (df *DataFrame) Mutate(name string, fn func(Row) (any, bool)) error {
	if df == nil {
		return fmt.Errorf("gods/dataframe: nil dataframe")
	}
	next, err := df.Assign(name, fn)
	if err != nil {
		return err
	}
	*df = *next
	return nil
}

// Pipe chains DataFrame transformations.
func (df *DataFrame) Pipe(fns ...func(*DataFrame) (*DataFrame, error)) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	current := df
	for i, fn := range fns {
		if fn == nil {
			continue
		}
		next, err := fn(current)
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: pipe step %d: %w", i, err)
		}
		current = next
	}
	return current, nil
}
