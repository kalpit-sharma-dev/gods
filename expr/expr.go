package expr

import (
	"regexp"
	"strings"
)

// Expr is a composable boolean predicate over a DataFrame row.
type Expr interface {
	// Eval evaluates the predicate for the provided row.
	Eval(row map[string]any) bool
}

// ColExpr is a column expression builder.
type ColExpr struct{ name string }

// Col starts an expression chain for a named column.
func Col(name string) *ColExpr {
	return &ColExpr{name: name}
}

type predicateExpr struct {
	fn func(row map[string]any) bool
}

func (p predicateExpr) Eval(row map[string]any) bool { return p.fn(row) }

func (c *ColExpr) cmp(val any, f func(a, b any) bool) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			v, ok := row[c.name]
			if !ok {
				return false
			}
			return f(v, val)
		},
	}
}

// Eq checks equality.
func (c *ColExpr) Eq(val any) Expr {
	return c.cmp(val, func(a, b any) bool { return compareAny(a, b) == 0 })
}

// Ne checks inequality.
func (c *ColExpr) Ne(val any) Expr {
	return c.cmp(val, func(a, b any) bool { return compareAny(a, b) != 0 })
}

// Gt checks strictly greater than.
func (c *ColExpr) Gt(val any) Expr {
	return c.cmp(val, func(a, b any) bool { return compareAny(a, b) > 0 })
}

// Gte checks greater than or equal.
func (c *ColExpr) Gte(val any) Expr {
	return c.cmp(val, func(a, b any) bool { return compareAny(a, b) >= 0 })
}

// Lt checks strictly less than.
func (c *ColExpr) Lt(val any) Expr {
	return c.cmp(val, func(a, b any) bool { return compareAny(a, b) < 0 })
}

// Lte checks less than or equal.
func (c *ColExpr) Lte(val any) Expr {
	return c.cmp(val, func(a, b any) bool { return compareAny(a, b) <= 0 })
}

// In checks membership in a value set.
func (c *ColExpr) In(values ...any) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			v, ok := row[c.name]
			if !ok {
				return false
			}
			for _, candidate := range values {
				if compareAny(v, candidate) == 0 {
					return true
				}
			}
			return false
		},
	}
}

// IsNull checks whether the column value is nil.
func (c *ColExpr) IsNull() Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			v, ok := row[c.name]
			return !ok || v == nil
		},
	}
}

// IsNotNull checks whether the column value is non-nil.
func (c *ColExpr) IsNotNull() Expr {
	return Not(c.IsNull())
}

// Contains checks string contains.
func (c *ColExpr) Contains(sub string) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			v, ok := row[c.name]
			if !ok || v == nil {
				return false
			}
			s, ok := v.(string)
			return ok && strings.Contains(s, sub)
		},
	}
}

// StartsWith checks string prefix.
func (c *ColExpr) StartsWith(prefix string) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			v, ok := row[c.name]
			if !ok || v == nil {
				return false
			}
			s, ok := v.(string)
			return ok && strings.HasPrefix(s, prefix)
		},
	}
}

// Matches checks regex match for string values.
func (c *ColExpr) Matches(pattern string) Expr {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return predicateExpr{fn: func(map[string]any) bool { return false }}
	}
	return predicateExpr{
		fn: func(row map[string]any) bool {
			v, ok := row[c.name]
			if !ok || v == nil {
				return false
			}
			s, ok := v.(string)
			return ok && re.MatchString(s)
		},
	}
}

// And combines expressions using logical AND.
func And(exprs ...Expr) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			for _, e := range exprs {
				if e == nil || !e.Eval(row) {
					return false
				}
			}
			return true
		},
	}
}

// Or combines expressions using logical OR.
func Or(exprs ...Expr) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			for _, e := range exprs {
				if e != nil && e.Eval(row) {
					return true
				}
			}
			return false
		},
	}
}

// Not negates an expression.
func Not(e Expr) Expr {
	return predicateExpr{
		fn: func(row map[string]any) bool {
			return e != nil && !e.Eval(row)
		},
	}
}

func compareAny(a, b any) int {
	switch av := a.(type) {
	case int:
		return compareFloat(float64(av), toFloat(b))
	case int8:
		return compareFloat(float64(av), toFloat(b))
	case int16:
		return compareFloat(float64(av), toFloat(b))
	case int32:
		return compareFloat(float64(av), toFloat(b))
	case int64:
		return compareFloat(float64(av), toFloat(b))
	case uint:
		return compareFloat(float64(av), toFloat(b))
	case uint8:
		return compareFloat(float64(av), toFloat(b))
	case uint16:
		return compareFloat(float64(av), toFloat(b))
	case uint32:
		return compareFloat(float64(av), toFloat(b))
	case uint64:
		return compareFloat(float64(av), toFloat(b))
	case float32:
		return compareFloat(float64(av), toFloat(b))
	case float64:
		return compareFloat(av, toFloat(b))
	case string:
		bs, ok := b.(string)
		if !ok {
			return 1
		}
		return strings.Compare(av, bs)
	case bool:
		bb, ok := b.(bool)
		if !ok {
			return 1
		}
		if av == bb {
			return 0
		}
		if !av && bb {
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
		if a == b {
			return 0
		}
		return 1
	}
}

func compareFloat(a, b float64) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

func toFloat(v any) float64 {
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
		return 0
	}
}
