package util

import "fmt"

// Dtype represents a logical data type for Series and DataFrame columns.
type Dtype int

const (
	// DtypeInt64 is a signed 64-bit integer type.
	DtypeInt64 Dtype = iota
	// DtypeFloat64 is a 64-bit floating-point type.
	DtypeFloat64
	// DtypeBool is a boolean type.
	DtypeBool
	// DtypeString is a UTF-8 string type.
	DtypeString
	// DtypeTime is a timestamp type.
	DtypeTime
	// DtypeAny is a dynamically typed fallback.
	DtypeAny
)

// String returns the human-readable name of the dtype.
func (d Dtype) String() string {
	switch d {
	case DtypeInt64:
		return "int64"
	case DtypeFloat64:
		return "float64"
	case DtypeBool:
		return "bool"
	case DtypeString:
		return "string"
	case DtypeTime:
		return "time"
	case DtypeAny:
		return "any"
	default:
		return fmt.Sprintf("dtype(%d)", d)
	}
}

// IsNumeric reports whether the dtype is numeric.
func (d Dtype) IsNumeric() bool {
	return d == DtypeInt64 || d == DtypeFloat64
}
