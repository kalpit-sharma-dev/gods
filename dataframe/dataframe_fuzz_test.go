package dataframe

import "testing"

func FuzzEncodeCompositeKeyDeterministic(f *testing.F) {
	f.Add(int64(1), "a", true)
	f.Add(int64(-5), "hello", false)
	f.Fuzz(func(t *testing.T, i int64, s string, b bool) {
		key := compositeKey{i, s, b, nil}
		encodedA := encodeCompositeKey(key)
		encodedB := encodeCompositeKey(key)
		if encodedA != encodedB {
			t.Fatalf("encoding is not deterministic: %q vs %q", encodedA, encodedB)
		}
	})
}

func FuzzJoinKeyStability(f *testing.F) {
	f.Add("x", int64(1))
	f.Add("a||b", int64(999))
	f.Fuzz(func(t *testing.T, s string, n int64) {
		left, err := FromMap(map[string]any{
			"k1": []string{s, s},
			"k2": []int64{n, n + 1},
			"v1": []int64{1, 2},
		})
		if err != nil {
			t.Fatalf("left from map: %v", err)
		}
		right, err := FromMap(map[string]any{
			"k1": []string{s, s},
			"k2": []int64{n, n + 1},
			"v2": []int64{10, 20},
		})
		if err != nil {
			t.Fatalf("right from map: %v", err)
		}
		out, err := Join(left, right, []string{"k1", "k2"}, InnerJoin, [2]string{"_l", "_r"})
		if err != nil {
			t.Fatalf("join: %v", err)
		}
		rows, _ := out.Shape()
		if rows != 2 {
			t.Fatalf("expected 2 rows after stable join, got %d", rows)
		}
	})
}
