package ml

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewZeroVector(t *testing.T) {
	v := NewZeroVector(5)

	assert.Equal(t, 5, len(v.val))
	length := v.GetLength()
	assert.Equal(t, 5, length)
}

func TestSetAndGetSingleValueToVector(t *testing.T) {
	v := NewZeroVector(5)

	_, err := v.GetSingleValue(9)
	assert.Error(t, err)

	err = v.SetSingleValue(9, 10)
	assert.Error(t, err)

	err = v.SetSingleValue(3, 10)
	assert.NoError(t, err)
	assert.Equal(t, float64(10), v.val[2])
	res, err := v.GetSingleValue(3)
	assert.NoError(t, err)
	assert.Equal(t, float64(10), res)
}

func TestSetValue(t *testing.T) {
	v := NewZeroVector(3)

	falseVal := []float64{1, 2, 3, 4}
	val := []float64{1, 2, 3}

	err := v.SetValue(falseVal)
	assert.Error(t, err)

	err = v.SetValue(val)
	assert.NoError(t, err)
	assert.Equal(t, val[2], v.val[2])
}

func TestAddVariableVector(t *testing.T) {
	input := []float64{1, 2, 3}

	v := NewVector(input)

	v.AddVariable(3)
	assert.Equal(t, float64(4), v.getSingleValue(1))
	assert.Equal(t, float64(5), v.getSingleValue(2))
	assert.Equal(t, float64(6), v.getSingleValue(3))
}

func TestMultiplyVariableVector(t *testing.T) {
	input := []float64{1, 2, 3}

	v := NewVector(input)

	v.MultiplyVariable(3)
	assert.Equal(t, float64(3), v.getSingleValue(1))
	assert.Equal(t, float64(6), v.getSingleValue(2))
	assert.Equal(t, float64(9), v.getSingleValue(3))
}

func TestAddVector(t *testing.T) {
	input := []float64{1, 2, 3}
	v := NewVector(input)

	input2 := []float64{1, 2, 3, 4}
	v2 := NewVector(input2)

	err := v.AddVector(v2)
	assert.Error(t, err)

	input3 := []float64{4, 3, 2}
	v3 := NewVector(input3)

	err = v.AddVector(v3)
	assert.NoError(t, err)
	for i := 1; i <= v.GetLength(); i++ {
		assert.Equal(t, float64(5), v.getSingleValue(i))
	}

	fmt.Printf("Result of result vector should be all 5\n%s\n\n", v)
}

func TestDotProduct(t *testing.T) {
	input := []float64{1, 2, 3}
	v := NewVector(input)

	input2 := []float64{1, 2, 3, 4}
	v2 := NewVector(input2)

	_, err := v.DotProduct(v2)
	assert.Error(t, err)

	input3 := []float64{4, 3, 2}
	v3 := NewVector(input3)

	result, err := v.DotProduct(v3)
	assert.NoError(t, err)
	assert.Equal(t, float64(16), result)
}

func TestPowerOfVector(t *testing.T) {
	input := []float64{1, 2, 3}
	v := NewVector(input)

	v.PowerOf(2)
	for i := 1; i <= v.GetLength(); i++ {
		assert.Equal(t, float64(math.Pow(float64(i), 2)), v.getSingleValue(i))
	}
	fmt.Printf("Result of calculated pow vector should be [1,4,9]\n%s\n\n", v)
}

func TestLog10Vector(t *testing.T) {
	input := []float64{1, 10, 100}
	v := NewVector(input)

	v.Log10()
	for i := 1; i <= v.GetLength(); i++ {
		assert.Equal(t, float64(i-1), v.getSingleValue(i))
	}
	fmt.Printf("Result of calculated log10 vector should be [0,1,2]\n%s\n\n", v)
}

func TestLoadNewVector(t *testing.T) {
	file := "data1.csv"

	v, err := LoadNewVector(file, ":", "3")
	assert.NoError(t, err)
	assert.Equal(t, 100, v.GetLength())
}

func TestAddValueVector(t *testing.T) {
	input := []float64{1, 10, 100}
	v := NewVector(input)
	v.AddValue(1000)
	v2 := NewVector([]float64{1, 10, 100, 1000})
	assert.Equal(t, v2, v)
}

func TestMultiplyVector(t *testing.T) {
	v1 := NewVector([]float64{1, 2, 3})
	v2 := NewVector([]float64{3, 2, 1})
	v3 := NewVector([]float64{3, 4, 3})

	err := v1.MultiplyVector(v2)
	assert.NoError(t, err)
	assert.Equal(t, v1.val, v3.val)
}
