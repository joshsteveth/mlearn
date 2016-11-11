package ml

import (
	"fmt"
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

	fmt.Println(v)
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
